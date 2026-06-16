# Section 4: Method — Deep Analysis

## Paper Reference
"Take a Step and Reconsider: Sequence Decoding for Self-Improved Neural Combinatorial Optimization"
Pirnay & Grimm, ECAI-2024
Section 4 (4.1–4.3)

---

## (a) Raw Content Summary

### 4.1 Sequence Decoding as Tree Traversal
- Decoding a solution sequence for a problem instance is modeled as traversing a **search tree**.
- **Root**: empty sequence. **Node at depth d**: partial solution (first d actions). **Children**: assignments to the (d+1)th variable. **Leaf**: complete solution.
- The tree is **maintained in memory**; nodes are expanded on demand.
- Expanding a node queries the **model (policy network)** for transition probabilities.
- Each node stores an **"unnormalized total probability"** `p(\bar a_{1:d})`, initially set to `\pi(\bar a_{1:d})` (the policy-assigned probability of that partial sequence).
- Normalized transition probability:
  ```
  \tilde\pi(a_d | \bar a_{1:d-1}) = p(\bar a_{1:d}) / \sum_{\bar b_{1:d} \in Ch(\bar a_{1:d-1})} p(\bar b_{1:d})
  ```
  This is the child's probability **mass** divided by the **total remaining mass across all siblings**.

### 4.2 Stochastic Beam Search
- Deterministic beam search with width k ranks nodes by total log-probability, yielding k high-probability but **often non-diverse** sequences.
- **Stochastic Beam Search (SBS)** samples k sequences **without replacement** by perturbing log-probabilities with **Gumbel noise**.
- The perturbation satisfies a crucial condition: **the maximum perturbed log-probability of sibling nodes equals their parent's**. This propagates perturbations consistently down the subtree.
- SBS can be **parallelized** like deterministic beam search (GPUs), making it computationally practical.

### 4.3 Take a Step and Reconsider

#### Algorithm
Two hyperparameters: **beam width k** and **step size s** (s <= n, where n is sequence length).

1. Build empty tree. Sample k leaves L = {seq^1, ..., seq^k} **without replacement via SBS** using current `\tilde\pi`.
2. Select the **best trajectory** `b` = seq^i according to objective `f` (to maximize).
3. For each sampled sequence matching the first s steps of `b`:
   - Remove its **probability mass** from its i-th ancestor's `p(...)` for i = s, s+1, ..., n (i.e., from the s-th ancestor onwards).
   - This marks the sequence as **sampled** — it cannot appear again.
4. **Shift the root** to `b`'s first s steps (i.e., commit to those s decisions).
5. **Repeat**: SBS now runs on the updated tree (pi_tilde from which seen sequences' mass has been removed), until the full tree is traversed.
6. Track best solution: update if newly sampled solutions improve upon it.

#### Why This Works
- Removing a leaf's probability mass from all ancestors makes SBS equivalent to **sampling without replacement from pi under the constraint that already-encountered sequences cannot be sampled again**. Originally by Kool et al. and Shi et al.
- Taking **only s steps** from the best solution and then **resampling** forces exploration of unseen alternatives.
- The **remaining sequence length shrinks** progressively as the root advances. This is beneficial because **sampling without replacement is much more potent on shorter sequences** (where the probability mass is less diluted).

#### Simplicity
- Only **two intuitive hyperparameters**: k (how many sequences to consider before committing — more is better) and s (how long to commit before exploring — shorter is better).
- s=1 leads to **MCTS-like search** (commit one step, deep explore). s=n reduces to **simple SBS with width k** (no reconsideration).
- Parameters adjust transparently to computational resources.

#### Inference Coupling
- The decoding method can be coupled with **Top-p (nucleus) sampling**: trim the unreliable tail of `\tilde\pi` at each SBS expansion step.

---

## (b) Key Ideas Unpacked

### Idea 1: Explicit Tree = Explicit Control Over Search State
The tree is not just a conceptual device — it is physically maintained in memory with `p(bar_a_{1:d})` stored at every node. This gives the algorithm **perfect state information** about what has been explored, what remains, and how much probability mass each unexplored branch carries. Contrast this with simple greedy decoding, where the search state is just "the current partial solution."

**Why this matters**: The tree is what enables WOR across multiple SBS rounds. Without it, you cannot track which sequences have been "used up."

### Idea 2: Unnormalized Total Probability as a "Resource" to Deplete
The `p(bar_a_{1:d})` value at each node doubles as:
1. The policy's probability of reaching that node.
2. The "remaining budget" of explorable mass at that node.

When a child is sampled, its mass is subtracted from the parent's total. The **normalization** (`\tilde\pi`) then implicitly redistributes probability among remaining children. This automatically increases the probability of exploring genuinely different alternatives.

### Idea 3: Gumbel-Max WOR Condition
The core of SBS is the condition that the maximum Gumbel-perturbed log-probability among siblings equals the parent's perturbed value. This propagates a **single global random seed** down the tree deterministically, enabling parallel sampling without replacement. The math ensures you get the same k sequences as if you had sampled sequentially without replacement.

### Idea 4: "Commit-and-Reconsider" as a Meta-Strategy
TaSaR's loop is: **commit to s good steps, then search for alternatives you haven't seen**. This is an **alternating optimization** pattern: exploitation via the best-so-far trajectory, exploration via WOR resampling of the suffix. The balance is controlled by s.

### Idea 5: WOR Effectiveness Scales with Sequence Shortening
This is the critical insight: WOR sampling from a discrete distribution over sequences of length L is effective when L is small (each remaining sequence has a detectable probability mass) but near-useless when L is large (every sequence's probability is vanishingly small). By progressively committing to the first s actions, TaSaR reduces L and makes WOR powerful.

---

## (c) Technical Specifics

### The `p(bar_a_{1:d})` variable
- **Initial value**: `\pi(bar_a_{1:d})` — the raw product of policy log-probabilities along the path.
- **After child sampling**: For each sampled child c, `p(sequence_with_c)` is subtracted from `p(bar_a_{1:d})` via `log_subtract` (numerically stable log-space subtraction: `log(exp(x) - exp(y))`).
- **Exhaustion**: When `p(bar_a_{1:d})` becomes `-inf` (in log space), the node is exhausted and cannot produce more samples.

### The trie structure (from code)
- `_TrieNode.children`: list of child nodes (None = not expanded, [] = leaf).
- `_TrieNode.unsampled_log_masses`: numpy array of log-probability mass for each child (unexpanded children get masses set lazily from policy).
- `_TrieNode.children_advantages`: accumulated advantage mass (for Gumbeldore policy improvement).
- `_TrieNode.children_visit_counts`: visit counts (for normalization).
- `mark_mass_sampled(log_mass)`: **recursive** — subtracts log_mass from parent's unsampled_log_masses at the child's index, then recurses up the tree.
- `exhausted()`: returns True when children list is empty or all unsampled_log_masses are -inf.

### SBS batching (from code)
- All internal beam nodes across all problem instances are **flattened** into one list, passed through `child_log_probability_fn` as a batch, then re-split per instance.
- Beam selection uses `np.argpartition` for O(N) top-k selection.
- Gumbel re-sampling uses `sample_gumbels_with_maximum`, which conditions the max sibling Gumbel to equal the parent's Gumbel.

### TaSaR loop mechanics (from code, `perform_tasar`)
- **step_count** cycles 0..s-1. When step_count == 0, fresh SBS is run. Otherwise, no fresh sampling — the algorithm follows the best trajectory found so far.
- After each replan cycle (step_count == 0):
  - The **best leaf** from SBS is identified by sorting beam leaves by `leaf_evaluation_fn` (descending).
  - All freshly sampled leaves have their mass marked as sampled via `mark_mass_sampled`.
  - The **root shifts** by one action (popping the first from `best_leaf_action_seqs_batch`). The trie above the new root is **severed** (`root_node.parent = None`).
  - When a root node becomes exhausted (no children or all mass depleted), it's set to None (done).
- The loop exits when all root nodes are None.

### Top-p in TaSaR vs Incremental SBS
- **TaSaR**: Top-p is held **constant** at `nucleus_top_p` across all replan steps (parameter `min_nucleus_top_p`).
- **Incremental SBS**: Top-p is **linearly annealed** from `min_nucleus_top_p` to 1.0 over rounds.
- This difference reflects the different search dynamics: TaSaR wants consistent exploration at every resampling; incremental SBS wants increasing exploitation.

### Deterministic first round
- Both methods support a `perform_first_round_deterministic` flag. In TaSaR, this is on the `deterministic` parameter. In the code, the first SBS round uses deterministic beam search, then switches to SBS. This ensures the mode of the policy is exploited first.

---

## (d) VNE Implications

### D1. The Search Tree Must Accommodate Variable-Depth Trajectories
In VNE, different virtual networks have different deployment lengths (chain length varies). Also, some decisions (path routing for a processing link) may require multiple sub-steps while others (assigning a source link to a computational link) are single-step. The tree structure naturally accommodates this — leaves can be at different depths. The SBS implementation already handles this.

**Takeaway**: No architectural change needed for VNE's variable trajectory length. The tree naturally handles it.

### D2. s Is the Key Lever for VNE's Large Action Space
VNE has a branching factor far larger than TSP/CVRP/JSSP at most steps (choosing among O(|V|) substrate nodes or O(|E|) edges vs. O(k) for TSP). With a large branching factor, SBS with beam width k samples only a tiny fraction of possible branches. By setting s small (e.g., 2), we commit to s good decisions and then **resample from a fresh perspective**, which is essential when the decision space is vast.

**Takeaway**: For VNE, expect optimal s to be small (likely 2-4) to maximize the benefit of frequent resampling. This matches the planned grid of {2, 4, 8}.

### D3. The WOR-on-short-sequences Insight Is Especially Important for VNE
VNE trajectories are longer than TSP/CVRP (8-15+ actions vs 20 for TSP50, but each VNE action has a larger branching factor). As the root advances, the remaining sequence shrinks from full length to just the last few decisions. The WOR mechanism becomes **dramatically more effective** for these last few decisions, allowing the algorithm to find high-quality completions for the suffix.

**Takeaway**: This is one of the strongest reasons TaSaR may outperform simple SBS for VNE. The last few routing decisions in VNE are where small optimizations matter most, and WOR on short suffixes can find these.

### D4. Feasibility Constraints Are Handled Naturally Through Log-Prob Masking
In VNE, at each step, only a subset of actions are feasible (constraints on CPU capacity, bandwidth, link compatibility). These naturally map to -inf log-probabilities for infeasible actions in the policy's output. The SBS/TaSaR framework automatically respects this — nodes producing -inf probabilities are excluded from the beam.

**Takeaway**: No special feasibility handling needed beyond the trajectory's `log_probability_fn` returning -inf for infeasible actions.

### D5. The Constant Top-p Is Important for VNE
VNE's policy outputs may have uncertain quality in the tail (especially early in training where the policy is weak). The constant Top-p (default 1.0 in the config, i.e., no truncation) means the policy itself determines exploration. For VNE, a slightly lower Top-p (< 1.0) might be beneficial to prune unreliable long-shot branches and focus the beam.

**Takeaway**: The plan uses `min_nucleus_top_p=1.0` (no truncation) by default. Consider lowering this (e.g., 0.9-0.95) for VNE to focus SBS on the more promising part of the action distribution.

### D6. Parallel Beam Evaluation Is Even More Important for VNE
VNE's policy network is larger and slower than per-problem networks for TSP/CVRP/JSSP. SBS evaluates all k beam candidates in parallel (batched). This amortizes the network overhead. The existing code's `candidate_token_budget=8000` caps memory usage which is crucial for VNE's larger action space.

**Takeaway**: The batch parallelization already in place is essential. Ensure `candidate_token_budget` scales with VNE's action space to maintain efficiency.

### D7. The "Probability Mass Removal" Mechanic Prevents Redundant Exploration
In VNE, many different action sequences lead to embedding solutions that differ only in minor ways (e.g., routing a processing link through slightly different paths with the same cost). The mass removal means that once a family of similar solutions has been sampled, its probability mass is gone — the algorithm must find **genuinely different** alternatives.

**Takeaway**: This is critical for VNE, where the number of near-equivalent solutions is large. Without mass removal, SBS might produce 8 near-identical embeddings; with mass removal, it must explore structurally different mappings.

### D8. The Deterministic First Round May Be Less Useful for VNE
The original method runs a deterministic beam search first (to get the mode of the policy), then switches to SBS. For problems where the model is already strong (after supervised pre-training), this makes sense. For VNE with a weaker initial policy, the deterministic mode might lock into a poor region too early.

**Takeaway**: The plan's config has `perform_first_round_deterministic=False`, which is correct for VNE. Keep this unless the supervised pre-training produces a very strong policy.

### D9. Trajectory Length for VNE Chain Formulation
In the extended VNE formulation (chain-style), each virtual network has:
- 1 source link placement (1 action)
- N-1 processing link routings (potentially multi-step path building)
- 1 destination link placement (1 action)

If path routing is a single action (choose the entire path), trajectory length = number of virtual links = virtual_nodes + 1 (for chain). For a 5-node chain: 4 routing actions + 1 source + 1 destination = approximately 6 actions.

If path routing is decomposed into individual edge choices (expanded during traversal), trajectory length = source_link_place + sum(path_lengths) + dest_link_place, which could be 12+ actions for a moderate network.

**Takeaway**: The actual trajectory length depends on the VNE trajectory implementation. This affects the choice of s — if trajectories are short (5-7 actions), s=2-3 is a large fraction; if long (12-20), s=2-4 is more appropriate.

### D10. Gumbeldore Training Loop with TaSaR
The overall training cycle (from `core/train.py::main_train_cycle`):
1. **Supervised pre-training** (current Phase 1): train on ILP-labeled data.
2. **Gumbeldore self-improvement** (Phase 2): each epoch, use TaSaR to search on fresh instances, collect best solutions, train policy on those solutions.

The training signal is: the policy learns to imitate the better solutions found by TaSaR search. This is **expert iteration** / **self-distillation**, not policy gradient.

**Takeaway**: For VNE Phase 2, the quality of TaSaR-generated solutions determines the training signal quality. If TaSaR finds solutions far better than the current policy, the policy receives a strong learning signal. If not, improvement stalls.

---

## (e) Critical Insights for Our Work

### Insight 1: The s Parameter Is Where VNE Will Win or Lose
The step size s controls how aggressively the algorithm alternates between exploitation (following the best solution) and exploration (resampling alternatives). For VNE:
- **Too large s (= n)**: Reduces to simple SBS — no advantage over the baseline.
- **Too small s (= 1)**: Near-MCTS behavior — computationally expensive for complex VNE instances.
- **Optimal s**: Strikes a balance where the model commits to enough decisions to make progress but resamples often enough to escape local optima.

The planned grid (s in {2, 4, 8}) is a reasonable starting point. But **note**: if the VNE trajectory is shorter than 8 steps, s=8 = simple SBS. The fraction s/n matters more than absolute s.

### Insight 2: The Beam Width k Has a Different Meaning in VNE
For TSP (n=20 cities), k=8 means 8 complete tours to compare, which is a significant fraction of the 20! permutation space (in probability terms, not count). For VNE, k=8 means 8 complete embeddings out of an astronomically large space. The beam width affects **diversity** more than **coverage**.

The planned grid (k in {8, 16, 32, 64}) is appropriate. But the marginal benefit of increasing k beyond 32 may be smaller for VNE than for TSP, because the action space is so much larger that even 64 samples are a drop in the ocean. The **quality** of the SBS samples (how diverse the Gumbel perturbations make them) matters more than the count.

### Insight 3: The Policy Network Must Produce Good Log-Probabilities for SBS to Be Effective
SBS works by perturbing log-probabilities. If the policy's conditional probabilities are not well-calibrated (e.g., overconfident on poor actions, underconfident on good ones), the Gumbel perturbations won't help — the beam will still be dominated by noise. This is why **supervised pre-training** (Phase 1) is essential: it gives the policy a reasonable starting point.

**Corollary**: If the supervised policy gives near-uniform probabilities over feasible actions (because the problem is hard and the model is confused), SBS will be essentially random. The quality of the pre-trained policy determines the ceiling of TaSaR improvement.

### Insight 4: The Trie Memory Footprint Scales with VNE Branching Factor
Each expanded node in the trie stores `unsampled_log_masses` (an array of size = number of children), `children_advantages`, and `children_visit_counts`. For VNE steps with large branching factors (choosing one of 30+ substrate nodes), this is memory overhead but not prohibitive (30 floats ~= 240 bytes per expanded node).

The more serious memory concern is the **state cache** (`sbs_child_state_cache`), which stores full trajectory snapshots for each child. With beam width k and trajectory depth d, up to k*d trajectory snapshots are cached. For VNE with large substrate networks, trajectory snapshots contain routing tables and resource assignments — this could be large. The `memory_aggressive` option in the code helps by clearing states after visiting.

**Corollary**: For VNE, consider enabling `memory_aggressive=True` or implementing a custom state-cache eviction policy if memory becomes an issue during training.

### Insight 5: TaSaR's Inference-Time Behavior Is the Actual Test of the Method
The paper emphasizes TaSaR as a **decoding method** — it's used to produce better solutions at inference time, even without retraining. The full training cycle just bootstraps the policy to be better for the next round of TaSaR search.

For Phase 3 (Multi-beam evaluation), the test is: does increasing beam_width at inference time improve solution quality? If yes, the reconsider mechanism is working for VNE. If no (beam_width=1 is as good as beam_width=64), then SBS/TaSaR is not producing diverse useful alternatives for VNE.

### Insight 6: The Gap Metric Must Compare Against Both the ILP Optimum and the Beam-1 Solution
The most informative metric for VNE is:
- **Beam-1 gap** vs ILP: How good is the trained policy's greedy decoding? (= supervised baseline)
- **Beam-k gap** vs ILP: How good is the policy + TaSaR search?
- **Delta**: Beam-1 gap minus Beam-k gap. This is the **value of search**. If delta is large, TaSaR is working for VNE. If delta is small, search is not helping.

The plan's Phase 3 focuses on tracking gap vs beam width, which is exactly right.

---

## (f) Open Questions from This Section

### Q1: What is the optimal s/n ratio for VNE?
The paper establishes that s controls the commit-explore tradeoff. But the optimal s depends on:
- The coupling strength between early and late decisions in the sequence.
- The total sequence length n.
- The reliability of the policy for early decisions.

For VNE, early decisions (virtual node placement) strongly constrain later decisions (virtual link routing). This suggests small s (2-4). But if the supervised policy makes reliable placement decisions, larger s might be better. This should be empirically determined.

### Q2: How does the VNE trajectory's variable-length structure interact with SBS batching?
The SBS implementation batches internal states across instances for the `child_log_probability_fn` call. VNE instances of different sizes (different virtual chain lengths, different substrate sizes) will have internal states with different action spaces. The batching returns variable-length arrays. How does this affect GPU utilization and efficiency?

### Q3: Should the policy network output be calibrated for VNE's larger action space?
The log-probabilities output by the policy network for VNE span many actions (up to 30+ substrate nodes for placement, many paths for routing). Do Gumbel perturbations maintain their WOR guarantees when many actions have near-equal probabilities? The Gumbel-max trick relies on well-separated Gumbel samples — with very many near-equal sibling probabilities, the order statistics of the Gumbels determine selection almost randomly.

### Q4: Is there a role for the Gumbeldore advantage mechanism alongside TaSaR's search?
The current VNE plan uses TaSaR for search but not the Gumbeldore advantage-based policy updates (the `log_prob_update_type` in `perform_incremental_sbs`). The pure TaSaR approach removes mass and relies on the policy to produce good conditional probabilities. Would combining TaSaR's tree search with Gumbeldore's policy improvement (via advantage accumulation in the trie) give better results for VNE?

### Q5: How does the memory_aggressive flag affect VNE performance?
The code offers a `memory_aggressive` mode that clears cached states after visiting nodes. For VNE's larger state representations, this could be a necessary memory saver. But it requires re-transitioning to states, adding computational overhead. What's the right tradeoff for VNE?

### Q6: Does Top-p truncation help or hurt for VNE?
The current config sets `min_nucleus_top_p=1.0` (no truncation). The paper discusses Top-p as an inference-time technique for "trimming the unreliable tail." For VNE's large action space, many actions might have small but non-zero probability. Top-p < 1.0 would prune these, potentially focusing the beam. But if the policy is well-calibrated, pruning the tail might lose valuable diversity. This needs empirical testing.

### Q7: What is the minimum beam width for meaningful diversity in VNE?
For TSP/CVRP, beam width 4-8 gives meaningful diversity. For VNE with its larger action space, how many SBS samples are needed before the beam contains at least a few structurally different embeddings? If the answer is "more than 64," the planned grid won't find the benefit.

### Q8: How should the trajectory handle multi-step path routing?
If a single "action" in the VNE trajectory is a complete path (source to destination), the branching factor is enormous (all paths in the substrate network). If path construction is decomposed into individual edge choices, the trajectory is much longer but branching is manageable. Which decomposition is better for TaSaR? The paper doesn't discuss variable-granularity actions.

---

## Connection to Codebase Implementation

The code in `core/incremental_sbs.py::IncrementalSBS.perform_tasar` closely implements the paper's description:
- Lines 265-367: The main TaSaR loop with step_count, SBS calls, best leaf tracking, mass removal, and root shifting.
- The trie operations (`mark_mass_sampled`, `exhausted`) implement the probability mass bookkeeping.
- SBS (`sbs.stochastic_beam_search`) implements the Gumbel-perturbed WOR sampling.

Key difference between paper text and code:
- The paper says "shift root to best's first s steps." The code shifts by **1 step per iteration** (line 299: `step_count = (step_count + 1) % replan_steps`), and the SBS + root-shift happens every s iterations. This is an implementation detail: the "shift by s" happens cumulatively over s iterations of the outer loop, with fresh SBS every s-th iteration and no fresh sampling in between. This is **equivalent** to the paper's description but allows checking intermediate leaf results.

The config (`vne/config.py` lines 119-160) maps paper parameters to code:
- `beam_width` = k
- `replan_steps` = s
- `min_nucleus_top_p` = Top-p threshold (constant for TaSaR)
- `perform_first_round_deterministic` = whether first SBS is deterministic
