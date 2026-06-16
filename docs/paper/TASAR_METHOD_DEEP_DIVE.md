# TaSaR Method Deep Dive: Definitive Reference for VNE Adaptation

> **Purpose:** This document is the single source of truth for adapting the "Take a Step and Reconsider" (TaSaR) sequence decoding method to Virtual Network Embedding (VNE). Every future Claude Code session should read this first before writing any VNE+TaSaR code. All claims are grounded in the actual codebase at `/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider` and the published TaSaR paper (ECAI-2024, DOI: 10.3233/FAIA240707).

---

## Table of Contents

1. [The Algorithm: Complete Explanation of TaSaR](#1-the-algorithm)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [VNE Adaptation Blueprint](#3-vne-adaptation-blueprint)
4. [Hyperparameter Strategy for VNE](#4-hyperparameter-strategy-for-vne)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Key Insights for Beating State-of-the-Art](#6-key-insights-for-beating-state-of-the-art)

---

## 1. The Algorithm

### 1.1 What TaSaR Is and Why It Exists

TaSaR is a **decoding method** for neural combinatorial optimization (NCO). It sits inside a **self-improved learning (SIL)** loop:

```
for epoch in 1..num_epochs:
    1. Generate instances                               (problem-specific)
    2. Decode solutions using current best policy       (THIS IS WHERE TaSaR RUNS)
    3. Keep best solution per instance                  (pseudo-label)
    4. Train policy via cross-entropy on pseudo-labels  (supervised learning)
    5. Update leader if validation improves
```

The decoding step (step 2) is the critical efficiency lever: it determines whether the SIL cycle is virtuous (policy improves, generates better labels, trains better policy) or vicious (policy stalls and generates uniformly poor labels).

TaSaR's key insight: **existing decoding methods fail the SIL decoding triple requirement** -- they must be (a) fast enough to generate thousands of instances per epoch, (b) diverse enough to explore alternative solutions, and (c) scalable to long planning horizons. Pure beam search is fast but not diverse. Monte Carlo sampling is diverse but sample-inefficient and correlated. MCTS is too slow. WOR sampling's diversity advantage degrades as solution length grows.

TaSaR solves this with a **commit-explore-reconsider** cycle that combines the speed of beam search with structured diversity enforcement.

### 1.2 Stochastic Beam Search: The Foundation

Stochastic Beam Search (SBS) is the building block of TaSaR. From `core/stochastic_beam_search.py`:

```python
def stochastic_beam_search(
    child_log_probability_fn,   # policy network: states -> log-probs over actions
    child_transition_fn,        # (state, action) -> (next_state, is_leaf)
    root_states,                # one initial state per problem instance
    beam_width: k,              # number of leaves to sample
    deterministic: False,       # if True = regular greedy beam search
    top_p: 1.0,                 # nucleus sampling threshold
) -> List[List[BeamLeaf]]:      # k leaves per instance
```

**How SBS works step-by-step:**

1. **Initialization:** Start with one root node per instance, carrying log-probability 0.0 and Gumbel value 0.0.

2. **Expand the beam:** For each internal node on the beam, get log-probabilities for all child actions from the policy network. Apply Top-p filtering if active. Compute the **total log-probability** of each candidate: `total_log_prob = child_log_prob + node_log_prob`.

3. **Sample Gumbels:** For SBS, for each candidate, sample a Gumbel perturbation `g ~ Gumbel(total_log_prob, 1)` conditioned on the parent's Gumbel maximum. This is the core trick (see Section 2): `sample_gumbels_with_maximum(log_probabilities, parent_max_gumbel)` ensures the perturbed values are consistent across tree depth. For deterministic mode, `g = total_log_prob` (no perturbation).

4. **Select top-k candidates:** The Gumbel-perturbed values from all expansions plus any already-reached leaf nodes are pooled. The top-k by Gumbel value are selected using `np.argpartition` in O(n) time.

5. **Transition and repeat:** Selected candidates are transitioned via `child_transition_fn`. Leaves are stored; internal nodes are expanded again. Repeat until all states on the beam are leaf nodes.

6. **Return sorted:** Leaves are returned sorted by decreasing Gumbel value, which is equivalent to the order obtained by sampling without replacement one-at-a-time.

### 1.3 The Trie and Probability Mass Removal

The search tree is stored as a **trie of `_TrieNode` objects** (defined in `core/incremental_sbs.py`). Each node maintains:

- `unsampled_log_masses`: numpy array of log-probabilities for each child, initially equal to the policy's output + parent's mass. **These are the residual probability mass that has not been sampled yet**.
- `children`: list of child `_TrieNode` objects.
- `children_advantages`: used by original Gumbeldore, **ignored in TaSaR**.
- `sbs_child_state_cache`: cached (state, is_leaf) tuples for efficient re-traversal.

**The critical operation: `mark_mass_sampled`**

```python
def mark_mass_sampled(self, log_mass: float) -> None:
    """Recursively subtract log_mass from this node and its ancestors."""
    if not self.parent:  # is the root
        return
    if self.exhausted():
        new_log_mass = -np.inf
    else:
        new_log_mass = log_subtract(
            self.parent.unsampled_log_masses[self.index_in_parent],
            log_mass
        )
    self.parent.unsampled_log_masses[self.index_in_parent] = new_log_mass
    self.parent.mark_mass_sampled(log_mass)
```

When a leaf is sampled, its mass is subtracted from all ancestor nodes using `log_subtract(x, y) = log(exp(x) - exp(y))`. This ensures that **subsequent SBS rounds cannot pick the same trajectory again** -- the probability mass for that trajectory has been removed from the tree. A node whose `unsampled_log_masses` are all `-inf` is marked as `exhausted()` and produces no more samples.

In the `wrap_child_log_probability_fn`, this is materialized: when computing child probabilities for a node, the wrapper normalizes the **remaining** unsampled masses:

```python
# Re-normalize the remaining masses to create a conditional distribution
unnormalized = np.exp(log_unnormalized - np.max(log_unnormalized))
results[i] = np.log(unnormalized / np.sum(unnormalized))
```

### 1.4 The TaSaR Loop: Step-and-Reconsider

From `core/incremental_sbs.py`, `IncrementalSBS.perform_tasar`:

```
Algorithm: perform_tasar(beam_width=k, replan_steps=s, nucleus_top_p=p)

Input: root_nodes (one per instance)
Output: best_leaf for each instance

1. Initialize best_leaf = None for each root
2. Initialize best_action_sequence = [] for each root
3. step_count = -1

4. Loop until all roots are exhausted or None:
   a. step_count += 1  (modulo replan_steps)

   b. If step_count == 0:                       // TIME TO RECONSIDER
      - Run full SBS from current roots with beam_width=k and top_p=p
      - deterministic = False (after first SBS, which may be deterministic)

   c. Else:                                     // STILL COMMITTED
      - No fresh SBS (sbs_leaves_batch = [None] * len(roots))

   d. For each root that is still active:
      i.   If we have fresh SBS leaves:
           - Find the best leaf by evaluation function
           - If it's better than best_leaf so far, update best_leaf and
             best_action_sequence (the action indices from root to that leaf)

      ii.  Get the next action from best_action_sequence (pop front):
           root_action = best_action_sequence.pop(0)

      iii. If we had fresh SBS leaves:
           - Mark all freshly sampled leaves' mass as sampled via
             mark_mass_sampled (so they can't be re-sampled)

      iv.  Shift the root: follow root_action one step
           - New root = child of current root after taking root_action
           - Detach parent: new_root.parent = None  (trie above is gone)

      v.   If new root is exhausted or is a leaf:
           - Set root to None (done with this instance)
           Else:
           - Set root to (new_root, new_state)
5. Return [best_leaf] for each instance
```

**Key design choices visible in the code:**

- **Ancestors tracking is disabled:** `node.ancestors = None` is set at startup. The TaSaR mode does not need to propagate advantages up the tree (unlike Gumbeldore's `add_simple_advantage` or `obtain_locally_estimated_advantage`), which is a major simplification.

- **Mass is subtracted only once per round:** All SBS leaves from the same round collectively have their mass removed. The best leaf is tracked separately and not affected by mass removal.

- **The root shifts by EXACTLY one action per iteration** (the first action of the best trajectory). If `replan_steps > 1`, subsequent iterations just follow the best trajectory without calling SBS again.

- **Fresh SBS runs every `replan_steps` iterations** from the (shifted) root, generating a fresh set of `k` completions from that intermediate state.

- **Deterministic first round:** If `deterministic=True`, the very first SBS round is standard beam search (no Gumbel noise, `top_p=1`). This exploits the policy's mode at the start, then switches to stochastic sampling for all subsequent rounds.

### 1.5 How TaSaR Differs from Original Gumbeldore

The original `perform_incremental_sbs` method (same class) is **strictly more complex**:

| Aspect | TaSaR (perform_tasar) | Gumbeldore (perform_incremental_sbs) |
|--------|----------------------|--------------------------------------|
| Rounds | Implicit: one SBS per replan interval | Explicit `num_rounds` parameter |
| Mass update | Only mark_mass_sampled (WOR) | WOR + advantage-based logit adjustment |
| Advantage tracking | Not used | `add_simple_advantage` + `obtain_locally_estimated_advantage` |
| Normalization | Not needed | Advantage normalization (min-max, visit count) |
| Top-p | Constant for all rounds | Annealed linearly from `min_nucleus_top_p` to 1.0 |
| Parameters | 3: k, s, top-p | 9+: beam_width, num_rounds, advantage_constant, various normalization flags |
| Root | Shifts progressively along best trajectory | Stays at original root (exhausts the full tree) |
| Complexity | O(k * l/s * t_transition) | O(num_rounds * k * l * t_transition) |

**Why TaSaR is simpler:**

1. **No advantage estimation:** TaSaR does not compute expected outcomes, importance weights, or advantage values. It simply samples without replacement and commits to the best solution found so far. The `children_advantages` array on `_TrieNode` is **never touched** during TaSaR runs.

2. **No round-based Top-p annealing:** TaSaR uses a constant `nucleus_top_p` throughout, set via `gumbeldore_config["min_nucleus_top_p"]` (confusingly named -- in TaSaR it is constant, not a minimum).

3. **No policy gradient:** TaSaR does not modify the original policy log-probs. It only removes sampled mass. The policy network's parameters change only through the supervised training step (cross-entropy on pseudo-labels), not through any advantage-weighted update.

4. **Root shifting:** By moving the root forward, TaSaR progressively shortens the remaining sequence, making WOR sampling exponentially more effective on the remaining steps.

**What is inherited from the codebase (unchanged):**

- The `_TrieNode` trie structure with `unsampled_log_masses`
- `log_subtract` for numerical stable probability mass removal
- `wrap_child_log_probability_fn` (the normalization wrapper) -- but without the advantage addition branch
- `wrap_child_transition_fn` for state caching

---

## 2. Mathematical Foundations

### 2.1 Policy Formulation

The policy network models an autoregressive conditional distribution:

```
pi_theta(a_{1:d}) = prod_{i=1}^d pi_theta(a_i | a_{1:i-1})
```

where:
- `a_{1:d}` is a partial solution of length d
- `pi_theta(a_i | a_{1:i-1})` is the conditional probability of action a_i given the partial solution so far
- The full solution is `a_{1:n}` where n is the sequence length

**In the codebase:** The `BaseTrajectory` abstract class (from `core/abstracts.py`) requires every problem to implement:

```python
@staticmethod
def log_probability_fn(trajectories, network, to_numpy):
    """Returns log pi_theta(a_i | a_{1:i-1}) for each trajectory's next step."""
```

For VNE: This returns log-softmax over all feasible candidate substrate paths for the current virtual link. Each trajectory has a different number of feasible candidates (variable-size action space).

### 2.2 Unnormalized Total Probability and Normalized Transition

The search tree stores **unnormalized total probability** at each node:

```
p(a_{1:d}) = pi_theta(a_1) * pi_theta(a_2 | a_1) * ... * pi_theta(a_d | a_{1:d-1})
              = exp(log pi_theta(a_1) + log pi_theta(a_2 | a_1) + ...)
```

In log space: `log p(child) = log p(parent) + log pi_theta(child_action | parent_state)`

This is stored in `unsampled_log_masses` on each `_TrieNode`. For the root, `log p(root) = 0.0`. For each child, `unsampled_log_masses[child_index] = log p(child)`.

**The normalized conditional probability** used during tree expansion:

```
pi_tilde(a_d | a_{1:d-1}) = exp(log p(child) - logsumexp(children's log p values))
```

This is computed by the wrapper in `wrap_child_log_probability_fn`:

```python
unnormalized = np.exp(log_unnormalized - np.max(log_unnormalized))
results[i] = np.log(unnormalized / np.sum(unnormalized))
```

After mass removal, this normalization automatically accounts for the removed mass: the conditional distribution is re-normalized over only the **remaining** (unsampled) probability mass at each node.

### 2.3 Gumbel-Top-k Trick and WOR Sampling

The SBS algorithm uses the **Gumbel-Top-k trick** to sample k items without replacement from a discrete distribution:

For a set of items with log-probabilities `log_p_i`, sample i.i.d. Gumbel(0,1) noise `g_i`, then compute `G_i = log_p_i + g_i`. The top-k items by `G_i` are a sample without replacement from the distribution.

**Consistency across tree depth:** The key insight in the SBS paper (Kool et al., 2019) is that the Gumbel perturbations must be **consistent**: the maximum perturbed log-probability among a node's children must equal the parent's perturbed log-probability. This is achieved by `sample_gumbels_with_maximum`:

```python
def sample_gumbels_with_maximum(log_probabilities, target_max):
    """
    Sample Gumbels conditioned on a given maximum.
    target_max = parent's Gumbel value (perturbed log-prob)
    """
    gumbels = np.random.gumbel(loc=log_probabilities)
    max_gumbel = np.max(gumbels)

    # Shift so that max(gumbels) = target_max, preserving ordering
    v = target_max - gumbels + np.log1p(-np.exp(gumbels - max_gumbel))
    ret_gumbels = target_max - np.maximum(v, 0) - np.log1p(np.exp(-np.abs(v)))
    return ret_gumbels
```

This ensures that `max(sample_gumbels_with_maximum(log_probs, parent_gumbel)) = parent_gumbel`, making the tree-wide perturbation consistent.

### 2.4 The WOR Sampling Condition via Mass Removal

The probability mass removal mechanism ensures that after sampling a leaf with probability mass `exp(log_mass)`, the remaining mass at each ancestor node is reduced:

```
remaining_log_mass = log_subtract(original_log_mass, sampled_log_mass)
```

This is equivalent to conditioning on NOT having sampled that leaf. After k rounds of SBS (in TaSaR's case, k = beam_width leaves per SBS run), the remaining distribution at each node is:

```
p_remaining(a) = p_original(a) / (1 - p_sampled)
```

where `p_sampled` is the total probability mass of all already-sampled trajectories. This ensures subsequent SBS rounds sample from a valid conditional distribution (sampling without replacement from the original policy).

**The `log_subtract` implementation:**

```python
def log_subtract(x, y):
    """Returns log(exp(x) - exp(y)), or -inf if x <= y."""
    return x + np.log1p(-np.exp(np.minimum(y - x, 0)))
```

This is numerically stable: when `y >> x`, it returns `-inf`. When `y << x`, `log1p(-exp(y-x))` is accurate. Only the regime where `exp(y - x)` is close to 1 loses precision, but in that regime the result should be very negative (the remaining mass is tiny), which is fine.

### 2.5 Cost Complexity

The paper defines `g(k, s)` as the number of node transitions evaluated for one TaSaR decoding pass:

```
g(k, s) = k * (t*l - (s*t^2 - s*t)/2)
```

where:
- `l` = solution sequence length (number of actions)
- `k` = beam width
- `s` = replan_steps
- `t = ceil(l / s)` = number of replan rounds

For the equivalent greedy or WOR comparison:

- **Greedy** (k=1): `g(1, l) = l` transitions (just follow one trajectory)
- **WOR** with beam k: `g(k, l) = k * l` transitions (k full independent trajectories)

TaSaR's computational cost relative to WOR at same k:

```
ratio = g(k,s) / (k*l) = t/l * (l - (s*t - s)/2) ≈ 1/s  (for l >> s)
```

This is the efficiency argument: by committing for s steps and only resampling every s-th step, TaSaR is approximately `1/s` the cost of running k independent trajectories.

**Numerical example for VNE:**
- `l = 12` (typical: 3 requests * 4 links each)
- `k = 64, s = 2`
- `t = ceil(12/2) = 6`
- `g(64, 2) = 64 * (6*12 - (2*36 - 2*6)/2) = 64 * (72 - 30) = 64 * 42 = 2688`
- WOR at k=64: `64 * 12 = 768` (cheaper per-round!)
- Wait -- this means TaSaR is ~3.5x MORE expensive for short sequences. However, TaSaR is designed for long sequences where WOR converges poorly.

**For long sequences (the paper's regime):**
- `l = 100, k = 64, s = 10`
- `t = ceil(100/10) = 10`
- `g(64, 10) = 64 * (10*100 - (10*100 - 10*10)/2) = 64 * (1000 - 450) = 35,200`
- WOR at k=64: `64 * 100 = 6,400`
- TaSaR is 5.5x more expensive per round.

**Important caveat for VNE:** The VNE trajectory length (`l`) is SMALL (maybe 6-12 actions), which changes the cost calculus. TaSaR may be more expensive per round for short sequences, but the **diversity benefit still compounds** across training epochs. The cost formula matters for wall-clock budget planning but should NOT be the primary concern -- the quality of the pseudo-labels is what drives the self-improvement cycle.

---

## 3. VNE Adaptation Blueprint

### 3.1 MDP Mapping

The combinatorial optimization as a Markov Decision Process (MDP) maps to VNE as follows:

| MDP Component | VNE Mapping | Current Implementation |
|---------------|-------------|----------------------|
| **State** | (substrate residual resources, current partial embedding of VN requests) | `Trajectory` dataclass (`vne/trajectory.py`) with `residual_bandwidth`, `residual_compute`, `current_request_idx`, `current_link_idx`, `action_candidates` |
| **Action** | Choose a feasible substrate path for the current virtual link | Selecting `action_candidates[action_index]` from the ordered list of feasible paths |
| **Transition** | Reserve bandwidth along chosen path, consume compute at endpoints, advance to next link/request | `transition_fn(action_index)` -- deep-copies trajectory, applies path reservation, computes next candidates |
| **Reward** | (Negative) embedding cost, collected only at terminal state | `to_max_evaluation_fn()` returns `-cost` (negative objective for maximization) |
| **Terminal** | All virtual links of all requests have been embedded, or no feasible path remains for the current link | `is_finished = True` when `next_request_idx` is `None`, or `action_candidates` is empty |
| **Constraint** | Bandwidth capacity on comm edges, compute capacity on comp links | Enforced during `_candidate_paths` computation -- paths violating residual capacity are excluded |

### 3.2 What "Actions" Mean in VNE

In TSP, an action is simple: choose the next city to visit from remaining cities. The action space at step d is `n - d` (polynomial, well-structured).

In VNE, an action is **structured and combinatorial**: choose a **substrate path** for the current virtual link. The path is a sequence of communication nodes defining the start, intermediate hops, and end. The feasibility depends on:

1. **Start node** (F_0 placement): Must be a communication node with an attached compute node having sufficient residual capacity for `source_link_demand` (if this is the first link of a request).

2. **End node** (F_{k-1} placement): Must be a communication node with an attached compute node having sufficient residual capacity for `destination_link_demand` (if this is the last link of a request).

3. **All edges on the path:** Must have sufficient residual bandwidth for the `processing_link_demands`.

4. **Path connectivity:** Must follow substrate edge direction.

5. **Chain continuity:** For links after the first in a request, the path must start at the end node of the previous link.

**How this translates to the trajectory:**

In `Trajectory._candidate_paths` (lines 121-192 of `vne/trajectory.py`), action candidates are computed by enumerating all feasible substrate paths for the current `(request_idx, link_idx)` pair. This is a DFS enumeration with pruning:

1. Filter start nodes by compute capacity (for first/last link).
2. For each start node, DFS along directed edges, pruning paths where intermediate edges lack bandwidth.
3. Check destination compute capacity (for last link).
4. **Key optimization:** `check_future_completion` recursively checks that at least one feasible completion exists for all remaining links, so dead-end actions are pruned before becoming candidates.

**Action indexing:** Actions are indexed (0, 1, ..., num_candidates-1) in the order `_candidate_paths` returns them. The policy network outputs logits for each candidate index. This is compatible with SBS, which only needs to know `num_actions()` and the log-probability distribution.

### 3.3 Handling VNE's Larger Action Space

VNE's action space is astronomically larger than TSP's:

- **TSP at step d:** `n - d` actions (at most 99 for n=100)
- **CVRP at step d:** varies but typically ~50-100 actions depending on remaining nodes
- **VNE at step d:** Up to `|V_comm| * |paths|` where paths can be exponential in the number of comm nodes. In practice, the DFS enumeration in `_candidate_paths` bounds this, but instances with 20-40 comm nodes and dense connectivity can produce hundreds to thousands of candidate paths.

**How the codebase handles this:**

1. **Candidate bucketing in the network forward:** The `VNEPolicyNetwork.forward` method (`vne/network.py`, line 304) groups states by candidate count and limits the `group_size * max_candidates` product to `candidate_token_budget = 8000`. This prevents GPU OOM from one instance with 5000 candidates padding the whole batch.

2. **The network output is always per-candidate logits:** The policy decoder attends to a sequence of `[global_context, current_virtual, candidate_1, ..., candidate_n]` and produces one logit per candidate. Complexity is O(candidates^2) due to self-attention in the decoder, but the candidate budget caps this.

3. **SBS is action-index-based:** SBS never inspects candidate content -- it only sees the log-probability vector. The index-based action in `transition_fn` handles all candidate complexity.

**Implications for TaSaR on VNE:**

- **Top-p filtering is more important:** With hundreds of candidates, many will have near-zero probability. Top-p truncation prevents the beam from wasting budget on near-uniform noise from the tail.
- **Beam width vs action space:** k=8 spans a tiny fraction of the action space. But Gumbel perturbations ensure diversity among the top-k by log-probability, which should include the truly distinct structural alternatives (e.g., placing F_0 on different substrate nodes).
- **Candidate enumeration cost dominates:** The DFS in `_candidate_paths` is called once per transition (via `wrap_child_transition_fn`), not once per candidate. However, it is called for every beam expansion. A larger beam means more DFS calls. For VNE, the transition cost is not trivial (unlike TSP where transition is just "remove index from list").

### 3.4 VNE Constraint Handling

VNE constraints are handled at TWO levels:

**Level 1: Candidate enumeration (pre-filtering)**
- `_candidate_paths` only returns paths that are feasible given current residual resources
- This is the primary constraint enforcement mechanism
- Infeasible actions never appear in the action list, so the policy never needs to learn to avoid them

**Level 2: Ordered action lists**
- Candidates are returned as an ordered list of Paths
- The order is deterministic (DFS from start node 0, then start node 1, etc.)
- The policy assigns logits to each position in this list

**When constraints cause trajectory termination:**
If `transition_fn` finds `action_candidates` is empty for the next step, it sets `objective = float("-inf")` and marks the trajectory as finished. This means:
- The trajectory is a **dead-end** (infeasible completion)
- Its evaluation in TaSaR will be very poor (negative infinity)
- It will never be selected as the best trajectory
- Its mass is still removed from the tree (WOR)

**Critical subtlety:** The current `_candidate_paths` already prunes dead-ends with `check_future_completion=True` (the default). This recursive check verifies that at least one feasible path exists for all remaining links. This means the action list at any step only contains paths that can lead to a complete solution. This prevents the SBS beam from filling with dead-end trajectories, which is essential for search efficiency. **However**, the recursive check is a significant computational cost that may need to be traded off against search speed. The parameter exists: setting `check_future_completion=False` in `_candidate_paths` would speed up enumeration but risk dead-end candidates.

### 3.5 What Changes Needed in the Trajectory Class

The VNE trajectory (`vne/trajectory.py`) already implements the full `BaseTrajectory` interface. To work with TaSaR/Gumbeldore, it needs:

**Existing (already implemented):**
- `init_batch_from_instance_list` -- creates trajectories from instances
- `log_probability_fn` -- returns log-softmax over candidates
- `transition_fn` -- applies path reservation, returns (new_traj, is_finished)
- `to_max_evaluation_fn` -- returns negative cost (objective for maximization)
- `num_actions` -- returns len of action_candidates

**What MUST be checked/verified for TaSaR compatibility:**

1. **Deterministic state copying:** `transition_fn` uses `copy.deepcopy(self)`. This must produce a fully independent trajectory (no shared mutable state). The current implementation appears correct, but any future modification must preserve deep-copy semantics.

2. **action_candidates ordering is deterministic:** The same partial state must always produce the same candidate list in the same order (path order: DFS from start node 0 ascending). This is required for the trie's child indexing to be consistent between policy computation and transition.

3. **leaf evaluation function signature:** `to_max_evaluation_fn` returns a float to **maximize**. Negative costs are already the convention.

4. **Empty candidate handling:** If `action_candidates` is empty, `log_probability_fn` must return an empty tensor/array. The transport wrapper in `wrap_child_log_probability_fn` handles this (returns existing `unsampled_log_masses` for already-computed nodes, or falls through to the empty policy output). The current implementation handles the empty case (line 212: `if logits.numel() == 0`).

**What NEEDS to be built (not yet needed for supervised training):**

1. **Variable-length trajectory handling for the Gumbeldore Dataset:** The `GumbeldoreDataset` (`core/gumbeldore_dataset.py`) needs to handle VNE's variable-length trajectories. The TSP always has exactly n actions, but VNE trajectories can terminate early if infeasible. The BeamLeaf's packed state must carry the full Trajectory dataclass, and `beam_leaves_to_result_fn` must unpack it.

2. **Admission-aware trajectory (parked, for Phase 3):** When admission control is enabled (some requests may be rejected), the action space needs a "skip/reject request" action. The current `_candidate_paths` always embeds every link. This is not needed for the current embed-all regime.

### 3.6 What the Policy Network Needs to Output

The VNE policy network (`vne/network.py`) is an LEHD-style architecture with:

- **Encoder:** Graph attention over substrate nodes, edges, and virtual network state. No positional encoding (pure set attention).
- **Decoder:** Self-attention over [global_context, current_virtual, candidate_1, ..., candidate_n].
- **Output:** One scalar logit per candidate.

**Required outputs for TaSaR:**

The `log_probability_fn` returns a **list of numpy arrays**, one per trajectory, where each array contains log-probabilities for each candidate action. This is exactly what the current `VNEPolicyNetwork` produces after `log_softmax`.

**CRITICAL: The policy must be evaluated in eval mode.** The current `log_probability_fn` does not set `network.eval()`; it assumes it has been set by the caller. During SBS, the call chain is:

1. `IncrementalSBS.__init__` receives `child_log_probability_fn`
2. Wrapped by `wrap_child_log_probability_fn`
3. Called in `stochastic_beam_search` as `child_log_probability_fn(list_of_states)`

The network's `forward` method returns a list of logit tensors. The `_forward_batch` method handles batching with candidate-count bucketing. **Neither method applies dropout or other training-only stochasticity**, so eval/train mode differences do not affect output quality for VNE. But for other problems (e.g., dropout-based architectures), eval mode matters.

---

## 4. Hyperparameter Strategy for VNE

### 4.1 Beam Width `k`

**Paper findings:**
- TSP: k=64 training, k=128 inference
- CVRP: k=64 training, k=128-256 inference
- JSSP: k=64 training and inference (k=256 for finetuning)
- General: "larger k is better" with diminishing returns

**VNE considerations:**

- **VNE's action space is larger than the paper's problems**, so a given k covers a smaller fraction of the space. k=8 vs hundreds of candidates is very sparse.

- **However, Gumbel perturbations provide diversity even at small k.** The paper's WOR-diminishes-with-length finding suggests that for VNE's shorter sequences (l ~ 6-12), WOR is already effective at small k. The TaSaR mechanism adds additional value through root-shifting, not just through large k.

- **Beam width interacts with candidate enumeration cost.** Each SBS expansion calls `child_transition_fn`, which calls `_candidate_paths` (with its expensive future-completion check). For VNE, this is the dominant cost, unlike TSP where transition is O(1). A larger k means more transitions per SBS round.

- **Practical recommendation based on existing config:**
  - Phase 1 (supervised pretraining): k not used (greedy validation with beam-1)
  - Phase 2 (TaSaR training): Start with **k=8** (the current config default). The planned grid is {8, 16, 32, 64}.
  - Phase 3 (inference with larger search): k=128, 256 as compute budget allows.

- **Wall-clock tracking is essential.** The paper's compute fairness formula does not account for VNE's non-uniform transition cost. Measure actual epoch time and report it alongside quality metrics.

### 4.2 Step Size `s` (replan_steps)

**Paper findings:**
- TSP (l=100): s=10 (commit for 10% of the sequence)
- JSSP (l=300): s=50 (commit for ~17% of the sequence)
- General: "smaller s is better" but more expensive (s=1 = MCTS-like)
- The fraction s/l matters more than absolute s

**VNE considerations:**

- **VNE sequence length `l` is SHORT** (6-12 actions). The planned grid {2, 4, 8} means:
  - s=2: t = ceil(12/2) = 6 replan rounds
  - s=4: t = ceil(12/4) = 3 replan rounds
  - s=8: t = ceil(12/8) = 2 replan rounds

- **The optimal s for VNE may be 2** because:
  - Each action (path choice) strongly constrains subsequent actions
  - Early mistakes compound severely (a bad F_0 placement affects all routing)
  - More replanning means more opportunities to correct early bad decisions
  - VNE's short sequence length makes s=2 computationally feasible (6 SBS rounds vs greedy's 12 single-update steps)

- **The expression `s/n` should guide the choice.** For VNE with l ~ 12:
  - s=2 => 17% of sequence (similar to JSSP's ~17% at s=50)
  - s=4 => 33% (more commitment, fewer replans)
  - s=8 => 67% (close to simple SBS)

- **Recommendation:** Default to s=2 (frequent replanning, consistent with the paper's "shorter is better" guidance and JSSP's fraction-of-length approach). Use the planned grid to find the Pareto frontier.

- **One caveat:** The current config has `replan_steps: 2` and `beam_width: 8`. For the initial TaSaR runs, keep these defaults and only vary one at a time.

### 4.3 Top-p (Nucleus Sampling)

**Paper findings:**
- TaSaR holds Top-p **constant** (unlike Gumbeldore which anneals it)
- The flag is confusingly named `min_nucleus_top_p` but in TaSaR mode it means "constant Top-p threshold"
- The paper uses Top-p=1.0 (no filtering) for small problems, smaller values for larger action spaces
- Top-p=0.8 was used in CVRP finetuning phase (k=256, s=1)

**VNE considerations:**

- **VNE's policy distribution over hundreds of candidates is likely flatter** than TSP's, because many paths have similar features. This means Top-p may exclude fewer candidates than in TSP.

- **Top-p = 1.0 (current config) means no truncation.** This is safe for initial TaSaR experiments. All candidates are considered.

- **Smaller Top-p might help** by pruning low-probability paths that the Gumbel perturbation might otherwise rank highly by chance. But this risks excluding the truly best path if the policy is poorly calibrated.

- **Recommendation:** Start with Top-p = 1.0 (no filtering). If the beam fills with near-random paths at larger k, reduce Top-p to 0.95 or 0.9. Treat this as a **second-order dial** after k and s are tuned.

### 4.4 Hyperparameter Interactions with VNE's Specific Challenges

**Challenge 1: Many candidates are near-equivalent (many paths between the same endpoints).**
- Gumbel perturbations will amplify small probability differences among similar paths.
- Top-p will not help much here (they all have similar probability).
- The trie's mass removal prevents re-sampling the exact same path but does NOT prevent sampling different paths between the same endpoints.

**Mitigation:** This is a feature, not a bug. The SBS diversity mechanism already ensures that different paths are sampled. The key question is whether the policy learns to distinguish between high-cost and low-cost paths between the same endpoints, which depends on the quality of candidate features (path length, residual bandwidth, etc.).

**Challenge 2: Dead-ends from resource exhaustion.**
- The `check_future_completion` in `_candidate_paths` prunes actions that lead to dead-ends.
- But there is a subtlety: the resource check is based on CURRENT residual resources, not the state after this action. An action may be completable from current resources but not from post-action resources (if other candidates in the beam also consume from the same polluted pool).

**Mitigation:** The recursive check (`check_future_completion=True`) already checks exactly this: it simulates path consumption and then checks completability. However, it only does this check once (don't double-check during transition). This is handled correctly.

**Challenge 3: Expensive transition function.**
- Each SBS expansion calls `_candidate_paths` (DFS) via the transition wrapper.
- For large beam widths, this creates a significant compute burden.

**Mitigation:** `sbs_child_state_cache` on `_TrieNode` caches (state, is_leaf) tuples, so transitioning to the same child again is O(1). However, the first transition for each unique (state, child_index) pair pays the full DFS cost.

**Challenge 4: Variable-length trajectories.**
- Unlike TSP where all trajectories have exactly n actions, VNE trajectories can be shorter (dead-end) or vary across instances (different numbers of requests/links).
- The `_TrieNode` structure handles this via `mark_leaf()` for nodes that complete early.

**Mitigation:** Already handled. The `_TrieNode` structure is designed for variable-length trajectories (line 154-157 in `incremental_sbs.py`).

---

## 5. Implementation Roadmap

### 5.1 Current Repository Status

The following VNE infrastructure is ALREADY IMPLEMENTED and working:

**Complete and tested:**
- `vne/config.py` -- Full config with all TaSaR hyperparameters in `gumbeldore_config`
- `vne/trajectory.py` -- Complete `BaseTrajectory` subclass with candidate enumeration, feasibility checking, and matching the PROBLEM_FORMULATION.md schema
- `vne/network.py` -- LEHD-style policy network with batched forward and candidate bucketing
- `vne/features.py` -- State input construction from trajectory state
- `vne/dataset.py` -- `RandomVNEDataset` for supervised replay
- `vne/validation_set_generator.py` -- MILP solver with multi-backend fallback, all objective modes
- `vne/instance_generator.py` -- Problem instance generation (substrate + VN requests)
- `vne_main.py` -- Full training loop wiring with env overrides for A/B experiments
- `vne/PROBLEM_FORMULATION.md` -- Formulation documentation (the extended VNE with comm/comp distinction)

**Trained and available:**
- 50k supervised training dataset (`vne_supervised_training_dataset_50k.pickle`)
- 1k validation dataset (`vne_validation_dataset_1k.pickle`) with ILP solutions
- 2k test dataset (`vne_test_dataset_2k.pickle`) with ILP solutions

**Experimental results so far:**
- Supervised plateau at val ~ -22.9, test ~ -25.6
- Gradient clipping (norm=1.0) fixed the epoch-6 divergence
- 50x more data did NOT improve the plateau -- the bottleneck is the decoding/solution quality, not data quantity

### 5.2 What Needs to Be Built/Modified

#### Phase 2: Enable TaSaR Inference (next step)

The config already has `gumbeldore_config["search_type"] = "tasar"` and all necessary parameters. To run TaSaR-based evaluation (not yet training):

1. **Set `gumbeldore_eval = True` with TaSaR config** -- The `evaluate` function in `vne_main.py` currently uses `search_type = "beam_search"` for the non-gumbeldore_eval path. To run TaSaR evaluation:

   ```python
   # In VNEConfig:
   gumbeldore_eval = True
   # gumbeldore_config already has search_type = "tasar"
   ```

   The `GumbeldoreDataset` generator will load `search_type` from config and dispatch to the correct SearchBehaviour enum value, which invokes `IncrementalSBS.perform_tasar`.

2. **Verify the signature compatibility** -- The `GumbeldoreDataset` (`core/gumbeldore_dataset.py`) calls `IncrementalSBS` with:
   - `initial_states` -- list of (trie_node, trajectory_state) tuples, produced by the worker
   - `child_log_probability_fn` -- callable that calls `Trajectory.log_probability_fn` wrapped to handle trie integration
   - `child_transition_fn` -- callable that calls `Trajectory.transition_fn` wrapped to handle trie integration
   - `leaf_evaluation_fn` -- callable that calls `Trajectory.to_max_evaluation_fn`

   The trajectory class already provides all three required static methods.

3. **Set up multi-worker generation** -- For the Gumbeldore training loop, configure `devices_for_workers` for parallel generation. Options:
   - `["cpu"] * N` -- N CPU workers (slow but available)
   - `["cuda:0"] * N` -- N GPU workers sharing via MPS (faster if GPU memory allows)
   - The current config has `["cpu"]` -- one CPU worker. Scale this up.

4. **Implement `generate_instances` properly for the config** -- The function `generate_instances` in `vne_main.py` already returns (instances, batch_size_per_worker, batch_size_per_cpu_worker). The instances generator uses `make_dataset` which generates fresh problem instances. This is ready.

#### Phase 2: Enable TaSaR Training (Gumbeldore mode)

To switch from supervised to Gumbeldore (self-improved) training:

1. **Set `learning_type = "gumbeldore"`** in `VNEConfig` or via env override `VNE_LEARNING_TYPE=gumbeldore`.

2. **The training loop will then call `train_for_one_epoch_gumbeldore_fn`** instead of the supervised version. In `vne_main.py`, this is `train_for_one_epoch_gumbeldore` which:
   - Generates a dataset via `GumbeldoreDataset.generate_dataset`
   - Creates a `RandomVNEDataset` from the generated data
   - Trains for one epoch on it

3. **The `GumbeldoreDataset` class** (`core/gumbeldore_dataset.py`) handles the Ray-based parallel dispatch. It:
   - Creates Ray workers (one per `devices_for_workers` entry)
   - Each worker: generates instances -> encodes with policy -> runs SBS/TaSaR -> saves results
   - Results are collected, processed by `beam_leaves_to_result_fn` and `process_search_results_fn`

4. **Verify `beam_leaves_to_result_fn`** -- The current `beam_leaves_to_result` in `vne_main.py` (line 102) extracts `processing_paths`, `f_placements`, `objective` from the best trajectory. This is called for each instance to produce the search result that gets stored in the dataset. **This function will be called with the BeamLeaf format** where `leaf.state` is the `Trajectory` object. The current implementation uses `trajectory.to_max_evaluation_fn(trajectory)` to find the best trajectory, then extracts fields. This should work.

5. **Data flow for Gumbeldore training:**
   ```
   generate instances  ->  TaSaR search  ->  extracts (processing_paths, objective)
       ->  pickle als "solved" dataset  ->  RandomVNEDataset reads it
       ->  DataLoader  ->  train_with_dataloader  ->  cross-entropy loss
   ```

#### Phase 2: Compute and Log Optimality Gap

The single most important metric to add:

```python
# In validate() or evaluate():
# ILP objective is stored in each test/val instance as instance["objective"]
# Model objective is obtained from the search result

ilp_objective = instance["objective"]    # from pickle
model_objective = result["objective"]    # from TaSaR search

# Optimality gap:
gap = (model_objective - ilp_objective) / abs(ilp_objective)
# For negative objectives (costs):
#   gap = 0.0 if model_objective == ilp_objective
#   gap > 0.0 means model is worse (less negative cost)
```

The val and test pickles contain `instance["objective"]` which is the ILP optimum. Logging the gap allows comparison with the paper's results (which report gap vs optimal/known solutions).

#### Phase 3: Training Stability and Scaling

- **Gradient clipping is already set** in `vne/config.py` (`optimizer["gradient_clipping"]: 1.0`).
- **LR schedule:** Currently `decay_factor=1.0` (no decay). Consider decay after initial experiments (e.g., 0.9 every 5 epochs).
- **Leader mechanism:** The SIL training loop already has `use_best_model_for_generation` and the leader checkpoint is managed by the training cycle. No changes needed.
- **Loss function:** The `CrossEntropyLoss` with `reduction="mean"` is used. This is compatible with SIL (maximum likelihood on demonstrated actions).

### 5.3 Known Pitfalls

1. **Training divergence from bad pseudo-labels.** In early Gumbeldore epochs, the policy is weak and TaSaR may not find better solutions than the policy's own greedy decode. If no trajectory improves, the generated pseudo-labels are no better than the previous epoch's and the SIL loop stalls. **Mitigation:** Start from a supervised-pretrained model (Phase 1 labels), not from random initialization.

2. **The search_type dispatch.** The `GumbeldoreDataset` uses a `SearchBehaviour` enum to map `search_type` strings to implementations. Make sure `"tasar"` maps to `perform_tasar` and that all config parameters (`beam_width`, `replan_steps`, `min_nucleus_top_p`, `perform_first_round_deterministic`) are correctly passed.

3. **Ray worker memory.** Each Ray worker loads the policy network onto its device. With `devices_for_workers = ["cpu"]`, all workers share CPU memory. For VNE with large networks, ensure worker count * model size does not exceed available RAM.

4. **Candidate enumeration cost in SBS.** The `_candidate_paths` DFS is called once per transition. With k=64 and l=12, this is ~768 DFS calls per instance per epoch. For 256 instances, that's ~200k DFS calls. Each DFS explores the full substrate graph. Profile this early to catch bottlenecks.

5. **The `candidate_token_budget` cap.** In the batched forward pass (`_forward_batch`), instances are grouped by candidate count. The token budget of 8000 caps `group_size * max_candidates_in_group`. If one instance has 5000 candidates, only 1 instance can be in that group (5000 * 1 = 5000 < 8000 but tight). This limits GPU utilization for instances with extremely large action spaces.

6. **Variable-length trajectories in the trie.** The `_TrieNode` structure handles leaf nodes via `mark_leaf()`. When a VNE trajectory reaches a dead-end (no feasible candidates remaining), `transition_fn` returns `is_finished=True`. The trie node for that step should have `children = []`, which makes `exhausted()` return True. **Verify that this happens correctly** in the wrapper code path.

### 5.4 Testing and Validation Strategy

**Before running TaSaR training:**

1. **Unit test: TaSaR inference on a single VNE instance.** Create a test that:
   - Loads one instance from the validation set
   - Runs `IncrementalSBS.perform_tasar` with the pretrained supervised model
   - Verifies that the returned trajectory is feasible (passes `_verify_solution`)
   - Verifies that the objective is at least as good as greedy (beam-1)
   - Logs the optimality gap vs the ILP label

2. **Unit test: Mass removal on a small VNE case.** Create a simple case with 2-3 candidates per step and verify that:
   - After sampling k leaves, re-running SBS produces different trajectories (WOR property)
   - The best trajectory's objective is at least as good as the mean of sampled trajectories
   - exhausted() returns True after all probability mass is consumed

3. **Unit test: End-to-end Gumbeldore generation.** Run `GumbeldoreDataset.generate_dataset` with:
   - 4 instances, beam_width=4, replan_steps=2, supervised-pretrained model
   - Verify that the output pickle has the correct schema (processing_paths, objective, f_placements)
   - Verify that all trajectories are feasible

**During training:**

4. **Log optimality gap every epoch.** For both validation and test sets:
   ```python
   val_gap = (val_model_obj - val_ilp_obj) / abs(val_ilp_obj)
   ```

5. **Monitor the pseudo-label quality.** Track the mean objective of generated solutions each epoch. It should improve or stay stable. A decreasing trend means the SIL cycle is failing.

6. **Monitor training loss.** The cross-entropy loss should decrease. If it increases while validation gap stays constant, the policy may be overfitting to bad pseudo-labels.

7. **Track wall-clock time per epoch.** VNE's expensive transition means TaSaR epochs may be much slower than supervised epochs. This is expected for the first run but should be optimized if it becomes the bottleneck.

---

## 6. Key Insights for Beating State-of-the-Art

### 6.1 Why TaSaR Outperforms on Harder Problems

The paper's experimental results show a clear trend: **TaSaR's marginal benefit increases with problem difficulty**:

| Problem | Greedy gap | TaSaR gap (k=64, s=10) | Improvement |
|---------|-----------|------------------------|-------------|
| TSP N=100 | 0.37% | 0.0009% (k=128, s=10) | 410x |
| CVRP N=100 | 2.96% | 0.14% (k=128, s=10) | 21x |
| JSSP 50x15 | 2.2% | 0.4% (k=64, s=50) | 5.5x |
| JSSP 100x20 | 1.0% | 0.1% (k=64, s=50) | 10x |

**The pattern:** TaSaR's advantage is largest relative to greedy when the greedy policy is weakest. TSP is nearly solved (0.37% greedy gap), so there's little room for search to improve. JSSP has much larger remaining gaps, so search-driven improvement is more dramatic.

For VNE, the current supervised gap is likely large (the val/test objectives of -23 to -25 may be far from the ILP optimum). **A large initial gap means TaSaR has more room to improve.**

**Why TaSaR outperforms simpler methods on hard problems:**

1. **WOR diversity degrades with length, but VNE has SHORT sequences (l ~ 6-12).** This is the opposite of the paper's finding. For VNE, WOR from the full sequence is already effective. TaSaR's root-shifting provides less marginal benefit over simple WOR than it does for l=100 TSP.

2. **BUT: VNE has a large action space with many near-equivalent alternatives.** The Gumbel noise amplifies small differences among hundreds of candidate paths. Simple greedy decoding gets stuck in one mode; TaSaR's SBS explores structurally different alternatives (different substrate node assignments).

3. **The commit-reconsider pattern directly addresses VNE's "early mistake" problem.** An early bad node placement cascades through all subsequent routing. By committing to the best prefix and resampling, TaSaR detects and corrects this.

### 6.2 Why VNE Is an Even Stronger Test Case for TaSaR

**VNE is structurally different from the paper's three benchmarks in ways that favor TaSaR:**

1. **Constraint-heavy with cascading feasibility.** Unlike TSP (always feasible) or JSSP (always feasible with resource constraints that are tight but don't fail), VNE has hard resource constraints that can make partial solutions infeasible. TaSaR's exploration of different completions from intermediate states is exactly what's needed to find feasible completions when the current partial solution has consumed too many resources.

2. **No efficient exact solver.** Concorde solves TSP optimally. HGS solves CVRP near-optimally. JSSP has efficient solvers for small instances. VNE has no such practically efficient solver -- even MILP formulations with commercial solvers struggle for instances beyond ~50 nodes. This means ANY improvement from TaSaR is more valuable because the alternative (ILP labels) is exponentially more expensive.

3. **Self-improvement is the only viable training paradigm.** The A/B result (50x data gave no improvement) shows that supervised training on ILP labels is bottlenecked at the policy's capacity, not data scarcity. SIL with TaSaR decoding is the escape from this plateau because it:
   - Generates pseudo-labels that are BETTER than the greedy policy
   - The pseudo-labels improve over epochs as the policy improves
   - This creates a positive feedback loop that external labels cannot provide

4. **The LEHD architecture is designed for generalizable representation learning.** LEHD was shown to generalize from N=100 to N=1000 for TSP. For VNE, this means the learned embeddings could generalize across different substrate topologies and sizes, making each training epoch more valuable.

5. **VNE's business relevance makes SOTA results publishable.** Any improvement over existing VNE methods is immediately impactful for network virtualization, 5G/6G network slicing, and cloud resource allocation.

### 6.3 What Advantages TaSaR Has Over Existing VNE Methods

**Existing VNE methods:**

1. **Exact MILP (HiGHS, Gurobi, CPLEX):** Optimal but computationally prohibitive for realistic instances. The current label generator uses HiGHS with 60s time limit, which is already stretched for the current small instances (20-40 nodes). Scaling to larger substrates is infeasible.

2. **Heuristic methods (greedy first-fit, particle swarm, genetic algorithms):** Fast but no quality guarantee, no learning, no generalization across instance distributions. Each new instance distribution requires re-tuning.

3. **Supervised NCO (the current approach):** Requires ILP labels (the bottleneck), plateaus at policy capacity. The 50k-instance plateau shows this approach hits a ceiling.

4. **RL-based NCO:** VNE's large action space makes REINFORCE-style training unstable and sample-inefficient. The LEHD architecture that enables generalization is too large for policy-gradient methods.

**TaSaR's advantages:**

1. **No ILP labels needed after initial pretraining.** The SIL cycle replaces the ILP solver as the source of training data. This is the decisive advantage: VNE's scalability bottleneck (ILP computation) is broken.

2. **Diverse training data from a fixed policy.** TaSaR generates many alternative solutions from each instance using the same policy. These alternatives have different structural properties (different node placements, different path routings), providing richer training signal than greedy decoding.

3. **Search-time quality improvement.** Even without retraining, TaSaR at inference time with larger k and smaller s finds better solutions than greedy or beam search alone. The current beam-1 validation gives ~-23 val / ~-25.6 test; TaSaR at k=64, s=2 is expected to improve both.

4. **Theoretical grounding in probability theory.** The WOR-with-mass-removal mechanism has rigorous guarantees (sampling without replacement from the policy), unlike heuristic search methods. The commitment mechanism has an intuitive interpretation (exploit the best prefix found so far).

5. **Transferability across instance distributions.** Because TaSaR explores structurally different solutions, it discovers pseudo-labels that may be valid for Out-of-Distribution instances. The paper demonstrates this for TSP (N=100 -> N=200,500) and JSSP (multi-size training).

### 6.4 Expected Impact: Numerical Targets

Based on the paper's results and the current VNE plateau:

| Metric | Current (supervised) | TaSaR target (Phase 2) | Evidence from paper |
|--------|---------------------|----------------------|-------------------|
| Val objective | ~ -22.9 | ~ -25 to -28 | CVRP gap reduced 21x (2.96% -> 0.14%) |
| Test objective | ~ -25.6 | ~ -28 to -32 | JSSP 50x15 gap reduced 5.5x (2.2% -> 0.4%) |
| Optimality gap | Unknown (needs measurement) | 50-80% reduction | Consistent across problems |
| Training epochs to plateau | ~5 | ~15-30 | SIL needs more epochs than supervised |

**Critical caveat:** These targets are aspirational. The paper's problems have well-calibrated policies (from expert data), clean MDPs, and established benchmarks. VNE has a newly-trained policy, a more complex MDP, and no established NCO baseline. A successful Phase 2 outcome is measurable progress on these metrics, not necessarily matching the paper's improvement ratios.

### 6.5 Success Criteria

The TaSaR experiment succeeds if:

1. **TaSaR-generated pseudo-labels have better mean objective than greedy decoding**, measured on the same instances. Without this, the SIL cycle cannot start.

2. **The optimality gap (vs ILP labels) shrinks over epochs**, indicating the self-improvement loop is working.

3. **Test-set performance improves beyond the supervised plateau**, showing that TaSaR's diverse search discovers solutions the greedy policy could not.

4. **Gradient clipping keeps training stable** (already confirmed).

5. **Wall-clock cost is acceptable** for the research timeline (epochs should complete within hours, not days, on the available GPU).

---

## Appendix A: Code Map (File Paths)

| File | Purpose |
|------|---------|
| `core/abstracts.py` | `BaseTrajectory` and `Config` type definitions |
| `core/stochastic_beam_search.py` | SBS algorithm, `sample_gumbels_with_maximum`, `BeamLeaf` namedtuple |
| `core/incremental_sbs.py` | `_TrieNode`, `IncrementalSBS` with `perform_tasar` and `perform_incremental_sbs` |
| `core/train.py` | `main_train_cycle` for supervised and Gumbeldore training |
| `core/gumbeldore_dataset.py` | Ray-based parallel data generation |
| `vne/config.py` | `VNEConfig` with supervised and Gumbeldore hyperparameters |
| `vne/trajectory.py` | `Trajectory(BaseTrajectory)` for VNE |
| `vne/network.py` | `VNEPolicyNetwork` with batched forward pass |
| `vne/features.py` | State input construction from trajectory |
| `vne/dataset.py` | `RandomVNEDataset` for replay |
| `vne/validation_set_generator.py` | MILP-based label generation |
| `vne/instance_generator.py` | Problem instance generation |
| `vne_main.py` | Training entry point |
| `vne/PROBLEM_FORMULATION.md` | VNE problem formulation documentation |
| `docs/HANDOFF.md` | Current experimental state and cluster setup |

## Appendix B: Config Parameter Reference for TaSaR

From `vne/config.py`, `gumbeldore_config` dict:

| Parameter | Current value | Effect in TaSaR mode | Suggested range for VNE |
|-----------|--------------|----------------------|------------------------|
| `search_type` | `"tasar"` | Selects `perform_tasar` over gumbeldore | `"tasar"` |
| `beam_width` | 8 | Number of leaves sampled per SBS round | {8, 16, 32, 64} |
| `replan_steps` | 2 | Actions committed before resampling | {2, 4, 8} |
| `min_nucleus_top_p` | 1.0 | Constant Top-p threshold for TaSaR | 1.0, 0.95, 0.9 |
| `perform_first_round_deterministic` | False | If True, first SBS round is deterministic beam search | False (start stochastic) |
| `num_instances_to_generate` | 256 | Fresh instances per Gumbeldore epoch | 256, 512 |
| `devices_for_workers` | `["cpu"]` | Parallel workers for data generation | `["cpu"] * N` or `["cuda:0"] * N` |
| `batch_size_per_worker` | 32 | SBS batch size per GPU worker | 32 (tune for GPU memory) |
| `batch_size_per_cpu_worker` | 32 | SBS batch size per CPU worker | 32 (smaller if CPU-bound) |

**Ignored in TaSaR mode:** `num_rounds`, `advantage_constant`, `min_max_normalize_advantage`, `expected_value_use_simple_mean`, `use_pure_outcomes`, `normalize_advantage_by_visit_count`.

---

*Generated on 2026-06-13. This document is the definitive reference for adapting TaSaR to VNE. File: `docs/paper/TASAR_METHOD_DEEP_DIVE.md`.*
