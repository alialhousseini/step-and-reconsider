# Section 2: Related Work — Deep Analysis

**Paper:** "Take a Step and Reconsider: Sequence Decoding for Self-Improved Neural Combinatorial Optimization"
**Authors:** Jonathan Pirnay, Dominik G. Grimm (ECAI-2024)
**Section:** 2 — Related Work

---

## (a) Raw Content Summary

This section has three logical clusters, each positioning the paper's method against prior work:

### Cluster 1: Constructive NCO (the "what came before")
- **Pointer Network (Vinyals et al.)** — the first neural approach to constructive combinatorial optimization.
- **REINFORCE with baseline (Bello et al.)** — applied policy gradient to NCO, replacing supervised learning.
- **Attention Model (AM)** — Transformers became the standard architecture for NCO.
- **POMO (Kwon et al.)** — exploits problem symmetries by starting from every possible first node, sampling diverse trajectories. Does not scale well.
- **BQ and LEHD** — identify the AM decoder as too light, causing poor generalization. Solution: heavy decoders (up to nine transformer blocks). Trained via **supervised learning on expert solutions** for 100-node instances; generalize to 1,000 nodes. But their size makes reinforcement-learning training difficult.

### Cluster 2: Self-Improved Learning (the "how we train without expert labels")
- **Central concept:** use the current policy to generate its own training data (solutions as pseudo-labels). This is the paradigm TaSaR operates in.
- **Luo et al.:** pre-train with RL on small problems, then generate solutions for larger instances, improve them via **random subtour re-unrolling**. Exploits subtour optimality property. **Limitation: routing-problem-specific** (does not generalize to VNE, JSSP, Gomoku).
- **Corsini et al.:** "self-labeling" for JSSP using **vanilla Monte Carlo sampling** — no search structure, just i.i.d. samples from the policy.
- **Pirnay and Grimm (prior Gumbeldore):** sample **WOut Replacement (WOR)** over multiple rounds with advantage-scaling hyperparameters. The authors themselves note this approach "can be complex to tune."

### Cluster 3: Sequence Decoding / Inference-Time Search (the "how we decode at test time")
- **Simulation-guided Beam Search:** beam search with greedy rollouts. Purely exploitative — unsuitable for self-improved learning (SIL) because it does not generate diverse training data.
- **AlphaZero-style MCTS:** effective, but time-consuming and hard to parallelize.
- **WOR sampling:** a key diversification approach. Standard method for generating diverse trajectories.
- **Stochastic Beam Search (SBS):** the method chosen by this paper. It parallelizes like regular beam search (key practical advantage) while providing the stochasticity needed for SIL.

---

## (b) Key Ideas Unpacked

### Key Idea 1: The decoder capacity / generalisation trade-off
BQ and LEHD argue that the Attention Model's light (single-transformer-block) decoder causes poor generalization to larger instances. Their fix — heavy multi-block decoders — works for SL-on-experts but makes RL training hard (large models are sample-inefficient). This reveals a fundamental tension: **you want enough capacity to generalize, but not so much that RL becomes infeasible.** TaSaR sidesteps this: instead of a heavier decoder, it improves inference-time search, keeping the policy network light enough for RL.

### Key Idea 2: Self-improvement requires *structured* search, not just i.i.d. sampling
The cluster of self-improved methods forms a spectrum:
- **Simple (Corsini):** vanilla MC sampling — no structure, low cost per sample, but poor quality per sample.
- **Moderate (Luo):** problem-specific improvement (subtour re-unrolling) — exploits domain structure, but non-transferable.
- **Complex (Gumbeldore WOR):** multi-round, advantage-weighted, WOR — generic but hyperparameter-heavy.
- **TaSaR's position:** structured search (SBS) with periodic commitment — generic, less tuning, better diversity-vs-quality balance.

The deep insight: **for self-improvement, the search method must produce BOTH high-quality solutions (to learn from) and diverse solutions (to explore). Pure exploitation (beam search) fails SIL. Pure exploration (MC sampling) wastes compute.**

### Key Idea 3: Parallelizability is a first-class design constraint
The paper explicitly chooses SBS over MCTS because SBS "parallelizes like regular beam search." This is not a minor engineering detail — it means the method scales on GPU hardware without custom parallelization schemes. For VNE, where instance solve times are already high, this is critical.

### Key Idea 4: The hyperparameter burden is a barrier
The paper's own prior work (Gumbeldore WOR) requires "advantage-scaling hyperparameters" that "can be complex to tune." TaSaR's design goal is to reduce this burden. Search-based methods should be **robust** across problem classes, not require per-problem tuning.

---

## (c) Technical Specifics

| Topic | Specificity | Implication |
|-------|-------------|-------------|
| **Pointer Network** | sequence-to-sequence with attention as decoder | First demonstration that learned construction can work |
| **REINFORCE baseline** | learned baseline for variance reduction | Made RL practical for NCO; still the default paradigm |
| **AM decoder** | single lightweight transformer block | Identified as bottleneck for generalization |
| **BQ decoder** | "heavy" — up to 9 transformer blocks | Improves generalization (100→1000 nodes) but makes RL hard |
| **LEHD decoder** | also heavy; heterogeneous decoder with enhanced attention | Same trade-off as BQ; used as the VNE policy network backbone |
| **POMO** | K starting points × L rollouts = K×L trajectories | Simple, effective, but O(KL) compute; doesn't scale to large K |
| **Luo's subtour re-unrolling** | RL pre-train small → generate big → fix subtours via shortest-path | Domain-specific; exploits subtour optimality principle |
| **Corsini's self-labeling** | vanilla MC sampling from policy; no search tree or beam | Simple but sample-inefficient; baseline for JSSP |
| **Gumbeldore WOR** | multi-round, WOR sampling, advantage scaling | Complex tuning; the starting point TaSaR improves upon |
| **Simulation-guided Beam Search** | roll out candidates with greedy policy during search | Exploitation-only; generates no diverse data for SIL |
| **AlphaZero MCTS** | game-tree search with NN-guided UCB | Slow, serial, not GPU-friendly |
| **SBS** | Gumbel-Top-k trick; parallel beam search with stochastic branching | Parallelizable, tunable stochasticity, suitable for SIL |

---

## (d) VNE Implications

This section is the **most consequential for VNE** because it defines the design space and justifies why TaSaR is a good fit. Here is a point-by-point analysis:

### VNE Implication 1: The BQ/LEHD trade-off directly applies to our architecture
VNE uses the **LEHD policy network** (inherited from the routing literature). The paper tells us:
- LEHD was designed for supervised learning on expert solutions (100-node TSP → 1000-node TSP).
- LEHD is "heavy" (many decoder blocks), which the authors say "makes RL training difficult."
- **Bottom line for VNE:** If the paper authors themselves note that LEHD is difficult for RL, and our VNE experiments already show training instability (epoch-6 divergence, now mitigated by gradient clipping), this architectural limitation may be a **fundamental ceiling** on VNE performance. The plateau at ~−23 val / ~−25.6 test may be a **LEHD capacity-for-RL problem**, not a TaSaR search problem.

**Action:** Consider whether VNE needs BQ/LEHD at all, or whether a lighter architecture + TaSaR's improved search might work better.

### VNE Implication 2: Problem-specific improvement methods (Luo) are ruled out for VNE
Luo's subtour re-unrolling exploits the fact that routing-problem subpaths are themselves routing problems. VNE has no such compositional structure:
- The embedding of a virtual node depends on which substrate node it maps to.
- The "subgraph" of a VNE solution is not independently valid.
- The cost structure (link embedding cost) is inherently global.

**Bottom line:** VNE cannot benefit from problem-specific heuristics like subtour re-unrolling. It MUST use a **general-purpose search method** (TaSaR, WOR, MCTS). This makes TaSaR particularly attractive because it's generic and does not rely on domain structure.

### VNE Implication 3: POMO does not scale, but VNE might need it less
POMO's approach (K starting positions × L samples) doesn't scale because VNE's "starting positions" are the substrate node choices, which are large (20-40 nodes) and interdependent. However, VNE also doesn't have clean symmetries to exploit — there's no natural "starting node" that would generate diverse solutions. **TaSaR's SBS provides an alternative diversity mechanism** (stochastic beam branching) that doesn't rely on start-position symmetries.

### VNE Implication 4: Self-improvement is the only viable training paradigm for VNE
The paper establishes that:
- **Supervised learning** on expert solutions works (BQ/LEHD for TSP) but requires oracle solutions. VNE has an ILP oracle (HiGHS, GUROBI), but generating expert solutions is computationally expensive (minutes per instance via MILP). The 50k dataset already took significant compute.
- **REINFORCE-style RL** is hard with heavy architectures (LEHD).
- **Self-improved learning** generates its own training data from the policy, removing the oracle bottleneck.

**Bottom line:** For VNE to scale to larger substrates (50+ nodes, more requests), self-improvement is the only practical path. TaSaR is the best-positioned method because it offers structured search (better sample quality than WOR) without the parallelization problems of MCTS.

### VNE Implication 5: SBS parallelizability is make-or-break for VNE
VNE instances are already slower than TSP/CVRP/JSSP per search step (due to larger action spaces, constraint checking, and link embedding cost calculation). MCTS would be prohibitively slow. **SBS's GPU-parallel beam search is the only inference-time method whose computational profile fits VNE's requirements.**

### VNE Implication 6: The hyperparameter story matters more for VNE
The paper notes Gumbeldore WOR has "complex to tune" hyperparameters. VNE adds another layer of complexity: the configuration space already includes substrate scaling, request scaling, admission settings, objective modes (lex, profit), solver choices. Adding a fragile search hyperparameter space on top would be unsustainable. **TaSaR's relative robustness** (fewer search hyperparameters, no advantage scaling) is a significant advantage for VNE research.

### VNE Implication 7: The "no problem-specific tricks" philosophy is VNE-aligned
The paper's entire approach is generic across TSP, CVRP, JSSP, and Gomoku. They deliberately avoid problem-specific search improvements. This is exactly what VNE needs — there is no established body of NCO-specific heuristics for VNE (compared to subtour re-unrolling for TSP). A generic method that "just works" on a new problem class is the research-relevant contribution.

---

## (e) Critical Insights for Our Work

### CRITICAL INSIGHT 1: The paper positions itself by what it ISN'T
The section is structured as a series of negations:
- "Not REINFORCE" (we use self-improvement, not RL from scratch)
- "Not SL-on-expert" (we don't need oracles or heavy decoders)
- "Not POMO" (we don't need symmetry exploitation)
- "Not Luo's subtour trick" (we don't need domain-specific improvement)
- "Not WOR" (we don't want complex advantage-scaling tuning)
- "Not beam search" (we want diversity, not pure exploitation)
- "Not MCTS" (we want GPU parallelization)

The positive statement: **TaSaR = structured stochastic search + periodic commitment + GPU parallelism + generic enough for any constructive combinatorial problem.**

### CRITICAL INSIGHT 2: The paper's key claim (falsifiable)
The implicit claim from this section is:
> "A generic, GPU-parallelizable, stochastic search method with periodic commitment can outperform problem-specific improvement heuristics and complex multi-round WOR sampling, across diverse problem classes, without per-problem hyperparameter tuning."

This is a strong claim. For VNE, we are testing the "across diverse problem classes" part. If TaSaR fails on VNE, it either means:
(a) VNE's action space / constraint structure falls outside the scope of "constructive combinatorial problems" TaSaR handles, or
(b) there is a VNE-specific issue (e.g., LEHD architecture not suitable for RL, as noted in point 1), or
(c) the search hyperparameters (beam width, replan step, top-p) need problem-specific tuning despite the paper's claim.

### CRITICAL INSIGHT 3: The gap between "self-improved learning" and "supervised oracle labels" is the key metric
The paper never says self-improvement *replaces* oracle-based supervision. It says it removes the oracle *dependency*. For VNE, we have oracles (ILP solvers). The research question is: **how much of the oracle gap can self-improvement close?** The HANDOFF.md already shows that with 50k supervised labels, VNE plateaus at ~−25.6 test objective. The ILP optimum is known (carried in the pickles). The gap is the headroom — and TaSaR should ideally close more of it than supervised pretraining alone.

### CRITICAL INSIGHT 4: LEHD's origins matter for VNE generalization
LEHD was designed for TSP generalization from 100 to 1,000 nodes. For VNE, generalization means: trained on substrate 20-40 nodes, tested on (a) similar sizes, (b) larger substrates, (c) different request distributions. The paper shows BQ/LEHD generalize across *scale* (small→large TSP). Whether this property transfers to VNE topology is an open question.

### CRITICAL INSIGHT 5: The self-improvement loop and data quality
The section establishes that self-improvement works when the current policy generates data that is *better* (higher quality, more diverse) than it was trained on. The SBS/TaSaR search is the mechanism for generating "improved" data. For VNE:
- If SBS generates solutions with objective −25 (vs supervised best of −25.6), self-improvement would wash out the supervised phase.
- If SBS generates solutions at −30 or worse (barely better than random), the self-improvement loop collapses.
- **Key unknown:** What is SBS/TaSaR's solution quality on VNE at zero-shot (before any self-improvement iterations)?

---

## (f) Open Questions Raised by This Section

1. **Architecture question:** If BQ/LEHD's heavy decoder is hard for RL, and TaSaR is a search-layer improvement, does TaSaR *sufficiently* compensate for a hard-to-train policy? Or does the policy still need to be RL-friendly independently?

2. **Comparison question:** The paper compares TaSaR against Gumbeldore WOR, AM+LCPO, and other methods on TSP/CVRP/JSSP/Gomoku. For VNE, there are no published baselines using TaSaR. What is the *appropriate baseline* for VNE? (a) Random construction? (b) ILP optimal? (c) The supervised-only VNE policy? (d) A greedy heuristic?

3. **Search budget question:** How does the VNE action space size affect the beam width / replan step trade-off? VNE's action space is larger than TSP's (|substrate nodes| per virtual node vs |remaining cities|). Does the optimal beam width scale with action space size?

4. **Diversity question:** The paper values search diversity for self-improvement. VNE's large action space may already provide "natural diversity" (many valid mappings). Does SBS's stochastic branching add value beyond what random sampling from a softmax would give?

5. **Scaling question:** If VNE instances are computationally heavier per search step, and self-improvement requires many search passes, does the wall-clock cost of one self-improvement iteration on VNE exceed the cost of generating ILP labels for a supervised dataset? (This is a practical, not theoretical, question — if the answer is yes, self-improvement may not be worthwhile for VNE at the current scale.)

6. **Transfer question:** Luo's method transfers from small to large via subtour re-unrolling. TaSaR is generic. Can TaSaR's search on small VNE instances (e.g., 10-node substrates) produce a policy that works on large instances (40+ nodes)? The paper covers scale generalization for TSP/CVRP, but VNE's combinatorial structure is different.

7. **Admission control question:** The section assumes constructive problems where every partial solution is feasible and completable. VNE with admission control (rejecting requests) breaks this assumption: the policy must learn *when* to stop embedding. How does SBS handle a variable-length trajectory with early stopping?

8. **The implicit assumption of "solution quality improves with search":** For VNE, is this true? The action mask ensures feasibility, but the search may not find *good* solutions if the policy is untrained. At low policy quality, is SBS better than random search? At high policy quality, does SBS find solutions near the ILP optimum? The answer determines whether TaSaR is worth implementing for VNE.

9. **Why not also compare to Monte Carlo Tree Search?** The paper dismisses MCTS as "time-consuming and hard to parallelize." But there has been significant progress on parallel MCTS (e.g., Vezhnevets et al., Liu et al.). Is the dismissal still valid for modern GPU-based MCTS implementations? For VNE, a well-tuned parallel MCTS might be competitive.

10. **Is the "periodic commitment" the essential innovation, or the SBS structure?** The section frames the contribution as SBS + periodic commitment. But commitment (taking `s` steps, then reconsidering) could be applied to MCTS, WOR, or even REINFORCE rollouts. What part of the method actually drives improvement over WOR?

---

## Synthesis for VNE Adaptation

### What we must borrow from Section 2

| Knowledge | How to use it |
|-----------|---------------|
| Self-improvement needs structured search, not i.i.d. sampling | Use SBS (not MC sampling) for VNE self-improvement |
| Problem-specific tricks don't transfer | Do not try to build VNE-specific heuristics; rely on TaSaR's generic search |
| Search parallelizability is critical | SBS's GPU-friendly beam search is the right choice for VNE |
| Heavy decoders (LEHD) are hard for RL | Monitor whether the LEHD-based VNE policy limits TaSaR effectiveness; consider alternatives |
| The paper's method is hyperparameter-robust by design | Exploit this: start with default parameters, only tune if VNE-specific issues arise |

### What we must verify independently

| Uncertainty | Test |
|-------------|------|
| SBS solution quality on VNE at zero-shot | Pre-experiment: run SBS on VNE with random policy vs pretrained policy |
| Optimal beam width for VNE's action space | Ablation: vary beam width at fixed replan step |
| Whether "periodic commitment" matters for VNE ablation | Compare: SBS-with-commitment vs SBS-without vs standard beam search |
| Cost of self-improvement vs ILP labeling | Wall-clock comparison: 1 epoch of Gumbeldore training vs generating 50k ILP solutions |

---

## Relationship to Other Sections (Preview)

- **Section 3 (Preliminaries):** Will formalize SBS, Gumbel-Top-k, and the beam search machinery. This section (Section 2) motivates WHY SBS was chosen.
- **Section 4 (TaSaR):** Describes the periodic commitment mechanism. This section explains WHY commitment is needed (pure beam search is exploitative, not diverse enough for SIL).
- **Section 5 (Experiments):** Tests TaSaR against the methods surveyed here. The key comparison is against Gumbeldore WOR (which has the complex tuning problem noted here) and against the LWKD baseline.

---

*Analysis prepared for VNE adaptation research. This document will feed into the synthesis phase alongside analyses of Sections 1, 3, 4, 5, and 6 of the paper.*
