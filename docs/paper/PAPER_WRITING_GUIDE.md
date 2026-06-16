# Paper Writing Guide: VNE + TaSaR

> Reverse-engineered from "Take a Step and Reconsider: Sequence Decoding for Self-Improved Neural Combinatorial Optimization" (Pirnay & Grimm, ECAI-2024).
>
> This guide synthesizes section-level analyses (see `docs/paper/section-analysis/`) into a practical
> writing reference for the future VNE+TaSaR paper. Each analysis was produced by reading that section of
> the ECAI paper and extracting format, content strategy, implicit assumptions, and VNE-relevant
> takeaways.

---

## Table of Contents

1. [Paper Structure Deconstruction](#1-paper-structure-deconstruction)
2. [Writing Patterns to Emulate](#2-writing-patterns-to-emulate)
3. [VNE Paper Outline](#3-vne-paper-outline)
4. [Results Presentation Strategy](#4-results-presentation-strategy)
5. [Publication Strategy](#5-publication-strategy)
6. [Common Pitfalls to Avoid](#6-common-pitfalls-to-avoid)

---

## 1. Paper Structure Deconstruction

### 1.1 The Narrative Arc

The ECAI paper follows a five-act arc:

| Act | Section | Function | Page share |
|-----|---------|----------|------------|
| Setup | Abstract + Intro (Section 1) | Frame the problem, state the trilemma, announce the solution | ~15% |
| Context | Related Work (Section 2) | Position against alternatives, acknowledge what exists | ~15% |
| Foundation | Preliminaries (Section 3) | Formalize CO-as-MDP and self-improved learning cycle | ~15% |
| Core | Method (Section 4) | The TaSaR algorithm, explained via tree metaphor | ~20% |
| Validation | Experiments (Section 5) | Empirical results, compute fairness, ablations | ~30% |
| Close | Conclusion (Section 6) | Summarize, claim generality, open the door | ~5% |

**Key observation**: The paper front-loads the problem (acts 1-2) but does not reveal the full algorithm until halfway through. The reader understands *why* a new decoding method is needed before seeing *what* it is.

### 1.2 How Each Section Is Organized

**Abstract** (one paragraph, ~15 lines):

- Sentence 1: Framing (CO as MDP, neural policy solves by decoding).
- Sentence 2-3: Shift to self-improved learning, its motivation.
- Sentence 4-5: The contribution (simple, problem-independent, sampling without replacement).
- Sentence 6: The mechanism (follow best solution, resample, block seen sequences).
- Sentence 7-8: Results (strong on TSP/CVRP, outperforms prior on JSSP).
- Sentence 9: Implicit claim (generality, no problem-specific heuristics).

Lessons: Every sentence carries weight. No background citations. Results are stated as fact, not hedged. The contribution is described before the mechanism.

**Introduction** (Section 1, ~2 pages):

1. Open with the CO problem class and its practical importance (1 paragraph).
2. Introduce the three-way paradigm trilemma: SL needs expensive labels, RL needs backward through trajectories (high memory), SIL avoids both (1 paragraph).
3. Establish that existing decoding methods fail SIL's needs: MCTS is too slow, MC sampling lacks diversity, WOR advantage diminishes with length (1 paragraph).
4. Introduce TaSaR as the solution: commit-explore-reconsider cycle, mechanism sketch, two hyperparameters (k, s) (2 paragraphs).
5. Summarize contributions as a bullet list (3-4 items).
6. Describe paper structure (1 short paragraph).

Lessons: The trilemma framing is powerful — it makes the reader feel the problem before offering the solution. The contribution list at the end of the introduction is the single most important paragraph for reviewers.

**Related Work** (Section 2, ~1.5 pages):

1. Paragraph sequence: RL-based methods -> SL-based methods -> search-based methods -> self-improved methods -> position TaSaR.
2. Each paragraph: name the approach, state what it achieves, state its limitation, transition to the alternative.
3. Key rhetorical move: TaSaR is positioned by *negation* — not REINFORCE, not SL-on-experts, not POMO, not MCTS, not vanilla WOR.
4. The last paragraph orients the reader: "this paper goes beyond [prior work] by introducing [contribution]."

Lessons: Never dismiss a baseline — always acknowledge its merits, then identify a specific gap that TaSaR fills. The paper positions TaSaR as the *generic* solution that the problem-specific methods cannot match.

**Preliminaries** (Section 3, ~2 pages):

1. Formalize the CO-as-MDP setup (1 paragraph).
2. Define autoregressive policy factorization (1 paragraph).
3. Describe the self-improved learning cycle in three phases (2 paragraphs).
4. Explain the leader mechanism for stability (1 paragraph).
5. Set up the notation used throughout the rest of the paper.

Lessons: This section is entirely definitional — no results, no opinions. It exists so that the Method section can reference formal notation. Every equation introduced here is used in Section 4. The paper does not *argue* the SIL cycle here; it only *describes* it.

**Method** (Section 4, ~3 pages):

1. Open with the tree traversal metaphor (partial solutions as nodes, complete solutions as leaves) — this is the conceptual hook (1 paragraph).
2. Define SBS via Gumbel perturbations (1-2 paragraphs).
3. Define the tree data structure (trie nodes, stored per-node probabilities) (1 paragraph).
4. The TaSaR loop: commit -> sample -> remove mass -> reconsider (the core algorithm, as pseudocode + walkthrough) (3-4 paragraphs).
5. Design decisions: mass removal mechanism, Top-p coupling, first-round determinism (2 paragraphs).
6. Hyperparameter discussion: k and s as intuitive dials (1 paragraph).
7. Contrast with prior SBS and WOR (1 paragraph).

Lessons: The tree traversal metaphor is the paper's single most effective explanatory device. Pseudocode is included. The mass removal mechanism (subtract in log space via `log_subtract`) is explained in detail, not glossed over. The hyperparameter discussion emphasizes simplicity (two knobs, clear intuition).

**Experiments** (Section 5, ~4 pages):

1. Section 5.1: Routing problems (TSP, CVRP) — setup, baselines, main results table, compute fairness analysis.
2. Section 5.2: JSSP — setup, multi-size training, baselines, main results table, Taillard benchmark.
3. Section 5.3: Ablation analysis — effect of k, effect of s, SIL convergence tracking, training details.
4. Section 5.4: Comparison with baselines.
5. Each subsection: Setup paragraph -> table of results -> interpretation paragraph.

Lessons: Tables are the centerpiece of this section. Each table is accompanied by analysis text that highlights specific numbers and explains why they matter. The compute fairness analysis (converting beam-search node expansions to wall-clock equivalents) is a signature move that preempts "unfair comparison" criticism.

**Conclusion** (Section 6, ~0.5 pages):

1. Reiterate the problem and the contribution (1 paragraph).
2. State the three problems where TaSaR succeeds (1 sentence).
3. Claim generality ("can in principle also be used in other problem-specific SIL approaches") (1 sentence).
4. Acknowledgments.

Lessons: The conclusion is short and confident. No hedging, no limitations section, no future work. This is a calculated choice — limitations and future work were either integrated into the experiments section or omitted entirely.

### 1.3 Transitions Between Sections

The paper uses explicit structural transitions:

- End of Introduction: "The rest of this paper is organized as follows."
- End of Related Work: "In this work we overcome these limitations by introducing..."
- End of Preliminaries: "With the preliminaries established, we now introduce..."
- End of Method: "We now test this method empirically."
- End of Experiments: transitions naturally to Conclusion.

Each transition is a single sentence. The paragraph before the transition summarizes what was just established and points to what comes next.

### 1.4 Length and Depth Balance

| Section | Estimated lines | What it prioritizes |
|---------|----------------|---------------------|
| Abstract | 15 | Maximum density |
| 1. Introduction | ~80 | Problem framing, contribution statement |
| 2. Related Work | ~60 | Coverage + differentiation |
| 3. Preliminaries | ~70 | Formal foundations |
| 4. Method | ~120 | Algorithm + design decisions |
| 5. Experiments | ~200 | Empirical evidence |
| 6. Conclusion | ~20 | Summary + generality claim |

The experiments section is the longest by a wide margin. The paper earns its claims through results, not argumentation.

---

## 2. Writing Patterns to Emulate

### 2.1 Problem Framing

**Pattern**: "X is important but existing methods have limitation Y."

Example (from the intro's implicit structure):

> "Neural combinatorial optimization has emerged as a promising approach... However, existing methods face a fundamental trilemma..."

The paper never says "our method is innovative" — it says "here is a problem that existing methods have not solved." The contribution is positioned as a solution to a clearly defined gap.

**TaSaR-specific framing**: The paper identifies a specific gap in self-improved learning: existing decoding methods fail the three criteria of speed, diversity, and generality. TaSaR fills this gap.

**VNE adaptation**: "Virtual Network Embedding is a critical problem for network virtualization, but neural approaches have been limited by the prohibitive cost of optimal labels (ILP is intractable at scale) and the instability of reinforcement learning in large action spaces. Self-improved learning offers a way out, but requires a decoding method that can effectively explore VNE's exponentially large solution space."

### 2.2 Contribution Positioning

**Pattern**: Simple + Problem-Independent + Effective.

The paper uses "simple" as a deliberate rhetorical strategy — it signals that the method is not a complex assembly of heuristics, that the results are attributable to the core mechanism, not to engineering tricks.

How to position in VNE paper:

> "We show that a generic sequence decoding method, applied without any problem-specific modifications, achieves state-of-the-art results on VNE — a problem that is structurally more complex than the routing and scheduling benchmarks on which the method was originally validated."

This positioning does three things: (1) establishes VNE's difficulty, (2) emphasizes the method's generality (no VNE-specific modifications), (3) implies that if TaSaR works on VNE, it truly is problem-independent.

### 2.3 Related Work Handling

**Pattern**: Acknowledge -> Differentiate.

For each related method:
1. One sentence describing what it does and what it achieves.
2. One sentence identifying a specific limitation relevant to the paper's contribution.
3. Transition to the alternative.

Example (paraphrasing the paper's pattern for SBS vs TaSaR):

> "Pirnay and Grimm (2023) introduced Gumbeldore, a stochastic beam search with multiple rounds and advantage-weighted importance sampling. However, the advantage-scaling hyperparameters can be complex to tune, and the method's iterative structure does not exploit the structure of self-improved learning. TaSaR addresses both limitations by removing the advantage mechanism entirely and introducing a deterministic commitment phase."

**Key differentiation move**: The paper positions prior work as having *more parameters and less structure*, not as *wrong*. TaSaR wins by being simpler, not by being more complex.

### 2.4 Technical Material Presentation

**Pattern**: Metaphor -> Formal definition -> Algorithm -> Walkthrough.

1. **Metaphor**: "Partial solutions are nodes in a tree, complete solutions are leaves."
2. **Formal definition**: Define p(bar_a_{1:d}) as node probability.
3. **Algorithm**: Pseudocode of `perform_tasar`.
4. **Walkthrough**: Step-by-step explanation of what happens in each iteration.

The metaphor is the most important element — it gives the reviewer a mental model before diving into details. The pseudocode is a reference, not the primary explanation.

**Notation conventions from the paper**:

- `a_i`: action at step i
- `bar_a_{1:d}`: partial solution (sequence of d actions)
- `pi_theta(a|bar_a_{1:d})`: policy's conditional distribution
- `p(bar_a_{1:d})`: unnormalized total probability of a partial solution (product of conditional logits)
- `k`: beam width
- `s` (or `replan_steps`): number of steps before reconsideration
- `l`: sequence length
- `f`: solution quality metric (to maximize)

### 2.5 Results Discussion

**Pattern**: Present -> Interpret -> contextualize -> acknowledge limitation.

Example (paraphrased):

> "Table 1 shows the main results. With k=64 and s=10, TaSaR achieves an optimality gap of 0.0009% on TSP N=100, improving from 0.37% with greedy decoding. For CVRP, the improvement is from 2.96% to 0.14%. Notably, the relative improvement is larger for harder problems. [Interpretation]: this is because the diversity-forcing mechanism is more valuable when the policy is less confident. [Context]: The improvement is consistent across all test sizes. [Limitation]: For TSP, the greedy gap is already small, so the absolute gain is modest."

**Key principles**:
- Always give the greedy baseline first, then the search-improved result.
- State the absolute improvement AND the relative improvement.
- Contextualize by problem difficulty.
- Acknowledge when the gain is modest (TSP).
- Use the compute fairness formula to preempt "unfair comparison" criticisms.

### 2.6 The Simplicity Narrative as Rhetorical Strategy

The paper tells a consistent story: complex methods (MCTS, advantage-weighted WOR, multi-round RL) are fragile, hard to tune, and problem-specific. TaSaR is deliberately simple — two hyperparameters, one loop, no problem-specific tricks.

This narrative is reinforced in every section:
- Abstract: "simple and problem-independent sequence decoding method"
- Introduction: "While beam search and MCTS are used for decoding in NCO... they either lack stochasticity or are too time-consuming"
- Method: "Our method is strikingly simple"
- Conclusion: "surprisingly simple"

**For VNE paper**: Continue this narrative. VNE's complexity (joint node-link allocation, tight constraints, exponential action space) makes the simplicity of TaSaR *more* impressive, not less. Frame it as: "Despite VNE's additional complexity, the same simple method works without modification."

---

## 3. VNE Paper Outline

### 3.1 Proposed Section Structure

| Section | Proposed title | Length estimate | Purpose |
|---------|---------------|-----------------|---------|
| Abstract | (untitled) | ~15 lines | Maximum-density summary |
| 1 | Introduction | 2 pages | Frame VNE challenge, identify plateau, announce TaSaR as solution |
| 2 | Related Work | 1.5 pages | VNE-NCO methods, SIL methods, decoding methods |
| 3 | Preliminaries | 2 pages | VNE problem formulation, SIL cycle, notation |
| 4 | Method | 2.5 pages | TaSaR algorithm (adapted from ECAI paper, with VNE adaptations) |
| 5 | VNE-Specific Adaptations | 1.5 pages | Structured actions, feasibility handling, action space bucketing |
| 6 | Experiments | 4 pages | Setup, baselines, results, ablations, generalization |
| 7 | Conclusion | 0.5 pages | Summary, generality, limitations |
| - | Acknowledgments | 2 lines | Funding, compute resources |

**Total**: ~14 pages (standard conference paper).

**Key structural changes from ECAI paper**:

1. The VNE problem is harder and less familiar to the NCO community, so the problem formulation needs more space (subsection within Preliminaries).
2. A dedicated "VNE-Specific Adaptations" section is needed because the method is *not* purely plug-and-play — the trajectory implementation, feasibility handling, and action bucketing are engineering contributions worth describing.
3. The "Related Work" section now covers both VNE-NCO literature and general NCO/SIL methods.

### 3.2 Framing VNE as a Harder Problem

**Narrative strategy**: VNE is to TSP/CVRP as JSSP is to TSP — a structurally more complex problem that tests the method's generality more severely.

**Specific points to make**:

1. **Joint allocation**: VNE requires simultaneous node mapping and link routing, unlike TSP (pure ordering) or CVRP (ordering with capacity).
2. **Exponential action space**: VNE's per-step actions combine substrate node selection and path routing, producing a branching factor that dwarfs TSP's n-remaining-cities.
3. **Tight feasibility constraints**: CPU, bandwidth, and adjacency constraints create many dead-end partial solutions, making exploration harder.
4. **Sequential interdependence**: Early node placements determine all later routing options — the exact failure mode that commit-and-reconsider addresses.
5. **No efficient optimal solver**: Unlike TSP (Concorde) or CVRP (HGS), VNE's ILP formulation does not scale to large instances, making SIL not just convenient but necessary.
6. **Variable-length trajectories**: Embedding may fail partway, unlike TSP's fixed N steps.

**Key claim**: "VNE combines every challenge that makes self-improved learning difficult — long trajectories, exponential action spaces, tight constraints, costly evaluation — making it the most stringent test of TaSaR's generality to date."

### 3.3 Positioning TaSaR as the Natural Solution

**For each VNE challenge, show how TaSaR addresses it**:

| VNE challenge | TaSaR mechanism that addresses it |
|---------------|-------------------------------------|
| Tight constraints cause infeasible branches | SBS naturally excludes -inf logits; mass removal forces exploration away from near-identical (and likely infeasible) completions |
| Early placements cascade | Commit-and-reconsider: follow the best prefix, then reconsider from intermediate states |
| Enormous action space | Gumbel-Top-k focuses computation on the most promising fraction; bucketing (Section 5) manages memory |
| No optimal labels | SIL cycle generates improving pseudo-labels from TaSaR's own search, bypassing ILP |
| Expensive evaluation (embedding simulation) | SBS parallelizes across batch dimension, amortizing network and simulation costs |
| Long trajectories | TaSaR's advantage grows with trajectory length (WOR diminishes with length; TaSaR's progressive shortening counters this) |

### 3.4 What to Emphasize

**Emphasis hierarchy**:

1. **MOST IMPORTANT**: TaSaR's problem-independence. If it works on VNE without modification, this is the strongest evidence yet for TaSaR as a general-purpose NCO tool.
2. **SECOND**: VNE's ILP gap. Show that self-improvement significantly closes the gap to exact optimality, which supervised pre-training on 50k ILP labels could not do (from the A/B results).
3. **THIRD**: The plateau finding. "Supervised learning on ILP-optimal labels saturates quickly — 1k and 50k labels achieve the same validation objective. Self-improved learning breaks this plateau."
4. **FOURTH**: Generalization. Show that TaSaR-trained policies generalize to larger substrates and different request distributions.
5. **FIFTH**: Hyperparameter robustness. Show that TaSaR works across a range of k and s without careful tuning, supporting the simplicity narrative.

### 3.5 Where to Place VNE-Specific Contributions

The paper should distinguish clearly between:
- **Method contribution**: TaSaR itself (attributed to Pirnay & Grimm).
- **Application contribution**: First application of TaSaR to VNE (our contribution).
- **Engineering contribution**: Structured-action trajectory, feasibility-aware SBS, candidate bucketing for VNE (our contribution).
- **Empirical contribution**: VNE results, ablation studies, plateau analysis (our contribution).

**Placement**:

- Method section (Section 4): Adapted TaSaR description — cite the original, then describe the implementation faithfully.
- VNE-Specific Adaptations (Section 5): Structured action decisions, feasibility checking within the SBS loop, candidate bucketing for memory management. These are what the reviewer needs to reproduce the VNE results.
- Experiments (Section 6): All results. The comparison to supervised pre-training is the headline result.

**Note on co-authorship and attribution**: The paper must clearly credit Pirnay and Grimm for TaSaR. Our contribution is the VNE application, the adaptation, and the empirical findings. The paper title should reflect this — something like "Take a Step and Reconsider for Virtual Network Embedding: A Self-Improved Neural Approach" (not "A New Method for NCO").

### 3.6 Detailed Section-by-Section Outline

**Abstract** (~15 lines):

1. VNE is critical but computationally intractable.
2. Neural approaches are promising but limited by label availability (ILP too expensive) and RL instability.
3. Self-improved learning bypasses both, but requires an efficient decoding method.
4. We apply TaSaR, a generic sequence decoding method based on commit-and-reconsider, to VNE — without any problem-specific modifications.
5. TaSaR achieves [X]% gap reduction over supervised pre-training, closing [Y]% of the gap to ILP optimality.
6. The method outperforms [baselines] and generalizes to [larger/more complex instances].
7. This is the first demonstration of self-improved learning for VNE and the most challenging test of TaSaR's generality to date.

**Section 1: Introduction** (~2 pages):

1. Paragraph 1: VNE importance (network virtualization, 5G, cloud computing). The problem is NP-hard. Current approaches are heuristic-based or ILP-limited.
2. Paragraph 2: Neural VNE. Past work uses supervised learning (limited by label cost), RL (limited by stability and architecture capacity).
3. Paragraph 3: Our finding: supervised VNE plateaus at the same objective with 1k or 50k ILP labels — data quantity is not the bottleneck. The plateau is a decoding problem.
4. Paragraph 4: Self-improved learning breaks this plateau by generating improving pseudo-labels. But it requires a decoding method that is fast, diverse, and general.
5. Paragraph 5: TaSaR (Pirnay & Grimm, 2024) is exactly such a method. Intuitive description: commit-explore-reconsider cycle.
6. Paragraph 6: Contributions (bullet list):
   - First application of TaSaR to VNE.
   - Demonstration that self-improved learning breaks the supervised plateau.
   - VNE-specific adaptations (structured actions, feasibility handling).
   - Extensive ablation study.
   - State-of-the-art VNE results.
7. Paragraph 7: Paper structure.

**Section 2: Related Work** (~1.5 pages):

1. VNE methods (heuristic, ILP, neural).
2. Neural NCO methods (Pointer Networks, AM, POMO, BQ/LEHD) — emphasize that BQ/LEHD architectures are VNE's backbone.
3. Self-improved learning (Luo et al., Corsini et al., Gumbeldore).
4. Decoding methods (beam search, MCTS, WOR, SBS).
5. Position TaSaR: generic, simple, parallel, and designed for SIL.

**Section 3: Preliminaries** (~2 pages):

1. VNE formulation: substrate graph, virtual network request, node mapping, link mapping, constraints (CPU, bandwidth, adjacency), objective (minimize embedding cost, maximize acceptance).
2. CO-as-MDP for VNE: state = current partial mapping, action = map next virtual node + route incident links, transition = deterministic update of resource availabilities, terminal = all virtual nodes mapped or infeasible.
3. Autoregressive policy: factorization over sequential actions.
4. Self-improved learning cycle: generate -> train -> validate -> repeat. Leader mechanism.
5. The decoding challenge: need for speed, diversity, and generality.

**Section 4: Method — TaSaR** (~2.5 pages):

1. Tree traversal metaphor for autoregressive decoding.
2. Stochastic Beam Search via Gumbel-Top-k.
3. The TaSaR loop: commit, sample, remove mass, reconsider.
4. Mass removal: `log_subtract`, trie data structure, recursive propagation.
5. Design decisions: Top-p coupling, first-round determinism.
6. Hyperparameters: k (beam width) and s (step size).

(This section is adapted from the ECAI paper. Cite the original. Keep it concise — the informed reader should recognize the algorithm. Focus on clarity, not novelty.)

**Section 5: Adapting TaSaR to VNE** (~1.5 pages):

1. **Trajectory design**: Decision order (chain ordering vs. general DAG), action representation for structured actions (substrate node + routing path).
2. **Feasibility handling**: Dead-end detection, -inf logit assignment for infeasible actions, recovery mechanisms.
3. **Candidate bucketing**: Managing VNE's enormous action space via batched logit scoring with bucketed candidate sets.
4. **Admission control** (if applicable): Handling variable-length trajectories via termination action.
5. **Policy network architecture**: LEHD encoder-decoder, substrate and request embedding, compatibility scoring.

This section is the paper's secondary contribution (after the empirical results). It should be sufficient for a reader to reproduce the VNE implementation.

**Section 6: Experiments** (~4 pages):

Subsection 6.1: Setup
- Dataset description (training/validation/test, instance distributions).
- ILP optimality labels (how they were generated, solver used).
- Training configuration (PyTorch, optimizer, hardware).
- Baseline methods (greedy, supervised pre-training, WOR sampling).
- TaSaR configuration (k, s, Top-p values tested).

Subsection 6.2: Main Results
- Table: Greedy vs Supervised vs WOR vs TaSaR.
- Table: TaSaR(k,s) configurations.
- Headline: gap to ILP, absolute acceptance rate, embedding cost.
- Key figure: training curve comparison (supervised vs SIL).

Subsection 6.3: Ablation Study
- Effect of beam width k (sweep: 8, 16, 32, 64).
- Effect of step size s (sweep: 2, 4, 8).
- Impact of Top-p (sweep: 0.8, 0.9, 1.0).
- Impact of first-round deterministic vs. stochastic.

Subsection 6.4: Generalization
- Performance on larger substrates (extrapolation).
- Performance on different request distributions.
- Performance with varying request counts.

Subsection 6.5: Compute Analysis
- Wall-clock time comparison.
- Compute fairness formula applied to VNE.
- Scaling analysis.

**Section 7: Conclusion** (~0.5 pages):

1. TaSaR, applied without VNE-specific modifications, achieves state-of-the-art VNE results.
2. Self-improved learning breaks the supervised plateau.
3. VNE provides the strongest evidence yet for TaSaR's generality.
4. Limitations: applicability to the admitted-VNE setting, ILP gap remaining at large sizes.
5. Future work: admission control, hybrid heuristics, larger-scale training.

---

## 4. Results Presentation Strategy

### 4.1 Making Results Look Compelling

**Principle**: Every number you present should answer an implicit question the reviewer has.

**Preempt reviewer questions**:

| Reviewer question | How to answer |
|-------------------|---------------|
| Does this beat simple baselines? | Show greedy decoding gap. |
| Does search help at all? | Show gap reduction from greedy to TaSaR. |
| Does training help beyond the initial pretraining? | Show SIL epoch curves (policy improves over time). |
| Are gains just from more compute? | Compute fairness analysis. |
| Does it generalize? | Out-of-distribution test results. |
| Is it reproducible? | Exact hyperparameter values, seed ranges. |

**Key numbers to extract from VNE**:

- Gap to ILP: `(model_objective - ilp_objective) / abs(ilp_objective)` — the paper's equivalent of optimality gap.
- Gap reduction: `(greedy_gap - tasar_gap) / greedy_gap`.
- Acceptance rate (if admission is on).
- Percentage of ILP gap closed: `(greedy_gap - tasar_gap) / (greedy_gap - 0)`.

**If the plateau at ~-23 is not broken** (realistic scenario):

Reframe the result: "TaSaR achieves the same objective as supervised pre-training but without requiring ILP-optimal labels" — this is still a significant result (demonstrating that the SIL loop is viable). Less flashy but publishable.

**If plateau IS broken**:

Lead with: "Self-improved learning with TaSaR reduces the optimality gap by X%, from Y% (supervised pre-training) to Z% (TaSaR after N epochs)."

### 4.2 Table Design (Matching the Paper's Style)

The ECAI paper uses clean, minimal tables. Each table has a short caption and footnoted values.

**Table template (main results)**:

```
Table 1: VNE embedding results. Gap = relative gap to ILP optimum.
Lower is better. Best results in bold.

| Method | Substrate 20-40 | Substrate 40-60 | Substrate 60-80 |
|--------|:-:|:-:|:-:|
| Greedy (beam=1) | X.X% | X.X% | X.X% |
| Supervised (pretrained) | X.X% | X.X% | X.X% |
| WOR (k=64) | X.X% | X.X% | X.X% |
| TaSaR (k=32, s=4) | X.X% | X.X% | X.X% |
| TaSaR (k=64, s=4) | **X.X%** | **X.X%** | **X.X%** |
| TaSaR (k=64, s=8) | X.X% | X.X% | X.X% |
| ILP optimum | 0% | 0% | 0% |

† Standard error across 5 seeds.  Confidence intervals via bootstrap.
```

**Table template (ablation)**:

```
Table 2: Ablation on beam width k (fixed s=4, Top-p=0.95).
Gap relative to ILP optimum.

| Beam width k | Valid instances | Gap | Wall-clock (s) |
|:-:|:-:|:-:|:-:|
| 8 | X% | X.X% | X.X |
| 16 | X% | X.X% | X.X |
| 32 | X% | X.X% | X.X |
| 64 | X% | X.X% | X.X |
| 128 | X% | X.X% | X.X |
```

**Prose pattern around tables**:

```
Table 1 presents the main results. The key finding is that TaSaR with k=64
achieves a gap of X%, compared to Y% for supervised pre-training — a Z%
relative reduction. Notably, WOR sampling (k=64) achieves only X%, confirming
that TaSaR's commit-and-reconsider mechanism provides additional value beyond
simple diversity-forcing.

For the largest instances (substrate 60-80), the improvement is largest:
this is consistent with TaSaR's design advantage on longer trajectories and
more constrained problems.
```

### 4.3 Figure Strategy

**Figure 1: Training curves** (Section 6.2):
- X-axis: Training epoch.
- Y-axis: Validation objective (or gap to ILP).
- Lines: Supervised pre-training (1 epoch, then flat), SIL with TaSaR (improving over epochs).
- Show that SIL improves while supervised plateaus.

**Figure 2: Gap vs compute** (Section 6.5):
- X-axis: Wall-clock time (or node transitions).
- Y-axis: Gap to ILP.
- Lines: Greedy, WOR(k), TaSaR(k,s).
- Show the Pareto frontier.

**Figure 3: Effect of k and s** (Section 6.3):
- 2D heatmap or paired bar chart.
- Columns: k values (8, 16, 32, 64).
- Within each column: bars for s values (2, 4, 8).
- Y-axis: Gap to ILP.

**Figure 4: Generalization** (Section 6.4):
- X-axis: Test instance complexity (substrate size, request count).
- Y-axis: Gap to ILP.
- Lines: Greedy, supervised, TaSaR.
- Show that the gap widens more slowly for TaSaR.

**Figure 5: SIL convergence** (potentially in Section 6.2):
- Per-epoch training loss and validation objective.
- Show the self-improvement cycles (train -> decode -> improve -> repeat).
- Include a secondary axis for pseudo-label quality (best decoding objective per epoch).

### 4.4 Generalization Story

**The generalization narrative should be**:

1. The supervised policy generalizes poorly from training to out-of-distribution instances (plateaus).
2. TaSaR's decoding improves generalization at inference time: by exploring multiple completions, the search finds better solutions even on unfamiliar instance types.
3. The SIL loop compounds this: better decoding produces better pseudo-labels, training a better policy, which decodes even better.

**Present generalization across three axes**:
1. **Larger substrates**: Train on 20-40 nodes, test on 40-60 and 60-80.
2. **More requests**: Train on 2-6 requests, test on 4-8 or 6-10.
3. **Different constraint regimes**: Train on high-capacity substrates, test on tighter substrates.

**For each axis**: Show that the gap increase from training to test distribution is smaller for TaSaR than for supervised or greedy baselines.

### 4.5 Statistical Presentation Best Practices

- Report **mean and standard error** across multiple seeds (at least 5).
- Use **confidence intervals** (95%) via bootstrap.
- Report **effect sizes** (gap reduction percentage, not just absolute numbers).
- Include **individual trial values** in supplementary material.
- **Always tie compute to quality**: "TaSaR(k=64, s=4) requires X node transitions per instance, equivalent to Y greedy decodes."
- **Two-column tables** in tables: first column for method name, subsequent columns for problem variants.
- **Footnotes** for important caveats (e.g., "Standard error across 5 seeds").
- **No vertical lines** in tables (matches the paper's style).
- **Bold the best result** in each column, optionally underline the second-best.

---

## 5. Publication Strategy

### 5.1 Target Venues

| Venue | Tier | What it values | VNE fit |
|-------|------|----------------|---------|
| ECAI | Top European AI | Problem-solving, rigorous empirical work, clarity | Strong: ECAI published the original TaSaR; VNE application fits the application track |
| AAAI | Top general AI | Novelty, broad impact, strong results | Good: positioning VNE as broader NCO test is compelling |
| IJCAI | Top general AI | Similar to AAAI, accepts more papers | Good: same positioning as AAAI |
| NeurIPS | Top ML | Methodological contribution, theoretical grounding | Weak: our contribution is application, not method |
| ICML | Top ML | Similar to NeurIPS | Weak: same reason |
| INFOCOM | Top networking | Networking relevance, practical impact | Strong: VNE is a networking problem; INFOCOM values operations research + ML |
| IEEE/ACM Trans. Networking | Top journal | Comprehensive evaluation, reproducibility | Good: more space for full SIL analysis |

**Recommendation**: Target ECAI (method venue) or INFOCOM (application venue) as primary. AAAI/IJCAI as secondary. The paper is an application paper with a strong method component — it fits ECAI's scope well.

### 5.2 What Each Venue Values

**ECAI**:
- Values problem-solving rigor over theoretical depth.
- Accepts both method and application papers.
- The original TaSaR was published here — reviewers will be familiar with the method.
- "Call for Papers" explicitly welcomes "AI for network and service management."
- Page limit: 8 pages (ECAI 2024).

**AAAI/IJCAI**:
- Values novelty and impact. Requires a clear "this is new" signal.
- Positioning as "first self-improved VNE" is novel enough.
- General AI audience needs more explanation of VNE.
- Page limit: 8 pages.

**NeurIPS/ICML**:
- Position VNE as a challenging test of a general method — "yet another application" may not pass.
- Only viable if the results are exceptionally strong (e.g., gap reduced from >50% to <5%).
- Have a "datasets and benchmarks" track that could fit.

**INFOCOM**:
- VNE is a core networking problem — natural fit.
- Values practical results and real-world relevance.
- Less emphasis on methodological novelty — emphasize VNE results.
- Network community may not know TaSaR — more background needed.

### 5.3 Positioning for Maximum Acceptance Probability

**Strategy A: ECAI (recommended primary)**

Positioning: "We extend TaSaR, a method published at this same venue, to a new and harder problem class (VNE). This provides the strongest evidence yet for TaSaR's problem-independence."

Advantages:
- Reviewers may include the original authors or their collaborators.
- The method is already accepted at this venue; novelty is in the application.
- The VNE community at ECAI will appreciate the problem coverage.

**Strategy B: AAAI/IJCAI**

Positioning: "We identify the fundamental limitation of supervised NCO (the data plateau) and show that self-improved learning with TaSaR escapes it. VNE is the case study, but the finding generalizes."

Advantages:
- Novel contribution (plateau finding) is method-independent.
- VNE as a severe test bed makes the generality claim stronger.
- Higher prestige venue.

**Strategy C: INFOCOM**

Positioning: "State-of-the-art neural VNE through self-improved learning — no ILP oracles needed."

Advantages:
- Natural fit for the networking audience.
- Less competition from ML methods.
- More space for VNE-specific details.

### 5.4 Timeline Planning

**Assumption**: 8-page ECAI format (references and appendix excluded from page count).

| Phase | Task | Timeline |
|-------|------|----------|
| Phase 1 | Complete TaSaR experiments on VNE | 2-4 weeks |
| Phase 2 | Run baseline comparisons (WOR, greedy, supervised) | 1-2 weeks |
| Phase 3 | Ablation studies (k, s, Top-p grid) | 1-2 weeks |
| Phase 4 | Generalization experiments | 1-2 weeks |
| Phase 5 | Compute fairness analysis | 3-5 days |
| Phase 6 | Write first draft | 2-3 weeks |
| Phase 7 | Internal review and iteration | 2 weeks |
| Phase 8 | Supplementary material (appendix, code release) | 1 week |
| Phase 9 | Final submission | 3 days |

**Total**: 10-14 weeks from start.

**Deadline planning**:

- ECAI: typically February/March deadline for September conference.
- AAAI: typically August deadline for February conference.
- IJCAI: typically January deadline.
- INFOCOM: typically July/August deadline.

**Recommendation**: Target a paper in 2026 or early 2027. Aim for ECAI 2026 or AAAI 2027.

### 5.5 Supplementary Material Strategy

The paper's page limit does not allow full detail. Supplementary material should include:

1. Full hyperparameter configuration (per the config.py).
2. All training curves for all seeds.
3. Complete results tables with per-instance values.
4. VNE instance generation details (seed ranges, distributions).
5. Reproducibility checklist (as required by many venues).
6. Code repository URL (prepare a clean version).

---

## 6. Common Pitfalls to Avoid

### 6.1 Overclaiming

**Pitfall**: Claiming "state-of-the-art" without exhaustive comparison, or claiming "problem-independent" when only tested on one problem.

**How to avoid**:
- Use precise language: "outperforms [specific baselines] on [specific instances]" not "state-of-the-art."
- For generality, present as evidence, not assertion: "These results provide evidence that TaSaR generalizes to VNE, a problem class with qualitatively different structure from the routing and scheduling problems in the original study."
- Never say "first" unless you have systematically searched for prior VNE+SIL work.
- Use hedges when appropriate: "to the best of our knowledge," "we observe that," "these results suggest."

**The paper's approach**: The conclusion says "surpasses existing SIL methods on three CO problems" and "new state-of-the-art NCO results on JSSP" — specific, measurable, defensible. Not "our method is the best ever."

### 6.2 Insufficient Baseline Comparison

**Pitfall**: Comparing only against naive baselines (greedy, random) and not against relevant prior work (WOR, Gumbeldore, supervised pre-training, RL VNE methods).

**Minimum baseline set**:
- Greedy decoding (beam=1) — establishes the floor.
- Supervised pre-trained policy (your Phase 1 result) — establishes what SIL must improve upon.
- WOR sampling (k=64, equivalent compute) — isolates TaSaR's commit-and-reconsider contribution.
- If available: a prior neural VNE method (e.g., a published RL-VNE result on compatible instances).

**How the paper handles it**: Four baselines for TSP/CVRP (SL, RL, Gumbeldore, SBS/base) and three for JSSP (GD, L2D, vanilla Monte Carlo). Each baseline is cited and its parameter configuration is reported.

### 6.3 Weak Ablation Studies

**Pitfall**: Reporting only the best configuration without showing the effect of each hyperparameter.

**Minimum ablation set**:
- Beam width k: sweep from 8 to 128 (or 4 to 64 for VNE given compute constraints).
- Step size s: sweep from 1 to n (or meaningful fractions: 1, 2, 4, n/2, n).
- Top-p: compare fixed (0.8, 0.95, 1.0) and no-Top-p.
- First-round deterministic vs. stochastic.
- With vs. without the mass removal mechanism (i.e., vanilla SBS vs. TaSaR).

**How the paper handles it**: The paper includes a full ablation (Section 5.3) with per-parameter sweeps, a figure showing the effect of k on CVRP, and a discussion of the s/l ratio.

### 6.4 Poor Reproducibility

**Pitfall**: Reporting results that cannot be reproduced because the exact configuration, seeds, or instance distributions are not specified.

**Minimum reproducibility checklist**:
- [ ] Exact Python package versions (torch, CUDA, HiGHS/Gurobi).
- [ ] Hardware description (GPU model, CPU, RAM).
- [ ] All hyperparameters with values (not ranges).
- [ ] Random seeds for each experiment.
- [ ] Instance generation logic (seed ranges, distributions, rejection criteria).
- [ ] Training duration and wall-clock time per epoch.
- [ ] Validation and test set composition.

**How the paper handles it**: The paper reports exact hyperparameter values (lr=2e-4, k=64, s=10, Top-p=0.95, 1000 batches of 1024 subtours, etc.) and includes compute fairness analysis. The acknowledgments name the cluster used.

### 6.5 Missing the Compute Fairness Analysis

**Pitfall**: Comparing TaSaR against baselines without accounting for the additional compute that search requires.

**The compute fairness formula** (from Section 5.3 of the paper):

For a sequence of length l, beam width k, and step size s:
- `t = ceil(l / s)` = number of replan rounds.
- Node expansions = `k * (t * l - (s * t^2 - s * t) / 2)`.

This quantifies how many greedy-equivalent decodes a TaSaR run costs, enabling fair comparison against WOR and naive MC sampling.

**For VNE**: VNE has a shorter effective sequence length (l ~ 12 for our instances) than TSP (l=100) or JSSP (l=300). The formula still applies — use it.

**Key point in the paper**: "In contrast, WOR can only compare a single depth for each sequence (one action per sequence) and Gumbeldore cannot easily parallelize multiple rounds." This justifies TaSaR's compute advantage even at equivalent node counts.

### 6.6 Ignoring the Negative Result

**Pitfall**: Reporting only the positive results and ignoring negative or null findings.

**VNE's potential negative results**:
- Gap to ILP remains large even after SIL training.
- SIL doesn't improve over supervised pre-training.
- TaSaR doesn't provide advantage over WOR for VNE.
- Large beam widths don't help because most branches are infeasible.

**How to handle**: Any of these is publishable if correctly framed. "Supervised pre-training on 50k ILP labels and self-improved learning with TaSaR achieve similar VNE objectives — suggesting that the policy architecture, not the training paradigm, is the current bottleneck." This is a valid scientific finding.

**The paper's approach**: The paper includes null results (CVRP LEHD SL 4.26% vs SIL 5.08% — supervised better than self-improved). It discusses this honestly: "We observe that supervised learning can still achieve better results... This may be because LEHD requires more optimization tuning to achieve its full generalization potential."

### 6.7 Writing Pitfalls Specific to VNE

1. **Assuming VNE knowledge**: The NCO audience may not know VNE. Define it clearly in 2-3 sentences. Use a figure for the substrate/request mapping.
2. **Mixing VNE and NCO terminology**: Be precise — "embedding cost" not "tour length," "substrate node" not "city," "request" not "instance."
3. **Overselling VNE difficulty**: VNE is hard, but don't say "unprecedentedly hard" — reviewers may know harder problems. Say "structurally more complex along these specific axes..."
4. **Ignoring the VHE community**: VNE has a separate literature (IEEE INFOCOM, IEEE/ACM ToN). Cite at least the major VNE-NCO papers to establish domain familiarity.
5. **Confusing instance size with difficulty**: VNE on 20-node substrates is not harder than TSP on 100 nodes in a comparable sense. Compare on appropriate difficulty axes (constraint density, action space branching).

### 6.8 The "Too Many Appendices" Problem

**Pitfall**: Shoving essential material into an appendix that the main paper does not reference.

**Rule**: Everything the reviewer needs to understand the method and its evaluation should be in the main paper. Appendices are for:
- Full hyperparameter tables.
- Additional per-instance results.
- Extended related work discussion.
- Reproducibility details.
- Broader impact statements.

The main paper should be self-contained enough that a reader who skips the appendix can still evaluate the contributions.

---

## Appendix A: Key Metrics Template

For each experiment, compute and report:

```
Metric                          | Definition                                     | How displayed
--------------------------------|------------------------------------------------|---------------
Mean objective                  | avg(model_objective)                           | Table, Figure 1
Optimality gap                  | (model - ilp) / abs(ilp)                      | Table (primary metric)
Gap reduction vs greedy         | (gap_greedy - gap_tasar) / gap_greedy          | In-text, Table footnote
Gap reduction vs supervised     | (gap_sup - gap_tasar) / gap_sup               | In-text, Table footnote
Percentage of ILP gap closed    | (gap_greedy - gap_tasar) / (gap_greedy - 0)   | Headline result
Valid solution rate             | count(feasible) / count(total)                | Table, Ablation
Wall-clock time per instance    | avg(instance_decode_time)                      | Table, Figure 2
Node transitions per instance   | avg(expansions)                                | Compute fairness analysis
Training epochs to convergence  | epochs_best_val + epochs_patience              | In-text
```

## Appendix B: Abstract Template for VNE+TaSaR Paper

> Virtual Network Embedding (VNE) is a fundamental challenge in network virtualization, requiring the joint allocation of computational and bandwidth resources across multiple virtual network requests. Neural approaches to VNE have been limited by the prohibitive cost of generating optimal labels via integer linear programming and the architectural constraints imposed by reinforcement learning. Self-improved learning, where a policy iteratively trains on its own decoded solutions, offers a path forward but depends critically on an efficient and diversity-enforcing decoding method. We adapt TaSaR -- a generic, problem-independent sequence decoding method based on a commit-and-reconsider mechanism -- to VNE without any problem-specific heuristics or architectural changes. Across diverse substrate topologies and request distributions, TaSaR reduces the mean optimality gap from X% (supervised pretraining) to Y% -- a Z% relative improvement -- while using no external optimal labels during training. Unlike supervised approaches that plateau at the same objective regardless of data quantity, self-improved training with TaSaR continues to improve over multiple epochs. This work provides the first demonstration of self-improved neural combinatorial optimization for VNE and the strongest evidence yet that TaSaR's commit-and-reconsider mechanism transfers to constraint-dominated allocation problems beyond the routing and scheduling benchmarks on which it was originally validated.

---

*Generated on 2026-06-13. Based on section analyses in `docs/paper/section-analysis/`. Update as experimental results solidify.*
