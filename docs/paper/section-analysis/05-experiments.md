# Section 5 Analysis: Experiments

**Source**: "Take a Step and Reconsider: Sequence Decoding for Self-Improved Neural Combinatorial Optimization" by Pirnay and Grimm (ECAI-2024)

**Analyzed**: 2026-06-13
**Context**: Adapting TaSaR to Virtual Network Embedding (VNE)

---

## Table of Contents

1. [Raw Content Summary](#1-raw-content-summary)
2. [Key Ideas Unpacked](#2-key-ideas-unpacked)
3. [Technical Specifics](#3-technical-specifics)
4. [VNE Implications](#4-vne-implications)
5. [Critical Insights for Our Work](#5-critical-insights-for-our-work)
6. [Questions This Section Raises](#6-questions-this-section-raises)

---

## 1. Raw Content Summary

### 5.1 Routing Problems (TSP and CVRP)

**Problem setup**:
- **TSP**: Euclidean TSP in the unit square with N nodes. Minimize total Euclidean tour length. Solutions constructed sequentially by choosing unvisited nodes.
- **CVRP**: Vehicle of capacity D visits N customer nodes (demand delta_i). Feasible solutions are subtours starting/ending at a depot with capacity constraints. Depot visits are implicit in the constructive formulation.

**Data generation and optimal solutions**:
- Sizes N in {100, 200, 500} for evaluation; **training only on N=100**.
- Nodes uniformly sampled from the unit square. CVRP demands uniform from {1,...,9}.
- Capacity D = 50, 80, 100 for N = 100, 200, 500 respectively.
- Supervised learning uses 1M random instances with **Concorde** (TSP) and **HGS** (CVRP) optimal solutions.
- Test sets: 10k instances for N=100; 128 instances for N in {200, 500}.

**Policy network architecture**:
- Two architectures: **BQ** (Drakulic et al.) and **LEHD** (Luo et al.), both Transformer-based.
- CVRP with BQ: 9 transformer blocks, 12 attention heads, latent dimension 192.
- TSP with BQ: 9 blocks, 8 heads, latent dimension 128.
- LEHD (both TSP/CVRP): 6 transformer blocks, 8 heads, latent dimension 128.
- Both use **ReZero normalization** (LEHD originally had none). Feedforward hidden dim = 512.

**Training details**:
- **Each epoch**: decode solutions for 1,000 random instances in parallel with k=64, s=10, taking ~2 minutes.
- Training uses 1,000 batches of 1,024 uniformly sampled subtours.
- **Adam optimizer**, learning rate **2e-4**, **gradient clipping to unit norm**.
- Early stopping: stop after 50 epochs without validation improvement.
- **CVRP finetuning**: 30k solutions decoded with k=256, s=1, Top-p=0.8, then 100 more epochs.
- BQ with SIL: ~3k epochs (TSP), ~3k epochs (CVRP). LEHD converges faster: ~2k (TSP), ~1k (CVRP).

**Baselines compared**:
- AM: beam search width 1,024
- MDAM: beam search width 50
- POMO: augmentation x8
- SGBS: POMO backbone (beta,gamma = (10,10) for TSP, (4,4) for CVRP)
- LEHD RL+SIL
- Gumbeldore SIL (BQ/LEHD)

**Results (Table 1) -- greedy decoding**:
- **TSP N=100**: BQ SIL 0.37% gap (vs 0.40% SL). LEHD SIL 0.39% (vs 0.58% SL).
- **TSP N=500**: BQ SIL 1.10% (vs 0.98% SL). LEHD SIL 1.40% (vs 1.72% SL).
- **CVRP N=100**: BQ SIL 2.96% (vs 3.03% SL). LEHD SIL 5.08% (vs 4.26% SL).
- **Inference-time decoding** (k=128, s=10): TSP N=100 achieves **0.0009%** gap; CVRP N=100 achieves **0.14%** gap.

### 5.2 Job Shop Scheduling Problem

**Problem setup**:
- JxM: J jobs, M operations each. Operations have processing times and require specific machines.
- Objective: minimize makespan. Solutions are sequences of jobs chosen at each step.

**Data**:
- Random instances: integer processing times [1,99], randomly permuted machine assignments.
- Validation: 100 instances size 20x20. Testing: **Taillard benchmark dataset**.

**Policy network architecture**:
- From Pirnay and Grimm (2024): **6 transformer blocks, 8 heads, hidden dimension 256**.
- Downside: quadratic complexity in total operations.

**Training**:
- SIL for 450 epochs. Each epoch decodes 512 instances with sizes randomly drawn from {15x10, 15x15, 15x20}.
- k=64, s=50. Largest size takes ~5 minutes per epoch.
- Adam, lr=2e-4, gradient clipping.
- Each epoch: 1,000 batches of 512 subschedules.

**Baselines**:
- L2D, ScheduleNet (RL approaches)
- L2S with 500/5000 improvement steps
- SPN (SIL with replacement sampling)
- **GD SIL** (same architecture -- this is the most directly comparable baseline)

**Results (Table 2) -- greedy**:
- Greedy outperforms ALL baselines at comparable runtime.
- 30x20: "improvement of >4% compared to GD".
- 20x20: 8.7% gap vs GD's 11.1%.
- 50x15: 2.2% vs GD's 2.7%.
- 100x20: 1.0% vs GD's 1.7%.
- **With inference decoding** (k=64, s=50): 15x15 achieves 3.0%, 30x20 achieves 6.2%, 50x15 achieves 0.4%, 100x20 achieves 0.1%.

### 5.3 Sampling Comparison

- Compared against SBS (sampling WOR) and GD at equivalent compute budgets.
- Number of node transitions for TaSaR with parameters k, s:
  - g(k,s) = k * (t*l - (s*t^2 - s*t)/2), where t = ceil(l/s).
  - For l=100, k=64, s=10: h(s)=5.5, so SBS gets 6k=384 sequences.
- **TSP**: "improvement over SBS and GD is small because 'the policy is already confident (<2% optimality gap for k=4)'."
- **CVRP and JSSP**: "we see a significant improvement of about 1-2% over sampling WOR."

---

## 2. Key Ideas Unpacked

### 2.1 The training-vs-inference gap is the central metric

The paper's primary measurement is the **optimality gap** between the model's solutions and the exact optimum (Concorde for TSP, HGS for CVRP, Taillard optimal/sota for JSSP). This reveals the true cost of the method's approximations. The gap is reported at two inference regimes: greedy (beam=1) and search (beam=k with TaSaR). The gap between greedy and search tells you how much TaSaR's search procedure improves over the raw policy.

**Why this matters**: Our VNE work currently tracks mean objective value (~-23 val, ~-25.6 test) without reporting the gap to the ILP optimum. As the HANDOFF notes (SS5 step 2), this gap is now the main lever to understand.

### 2.2 "Policy confidence" determines TaSaR's marginal benefit

Section 5.3 reveals a critical insight: **the harder the problem, the more TaSaR helps**. For TSP, where the policy already achieves <2% gap with k=4, TaSaR yields only small improvements. For CVRP and JSSP, where the policy is less confident, TaSaR gives a significant 1-2% improvement over WOR sampling. The method's value grows with problem difficulty.

**Why this matters**: VNE is almost certainly harder than CVRP and JSSP (larger action space, tighter constraints, heterogeneous resources). This suggests TaSaR may be particularly valuable for VNE -- but also that the baseline policy must be good enough for TaSaR to build on. If the policy is too poor, neither search nor TaSaR will help.

### 2.3 Scaling from small training to large testing

All problems train only on the smallest size (N=100 for TSP/CVRP; random {15x10, 15x15, 15x20} for JSSP) and generalize to larger sizes at test time (N=200,500; JSSP up to 100x20). This is feasible because the constructive MDP formulation generalizes naturally -- the policy learns a sequential decision process that transfers to longer sequences.

**Why this matters**: For VNE, we could train on smaller substrate networks (e.g., 10-20 communication nodes) and test on larger ones (40-60). However, VNE's structure involves multiple virtual requests over a substrate, which changes the combinatorial structure more drastically than adding nodes to a TSP tour.

### 2.4 Gradient clipping is universal and essential

Every single problem (TSP, CVRP, JSSP) uses gradient clipping to unit norm with Adam lr=2e-4. This is not incidental -- it is a core training stability mechanism that the paper treated as standard practice. Our VNE work independently confirmed this: without clipping, training diverges at epoch 7-9; with clipping, it stabilizes.

### 2.5 CVRP finetuning with wider beam, narrower replan

CVRP has a dedicated **finetuning phase**: after SIL training, 30k solutions are decoded with k=256, s=1, Top-p=0.8, then 100 more training epochs. The parameters here are telling: wider beam (256 vs 64), minimal replan (s=1 vs 10), and a tighter Top-p (0.8 vs presumably 1.0). This suggests that once the policy is reasonable, you can use a different search configuration to squeeze out additional performance.

### 2.6 JSSP is the most relevant reference problem

JSSP has several parallels to VNE:
- **Longer sequences**: s=50 (vs 10 for TSP) because JSSP has longer action sequences.
- **Higher per-instance cost**: ~5 min per epoch for largest JSSP vs ~2 min for TSP.
- **Quadratic complexity**: the JSSP network scales quadratically in total operations, similar to how VNE's attention grows with problem size.
- **Complex constraints**: JSSP has machine-operation constraints; VNE has resource-capacity constraints.

### 2.7 Multiple sizes during training

JSSP trains on **randomly drawn sizes** from {15x10, 15x15, 15x20}, not a single fixed size. This is different from TSP/CVRP which train only on N=100. This multi-size training likely improves generalization to the Taillard benchmark which spans various sizes.

---

## 3. Technical Specifics

### 3.1 Compute fairness formula (Section 5.3)

The number of node transitions for TaSaR with parameters k (beam width) and s (replan steps):

```
g(k,s) = k * (t*l - (s*t^2 - s*t)/2)

where:
  l = sequence length (number of decisions)
  t = ceil(l/s)  (number of replan rounds)
  k = beam width
  s = replan step size
```

This formula accounts for the fact that within each replan round, the beam explores the first s actions for all k candidates, then replans. The total transitions are less than k * l (full expansion) because the beam narrows between replans.

For l=100, k=64, s=10:
- t = ceil(100/10) = 10
- g = 64 * (10*100 - (10*100 - 10*10)/2) = 64 * (1000 - 450) = 64 * 550 transitions

The paper states "h(s)=5.5, so SBS gets 6k=384 sequences" -- this means the effective expansion factor is 5.5x compared to greedy, so at equivalent compute, SBS can explore 6k=384 complete sequences (6 * 64 = 384) while TaSaR explores k=64 sequences with more internal transitions.

**Important for VNE**: The VNE action sequence length depends on the number of virtual requests and the number of virtual nodes per request. A typical instance with 4 requests of 3 nodes each might have l = 4 * 3 = 12 decisions. This is much shorter than TSP (l=100) or JSSP (l=300 for 15x20). The compute trade-off will look very different.

### 3.2 Exact parameter values

| Parameter | TSP | CVRP | CVRP (finetune) | JSSP |
|-----------|-----|------|-----------------|------|
| k (beam width, training) | 64 | 64 | 256 | 64 |
| s (replan steps) | 10 | 10 | 1 | 50 |
| Top-p | ~1.0 | ~1.0 | 0.8 | ~1.0 |
| Instances per epoch | 1,000 | 1,000 | 30k total | 512 |
| Batches per epoch | 1,000 | 1,000 | n/a | 1,000 |
| Batch size (subtours) | 1,024 | 1,024 | n/a | 512 |
| Epochs | ~3,000 | ~3,000 | +100 | 450 |
| LR | 2e-4 | 2e-4 | 2e-4 | 2e-4 |
| Gradient clipping | unit norm | unit norm | unit norm | unit norm |
| Early stopping | 50 epochs no improvement | 50 epochs | n/a | none stated |

### 3.3 Architectural choices by problem

| Property | TSP (BQ) | CVRP (BQ) | Both (LEHD) | JSSP |
|----------|----------|-----------|-------------|------|
| Transformer blocks | 9 | 9 | 6 | 6 |
| Attention heads | 8 | 12 | 8 | 8 |
| Latent dimension | 128 | 192 | 128 | 256 |
| Feedforward dim | 512 | 512 | 512 | n/a |
| Normalization | ReZero | ReZero | ReZero | ReZero |

### 3.4 Supervised pretraining data

- TSP: 1M instances with Concorde solutions
- CVRP: 1M instances with HGS solutions
- JSSP: No supervised pretraining (SIL directly)

For comparison, our VNE supervised dataset: 50k instances with HiGHS solutions.

### 3.5 Test set sizes

| Problem | N=100 (in-distribution) | N=200/500 (out-of-distribution) |
|---------|------------------------|----------------------------------|
| TSP | 10,000 instances | 128 instances each |
| CVRP | 10,000 instances | 128 instances each |
| JSSP | Taillard benchmark | Taillard benchmark |

Small out-of-distribution test sets (128) suggest that evaluation on large instances is **expensive** (likely because decoding takes longer per instance), so they trade statistical power for feasibility.

---

## 4. VNE Implications

### 4.1 The training-vs-inference gap is our next priority

The paper's core metric (optimality gap) is currently not tracked in our VNE work. The HANDOFF (SS5 step 2) correctly identifies this as the next lever. Our val/test pickles carry the ILP objective, so we can compute:
```
gap_per_instance = (model_cost - ilp_cost) / abs(ilp_cost)
```

This is essential because:
- The mean objective (~-23 val, ~-25.6 test) is an absolute measure that conflates problem difficulty with model quality.
- The gap controls for difficulty, telling us "how close to optimal" the model is.
- The paper shows that for harder problems (JSSP), even large gaps (8.7% at greedy) can be improved substantially.

### 4.2 TaSaR's value grows with problem difficulty

The paper explicitly states: TSP improvement is "small because the policy is already confident." CVRP and JSSP show "significant improvement of about 1-2%." As VNE is more complex than any of these, TaSaR could be even more beneficial.

**However**, there is a threshold effect: if the policy is too poor (e.g., >50% gap), TaSaR may not help because the search space is too large and the policy's guidance is too weak. The paper's JSSP experiment suggests the method works well even with ~8-11% gaps, but it's unclear where the threshold lies.

**Key corollary**: Our Phase 1 supervised baseline must achieve a gap that is "good enough" for TaSaR to build on. The paper's JSSP starting gap (GD SIL) is ~11% on 20x20. If our VNE supervised gap is >20%, we may need a stronger network before TaSaR activation.

### 4.3 Training on smaller, generalizing to larger

TSP/CVRP train on N=100 and test on N=200,500. For VNE, we could train on smaller substrate networks (e.g., 10-20 communication nodes, 2-3 requests) and test on larger ones (e.g., 40-60 nodes, 4-6 requests). But the generalization demands are higher because VNE's structure scales combinatorially (more possible embeddings vs. more possible tours).

**Actionable**: After Phase 1 supervised training, evaluate on held-out larger instances to measure the generalization gap. If the gap is reasonable, the multi-size training approach (like JSSP's random sizes) could further improve generalization.

### 4.4 Replan step size (s) must be calibrated to VNE's sequence length

TSP uses s=10 with l=100 (10%). JSSP uses s=50 with l=300+ (also ~10-15% of sequence length).

For VNE, if a typical instance has l = num_requests * avg_virtual_nodes = 4 * 3 = 12 actions, then s=2 (our current default) is ~17% of the sequence, similar to the paper's ratios. But if VNE sequences are much shorter (l=6-12), the number of replan rounds t = ceil(l/s) will be very small (3-6), limiting TaSaR's ability to benefit from the reconsider mechanism.

**If VNE sequences are short**: Consider modifying the action space to create longer sequences (e.g., embedding node-link pairs sequentially rather than request-by-request).

### 4.5 The compute budget formula for VNE

For fair comparison against WOR or beam search baselines, we need to compute TaSaR's transition count:

```
Let VNE sequence length = l (decisions per instance)
With k = beam width, s = replan steps:
t = ceil(l/s)
transitions = k * (t*l - (s*t^2 - s*t)/2)
```

For our current config (k=8, s=2) and l=12:
t = ceil(12/2) = 6
transitions = 8 * (6*12 - (2*36 - 2*6)/2) = 8 * (72 - 30) = 336

At equivalent compute, WOR could produce: transitions / l = 336 / 12 = 28 complete sequences. So WOR with 28 sequences x 8 width = 224 total sequences (by the formula in the paper). This comparison should be made when reporting results.

### 4.6 Gradient clipping is confirmed critical

The paper's universal use of gradient clipping (unit norm) independently validates our finding that gradient_clipping=1.0 eliminated the epoch-6 divergence. The paper treats this as standard practice, not an optional add-on.

### 4.7 CVRP finetuning suggests a two-phase VNE approach

For CVRP, the paper does: SIL training -> finetuning (k=256, s=1, Top-p=0.8). This two-phase approach could map to our VNE plan:
- **Phase 1**: Supervised training (our current focus)
- **Phase 2**: TaSaR SIL (gumbeldore mode)
- **Phase 2b**: Finetuning with wider beam, tighter Top-p, smaller replan steps

### 4.8 JSSP's multi-size training

JSSP's training uses instances of randomly drawn sizes {15x10, 15x15, 15x20}. This is different from TSP/CVRP which fix N=100. The paper doesn't explain why, but likely reasons:
- The Taillard benchmark spans a wide range of sizes
- JSSP has no supervised pretraining data (so no fixed-size training set)
- Multi-size training improves robustness

For VNE: we could train on a range of substrate/virtual network sizes simultaneously. This might improve generalization and could be done even in the supervised phase if we generate training data with varying sizes.

### 4.9 Early stopping and epoch counts

TSP/CVRP train for 2,000-3,000 epochs with early stopping (50 epochs without validation improvement). JSSP trains for 450 epochs. Our VNE runs use 15-30 epochs. If each epoch is cheap enough, we should increase the epoch budget. But there's a catch: each Gumbeldore epoch requires generating training data, which is expensive for VNE (solving with HiGHS or running search).

The paper's ~2 min/epoch for TSP at k=64 is fast because TSP decoding is cheap. Our VNE forward is now vectorized (4.4x faster), but each epoch still involves HiGHS ILP solving for label generation (supervised) or search (Gumbeldore).

### 4.10 The marginal return of wider search

TSP N=100 goes from 0.37% (greedy) to 0.0009% (k=128, s=10) -- a 400x improvement in gap. CVRP goes from 2.96% to 0.14% -- a 20x improvement. JSSP on 50x15 goes from 2.2% to 0.4% -- a 5.5x improvement.

The ratio of improvement varies by problem. This suggests that the search procedure's benefit is not uniform but depends on the problem's structure. For VNE, the benefit could be anywhere in this range.

### 4.11 Inference-time search is more aggressive than training

At inference, the paper uses k=128, s=10 (TSP/CVRP) or k=64, s=50 (JSSP). Compare to training: k=64, s=10 (TSP/CVRP) or k=64, s=50 (JSSP). TSP/CVRP double the beam at inference. This is a simple way to get better results without retraining.

For VNE: after training is complete, we should evaluate at multiple beam widths to measure the improvement trajectory and find the cost-quality Pareto frontier.

### 4.12 The JSSP wall-clock comparison is most relevant

The paper states "Greedy results outperform all baselines at comparable runtime" for JSSP. Given that VNE will likely have similar per-instance decoding costs to JSSP (complex constraints, graph-structured action space), the JSSP runtime comparison methodology is the most relevant template for our VNE results.

---

## 5. Critical Insights for Our Work

### 5.1 ESSENTIAL: Report the optimality gap before TaSaR activation

The single highest-value action is to compute and log the per-epoch optimality gap on the validation and test sets. The raw objective (~-23) conflates problem difficulty with model quality. The gap to ILP optimum reveals the true performance and tells us if the supervised baseline is strong enough for TaSaR.

**Implementation**: In `vne_main.py`, during validation, after running beam search, read the `objective` field from the validation pickle's ILP labels and compute:
```python
gap = (model_objective - ilp_objective) / abs(ilp_objective)
```
Both values are negative (cost to minimize), so this needs careful handling. But the key insight is: the absolute gap (not the mean objective) is the metric that maps to the paper's results.

### 5.2 ESSENTIAL: Gradient clipping is non-negotiable

The paper uses it universally. Our experiments confirm it fixed the divergence. It should be treated as a fixed hyperparameter, not something to tune.

### 5.3 HIGH VALUE: Plan for a multi-phase approach

The paper's CVRP experiment has a natural two-phase structure (SIL -> finetuning). Our VNE experimental plan already has phases (supervised -> Gumbeldore -> multi-beam eval). We should consider a **third finetuning phase** analogous to the paper: after Gumbeldore training converges, run a finetuning round with k=32-64, s=1-2, Top-p=0.8-0.9 for 50-100 epochs.

### 5.4 HIGH VALUE: Calibrate k and s to VNE's sequence length

Our current config (k=8, s=2) was chosen before this analysis. Given that VNE sequences are short (l~12), s=2 gives t=6 replan rounds, which is reasonable. But k=8 is on the low end compared to the paper (k=64 for all problems). The grid search in our experimental plan (k=8,16,32,64 and s=2,4,8) is well-motivated and should be executed.

**However**, VNE's per-epoch cost is higher than TSP's, so running k=64 may be very expensive. The Phase 2 grid should include wall-clock tracking to identify the cost-quality Pareto frontier.

### 5.5 HIGH VALUE: Multi-size training

Following JSSP's lead, consider training on a range of VNE instance sizes simultaneously. This could improve generalization and reduce overfitting. In practice:
- Supervised phase: generate training data with varying substrate sizes (e.g., 15-40 nodes) and varying virtual request counts (e.g., 2-6).
- Gumbeldore phase: generate instances with varying sizes each epoch.

### 5.6 MEDIUM VALUE: Inference-time search scaling

After training, evaluate at multiple beam widths to create a cost-quality Pareto curve. This is Phase 3 in our experimental plan and is well-justified by the paper's results showing dramatic improvements from wider search (TSP: 0.37% -> 0.0009%).

### 5.7 MEDIUM VALUE: Learn from JSSP's quadratic complexity

The paper notes that JSSP's policy network has "quadratic complexity in total operations." This is a limitation they accept. Our VNE network also has quadratic complexity in the number of candidates (via attention). If scaling to larger instances becomes a bottleneck, we may need to consider sparse attention or hierarchical decoding.

### 5.8 MEDIUM VALUE: The "policy confidence" threshold

The paper suggests TaSaR is most valuable when the policy is "not too confident" (i.e., has >2% gap). Our VNE supervised baseline should be measured on gap to determine if it's in the "sweet spot" for TaSaR. If the gap is <2%, TaSaR may not help much. If >20%, the policy may be too weak for search to help.

Given our current val objective (~-23 with standard deviation not yet reported), we need the gap to the ILP optimum to interpret this.

### 5.9 LOWER VALUE (for now): CVRP-style heavy finetuning

The CVRP finetuning protocol (30k solutions, k=256, s=1, 100 epochs) is expensive. It should be done only after the main TaSaR training cycle shows clear diminishing returns.

### 5.10 LOWER VALUE (for now): Early stopping logic

The paper stops after 50 epochs without validation improvement. Our 15-30 epoch runs are too short for early stopping to matter. Increase epoch count first.

---

## 6. Questions This Section Raises

### Open questions for the VNE adaptation

1. **What is our current optimality gap?** We must compute gap_to_ilp on val/test before any further tuning. This is the critical unknown that gates all subsequent decisions.

2. **How does the gap vary with instance difficulty?** The paper reports average gaps, but VNE instances vary widely in difficulty (more requests = harder). Does the gap concentrate on hard instances or is it uniform?

3. **What is the effective sequence length (l) for VNE?** Is it simply num_requests * avg_virtual_nodes? Or does the embedding of a single virtual request involve multiple decisions (node placement + link routing)? This affects the k, s calibration.

4. **Can we use supervised pretraining for Gumbeldore initialization?** The paper uses SIL directly for JSSP (no SL) but SL initialization for TSP/CVRP. Our Phase 1 supervised training initializes the Gumbeldore phase. Is this necessary or could we go directly to Gumbeldore?

5. **How does ReZero normalization affect VNE training?** The paper uses ReZero for both BQ and LEHD (adding it to LEHD which originally had none). Our VNE config has `use_rezero_transformer = False`. Should we enable it?

6. **Why does CVRP use 12 attention heads vs 8 for TSP with the same architecture?** Is this because CVRP's heterogeneous state (customer demands, capacity, depot) requires more expressive attention? VNE's state is equally heterogeneous (computational nodes, communication nodes, multiple resource types) — does it need more heads?

7. **What is the wall-clock budget for meaningful VNE experiments?** The paper's TSP experiments take ~3,000 epochs * 2 min = 100 hours. JSSP takes 450 epochs * 5 min = 37.5 hours. Our cluster time may be a bottleneck. What is a reasonable epoch count for VNE?

8. **Does TaSaR's relative improvement over WOR scale with the number of replan rounds (t)?** The paper shows this for t=ceil(100/10)=10 rounds. For VNE with l=12 and s=2, t=6. Is 6 rounds enough, or does TaSaR need more?

9. **Could VNE's action space be reformulated for longer sequences?** If the current MDP gives l~12, could we decompose it into finer-grained actions (e.g., embedding a single virtual node's allocation step-by-step) to create longer sequences and more replan opportunities?

10. **How does the paper's "decode 1,000 instances in parallel" map to VNE?** TSP decoding is fast because each step selects one of N remaining nodes. VNE decoding involves feasibility checks for each action (capacity, bandwidth, path existence). Is parallel decoding even feasible for VNE at the same scale?

### Questions about the paper itself

11. **Why does CVRP SL outperform SIL on some metrics?** LEHD CVRP N=100: SL 4.26% vs SIL 5.08%. The SL performance is *better* than SIL despite SIL being a self-improvement method. This is not explained. Possible explanations: (a) CVRP's SIL training hasn't converged; (b) the CVRP SIL training hyperparameters are suboptimal; (c) supervised data quality (HGS solutions) is harder to beat for CVRP. This matters because it shows SIL is not guaranteed to beat SL.

12. **What is the variance across runs?** The paper reports single numbers (no standard deviations) for all results. Are the reported differences statistically significant?

13. **How were the k and s values chosen?** Was there a hyperparameter search, or are these ad-hoc? The paper doesn't say. The JSSP s=50 (vs s=10 for TSP) is justified by sequence length, but no ablation is shown.

14. **What does "comparable runtime" mean for JSSP baselines?** The paper says greedy results outperform all baselines "at comparable runtime" but doesn't give actual wall-clock numbers for the baselines.

15. **What hardware was used?** No GPU model or CPU specification is given. This makes it hard to extrapolate wall-clock times to our cluster.

---

## Appendix: Mapping Paper Config to VNE Config

| Paper Parameter | TSP/CVRP Value | JSSP Value | Our VNE Config | Notes |
|----------------|----------------|------------|----------------|-------|
| lr | 2e-4 | 2e-4 | 2e-4 | Match |
| gradient_clipping | unit norm | unit norm | 1.0 | Match |
| k (training) | 64 | 64 | 8 | Much lower; grid planned |
| s (replan steps) | 10 | 50 | 2 | Proportional to l? |
| Top-p | ~1.0 | ~1.0 | 1.0 | Match |
| Num instances/epoch | 1,000 | 512 | 256 | Lower; budget constraint |
| Batches/epoch | 1,000 | 1,000 | 10,000 | Higher (but per-decision) |
| Batch size (subtours) | 1,024 | 512 | 128 | Lower |
| Transformer blocks | 6-9 | 6 | 1 encoder + 9 decoder | Different architecture |
| Attention heads | 8-12 | 8 | 8 | In range |
| Latent dim | 128-192 | 256 | 128 | Lower end |
| Feedforward dim | 512 | n/a | 512 | Match |
| ReZero normalization | Yes | Yes | False | **Mismatch** |
| Early stopping (epochs) | 50 | n/a | n/a | Not implemented |
| Num epochs | ~3,000 | 450 | 15-30 | Much lower |

---

*End of analysis. Completed 2026-06-13.*
