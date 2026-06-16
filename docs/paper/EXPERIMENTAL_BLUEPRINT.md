# VNE+TaSaR Experimental Blueprint

**Document:** `docs/paper/EXPERIMENTAL_BLUEPRINT.md`
**Date:** 2026-06-13
**Status:** Planning doc — drives all subsequent experiments
**Based on:** TaSaR paper ("Take a Step and Reconsider," ECAI-2024), codebase state at commit `594e733` + local modifications

---

## Table of Contents

1. [Paper's Experimental Design](#1-papers-experimental-design)
2. [VNE Experimental Design](#2-vne-experimental-design)
3. [Baselines to Beat](#3-baselines-to-beat)
4. [Metrics and Reporting](#4-metrics-and-reporting)
5. [Ablation Studies](#5-ablation-studies)
6. [Resource Planning](#6-resource-planning)

---

## 1. Paper's Experimental Design

This section extracts the full experimental methodology from the TaSaR paper (Sections 4-5 and Appendix) as a template for the VNE experiments.

### 1.1 Datasets

| Problem | Train Instances/Epoch | Train Decisions/Epoch | Test Set | Generalization Test |
|---------|----------------------|----------------------|----------|-------------------|
| TSP | 1,000 | 1,000 batches of 1,024 subtours (~1M decisions) | 10k for N=100; 128 for N=200, 500 | Train on N=100, test on N=200, 500 |
| CVRP | 1,000 | 1,000 batches of 1,024 subtours (~1M decisions) | 10k for N=100; 128 for N=200, 500 | Train on N=100, test on N=200, 500 |
| JSSP | 512 per epoch, sizes randomly from {15x10, 15x15, 15x20} | 1,000 batches of 512 subschedules (~512k decisions) | Taillard benchmarks (various sizes up to 100x20) | Train on smaller sizes, test on Taillard up to 100x20 |
| SL pretrain data | 1M instances (TSP/CVRP), solved with Concorde/HGS | N/A (offline) | Same test sets | Same |

**Key design choices:**
- Training instances are generated fresh each epoch (not a fixed dataset) for self-improved learning
- The 1M-instance SL pretrain is an offline phase distinct from the SIL epochs
- JSSP's multi-size training (randomly drawn per epoch) directly inspired our VNE approach
- Test sets are fixed benchmarks for comparability

### 1.2 Training Hyperparameters

| Parameter | TSP | CVRP | JSSP | VNE (current) |
|-----------|-----|------|------|--------------|
| Optimizer | Adam | Adam | Adam | Adam |
| Learning rate | 2e-4 | 2e-4 | 2e-4 | 2e-4 |
| Weight decay | Not reported | Not reported | Not reported | 0.0 |
| Gradient clipping | Unit norm | Unit norm | Unit norm | 1.0 (max_norm) |
| LR schedule | None reported | None reported | None reported | decay_factor=1.0 (flat) |
| Batch size (training decisions) | 1,024 | 1,024 | 512 | 128 |
| Epochs | 1,000 (with early stopping) | 1,000 (with early stopping) | 450 total | 15–30 |
| Early stopping | 50 epochs no improvement | 50 epochs no improvement | Not reported | None |
| Validation frequency | Every epoch | Every epoch | Every epoch | Every epoch |

**Critical finding:** Gradient clipping is universal across all paper problems. Our independent A/B confirmed it fixed the epoch-6 training divergence. This is a fixed hyperparameter, not tunable.

### 1.3 Architecture Sizes

| Architecture | Blocks | Heads | Dim | Feedforward | ~Params |
|-------------|--------|-------|-----|-------------|---------|
| TSP BQ | 9 enc + 9 dec | 8 | 128 | 512 | ~1.8M |
| CVRP BQ | 9 enc + 9 dec | 12 | 192 | 512 | ~3.8M |
| LEHD (CVRP) | 6 enc + 6 dec | 8 | 128 | 512 | ~1.2M |
| JSSP | 6 enc + 6 dec | 8 | 256 | 512 | ~3.0M |
| VNE (current) | 1 enc + 9 dec | 8 | 128 | 512 | ~2.0M |

**Notable observations:**
- All architectures use ReZero normalization (currently disabled in VNE config: `use_rezero_transformer=False`)
- Feedforward dimension is always 4x embedding dimension
- JSSP (the most relevant VNE analog) uses 6+6 blocks with dim 256
- Our VNE network has an asymmetric 1+9 architecture — only 1 encoder layer with 9 decoder layers. This deviates from the symmetric pattern of the paper's architectures

### 1.4 Inference (Decoding)

| Problem | Beam widths | Top-p | Step size (s) | Comment |
|---------|-------------|-------|---------------|---------|
| TSP | {1, 4, 16, 64, 128} | 0.8 | 10 | s=10 for l=100 |
| CVRP | {1, 4, 16, 64, 128, 256} | 0.8 | 10 | s=10 for l=100 |
| JSSP | {1, 64} | 0.8 | 50 | s=50 for l~300 |
| CVRP finetune | 256 | 0.8 | 1 | 100 extra epochs after SIL |

**Key insights:**
- Step size s scales with sequence length: s ~ l/10 (TSP: 100/10=10, JSSP: 300/50≈6 replans)
- For VNE with l~12-30, target s ≈ 2-4 gives comparable replan density
- Top-p=0.8 is used universally across all problems
- The CVRP finetuning phase (k=256, s=1, 100 epochs after SIL convergence) provides a template for a potential Phase 3a

### 1.5 Compute Fairness Formula

The paper introduces a node-transition budget for fair comparison:

```
g(k, s) = k * (t*l - (s*t^2 - s*t) / 2)

where:
- l = sequence length (num actions)
- k = beam width
- s = replan step size
- t = ceil(l / s) = number of replan rounds
```

This counts the number of policy evaluations. For VNE:
- l ≈ 12 (3 requests x 4 link actions each, depending on instance)
- k ∈ {8, 16, 32, 64}
- s ∈ {2, 4, 8}
- t = ceil(12/s) ∈ {6, 3, 2}

At k=64, s=2: g = 64 * (6*12 - (2*36 - 2*6)/2) = 64 * (72 - 30) = 2,688 transitions
At k=64, s=4: g = 64 * (3*12 - (4*9 - 4*3)/2) = 64 * (36 - 12) = 1,536 transitions
At k=8, s=2: g = 8 * (6*12 - 30) = 336 transitions

**Note:** VNE's per-transition cost is higher than TSP's (candidate path enumeration plus policy evaluation over many candidate paths), so the fairness comparison should also track wall-clock time, not just transition count.

### 1.6 Training/Compute Hardware

| Spec | Value |
|------|-------|
| Paper hardware | "moderate academic compute resources (DFG SPP 2331, KoDA cluster)" |
| GPU type | Not specified (likely V100/A100-class) |
| Per-epoch TSP (1k instances) | Not reported |
| Per-epoch JSSP (512 instances) | ~5 min (mentioned in paper) |
| Total training TSP | 1,000 epochs with early stopping at ~50 no-improvement |
| Total training JSSP | 450 epochs |
| Our cluster | gnode03-04 (A100 80GB), gnode05 (A30 24GB), gnode06 (L40S 46GB), gnode09-10 (H200 144GB) |

### 1.7 Metrics and Reporting

**Primary metric:** Optimality gap as percentage:
```
gap = (model_objective - optimal_objective) / |optimal_objective| * 100%
```

**Reporting style:**
- Table: Gap % for each method (SL, SIL, RL baselines) at each beam width
- Table: Comparison against SOTA for each problem (e.g., JSSP vs GD, ACO, CPLEX)
- Table of compute costs (policy evaluations or wall-clock)
- Figure: Gap vs beam width (log scale)
- Figure: Gap vs compute budget

---

## 2. VNE Experimental Design

### 2.1 Instance Generation

#### Current Setup (Phase 0 completed)

| Parameter | Current Value | Rationale |
|-----------|--------------|-----------|
| Substrate topology | Line | Simplest — path-based routing, easy ILP |
| Substrate comm nodes | 20–40 | Matches current datasets |
| Substrate bandwidth | 4–12 per edge | Loose regime — all requests fit |
| Substrate compute | 3–10 per attachment | Matched to demand range |
| Virtual requests per instance | 2–6 | Statistical multiplexing |
| Virtual nodes per request | 2–5 | Chain length |
| Link demands | 1–4 | Small integer ranges for feasibility |
| Admission | OFF | Embed-all — every request accepted |
| Objective | Min-cost (lex, but single request block) | Matching prior VNE work |

#### Phase 1 (Network Scaling) Setup

Same as current — no instance generation changes. The Phase 1 supervised baseline reuses the existing 50k/1k/2k datasets.

#### Phase 2 (TaSaR SIL) Setup

When entering the Gumbeldore self-improvement phase, instances are generated fresh each epoch. The generation follows the same distribution as Phase 1, but with this key change:

- Fresh instances are generated per epoch (not drawn from the 50k pool)
- The Gumbeldore pipeline: generate 256 instances -> decode via TaSaR -> add to training buffer -> train for one epoch -> repeat
- The leader mechanism (theta') selects the best model seen so far for the next generation round

**Plan for generalization testing (Phase 3+):**

| Training Regime | Test Size | Substrate Nodes | Requests | Nodes/Request |
|----------------|-----------|----------------|---------|--------------|
| In-distribution | 2k current test set | 20–40 | 2–6 | 2–5 |
| Larger substrate | New generation | 40–80 | 2–6 | 2–5 |
| More requests | New generation | 20–40 | 6–12 | 2–5 |
| Combined scale-up | New generation | 40–80 | 6–12 | 2–5 |

Generate 500 test instances per out-of-distribution regime, solved via ILP (HiGHS, time_limit=120s).

### 2.2 Training Set Size

| Phase | Instances/Epoch | Decisions/Epoch | Source |
|-------|----------------|-----------------|--------|
| Phase 0 (done) | 50k (total) | 10k sampled per epoch | Fixed pickle, custom_num_batches=("absolute", 10000) |
| Phase 1 (supervised) | 50k (total) | 10k sampled per epoch | Reuse existing pickle, same sampling |
| Phase 2 (Gumbeldore) | 256 per epoch | TBD (depends on TaSaR output) | Fresh generation each epoch |
| Phase 2+ (larger gen) | 512–1024 per epoch | TBD | Scale up if 256 is insufficient |

**Key difference from paper:** The paper uses 1,000 fresh instances/epoch for TSP/CVRP and 512 for JSSP. Our Phase 2 starts at 256 due to VNE's higher per-instance decoding cost. If this proves insufficient for stable self-improvement, scale to 512 or 1,024.

The `custom_num_batches=("absolute", 10000)` mechanism caps epoch length in Phase 1. For Phase 2, the Gumbeldore dataset naturally determines its length from the number of generated instances and their decision counts.

### 2.3 Validation Set

| Property | Value |
|----------|-------|
| Size | 1,000 instances in pickle; 128 used per epoch |
| Composition | Same distribution as training (20–40 substrate nodes, 2–6 requests) |
| Labels | Exact ILP optima (HiGHS, mip_gap=0.0) |
| Use | Best-model tracking, early stopping |
| Frequency | Every epoch |
| Metric | Mean objective (negative cost) -> convert to gap % |

### 2.4 Test Set

| Property | Value |
|----------|-------|
| Primary test set | 2,000 instances (existing `vne_test_dataset_2k.pickle`) |
| Labels | Exact ILP optima (HiGHS, mip_gap=0.0) |
| Generalization set 1 | 500 instances, substrate 40–80 nodes, 2–6 requests |
| Generalization set 2 | 500 instances, 20–40 nodes, 6–12 requests |
| Generalization set 3 | 500 instances, 40–80 nodes, 6–12 requests |
| Generalization set 4 (admission) | 500 instances, tighter resources, admission ON |

Generation plan for generalization sets:
```
# Generate each generalization set as a SLURM array job (CPU, ILP)
VNE_SPLIT=gen_large VNE_NUM_PER_SHARD=100 VNE_SEED_BASE=4000000 VNE_TIME_LIMIT=120
sbatch --array=0-4 scripts/vne_gen_array.sbatch   # 500 instances
```

### 2.5 What "Optimal Solutions" Means for VNE

For our experimental setup, "optimal" means:

- **Training labels (Phase 1):** ILP optima from HiGHS at mip_gap=0.0. Since the embed-all regime is loose (all requests fit), these provably minimize routing cost. The ILP is solved quickly (seconds per instance).
- **Validation/test labels:** Same — exact ILP optima. Every instance in our 1k val and 2k test sets has a verified optimal solution.
- **SIL pseudo-labels (Phase 2):** Solutions found by TaSaR search — NOT optimal, but the best found by the current policy. The gap between these and ILP optima measures the search quality.
- **Optimality gap floor:** 0% (matching ILP exactly). Since our ILP solves to exact optimality (mip_gap=0.0), TaSaR cannot beat the label.

**Critical nuance:** The ILP labels are exact only for the embed-all regime. If admission is enabled (Phase 4+), the ILP solves a lexicographic objective (max acceptance -> min cost), and the optimality definition becomes more nuanced. For now, the metric is clean: minimize cost subject to embedding all requests.

### 2.6 Phased VNE Experimental Protocol

#### Phase 0: Verification (DONE)
- [x] Gradient clipping at 1.0 — fixed training divergence
- [x] 1k-vs-50k A/B — data quantity is NOT the lever
- [x] Batched network forward (4.4x speedup)
- [x] Instance generation pipeline (ILP, sharding, merging, self-check)

#### Phase 1: Network Scaling (NEXT — HIGHEST PRIORITY)

Goal: Find the network architecture that gives the strongest supervised baseline, then freeze it for Phase 2.

**Architecture candidates** (in priority order):

| Run | emb_dim | dec_layers | enc_layers | heads | ff_dim | ~Params | Notes |
|-----|---------|-----------|-----------|-------|--------|---------|-------|
| P1.0 | 128 (current) | 9 | 1 | 8 | 512 | ~2.0M | Current config, asymmetric 1+9 |
| P1.1 | 128 | 6 | 6 | 8 | 512 | ~1.8M | Symmetric, matches LEHD pattern |
| P1.2 | 192 | 6 | 6 | 12 | 768 | ~3.8M | Scales up: 2x CVRP BQ size |
| P1.3 | 256 | 6 | 6 | 8 | 1024 | ~4.2M | Larger dim, matches JSSP gap |
| P1.4 | 128 | 9 | 9 | 8 | 512 | ~3.0M | Symmetric deep, scales to JSSP dim |
| P1.5 | 256 | 9 | 9 | 16 | 1024 | ~10M+ | Full scale: 3x TSP (if needed) |

**(Optional) ReZero variant for each:** Run once with `use_rezero_transformer=True` for the best-performing architecture, to test if the paper's universal ReZero improves VNE.

**Training protocol (same for each run):**
1. Use existing 50k supervised training set
2. 30 epochs (config already set), lr=2e-4, gradient_clipping=1.0
3. custom_num_batches=("absolute", 10000) — caps epoch length
4. Validate every epoch (beam-1, 128 instances)
5. Track: training loss curve, validation objective, test objective at end
6. Primary metric: optimality gap % on validation and test sets

**Go/no-go for Phase 2:** If best architecture achieves gap < 1% on the test set, proceed to Phase 2. If gap remains > 1% even at 256-dim/9-layer, investigate further (LR schedule, longer training, ReZero) before proceeding.

#### Phase 2: Gumbeldore/TaSaR Activation

Goal: Activate the self-improvement loop and find the best (beam_width, replan_steps) configuration.

**Grid search** (12 runs):

| Variable | Values |
|----------|--------|
| beam_width (k) | 8, 16, 32, 64 |
| replan_steps (s) | 2, 4, 8 |

**Fixed parameters:**
- learning_type = "gumbeldore"
- search_type = "tasar"
- min_nucleus_top_p = 1.0 (no Top-p truncation initially; relax to 0.8 if needed)
- perform_first_round_deterministic = False
- num_instances_to_generate = 256 per epoch
- devices_for_workers = ["cpu"] x N (Ray CPU workers)
- Network architecture: best from Phase 1
- lr = 2e-4, gradient_clipping = 1.0
- 15 Gumbeldore epochs + 1 supervised pre-training epoch to initialize

**Per run tracking:**
- Per epoch: training loss, mean generated objective, validation objective (beam-1, 128 instances)
- Final: test metrics (beam-1, beam-4, beam-16)

**Go/no-go:** If any configuration improves validation gap over the Phase 1 best, proceed to Phase 3. If none improves, consider: (a) more generation instances (512/epoch), (b) Top-p=0.8, (c) first-round deterministic, (d) larger beam widths (128+).

#### Phase 2.5: CVRP-style Finetuning (conditional)

If Phase 2 shows improvement but plateaus quickly, add a finetuning phase modeled on the paper's CVRP approach:
- Take the best model from Phase 2
- Run 50-100 extra Gumbeldore epochs with k=128, s=1, Top-p=0.8
- This is the "beam search with constant reconsideration" configuration

#### Phase 3: Multi-beam Evaluation

Goal: For the best model(s) from Phase 2, measure search quality as a function of beam width.

**Evaluation protocol:**
1. Load best model from Phase 2
2. Evaluate on 2k test set at beam_width ∈ {1, 4, 8, 16, 32, 64}
3. Each evaluation with deterministic beam search (not TaSaR) for clean comparison
4. Report: gap % vs ILP optimum at each beam width

**Key hypothesis test:** Does gap % decrease monotonically with beam width? If not (i.e., beam-64 is no better than beam-1), the TaSaR search is not producing diverse useful alternatives — a fundamental failure mode.

#### Phase 4: Generalization Testing (optional, based on results)

- Test best model on out-of-distribution sizes (generalization sets 1-4)
- Compare generalization of Phase 1 (supervised) vs Phase 2 (SIL) models
- Test whether SIL's diverse pseudo-labels improve out-of-distribution performance

---

## 3. Baselines to Beat

### 3.1 VNE Literature Baselines

The VNE literature reports on different metrics and problem variants than our setup. Here are the relevant baselines:

| Method | Type | Metric Reported | Adaptable to Our Setup? | Notes |
|--------|------|----------------|------------------------|-------|
| **DeepViNE** (Blenk et al., 2018) | ML-based, node classification | Acceptance ratio, revenue/cost | Partially — uses different problem definition | Predecessor to NCO-VNE |
| **NCO-VNE** (Yousefnezhad et al., 2020) | RL-based, sequential | Acceptance ratio, node/link utilization | Partially — uses original VNE (not extended) | Different topology assumptions |
| **ILP baseline** (our own labels) | Exact | Cost optimality (gap = 0%) | Directly | Our gap % already measures vs this |
| **Greedy heuristic** | Simple first-fit | Revenue/cost | Construct from our trajectory | Run baseline: assign each virtual link to the shortest feasible path |
| **Beam-1 policy (greedy decode)** | Our trained model, greedy | Gap % | Directly | This IS our Phase 1 baseline |
| **Worst-fit / best-fit heuristics** | Classical | Utilization stats | Manually implement | Lower priority |
| **HGA/VNE** (John et al., hybrid GA) | Metaheuristic | Acceptance, cost | Not directly | GA with VNE-specific operators |

### 3.2 Internal Baselines (Our Control Groups)

| Baseline ID | Description | What It Tests |
|-------------|-------------|---------------|
| B1 | Phase 1 supervised model, beam-1 (greedy) | Policy quality without search |
| B2 | Phase 1 supervised model, beam-64 (deterministic BS) | Search improvement over greedy |
| B3 | SIL model (same epochs, WOR sampling not TaSaR) | TaSaR-specific improvement vs simple WOR |
| B4 | Random model, beam-1 | Untrained policy lower bound |

### 3.3 Fair Comparison Protocol

To ensure fair comparison:

1. **Architecture parity:** All baselines use the same network architecture (best from Phase 1)
2. **Compute parity:** Compare TaSaR at g(k,s) transitions vs beam search at the same budget (if B2 at beam-64 costs ~N evaluations, compare TaSaR at a configuration that also costs ~N)
3. **Data parity:** SIL baselines use the same initial policy (pretrained from Phase 1)
4. **Seed control:** All random operations (instance generation, weight init, data shuffling) seed-controlled
5. **Evaluation harness:** Identical val/test sets for all comparisons

### 3.4 What Constitutes "Beating" Baselines Convincingly

| Criteria | Threshold | Interpretation |
|----------|-----------|---------------|
| Gap reduction vs B1 | > 1 percentage point absolute reduction | TaSaR search significantly improves over greedy |
| Gap reduction vs B2 | > 0.5 pp at same compute | TaSaR specifically better than deterministic BS |
| SIL improvement | Phase 2 best > Phase 1 best on val | Self-improvement loop is working |
| Pareto dominance | TaSaR achieves lower gap than B2 in less wall-clock time | TaSaR is more compute-efficient |
| Statistical significance | p < 0.05 over 5 random seeds per condition | Effect is robust |

**Minimum publishable result:** TaSaR at k=64, s=2 achieves gap < 0.5% vs ILP, beating beam-64 at gap > 1.0%. This shows that the reconsider mechanism is specifically improving search quality for VNE.

**Home-run result:** TaSaR SIL model achieves gap < 0.1% (essentially matching ILP) at inference time, while generalizing to larger instances where the supervised-only model degrades.

---

## 4. Metrics and Reporting

### 4.1 Primary Metric: Optimality Gap

Calculated per-epoch for both validation and test:

```
optimality_gap = (model_cost - ilp_cost) / ilp_cost * 100
```

Where:
- `model_cost = -mean_objective` (since stored objectives are negative costs)
- `ilp_cost = -ilp_objective` (same convention)
- Gap is always >= 0% (model cannot beat exact ILP optimum at mip_gap=0.0)

**Recording:** Logged via the logger (file + optionally MLflow) per epoch. The validation function currently returns the mean objective; modify to also return the per-instance cost for gap computation.

**Required code change:** In `vne_main.py::evaluate()`, compute and log `optimality_gap` alongside the raw objective. The test/validation pickles carry `"objective"` (the ILP optimum), so each instance's gap can be computed.

### 4.2 Secondary Metrics

| Metric | Definition | When to Report |
|--------|------------|----------------|
| Acceptance ratio | fraction of requests embedded | Phase 4+ (admission ON) |
| Revenue-to-cost ratio | revenue_units / cost_units | If using "profit" objective |
| Node utilization | avg computational resource used / total | Phase 1-3 (diagnostic) |
| Link utilization | avg bandwidth used / total | Phase 1-3 (diagnostic) |
| Feasibility rate | fraction of instances where model finds ANY feasible solution | Phase 1-3 (critical for VNE) |
| Wall-clock time | seconds per epoch, total training time | All phases |
| Policy evaluations | count of network forward passes | Fairness comparisons |

**Critical VNE-specific metric:** Feasibility rate. Unlike TSP (all actions always feasible), VNE can dead-end (infeasible partial solution). Track how often the model produces a complete feasible embedding. An optimality gap of 1% means nothing if it only achieves 60% feasibility.

### 4.3 Runtime Comparisons

Record for each experiment:
- Per-epoch generation time (Phase 2: TaSaR decode time)
- Per-epoch training time (forward/backward)
- Total wall-clock time
- Number of GPU hours consumed
- Peak GPU memory usage

For the Pareto frontier plot, normalize by compute budget (g(k,s) from the paper's formula) AND wall-clock time, in case the two diverge.

### 4.4 Statistical Significance

For each experimental condition (Phase 2 k/s combination):

1. Run with 3-5 random seeds (where applicable: network initialization, instance generation)
2. Report mean and standard deviation of optimality gap
3. Paired t-test between TaSaR and baseline on the same test instances
4. If feasible: bootstrap confidence intervals for gap differences

### 4.5 Reporting Templates

**Table 1: Supervised Baseline Comparison (Phase 1)**

| Architecture | Params | Val Gap % | Test Gap % | Feasibility % | Best Epoch | Train Loss |
|-------------|--------|-----------|------------|---------------|------------|------------|
| 128/9d/1e (current) | ~2.0M | 1.7% | 1.7% (from A/B) | TBD | 5 | 1.34 |
| 128/6d/6e (symmetric) | ~1.8M | TBD | TBD | TBD | TBD | TBD |
| 192/6d/6e (scaled) | ~3.8M | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... |

**Table 2: TaSaR Grid Results (Phase 2)**

| k | s | Val Gap % | Test Gap % | Feasibility % | Epochs to Converge | Wall Time |
|---|----|------------|-------------|----------------|--------------------|------------|
| 8 | 2 | TBD | TBD | TBD | TBD | TBD |
| 8 | 4 | TBD | TBD | TBD | TBD | TBD |
| 8 | 8 | TBD | TBD | TBD | TBD | TBD |
| 16 | 2 | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... |
| 64 | 8 | TBD | TBD | TBD | TBD | TBD |

**Table 3: Multi-beam Evaluation (Phase 3)**

| Beam Width | Supervised Gap % | SIL (Best k,s) Gap % | Improvement |
|------------|-----------------|---------------------|-------------|
| 1 (greedy) | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD |

**Figure 1: Gap vs Beam Width (log-log)**
- X-axis: beam_width (log scale: 1, 4, 16, 64)
- Y-axis: optimality gap % (log scale)
- Two lines: supervised model (B1), TaSaR SIL model (best config)
- Lower is better; gap should decrease with beam width

**Figure 2: Training Convergence**
- X-axis: epoch
- Two Y-axes: training loss, validation gap %
- Compare: supervised Phase 1 (50k), TaSaR Phase 2 (best config)
- Show where the SIL model overtakes the supervised model

**Figure 3: Pareto Frontier**
- X-axis: compute budget (policy evaluations, log scale)
- Y-axis: optimality gap % (log scale)
- Points: each Phase 2 (k, s) configuration, Phase 3 beam search at each beam width
- Line: the efficient frontier

---

## 5. Ablation Studies

### 5.1 Paper's Ablations (Section 5.3)

The paper's main ablation compares sampling methods:
- **TaSaR** vs **SBS** (single round) vs **WOR** (Gumbeldore with multiple rounds) vs **greedy** (beam-1)
- Controlled for compute budget using the g(k,s) formula
- Key finding: TaSaR outperforms SBS and WOR at equivalent compute for longer sequences (JSSP)
- For short sequences (TSP l=100): WOR and TaSaR are similar at low compute; TaSaR pulls ahead at high compute

Additional ablations implied but not shown:
- Top-p sensitivity (fixed at 0.8 throughout — not ablated)
- First-round deterministic vs not (used for CVRP, but not systematically ablated)
- Architecture comparison (BQ vs LEHD, but only for CVRP — LEHD underperforms BQ in SL but catches up in SIL)

### 5.2 VNE Ablations

**Ablation 1: Sampling Method (Primary)** — Mirror the paper's Section 5.3

Compare on VNE test set (same architecture, same compute budget):
1. Greedy (beam-1)
2. Deterministic beam search (beam-k)
3. WOR sampling (Gumbeldore search_type="wor", num_rounds=4)
4. TaSaR (search_type="tasar", s=2) at matched compute

Control for: total policy evaluations (using g(k,s) formula). Each method gets approx equal compute.

**Ablation 2: Step Size s (Secondary)** — Already part of Phase 2 grid

s ∈ {2, 4, 8} at fixed k=32. Expected: s=2 gives most replan rounds (> fine-grained reconsideration). If s=2 is worse than s=4, VNE benefits from longer commitment periods, suggesting the policy is confident in early actions.

**Ablation 3: Beam Width k (Secondary)** — Already part of Phase 2 grid

k ∈ {8, 16, 32, 64} at fixed s=4. Expected: larger k monotonically improves gap, but with diminishing returns. The point of diminishing returns is the recommended k for future experiments.

**Ablation 4: Top-p Threshold**

After finding the best (k, s) from Phase 2, test Top-p ∈ {0.6, 0.7, 0.8, 0.9, 1.0} at that configuration. Paper uses 0.8 universally; VNE's different action space may benefit from a different value.

**Ablation 5: Architecture Comparison (if Phase 1 identifies a winner)**

Compare the best Phase 1 architecture against the paper's closest match (e.g., symmetric 6+6 LEHD at dim 128) on TaSaR performance. Is the best supervised architecture also the best SIL architecture? (The paper's CVRP results suggest no: BQ wins in SL, LEHD wins in SIL.)

**Ablation 6: First-round Deterministic**

Compare `perform_first_round_deterministic=True` vs False at the best (k, s). The paper notes this helps initially exploit the policy mode; may help VNE bootstrapping if initial pseudo-labels are very noisy.

**Ablation 7: ReZero Normalization**

Compare `use_rezero_transformer=True` vs False in the Phase 1 best architecture. All paper architectures use ReZero; our VNE config has it disabled. This could affect gradient stability during SIL training.

### 5.3 Ablation Prioritization

| Priority | Ablation | Cost | Impact if Wrong |
|----------|----------|------|----------------|
| P0 | Sampling method (Ablation 1) | 4 eval runs | Core claim |
| P1 | k, s grid (Ablations 2, 3) | 12 Phase 2 runs | Hyperparameter sensitivity |
| P2 | Top-p (Ablation 4) | 5 eval runs | Search quality |
| P3 | Architecture (Ablation 5) | 2 training runs | Future-proofing |
| P4 | First-round det. (Ablation 6) | 2 training runs | Bootstrapping |
| P5 | ReZero (Ablation 7) | 2 training runs | Training stability |

---

## 6. Resource Planning

### 6.1 Estimated GPU Hours

#### Phase 1: Network Scaling (supervised)

| Run | Est. Time/Epoch | Epochs | Est. Total |
|-----|----------------|--------|-----------|
| P1.0 (128/9d/1e baseline) | ~8 min | 30 | ~4 GPU-hr |
| P1.1 (128/6d/6e symmetric) | ~8 min | 30 | ~4 GPU-hr |
| P1.2 (192/6d/6e scaled) | ~12 min | 30 | ~6 GPU-hr |
| P1.3 (256/6d/6e) | ~15 min | 30 | ~7.5 GPU-hr |
| P1.4 (128/9d/9e) | ~12 min | 30 | ~6 GPU-hr |
| P1.5 (256/9d/9e) | ~25 min | 30 | ~12.5 GPU-hr |
| Subtotal | | | ~40 GPU-hr |
| ReZero variant (optional) | +20% | | +8 GPU-hr |

**Note:** Times are extrapolated from current ~7.7 min/epoch on A30 (MPS 40%) with 2M-param model. Scaled architectures may reduce per-epoch instances due to GPU memory limits.

#### Phase 2: Gumbeldore Grid (12 runs x 15 epochs)

Each run = data generation (CPU, Ray workers) + training (GPU).

| Component | Est. Time | Notes |
|-----------|-----------|-------|
| TaSaR generation (256 inst, cpu-ray, k=8) | ~10 min | Depends on k and worker count |
| TaSaR generation (256 inst, cpu-ray, k=64) | ~30 min | k scales compute linearly |
| Training per epoch | ~5 min | Smaller batch (256 generated inst) vs 50k |
| Total per run (15 epochs) | ~3-6 hr | Varies with k and worker count |
| **Total Phase 2 (12 runs)** | **~36-72 GPU-hr** | Runs can be parallelized |

**Parallelization strategy:**
- Run up to 4 Phase 2 configurations simultaneously on different partitions
- gpuISIN (A30 or L40S), gpu (A100), gpu_AMD (H200/Blackwell)
- Each uses MPS (30-50% share) for training + 4-8 CPU cores for Ray workers
- Total wall clock: concurrent execution brings 12 runs to ~2-3 days

#### Phase 3: Multi-beam Evaluation

| Evaluation | Instances | Beam widths | Est. Time |
|------------|-----------|-------------|-----------|
| Test set eval (2k inst) | 2,000 | 6 widths | ~2 hr |
| Per model (best config) | 2,000 | 6 widths | ~2 hr |
| **Total Phase 3** | | | **~4 GPU-hr** |

#### Phase 4: Generalization + Baselines

| Component | GPU-hr |
|-----------|--------|
| Generate generalization sets (CPU, no GPU) | 0 |
| Evaluate on 4 generalization sets | ~4 |
| Greedy heuristic baseline (CPU) | ~1 (CPU) |
| WOR baseline training (1 run) | ~10 |
| **Total Phase 4** | **~5 GPU-hr + CPU time** |

#### Grand Total

| Phase | GPU-hr | Wall Clock |
|-------|--------|------------|
| Phase 1 (network scaling) | ~50 | ~2 days (parallel) |
| Phase 2 (Gumbeldore grid) | ~50 | ~3 days (parallel) |
| Phase 3 (multi-beam eval) | ~4 | ~4 hr |
| Phase 4 (generalization) | ~5 | ~1 day |
| **Total** | **~110** | **~7-10 days** |

### 6.2 SLURM Job Configurations

#### Phase 1: Supervised Training

```
# Single architecture, A30 via MPS
# Resources: small GPU + moderate CPU
sbatch --partition=gpuISIN --gres=mps:a30:40 \
  --cpus-per-task=6 --mem=24G --time=1-00:00:00 \
  --export=ALL,VNE_RUN_TAG=p1-128x9d-1e,VNE_NUM_EPOCHS=30,VNE_GPU_MEM_FRACTION=0.25 \
  scripts/vne_train.sbatch

# Larger architecture, A100 via MPS
sbatch --partition=gpu --gres=mps:a100:50 \
  --cpus-per-task=6 --mem=32G --time=1-00:00:00 \
  --export=ALL,VNE_RUN_TAG=p1-256x9d-9e,VNE_NUM_EPOCHS=30,VNE_EMBEDDING_DIM=256,\
VNE_NUM_DECODER_LAYERS=9,VNE_NUM_ENCODER_LAYERS=9,VNE_NUM_HEADS=16,VNE_FF_DIM=1024,\
VNE_GPU_MEM_FRACTION=0.35 \
  scripts/vne_train.sbatch
```

#### Phase 2: Gumbeldore Grid

```
# Example: k=32, s=4, CPU workers for Ray generation
sbatch --partition=gpuISIN --gres=mps:a30:40 \
  --cpus-per-task=8 --mem=32G --time=1-12:00:00 \
  --export=ALL,VNE_RUN_TAG=p2-k32-s4,VNE_LEARNING_TYPE=gumbeldore,\
VNE_BEAM_WIDTH=32,VNE_REPLAN_STEPS=4,VNE_NUM_EPOCHS=15,\
VNE_NUM_GENERATE=256,VNE_NUM_CPU_WORKERS=4,\
VNE_GPU_MEM_FRACTION=0.25 \
  scripts/vne_train.sbatch
```

**Launch all 12 grid jobs in parallel:**

```bash
# Phase 2 grid launcher (run from repo root)
for k in 8 16 32 64; do
  for s in 2 4 8; do
    sbatch --partition=gpuISIN --gres=mps:a30:40 \
      --cpus-per-task=8 --mem=32G --time=1-12:00:00 \
      --export=ALL,VNE_RUN_TAG="p2-k${k}-s${s}",VNE_LEARNING_TYPE=gumbeldore,\
VNE_BEAM_WIDTH=$k,VNE_REPLAN_STEPS=$s,VNE_NUM_EPOCHS=15,\
VNE_NUM_GENERATE=256,VNE_NUM_CPU_WORKERS=4,VNE_GPU_MEM_FRACTION=0.20 \
      scripts/vne_train.sbatch
    sleep 2  # avoid submit burst
  done
done
```

**GPU allocation strategy:**
- If gpuISIN has capacity: run 4 jobs on gnode06 (2xL40S, MPS: 4x50%) + remaining on gnode05 (A30, MPS: 30-40%)
- Overflow to gpu partition (gnode03-04, A100)
- If H200/Blackwell available (gpu_AMD): use for largest architecture runs (Phase 1.5, P1.5)

#### Array Job for Generalization Set Generation

```
# Generate 500 instances for generalization test set 1
sbatch --array=0-4 --partition=compute --cpus-per-task=4 --mem=8G --time=4:00:00 \
  --export=ALL,VNE_SPLIT=gen_large_substrate,VNE_NUM_PER_SHARD=100,\
VNE_SEED_BASE=4000000,VNE_TIME_LIMIT=120,VNE_OUT_DIR=./data/vne/gen_large_substrate/shards \
  scripts/vne_gen_array.sbatch

# Merge after completion
./.venv/bin/python scripts/vne_merge_shards.py \
  --shard-glob "./data/vne/gen_large_substrate/shards/*.pickle" \
  --out "./data/vne/gen_large_substrate.pickle"
```

### 6.3 Checkpointing Strategy

The training loop already saves `last_model.pt` and `best_model.pt` per run (in `results_path`). Extend for the Gumbeldore phase:

| File | When Saved | Contents | Size (est) |
|------|-----------|----------|-----------|
| `last_model.pt` | Every epoch | model weights, optimizer state, epoch count | ~20 MB |
| `best_model.pt` | When val metric improves | same | ~20 MB |
| `generated_dataset_epochN.pickle` | After each Gumbeldore gen round | training data for that epoch | ~5 MB |
| `config_snapshot.txt` | Run start | full config dump | < 1 KB |
| `final_test_metrics.json` | Run end | test metrics at all beam widths | < 1 KB |

**For Phase 2 grid,** prefix all files with the run tag to avoid collisions:
```
model_checkpoints/vne/results/p2-k8-s2_<jobid>/
```

### 6.4 MLflow Tracking Setup

Currently disabled (`log_to_mlflow=False`). For the full experiment:

1. **Option A (recommended):** Enable MLflow on a local tracking server or cluster filesystem
   - Set `mlflow_server_uri` to a shared path (e.g., `file:///mnt/beegfs/scratch/ali.alhousseini/mlruns`)
   - All runs automatically log: hyperparameters, per-epoch metrics, final metrics
   - Use MLflow's UI for comparing runs: `mlflow ui --port 5000`

2. **Option B (minimal):** Keep file logging + structured results
   - Each run produces a `results/` directory with checkpoints + a log file
   - Parse logs with `vne_plot_training.py` for visualization
   - Aggregate results manually or with a post-processing script

**For Phase 2 grid comparison,** the ability to query runs by hyperparameters (k, s, seed) across 12+ runs makes MLflow worthwhile. Strongly recommend Option A.

### 6.5 Experiment Naming Convention

```bash
VNE_RUN_TAG = "{phase}-{descriptor}_{date}"
```

Examples:
- `p1-128x6d-6e` — Phase 1, 128-dim, 6 decoder + 6 encoder layers
- `p1-192x6d-6e-rezero` — Phase 1, scaled + ReZero
- `p2-k32-s4` — Phase 2, beam_width=32, replan_steps=4
- `p2-k64-s2-top08` — Phase 2, k=64, s=2, top_p=0.8
- `p3-eval-best` — Phase 3, multi-beam evaluation of best model

### 6.6 Post-processing Pipeline

After each experiment phase, run the analysis script:

```
./.venv/bin/python scripts/vne_plot_training.py \
  --log logs/vne_train_<tag>_<jobid>.log
```

This produces:
- `artifacts/<tag>-loss.png` — training loss vs epoch
- `artifacts/<tag>-val-obj.png` — validation objective vs epoch
- `artifacts/<tag>.csv` — raw per-epoch metrics

For Phase 2 grid comparison, build an aggregation script that reads all run logs and produces:
- Comparison table (Markdown)
- Gap vs beam width plot
- Learning curves overlay

---

## Appendix A: Required Code Changes

### A.1. Gap Reporting (HIGH priority — must do before Phase 1)

Modify `vne_main.py::evaluate()` to compute and log per-instance optimality gap:

```python
# In the evaluate function, after computing mean_obj:
ilp_objectives = np.array([
    inst.get("objective", inst.get("sol", None))
    for inst in instances
], dtype=float)
# Filter instances with valid ILP objectives
valid = ~np.isnan(ilp_objectives)
model_costs = -objectives  # stored as negative costs
ilp_costs = -ilp_objectives[valid]
model_costs_valid = model_costs[valid]
gap = (model_costs_valid - ilp_costs) / ilp_costs * 100.0
loggable[f"{eval_type} gap %"] = float(np.mean(gap))
loggable[f"{eval_type} gap std %"] = float(np.std(gap))
loggable[f"{eval_type} feasibility %"] = float(
    np.mean([obj != float("-inf") for obj in objectives]) * 100.0
)
```

### A.2. Hyperparameter Overrides (done — in `_apply_env_overrides`)

The env-driven overrides system already supports:
- `VNE_EMBEDDING_DIM`, `VNE_NUM_DECODER_LAYERS`, `VNE_NUM_ENCODER_LAYERS`
- `VNE_NUM_HEADS`, `VNE_FF_DIM`, `VNE_LR`
- `VNE_LEARNING_TYPE`, `VNE_BEAM_WIDTH`, `VNE_REPLAN_STEPS`
- `VNE_NUM_GENERATE`, `VNE_NUM_CPU_WORKERS`

This is sufficient for the full experimental grid without touching `config.py`.

### A.3. ReZero Activation (Phase 1 optional)

To test ReZero, add env override path `VNE_USE_REZERO` to `_apply_env_overrides`:

```python
if env.get("VNE_USE_REZERO") is not None:
    config.use_rezero_transformer = env["VNE_USE_REZERO"].lower() in ("1", "true", "yes")
```

### A.4. Test Set Gap in Validation Function

The validation function currently logs mean objective only. Extend to log gap % and feasibility %. This is critical for Phase 1 architecture comparison and Phase 2 grid analysis.

---

## Appendix B: References

1. Pirnay, J., & Grimm, V. (2024). "Take a Step and Reconsider: Sequence Decoding for Self-Improved Neural Combinatorial Optimization." ECAI-2024. DOI: 10.3233/FAIA240707.
2. Kool, W., van Hoof, H., & Welling, M. (2019). "Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement." ICML 2019.
3. Luo, F., et al. (2023). "Learning to Solve Routing Problems via Sub-tour Re-unrolling." (Self-improved learning with subtour-based generalization.)
4. Corsini, A., et al. (2023). "Self-improved Learning for Job Shop Scheduling." (Self-labeling via MC sampling for JSSP.)
5. Haarnoja, T., et al. (2018). "Soft Actor-Critic." (Broader RL context for off-policy learning.)
6. Kool, W., van Hoof, H., & Welling, M. (2019). "Attention, Learn to Solve Routing Problems!" ICLR 2019. (Attention Model + POMO.)
7. Blenk, A., et al. (2018). "DeepViNE: Virtual Network Embedding with Deep Reinforcement Learning."
8. Yousefnezhad, N., et al. (2020). "NCO-VNE: Neural Combinatorial Optimization for Virtual Network Embedding."
9. Fischer, A., et al. (2013). "A Mathematical Model for the Virtual Network Embedding Problem." (Original VNE formulation.)
