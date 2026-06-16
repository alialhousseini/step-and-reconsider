# VNE+TaSaR Experimental Plan

> **Status:** Planning — 2026-06-15
> **Based on:** TaSaR paper (Pirnay & Grimm, ECAI-2024), pilot results (job 40437), HANDOFF.md
>
> This document defines the experimental protocol for the VNE+TaSaR paper.
> **Part 1 (NOW)** is what we execute immediately.
> **Part 2 (SKIP/LATER)** is parked for the journal version.

---

## 0. Architecture Families: BQ and LEHD

The TaSaR paper uses exactly **two** named architecture families. Both are encoder-decoder transformers,
but they differ fundamentally in HOW they encode the problem state and decode actions.

### BQ — Bisimulation Quotienting (Drakulic et al.)

**Core idea:** A single unified transformer encoder processes ALL information (problem state + candidates)
in one flat sequence. No separate encoder/decoder — one stack of N transformer blocks, then a linear
projection to logits. Simpler, monolithic, one forward pass per decision.

**Paper configurations:**

| Problem | Blocks | Heads | Dim | FF | ~Params |
|---------|--------|-------|-----|-----|---------|
| TSP | 9 | 8 | 128 | 512 | ~1.8M |
| CVRP | 9 | 12 | 192 | 768 | ~3.8M |

**How it works (from `tsp/bq_network.py`, `cvrp/bq_network.py`):**
1. Embed all entities (nodes, demands, capacities) into a flat sequence
2. Add positional markers (first/last node, depot)
3. Pass through N `TransformerEncoderLayer` (or `RZTXEncoderLayer` with ReZero) blocks
4. Linear projection to one logit per action
5. Apply action mask (set infeasible actions to -inf)

### LEHD — Heavy Decoder (Luo et al.)

**Core idea:** Separate encoder and decoder. The encoder processes the global problem structure
(graph attention). The decoder processes the partial solution + candidates with cross-attention
to the encoded global context. More structured, but encoder-decoder split adds flexibility
for heterogeneous state.

**Paper configurations:**

| Problem | Enc | Dec | Heads | Dim | FF | ~Params |
|---------|-----|-----|-------|-----|-----|---------|
| TSP/CVRP | 1 | 6 | 8 | 128 | 512 | ~1.2M |
| JSSP (scaled) | 6 | 6 | 8 | 256 | 1024 | ~3.0M |

**How it works (from `tsp/lehd_network.py`, `vne/network.py`):**
1. **Encoder:** Graph/node attention over the substrate/graph → latent representation
2. **Decoder:** Self-attention over [context, partial_solution, candidate_1, ..., candidate_K]
3. Special projections for start/destination markers
4. Linear projection to one logit per candidate

### Paper Finding: Architecture × Training Paradigm Interaction

The paper's CVRP comparison reveals: **BQ wins in supervised learning, but LEHD wins in SIL**.
This is a critical and interesting result — the architecture that works best with expert labels
is not the same as the one that benefits most from self-improvement.

**For VNE, we MUST replicate this comparison.** The paper's Table 1 explicitly compares BQ vs LEHD
for TSP/CVRP. Our equivalent table for VNE must include both families.

### Implementation Status

| Architecture | TSP | CVRP | JSSP | VNE |
|-------------|-----|------|------|-----|
| BQ | ✅ `tsp/bq_network.py` | ✅ `cvrp/bq_network.py` | — | ✅ **`vne/bq_network.py`** (2026-06-15) |
| LEHD | ✅ `tsp/lehd_network.py` | ✅ `cvrp/lehd_network.py` | ✅ (custom) | ✅ `vne/network.py` |

**BQ for VNE is implemented.** Single unified transformer over `[nodes | edges |
virtuals | candidates]` with type embeddings, 9 ReZero blocks, linear output.
Verified: forward pass produces correct logit shapes, `get_weights()` works for
Ray serialization, architecture switch in `vne_main.py::get_network` selects
between BQ and LEHD.

### Architecture Design for VNE BQ

Following the paper's BQ pattern applied to VNE:

1. **Input sequence:** [substrate_node_1_features, ..., substrate_node_N_features,
   virtual_request_context, candidate_1_features, ..., candidate_K_features]
2. Each token: features (type embedding + normalized attributes)
3. **N=9 transformer blocks** (matching paper's BQ) with ReZero
4. **Output:** one logit per candidate position (mask infeasible)
5. No encoder-decoder split — everything attends to everything in one stack

This is architecturally simpler than the LEHD version (no separate graph encoder,
no cross-attention decoder), which makes it interesting: does the simpler BQ approach
work better with expert labels but worse with self-generated labels, as in the paper?

---

## PART 1: NOW — Immediate Experiments

### 1. New VNE Problem Scale

**Motivation:** The pilot (20-40 nodes, 2-6 requests) proved TaSaR works and matches supervised
with 156× less data. But the sequences were too short (l ≈ 12) — TaSaR's reconsider mechanism
is designed for longer horizons. Scaling up strengthens the paper by showing TaSaR handles
genuinely challenging VNE instances.

**Paper analog:** TSP trained on N=100, tested up to N=500 (5×). JSSP trained on 15×{10,15,20},
tested up to 100×20 (6.7×).

**New parameter ranges:**

| Parameter | Training (Base) | Test 2× | Test 3× | Test 4-5× |
|-----------|----------------|---------|---------|-----------|
| Substrate comm nodes | 60–80 | 120–160 | 180–240 | 240–400 |
| # Virtual requests | 10–20 | 10–20 | 10–20 | 10–20 |
| VNR nodes/request | 2–5 | 2–5 | 2–5 | 2–5 |
| Sequence length (l) | ~30–100 | ~30–100 | ~30–100 | ~30–100 |

**Fixed across all scales:**

| Parameter | Range | Distribution |
|-----------|-------|-------------|
| BW (comm links) | 4–12 | Uniform int |
| CAP (comp links) | 3–10 | Uniform int |
| Link demands (processing) | 1–4 | Uniform int |
| Source/dest demands | 1–4 | Uniform int |
| Substrate topology | line | — |
| VNR topology | chain | — |
| Comp attach probability | 1.0 | — |
| Admission | OFF (embed-all) | — |
| Objective | min-cost | — |

**Feasibility notes:**
- At 60-80 nodes + 10-20 requests: l ≈ 30-100, giving t ≈ 8-25 replan rounds for s=4. Much better than pilot's t≈3.
- ILP at 60-80n: estimated 5-60s/instance on HiGHS. Parallel generation (SLURM array, 100+ cores) makes 10k labels feasible (~0.5-2h wall-clock).
- ILP at 4-5× scale (240-400n): may need time limits. Accept best feasible + report MIP gap transparently.

---

### 2. Architecture Design: Paper-Exact Replication + Motivated Extensions

**Principle:** Replicate the paper's BQ vs LEHD comparison for VNE. The paper tests each family
at the problem-appropriate scale. We do the same.

**Phase 1 runs (supervised baseline):**

| Run ID | Family | Blocks | Dim | Heads | FF | ReZero | ~Params (actual) | Paper Precedent |
|--------|--------|--------|-----|-------|-----|--------|------------------|-----------------|
| **BQ-128** | BQ | 9 | 128 | 8 | 512 | Yes | **1.78M** | BQ-TSP (dim 128, 9 blocks) |
| **BQ-192** | BQ | 9 | 192 | 12 | 768 | Yes | **4.00M** | BQ-CVRP (dim 192, 9 blocks) |
| **LEHD-128** | LEHD | 6e+6d | 128 | 8 | 512 | Yes | **2.43M** | LEHD TSP/CVRP (dim 128) |
| **LEHD-192** | LEHD | 6e+6d | 192 | 12 | 768 | Yes | **5.45M** | Scaled LEHD |
| **LEHD-256** | LEHD | 6e+6d | 256 | 8 | 1024 | Yes | **9.67M** | JSSP scale (dim 256) |

**Note:** LEHD has more params than BQ at the same embedding dim because LEHD has
separate encoder + decoder stacks (12 blocks total vs BQ's 9) plus extra projection
layers (global, current_virtual, candidate). This is inherent to the architecture
families and consistent with the paper — BQ is leaner but may benefit less from SIL.

**This is a 2 (family) × 2 (capacity) + 1 (extra LEHD scale) = 5 runs.**

**Why these specific configurations:**
- BQ-128 and LEHD-128 match the paper's TSP scale (~1.8M params). Direct comparison at equal capacity.
- BQ-192 and LEHD-192 match the paper's CVRP scale (~3.8M params). Tests whether scaling helps.
- LEHD-256 matches the paper's JSSP (dim 256). Tests whether VNE benefits from JSSP-level capacity.
- BQ-256 is omitted: the paper never uses BQ at dim 256; BQ's strength is at moderate dims.

**Motivation for the paper:**
> "We evaluate both architecture families from the original TaSaR paper — BQ and LEHD — at
> two capacity scales (1.8M and 3.8M parameters), plus an extended LEHD at JSSP-scale capacity
> (4.2M). This mirrors the paper's architecture comparison for CVRP and tests whether the
> BQ-vs-LEHD interaction with training paradigm (supervised vs SIL) replicates for VNE."

**Implementation plan for BQ-VNE:**
- New file: `vne/bq_network.py`
- Template: `tsp/bq_network.py` and `cvrp/bq_network.py`
- Key adaptation: VNE's heterogeneous state (substrate nodes, edges, residual capacities, virtual request features, candidate paths) must be flattened into a unified token sequence
- ReZero: enabled (paper standard, `modules/rztx.py` already available)

---

### 3. Dataset Generation Plan

**We start clean.** Move existing datasets (based on 20-40 nodes) aside and generate fresh ones.

```bash
mkdir -p data/vne/archive_20-40n
mv data/vne/vne_supervised_training_dataset_50k.pickle data/vne/archive_20-40n/
mv data/vne/vne_validation_dataset_1k.pickle data/vne/archive_20-40n/
mv data/vne/vne_test_dataset_2k.pickle data/vne/archive_20-40n/
mv data/vne/*.manifest.json data/vne/archive_20-40n/
```

**New datasets:**

| Dataset | # Instances | ILP Labels | Seed Band | Est. Generation Time |
|---------|------------|------------|-----------|----------------------|
| **Training (supervised)** | 10,000 | Yes (HiGHS, mip_gap=0.0) | 1e6+ | ~0.5-2h wall (100-core SLURM array) |
| **Validation** | 1,000 | Yes (HiGHS, mip_gap=0.0) | 2e6+ | ~0.1-0.3h wall |
| **Test (in-dist)** | 2,000 | Yes (HiGHS, mip_gap=0.0) | 3e6+ | ~0.2-0.5h wall |
| **Test (2×: 120-160n)** | 500 | Yes (time_limit=120s) | 4e6+ | ~0.5-3h wall |
| **Test (3×: 180-240n)** | 500 | Best feasible (time_limit=300s) | 5e6+ | ~1-7h wall |
| **Test (4-5×: 240-400n)** | 500 | Best feasible (time_limit=600s) | 6e6+ | ~2-14h wall |

**Why 10k not 50k:** Pilot conclusively showed 1k = 50k for supervised plateau.
10k is generous headroom at the new scale. Scale up only if Phase 1 shows data-hunger.

---

### 4. Phase 1: Supervised Baselines (Architecture Comparison)

**Goal:** Establish the supervised-learning ceiling for each architecture at the new scale.
Replicate the paper's BQ vs LEHD comparison for VNE.

**Protocol (per architecture):**
1. Train on 10k ILP-labeled instances
2. 30 epochs, lr=2e-4, gradient_clipping=1.0, flat LR
3. `custom_num_batches = ("absolute", 10000)` — fixed epoch length for fair comparison
4. Validate every epoch: beam-1 greedy, 128 instances, log gap% and feasibility%
5. Final test: beam-1 on 2k test set
6. Track: train loss, val gap%, val feasibility%, wall-clock per epoch

**Go/no-go gate:** Best architecture gap < 10% on ID test. If all > 10%, investigate before SIL.

**Architecture selection:** Pick the best architecture by val gap%. If BQ and LEHD are close
(within 1%), advance BOTH to Phase 2 — the paper's key finding is the interaction with SIL.

**Table produced:** Table 1 (§8) — Architecture comparison, paper-style.

---

### 5. Phase 2: TaSaR SIL — k × s Grid

**Goal:** Find the best (beam_width, replan_steps) for TaSaR self-improved learning.
If BQ and LEHD both advanced from Phase 1, run the grid on BOTH — the paper's key finding
is that architecture ranking can invert under SIL.

**Fixed parameters:**
- learning_type = "gumbeldore", search_type = "tasar"
- min_nucleus_top_p = 1.0, perform_first_round_deterministic = False
- 256 instances generated per epoch, 15 Gumbeldore epochs
- lr = 2e-4, gradient_clipping = 1.0
- Initialize from Phase 1 supervised checkpoint

**Grid:**

k ∈ {16, 32, 64} × s ∈ {2, 4, 8, 16} = **12 runs** (×2 if both architectures = 24 runs).

| k | s=2 | s=4 | s=8 | s=16 |
|---|-----|-----|-----|------|
| 16 | ? | ? | ? | ? |
| 32 | ? | ? | ? | ? |
| 64 | — | ? | ? | ? |

**Why s up to 16:** With l ≈ 30–100, s=16 gives t ≈ 2–6. Paper's JSSP uses t≈6.
Full range from MCTS-like (s=2, t≈15-50) to near-greedy (s=16, t≈2-6).

**Per run tracking:**
- Per epoch: train loss, generated objective, val gap% (beam-1), val feasibility%
- Final: test gap% at beam-1 AND beam-64, wall-clock/epoch, total wall-clock

**Output:** Table 2 (§8) + Pareto frontier plot.

---

### 6. Phase 3: Multi-beam Evaluation

**Goal:** Measure search quality vs beam width for best model(s).

**Protocol:**
1. Load best SIL model(s) from Phase 2
2. Evaluate on 2k test set at beam_width ∈ {1, 4, 8, 16, 32, 64}
3. Deterministic beam search (not TaSaR) for clean comparison
4. Also evaluate Phase 1 supervised model at same beam widths

**Output:** Table 3 (§8) + gap vs beam_width figure.

---

### 7. Phase 4: Generalization

**Goal:** Test generalization to larger substrates — the paper's headline result.

**Evaluation only (no retraining):**

| Test Set | Nodes | Requests | Instances | Labels |
|----------|-------|----------|-----------|--------|
| ID | 60–80 | 10–20 | 2,000 | Exact (mip_gap=0.0) |
| 2× | 120–160 | 10–20 | 500 | Exact (time_limit=120s) |
| 3× | 180–240 | 10–20 | 500 | Best feasible (time_limit=300s) |
| 4-5× | 240–400 | 10–20 | 500 | Best feasible (time_limit=600s) |

**Methods:** Best supervised + best SIL, beam-1 and best-k.

**Headline:** "TaSaR trained on 60-80 node substrates achieves X% gap on 240-400 node substrates
— a 4-5× scale increase — without retraining."

**Output:** Table 4 (§8).

---

### 8. Target Tables (Paper-Ready Format)

**Table 1: Supervised Baseline — Architecture Comparison (cf. paper Table 1)**

10k ILP-labeled instances, 60-80n substrates, 10-20 requests, beam-1 greedy.

| Architecture | Family | ~Params | Val Gap% | Test Gap% | Feasibility% | Best Epoch | Wall/Epoch |
|-------------|--------|---------|----------|-----------|-------------|------------|------------|
| BQ-128 (9 blocks) | BQ | ~1.8M | ? | ? | ? | ? | ? |
| BQ-192 (9 blocks) | BQ | ~3.8M | ? | ? | ? | ? | ? |
| LEHD-128 (6e+6d) | LEHD | ~1.8M | ? | ? | ? | ? | ? |
| LEHD-192 (6e+6d) | LEHD | ~3.8M | ? | ? | ? | ? | ? |
| LEHD-256 (6e+6d) | LEHD | ~4.2M | ? | ? | ? | ? | ? |

**Table 2: TaSaR k × s Grid**

Best architecture(s), 15 Gumbeldore epochs. Metric: test gap% at beam-1.

| k | s=2 | s=4 | s=8 | s=16 |
|---|-----|-----|-----|------|
| 16 | ? | ? | ? | ? |
| 32 | ? | ? | ? | ? |
| 64 | — | ? | ? | ? |

(If both BQ and LEHD advanced, show two tables or a split table.)

**Table 3: Multi-beam Evaluation — Search Scaling (cf. paper Table 1 "inference decode" rows)**

2k test set, deterministic beam search.

| Beam Width | Supervised Gap% | TaSaR SIL Gap% | Δ Improvement |
|------------|----------------|----------------|---------------|
| 1 (greedy) | ? | ? | ? |
| 4 | ? | ? | ? |
| 8 | ? | ? | ? |
| 16 | ? | ? | ? |
| 32 | ? | ? | ? |
| 64 | ? | ? | ? |

**Table 4: Main Results — Generalization (cf. paper TSP N=100→200→500 columns)**

| Method | Val (60-80n) | Test (60-80n) | 2× (120-160n) | 3× (180-240n) | 4-5× (240-400n) |
|--------|-------------|---------------|---------------|---------------|-----------------|
| ILP / best feasible | 0% | 0% | 0%† | 0%† | best feasible† |
| Supervised (best arch, beam-1) | ? | ? | ? | ? | ? |
| **TaSaR SIL (best k,s, beam-1)** | ? | ? | ? | ? | ? |
| + inference decode (k>1) | ? | ? | ? | ? | ? |

† Transparently report MIP gap for large instances where exact optimum not reached.

**Table 5: Wall-Clock & Compute (cf. paper's compute fairness discussion, §5.3)**

| Method | GPU | Train Wall Time | Per-Epoch Time | Inference (per inst, k=1) | Inference (per inst, best k) | Speedup vs ILP |
|--------|-----|----------------|----------------|--------------------------|-----------------------------|----------------|
| Supervised (best) | A100/40% MPS | ? h | ? min | ? ms | ? ms | ?× |
| TaSaR SIL (best) | A100/40% MPS | ? h | ? min | ? ms | ? ms | ?× |
| HiGHS ILP | CPU only | N/A | N/A | 5-60s (ID) | N/A | 1× |

**Figures:**
1. **Gap vs Beam Width** (log-log, supervised vs SIL, both architectures if applicable)
2. **Training Convergence** (train loss + val gap vs epoch, supervised vs SIL overlay)
3. **Pareto Frontier** (gap% vs g(k,s), each grid point labeled, both architectures if applicable)
4. **Generalization Degradation** (gap% vs substrate scale, 1×→2×→3×→4-5×)

---

### 9. Wall-Clock Time Methodology

#### What the Paper Actually Does

The paper's wall-clock reporting is **informal — mentioned in prose, not systematically tabulated:**

1. **Per-epoch generation time** in prose: "decode solutions for 1,000 random instances
   in parallel with k=64, s=10, taking ~2 minutes" (TSP), "~5 minutes per epoch" (JSSP).
2. **Total epochs to convergence** in prose: "BQ SIL ~3k epochs, LEHD SIL ~2k epochs".
3. **No GPU model specified** — makes wall-clock irreproducible across clusters.
4. **No per-instance inference time** — only per-epoch aggregate.
5. **"Comparable runtime" for baselines** — stated but no numbers given (JSSP §5.2).
6. **The g(k,s) formula (§5.3)** is their PRIMARY compute-fairness mechanism — a theoretical
   node-transition budget, not wall-clock. This is how they fairly compare TaSaR vs SBS vs WOR:
   all methods get the same g(k,s) budget, then quality is compared.

**In short:** The paper uses g(k,s) for theoretical fairness between decoding methods, and
mentions wall-clock only as rough context ("~2 min/epoch"). They do NOT report GPU specs,
per-instance timing, or systematic speedup factors.

#### Our Protocol (More Rigorous Than the Paper)

**For compute fairness between methods (matching the paper):**
- Use g(k,s) to equalize node-transition budgets when comparing TaSaR vs SBS vs WOR
- This is theoretically grounded and directly comparable to the paper

**For practical wall-clock (beyond the paper):**
- Record per-epoch: `t_gen` (data generation/loading) + `t_train` (forward+backward) + `t_val`
- Record per-instance inference: `t(k)` for each beam width k
- Record GPU: model, MPS fraction, peak memory
- Report speedup: `t_ILP / t_model(k)` at each beam width
- Report total training wall time to convergence

**How to record in code:**
```python
import time

# In training loop:
t0 = time.time()
# ... generate/load data ...
t_gen = time.time() - t0

t0 = time.time()
# ... train one epoch ...
t_train = time.time() - t0

# In evaluate():
t0 = time.time()
# ... beam search over N instances ...
t_inference = (time.time() - t0) / N  # per-instance
```

**Reporting template (in paper prose, matching the paper's style):**
> "At inference, TaSaR with k=64, s=4 embeds a 60-80 node VNE instance in X ms on an
> NVIDIA A100 GPU — a Z× speedup over HiGHS ILP (Y s/instance) with only W% optimality gap."

---

### 10. Code Changes Required

1. **`vne/bq_network.py`** ✅ (NEW) — BQ architecture for VNE. Single unified transformer
   over `[nodes | edges | virtuals | candidates]`, 9 ReZero blocks, linear output.
   Batched forward with sub-batching budget (8000 token limit). ~1.8M params at dim=128.
2. **`vne/config.py`** ✅ — Updated parameter ranges (60-80n, 10-20 req), added
   `architecture` switch ("bq"/"lehd"), `num_transformer_blocks=9`, symmetric
   encoder/decoder (6+6), ReZero enabled, dataset paths updated to 10k.
3. **`vne_main.py`** ✅ — Architecture-aware `get_network`, gap% + feasibility% +
   per-instance timing in `evaluate()`, wall-clock tracking in `train_with_dataloader`
   and `train_for_one_epoch_gumbeldore`, env var overrides for VNE_ARCHITECTURE,
   VNE_VALIDATION_SET_PATH, VNE_TEST_SET_PATH, VNE_NUM_TRANSFORMER_BLOCKS.
4. **`scripts/vne_gen_datasets.sh`** ✅ (NEW) — Convenience script to submit all 3
   SLURM array jobs for dataset generation (10k train, 1k val, 2k test).
5. **`scripts/vne_phase1_launch.sh`** ✅ (NEW) — Launches all 5 Phase 1 supervised
   architecture runs in parallel with correct env vars.
6. **`scripts/vne_train.sbatch`** ✅ — Added VNE_ARCHITECTURE env passthrough.
7. **`docs/paper/VNE_EXPERIMENTAL_PLAN.md`** ✅ — Updated with actual param counts
   and implementation status.

---

## PART 2: SKIP / LATER — Journal Extensions

### Ablations

| Item | Reason to Skip |
|------|---------------|
| **Top-p threshold** | Paper uses 0.8 at inference without ablation. Our Phase 2 uses 1.0 for training diversity. Ablating Top-p is inference-only (no retraining needed) — low cost but not essential for core story. **Journal add-on.** |
| First-round deterministic | Implementation detail, not a conceptual contribution. Skip. |
| Encoder-layer count ablation | Current 1+9 vs symmetric is a diagnostic for us, not a paper contribution. |

### Problem Extensions

| Item | Reason to Skip |
|------|---------------|
| **Admission control (Direction b)** | New action (reject), new trajectory logic, new objective, new datasets. Different problem → different paper. **Follow-up paper.** Current embed-all (Direction a) is the clean paper analog. |
| **Topology generalization (ER, mesh)** | Paper tests exactly 1 axis: size scaling. Topology is genuinely novel but adds risk. Test feasibility on supervised model first; add only if gap < 15%. **Journal add-on.** |
| Multi-size training (JSSP-style) | Refinement, not headline. Add only if single-size generalization is weak. |
| Request-count scaling | Only 10→20 (2×), weak dynamic range vs substrate 60→400 (5-7×). |
| RL baselines (REINFORCE) | New `core/rl_train.py` needed. Valuable but not essential — paper's main comparison is SIL vs SL. **Journal add-on.** |
| Greedy heuristic baselines | Quick but not needed for core story. **Journal add-on.** |

### Additional Datasets

| Item | Reason to Skip |
|------|---------------|
| 50k training set | Pilot: 1k = 50k. 10k is generous. Scale up only if Phase 1 data-hungry. |
| 10k test set (paper-style) | Paper uses 10k for TSP ID. Our 2k adequate; scale if borderline significance. |

---

## Appendix A: Decision Tree

```
Phase 1: BQ-128, BQ-192, LEHD-128, LEHD-192, LEHD-256 supervised on 10k
  │
  ├─ Best gap < 3%  ──→  Proceed. Policy strong. Both archs if close (<1% apart).
  ├─ Best gap 3-10%  ──→  Proceed. TaSaR designed for this (paper JSSP ~8-11%).
  └─ Best gap > 10%  ──→  STOP. Investigate architecture/features/training.

Phase 2: k×s grid on best architecture(s)
  │
  ├─ SIL gap < supervised gap  ──→  Core claim validated. Note if arch ranking inverted.
  ├─ SIL gap ≈ supervised gap   ──→  "No solver needed" + speed story.
  └─ SIL gap > supervised gap   ──→  STOP. Tune instances/epoch, LR, Top-p.

Phase 4: Generalization
  │
  ├─ Gap at 4-5× < 10%   ──→  Strong headline result.
  └─ Gap at 4-5× > 10%   ──→  Report anyway. VNE is harder than TSP/CVRP.
```

---

## Appendix B: Run Checklist

### Pre-Phase 0 (BQ implementation) ⟸ RUNS IN PARALLEL WITH PRE-PHASE 1

BQ development and dataset generation happen concurrently — no dependency between them.

- [ ] Create `vne/bq_network.py` (template from `tsp/bq_network.py`)
- [ ] Design unified token sequence for VNE state (substrate + virtual + candidates)
- [ ] Verify forward pass produces correct logit shape
- [ ] Verify BQ works with existing `log_probability_fn` interface
- [ ] Add `architecture` switch to config (BQ/LEHD)

### Pre-Phase 1 (infrastructure) ⟸ RUNS IN PARALLEL WITH PRE-PHASE 0

Dataset generation is independent of BQ implementation — run concurrently.

- [ ] Update `vne/config.py` parameter ranges (60-80n, 10-20 req)
- [ ] Generate 10k training set (SLURM array, HiGHS, mip_gap=0.0)
- [ ] Generate 1k validation set
- [ ] Generate 2k test set (in-distribution)
- [ ] Self-check: 0 duplicates, feasible, mip_gap=0.0
- [ ] Archive old datasets

### Phase 1 (supervised baselines) — Launch all 5 runs in parallel

All architectures are independent — submit as a batch of SLURM jobs simultaneously.
Use gpuISIN (A30/L40S) and gpu (A100) partitions with MPS sharing.
See §"Execution Strategy" below for workflow-based orchestration.

- [ ] Run BQ-128: 9 blocks, dim=128, ReZero
- [ ] Run BQ-192: 9 blocks, dim=192, ReZero
- [ ] Run LEHD-128: 6e+6d, dim=128, ReZero
- [ ] Run LEHD-192: 6e+6d, dim=192, ReZero
- [ ] Run LEHD-256: 6e+6d, dim=256, ReZero
- [ ] Compute gap% for all; select best architecture(s)

### Phase 2 (k × s grid)
- [ ] Launch 12 (or 24) SLURM jobs
- [ ] Monitor convergence
- [ ] Aggregate; identify best (arch, k, s)

### Phase 3 (multi-beam eval)
- [ ] Evaluate best supervised model at beam ∈ {1,4,8,16,32,64}
- [ ] Evaluate best SIL model at beam ∈ {1,4,8,16,32,64}

### Phase 4 (generalization)
- [ ] Generate 2×, 3×, 4-5× test sets
- [ ] Evaluate supervised + SIL on all test sets
- [ ] Compute gap% degradation curves

---

---

## Appendix C: Execution Strategy — Parallelism & Superpowers

This project has access to Claude Code's multi-agent orchestration capabilities:
**workflows** (deterministic fan-out with `Workflow` tool), **teams** (persistent agent groups
with task lists), **memory** (persistent project knowledge across sessions), and **skills**
(reusable slash-command behaviors).

### Parallelism by Design

Every phase is structured for maximum parallelism:

| Phase | Parallelism | Mechanism |
|-------|------------|-----------|
| Pre-Phase 0 + Pre-Phase 1 | **Concurrent** — BQ dev and dataset generation are independent | Two workstreams, no dependency |
| Phase 1 (5 arch runs) | **All 5 simultaneously** — each architecture is independent | SLURM batch submit to different GPU partitions |
| Phase 2 (12-24 grid runs) | **All grid cells simultaneously** — each (k,s) is independent | SLURM batch submit; `scripts/vne_grid_submit.sh` |
| Phase 3 (multi-beam eval) | **Sequential per model, parallel beams** — eval is fast | Single GPU, batch evaluation |
| Phase 4 (generalization) | **All test sets in parallel** — each scale is independent eval | SLURM array for ILP label generation first, then parallel eval |

### When to Use Workflows

Workflows (`Workflow` tool) are the primary orchestration mechanism. Use them when:

1. **Code generation with verification** — e.g., generate BQ network, then adversarially review
   against `tsp/bq_network.py` template and `vne/network.py` interface requirements.
2. **Multi-file consistency audit** — after implementing BQ, dispatch `vne-surveyor` agents
   across `vne/` to verify all files are consistent with the new architecture switch.
3. **Pre-submission review** — before launching Phase 1/2 jobs, a workflow of `code-reviewer`
   agents checks all changed files for bugs.
4. **Results synthesis** — after Phase 2 completes, a workflow fans out across all 12-24 run
   logs, extracts metrics, and synthesizes the comparison table.

### When to Use Teams

Teams (`TeamCreate` + agent spawning) are for long-running collaborative work:

1. **BQ implementation** — spawn a Developer + Reviewer team. Developer writes `bq_network.py`,
   Reviewer checks against paper spec and existing BQ implementations.
2. **Dataset generation monitoring** — a Monitor agent watches SLURM job completion and
   alerts when all shards are ready for merging.

### Memory Hygiene

Key decisions are persisted in `.claude/projects/.../memory/`. After each phase:
- Write a memory file with results, go/no-go decisions, and updated parameters.
- Link to relevant memories (`[[vne-experimental-plan]]`, `[[gumbeldore-pilot-results]]`).
- This ensures future sessions pick up exactly where we left off.

### Skills

Available skills are listed in system prompts. Relevant ones for this project:
- `/loop` — for recurring tasks (e.g., polling SLURM job status)
- `/code-review` — pre-commit review of changed files
- `/deep-research` — literature search for VNE baselines or related work

---

*End of plan. To be updated as experiments progress.*
