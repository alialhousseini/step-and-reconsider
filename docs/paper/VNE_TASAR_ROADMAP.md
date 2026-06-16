# VNE + TaSaR: Complete Research Roadmap

> **Status**: Diagnostics Phase (Tier 0) imminent. Supervised training stabilized with gradient clipping. 50k/1k/2k datasets ready. Key finding: training divergence fixed, plateau persists regardless of data quantity.

---

## 1. Current Positioning — Where We Are Right Now

### 1.1 What Has Been Built

The VNE extension is **fully implemented** as a self-contained problem package under `vne/`, following the same structure as `tsp/`, `cvrp/`, `jssp/`. The codebase compiles and runs end-to-end for supervised training and validation.

**Working components (verified by execution on 53k-instance pipeline):**

| Component | File | Status |
|-----------|------|--------|
| Instance generator (line topology, chain VNRs) | `vne/instance_generator.py` | Verified. 50k instances generated with feasibility guarantee |
| MILP solver pipeline (HiGHS, multi-solver fallback) | `vne/validation_set_generator.py` | Verified. Self-check passes on all 53k instances |
| Supervised dataset/replay | `vne/dataset.py` | Verified. `RandomVNEDataset` replays correctly with candidate reconstruction |
| State feature engineering | `vne/features.py` | Verified. Builds node/edge/virtual/candidate tensors with normalization |
| VNEPolicyNetwork (batched forward) | `vne/network.py` | Verified. `_forward_batch` numerically identical to per-example, 4.4x faster |
| Supervised training loop | `vne_main.py:train_with_dataloader` | Verified. 30 epochs stable with gradient clipping=1.0 |
| Validation beam search | `vne_main.py:evaluate` | Verified. Runs beam-1 search, returns -22.9 val objective |
| Env-driven config overrides | `vne_main.py:_apply_env_overrides` | Verified. SLURM env vars control A/B experiments |
| SLURM scripts | `scripts/vne_train.sbatch` | Verified. MPS sharing, GPU memory caps |

**Untested components (compile by inspection, never executed for VNE):**

| Component | File | Risk |
|-----------|------|------|
| `IncrementalSBS.perform_tasar()` with VNE trajectory | `core/incremental_sbs.py:265` | HIGH — variable-length trajectories, dead-end paths untested |
| GPU-based generation workers | `core/gumbeldore_dataset.py:148` | MEDIUM — MPS memory cap not applied to Ray workers |
| Gumbeldore data round-trip | `vne_main.py:save_search_results_to_dataset` | LOW — data format verified, `_has_nested_paths` handles it |
| Full SIL training loop | `core/train.py:main_train_cycle` | MEDIUM — tested for TSP/CVRP/JSSP, not VNE |

### 1.2 The Supervised Plateau — Facts, Not Speculation

The A/B experiment (jobs 40019 clip1k, 40020 clip50k) established:

```
Metric               clip1k (1k data)    clip50k (50k data)
Best val objective   -22.91 (e5)          -22.97 (e2)
Test objective       -25.594              -25.631
Stable loss band     1.28–1.43            1.28–1.43
```

**Established facts:**

1. **Gradient clipping (1.0) fixes divergence completely.** The epoch 7-9 explosion (train 8.97, val -140) is gone. Both runs hold ~-23 for all 15 epochs.

2. **Data quantity is not the lever.** 50k instances produce the same plateau as 1k instances. The model saturates at the same objective regardless of 50x more data.

3. **The plateau is real and robust.** The model reaches best val by epoch 2-5 and does not improve thereafter. No amount of additional supervised data shifts it.

4. **The optimality gap has never been computed.** The val/test pickles carry `instance["objective"]` (ILP optimum, same negative-cost convention). Preliminary estimate: model test cost 25.59 vs ILP cost 25.16 = ~1.7% gap. But this is UNCONFIRMED and the full distribution is unknown.

### 1.3 The Gap to a Publishable Paper

**Minimum publishable result**: A neural VNE method that (a) uses no ILP solver at inference time, (b) is within 5% of ILP optimal, (c) provides 100-1000x speedup vs ILP, and (d) generalizes to unseen substrate sizes/topologies. This is achievable with the supervised model alone.

**Ambitious result**: TaSaR self-improvement breaks the supervised plateau, achieving <1% gap. Generalization to 3x larger substrates. This would be publishable at a top ML venue.

**Critical gap**: The optimality gap is unmeasured. Every strategic decision depends on knowing: is the model at 1.7% from optimal or 17%? Until this is measured, we are flying blind.

---

## 2. What We've Done (Completed Milestones)

### Milestone 1: Reproducible GPU environment (completed)
- Built uv-managed Python 3.11 venv with torch 2.5.1+cu121
- MPS GPU sharing with `torch.cuda.set_per_process_memory_fraction()`
- SLURM scripts for GPU and CPU partitions

### Milestone 2: MILP solver pipeline (completed)
- Multi-solver layer (HiGHS default, Gurobi/CPLEX/CBC fallback)
- Keep-best-feasible-on-timeout (no timeout bias)
- Lexicographic (max-acceptance -> min-cost) and profit objectives
- 53k solved instances with self-check passed

### Milestone 3: 50k/1k/2k datasets (completed)
- `vne_supervised_training_dataset_50k.pickle` (~501k replay decisions)
- `vne_validation_dataset_1k.pickle` (1,000 instances)
- `vne_test_dataset_2k.pickle` (2,000 instances)
- Disjoint seed bands (train 1e6+, val 2e6+, test 3e6+)
- Manifest files with distribution data, zero duplicates

### Milestone 4: Vectorized network forward (completed)
- `_forward_batch` with block-padding and candidate-count bucketing
- Numerically identical to per-example path, 4.4x faster
- Single host-to-device copy per batch

### Milestone 5: Training stability fix (completed)
- `gradient_clipping = 1.0` eliminated the epoch-7 divergence
- Loss band 1.28-1.43 vs previous 1.3->8.97 explosion
- Both 1k and 50k runs stable

### Milestone 6: Key negative finding (completed)
- 50x more data does not improve the plateau
- Training stability (not data quantity) was the real lever
- This rules out the "more data" hypothesis and redirects focus to architecture/SIL/objective

---

## 3. The Path Forward — Step-by-Step Plan

### Phase A: Diagnostics (measure everything we don't know)

**Duration**: ~2-4 hours wall-clock, no new training needed.

#### Experiment A1: Compute the Optimality Gap (IMMEDIATE, HIGHEST PRIORITY)

**What**: Modify `vne_main.py:evaluate()` to compute `model_cost / ilp_cost` per instance.

**Implementation**:
```python
# In vne_main.py:evaluate(), inside process_search_results:
with open(to_evaluate_path, "rb") as f:
    all_instances = pickle.load(f)
ilp_objectives = np.array([inst.get("objective", np.nan) for inst in all_instances])
model_objectives = np.array([result["objective"] for result in results])
valid_mask = ~np.isnan(ilp_objectives)
gap = float(np.mean(model_objectives[valid_mask] / ilp_objectives[valid_mask]))
return {"mean_obj": float(objectives.mean()), "opt_gap": gap}
```

Also log histogram of per-instance gaps: min, max, median, p25, p75, std.

**Expected output**: A single number: the optimality gap of the current best supervised model. Plus per-instance distribution.

**Go/No-Go gate**: If gap < 5%, proceed to Phase B (SIL targeting small remaining gap). If gap > 20%, redirect to architecture scaling first (Phase A3).

**File location**: `vne_main.py`, function `process_search_results` inside `evaluate()` (line 151-155).

#### Experiment A2: Candidate Path Statistics

**What**: Add a counter to `vne/trajectory.py:_candidate_paths` that logs (num_candidates, path_length, request_idx, link_idx) for every call during a single validation pass. Run `evaluate` on validation set with logging enabled.

**Purpose**: Determine what beam_width is appropriate. If mean candidates < 10, beam_width=8 is fine. If > 50, beam_width needs to be higher.

**Expected duration**: <10 minutes (one validation pass with logging).

**Go/No-Go gate**: If mean candidates > 30, the beam_width grid minimum should be 16, not 8.

**File location**: `vne/trajectory.py:_candidate_paths` (line 121-192).

#### Experiment A3: Dead-end Path Frequency

**What**: During the same validation pass, count how many trajectories hit `objective = float("-inf")` (line 272 in `vne/trajectory.py`). Also count instances where `_candidate_paths` returns empty at a non-terminal decision.

**Purpose**: If dead-ends are rare (<1% of trajectories) in the current loose-regime, they are not a practical concern for TaSaR. If common, the SBS trie needs special handling.

**Expected duration**: <10 minutes (same validation pass as A2).

#### Experiment A4: Minimal Achievable Loss (Information-Theoretic Ceiling)

**What**: For each decision in the validation set, count how many candidate paths achieve the minimum cost. Compute the "degeneracy entropy": if there are always 2+ equally-good paths, the irreducible CE floor may be above 1.0.

**Implementation**: During dataset replay, after computing candidates and the chosen (optimal) path, count how many candidates achieve the same cost as the chosen path. Aggregate statistics across the validation set.

**Purpose**: If the CE floor is ~1.3 (the current training loss), the model may already be at the information limit. This would explain why supervised training cannot improve further and would mean SIL is the only escape (because SIL replaces CE targets with beam-search decisions rather than ILP optimums).

**Expected duration**: ~30 minutes of analysis on existing data.

**File location**: `vne/dataset.py:RandomVNEDataset.__getitem__` (line 195-220).

#### Experiment A5: Architecture Bottleneck Diagnostic (requires training)

**What**: Train a symmetric 6+6 encoder+decoder architecture (same total layers as current 1+9, rebalanced). Use same LR/clipping/data for 15 epochs. Compare: validation objective, gap, training loss convergence.

**Config changes**:
```python
config.num_encoder_layers = 6
config.num_decoder_layers = 6
# Keep: embedding_dim=128, heads=8, ff_dim=512
```

**Expected duration**: ~2 hours on MPS A30.

**Go/No-Go gate**: If 6+6 achieves val objective > -21.x (vs current -22.9), the encoder was the bottleneck. Use the 6+6 architecture for all subsequent experiments. If plateau unchanged, the limitation is elsewhere (decoder capacity, feature representation, or data).

**File location**: `vne/config.py` lines 88-89. Overridable via `VNE_NUM_ENCODER_LAYERS` and `VNE_NUM_DECODER_LAYERS` env vars.

#### Experiment A6: ReZero + Architecture Scaling (contingent on A5)

**What**: If A5 shows the encoder is the bottleneck, test a medium-scale architecture: dim=192, heads=12, 6+6 layers, ff_dim=768, ReZero enabled. Train supervised for 15 epochs.

**Config changes**:
```python
config.embedding_dim = 192
config.hidden_dim = 384
config.num_attention_heads = 12
config.num_encoder_layers = 6
config.num_decoder_layers = 6
config.feedforward_dimension = 768
config.use_rezero_transformer = True
```

**Expected duration**: ~4 hours on MPS A30 or L40S.

**Purpose**: Establish the strongest feasible supervised baseline before activating SIL. A stronger supervised baseline gives SIL more room to improve.

**File locations**: All overridable via `VNE_*` env vars in `vne_main.py:_apply_env_overrides` (lines 291-338).

### Phase B: First SIL Run (Proof of Life)

**Duration**: ~2-8 hours wall-clock for the first epoch, ~6-24 hours for 15 epochs.

#### Experiment B1: Single-Epoch SIL Feasibility Check

**What**: Run ONE epoch of Gumbeldore training. Config:
- `learning_type = "gumbeldore"` (via `VNE_LEARNING_TYPE=gumbeldore`)
- `num_instances_to_generate = 64` (reduced from 256 for speed)
- `beam_width = 8`, `replan_steps = 2` (current defaults)
- `devices_for_workers = ["cpu"]` (CPU generation for this test)
- `perform_first_round_deterministic = True`
- Load best supervised checkpoint as starting policy

**What to measure**:
1. **Does it crash?** Any errors in SBS/trie interaction with VNE's variable-length sequences
2. **Wall-clock time** for generation + training combined
3. **Generated solution quality**: mean objective of TaSaR-generated solutions vs beam-1 of the supervised model
4. **Post-training validation**: did it improve, degrade, or stay the same?

**SLURM command**:
```bash
sbatch --partition=gpuISIN --gres=mps:30 \
  --export=ALL,VNE_RUN_TAG=tasar_test,VNE_LEARNING_TYPE=gumbeldore,\
VNE_NUM_GENERATE=64,VNE_BEAM_WIDTH=8,VNE_REPLAN_STEPS=2,VNE_NUM_EPOCHS=1 \
  scripts/vne_train.sbatch
```

**Go/No-Go gate**: 
- Crash -> Debug SBS/variable-length interaction before proceeding. Fix most likely in `perform_tasar` where it assumes uniform-depth trie nodes.
- Generation > 4 hours -> Switch to GPU workers before full grid.
- Generated solutions same or worse than supervised model's beam-1 (expected at k=8,s=2) -> Acceptable, proceed to B2.
- Generated solutions much worse (e.g., 2x cost) -> Increase beam_width or check for dead-end path contamination.

#### Experiment B2: Full TaSaR Run (15 epochs)

**What**: Run a complete 15-epoch Gumbeldore training cycle with modest settings:
- `num_instances_to_generate = 256`
- `beam_width = 16`, `replan_steps = 4`
- `perform_first_round_deterministic = True`
- `check_future_completion = True` (default)

**SLURM command**:
```bash
sbatch --partition=gpuISIN --gres=mps:40 \
  --export=ALL,VNE_RUN_TAG=tasar_base,VNE_LEARNING_TYPE=gumbeldore,\
VNE_NUM_GENERATE=256,VNE_BEAM_WIDTH=16,VNE_REPLAN_STEPS=4,VNE_NUM_EPOCHS=15 \
  scripts/vne_train.sbatch
```

**What to track per epoch**:
- Mean generated objective (are pseudo-labels improving?)
- Training loss stability (is the SIL loop stable?)
- Validation objective (is it improving over supervised baseline?)
- Wall-clock time per epoch

**Success criteria**:
- **Minimum**: No crashes, 15 epochs complete, val objective does not degrade below -23.5
- **Good**: Val objective improves by >1% over supervised baseline (e.g., -22.9 to -22.6 or better)
- **Outstanding**: Val objective improves by >5% (e.g., -22.9 to -21.8 or better)

**Contingency**: If val diverges or training loss becomes unstable, try:
- Reduce LR to 1e-4 (via `VNE_LR=0.0001`)
- Increase gradient clipping to 0.5
- Use `use_best_model_for_generation = True` (already default)

#### Experiment B3: GPU Workers for Generation

**What**: If B1 shows CPU generation is too slow, modify `async_sbs_worker` to apply MPS memory cap and use GPU workers.

**Implementation**: In `core/gumbeldore_dataset.py:async_sbs_worker` (line 148), after setting CUDA_VISIBLE_DEVICES (line 172), add:
```python
mps_frac = os.environ.get("VNE_WORKER_GPU_MEM_FRACTION", "0.15")
if device != "cpu" and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(mps_frac), 0)
```
Add `VNE_WORKER_GPU_MEM_FRACTION` handler in `_apply_env_overrides` with default 0.15.

**Then run with**:
```bash
sbatch --partition=gpuISIN --gres=mps:50 \
  --export=ALL,VNE_RUN_TAG=tasar_gpu,VNE_LEARNING_TYPE=gumbeldore,\
VNE_NUM_GENERATE=256,VNE_BEAM_WIDTH=16,VNE_REPLAN_STEPS=4,\
VNE_GEN_DEVICE=cuda:0,VNE_WORKER_GPU_MEM_FRACTION=0.20 \
  scripts/vne_train.sbatch
```

**File locations**: 
- `core/gumbeldore_dataset.py:169-176` (MPS cap for workers)
- `vne_main.py:291-338` (env override handler)
- `vne/config.py:127` (devices_for_workers default)

### Phase C: Hyperparameter Optimization

**Duration**: ~12-24 GPU-hours, ~1-2 days wall-clock with parallel partitions.

#### Experiment C1-C6: (k, s) Grid

**Grid** (6 experiments):

| Run | beam_width (k) | replan_steps (s) | Notes |
|-----|---------------|-------------------|-------|
| C1  | 16            | 2                 | Wide beam, short commit |
| C2  | 16            | 8                 | Wide beam, long commit |
| C3  | 32            | 4                 | Middle ground |
| C4  | 32            | 8                 | Matches paper ratio for VNE (~1/5 sequence length) |
| C5  | 64            | 4                 | Aggressive beam, moderate commit |
| C6  | 64            | 8                 | Max beam, max commit (most expensive) |

**Fixed params**: `perform_first_round_deterministic=True`, `num_instances_to_generate=512`, `num_epochs=15`, `lr=2e-4`, `gradient_clipping=1.0`.

**Adaptive early stopping**: If validation objective hasn't improved for 5 epochs, terminate that run and allocate resources elsewhere.

**Parallel submission**:
```bash
# C1 on gpuISIN (gnode05 A30)
sbatch --partition=gpuISIN --gres=mps:30 \
  --export=ALL,VNE_RUN_TAG=tasar_k16s2,VNE_LEARNING_TYPE=gumbeldore,\
VNE_NUM_GENERATE=512,VNE_BEAM_WIDTH=16,VNE_REPLAN_STEPS=2,VNE_NUM_EPOCHS=15 \
  scripts/vne_train.sbatch

# C3 on gpuISIN (gnode06 L40S)
sbatch --partition=gpuISIN --gres=mps:20 \
  --export=ALL,VNE_RUN_TAG=tasar_k32s4,VNE_LEARNING_TYPE=gumbeldore,\
VNE_NUM_GENERATE=512,VNE_BEAM_WIDTH=32,VNE_REPLAN_STEPS=4,VNE_NUM_EPOCHS=15 \
  scripts/vne_train.sbatch
```

**Go/No-Go gate** for proceeding to full grid: At least 2 runs from {C1, C2, C3} must show val objective improvement over supervised baseline after 5 epochs. If none show improvement after 5 epochs, abort the grid and investigate root cause.

#### Experiment C7: Gumbeldore (WOR) Without TaSaR

**What**: If the (k,s) grid shows no improvement, try the original WOR method (`search_type="wor"`) which uses all beam leaves weighted by Gumbel perturbation rather than only the best leaf with replanning.

**Config change**: `gumbeldore_config["search_type"] = "wor"`, `gumbeldore_config["num_rounds"] = 4`.

**Expected output**: Comparison of WOR vs TaSaR for VNE. If WOR helps but TaSaR doesn't, the replan mechanism is misfiring for VNE (likely due to variable-length sequences).

**File location**: `vne/config.py` line 137 (`search_type` key).

### Phase D: Scaling and Generalization

**Duration**: ~4-8 hours.

#### Experiment D1: Multi-Beam Evaluation

**What**: Evaluate the best model from Phase C at beam_width = {1, 4, 8, 16, 32, 64} on the 2k test set.

**Implementation**: Modify `vne/config.py` to add beam widths:
```python
beams_with_batch_sizes = {1: 64, 4: 32, 8: 16, 16: 8, 32: 4, 64: 2}
validation_relevant_beam_width = 1
```

**Purpose**: Create the gap-vs-beam_width curve. This directly tests whether the beam search mechanism helps VNE inference.

#### Experiment D2: OOD Generalization (Substrate Size)

**What**: Generate test sets at larger substrate sizes (40-60 and 60-100 nodes) using `vne_gen_array.sbatch`. Evaluate the best model without retraining. Compare gap to ILP (for 40-60 where ILP is still tractable) and to heuristic baselines (for 60-100 where ILP may timeout).

**Implementation**:
```bash
# Generate medium-scale test set
sbatch --partition=compute --cpus-per-task=8 \
  --export=ALL,config_num_instances=500,\
config_substrate_range="(40,60)",config_request_range="(4,10)" \
  scripts/vne_gen_array.sbatch
```

**Expected duration**: ~1 hour generation + 30 min evaluation.

**Go/No-Go gate**: If gap at 40-60 nodes is <5% (same model, no retraining), the model generalizes well. If >10%, the model's learned representations are fragile.

#### Experiment D3: OOD Generalization (Topology)

**What**: Generate a test set with ER (Erdos-Renyi) topology instead of line topology. Evaluate the best model without retraining.

**Config change**: `substrate_topology = "er"` with `substrate_edge_probability = 0.3`.

**Expected duration**: ~1 hour generation + 30 min evaluation.

**Purpose**: Tests whether the model learns topology-agnostic strategies or overfits to the line topology.

### Phase E: Paper-Ready Results

**Duration**: ~2-4 days for analysis, figure generation, and writing.

#### Deliverables:
1. **Table 1 — Main Results**: Gap at beam-1, beam-16, beam-64 for supervised vs TaSaR, with inference time.
2. **Figure 1 — Training Curves**: Val objective vs epoch for supervised and best TaSaR config, with optimality gap overlay.
3. **Figure 2 — Parameter Sensitivity**: Heatmap of (k, s) vs final gap.
4. **Figure 3 — Pareto Frontier**: Gap vs inference time for all methods.
5. **Figure 4 — Generalization**: Gap bar chart across substrate sizes (20-40, 40-60, 60-100).
6. **Table 2 — Ablation**: Effect of removing components (deterministic first round, future completability check).

---

## 4. Optimization Goals & Metrics

### Primary Metric

**Optimality gap**: `model_objective / ilp_objective` (both negative, ratio > 1 means worse than ILP).

Report: mean, median, p25, p75 across test set. Also report per-instance gap histograms.

### Secondary Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Acceptance ratio | % of instances where all requests embedded | 100% (embed-all regime) |
| Inference time | ms per VNE instance at beam width w | <100ms for beam-64 |
| Training time | Wall-clock hours to converge | <24h total |
| Generalization gap | Gap on 40-60 node substrates without retraining | <5% |
| Training loss stability | Std of train loss across epochs | <0.1 after epoch 3 |

### What Numbers Would Be "Publishable"

| Scenario | Gap | Speedup vs ILP | Venue |
|----------|-----|----------------|-------|
| Minimum viable | <5% | 100x-1000x | Networking workshop |
| Good | <2% | 1000x | INFOCOM/ToN |
| Outstanding | <1% + generalizes | 10000x | NeurIPS/ICML |

---

## 5. Constraints & Resources

### Cluster Resources

| Partition | Nodes | GPUs | MPS | Best for |
|-----------|-------|------|-----|----------|
| gpuISIN | gnode05 | A30 (24GB) | mps:a30:100 | Primary training (lowest contention) |
| gpuISIN | gnode06 | 2xL40S (46GB each) | mps:l40s:200 | Heavy runs, parallel jobs |
| gpu_AMD | gnode11-12 | 2xRTX PRO 6000 (98GB each) | mps:rtx_pro_6000:200 | Largest experiments |
| compute | cnode01-04 | CPU only | N/A | Data generation, Ray CPU workers |

**MPS sharing guidelines**:
- VNE model is ~2M params (< 50 MB). One epoch needs < 4 GB GPU memory.
- Request `--gres=mps:20` to `--gres=mps:40` (20-40% of a GPU).
- Current `VNE_GPU_MEM_FRACTION=0.15` (capped to 15%) is conservative.
- For beam-64 generation workers: increase to 0.25.

**CUDA compatibility**:
- Current torch 2.5.1+cu121 works on A30 (SM80), L40S (SM89).
- RTX PRO 6000 Blackwell (SM120) needs torch 2.6+ with CUDA 12.5+. **Verify before using.**

### Solver Constraints

- HiGHS is the default solver (free, no license needed).
- Timeout at 60s/instance; best feasible incumbent is kept (no timeout bias).
- MIP gap = 0.0 (prove exact optimality). For large instances (40-60+ nodes), may need to relax to 0.01-0.03 to avoid timeout.
- Gurobi/CPLEX available if licenses permit, but HiGHS is sufficient for the current regime.

### Code Stability Map

| Area | Status | Needs modification for SIL? |
|------|--------|---------------------------|
| Instance generation | Stable | No |
| Solver pipeline | Stable | No |
| Dataset/replay | Stable | No |
| State features | Stable | No |
| Policy network | Stable | Minor (MPS fraction in worker) |
| Supervised training loop | Stable | No |
| TaSaR core (IncrementalSBS) | Stable (for TSP/JSSP) | Possible (variable-length tree) |
| SBS mechanics | Stable (for TSP/JSSP) | Verify for VNE |
| GumbeldoreDataset / Ray workers | Stable (for TSP/JSSP) | Add MPS cap |
| Validation evaluation | Needs modification | Add opt gap computation |
| Config | Stable | Override via env vars |
| SLURM scripts | Stable | Add env vars for new params |

### Estimated Wall-Clock

| Phase | GPU-hours | Wall-clock (parallel) | Dependencies |
|-------|-----------|----------------------|--------------|
| A1-A4 (diagnostics) | 0 | 2-4h | None |
| A5 (architecture) | 3 | 3h | None |
| A6 (scaling) | 6 | 6h | A5 results |
| B1 (SIL feasibility) | 3-8 | 3-8h | A1-A4 complete |
| B2 (full run) | 10-20 | 10-20h | B1 success |
| C1-C6 (grid) | 20-40 | 8-16h (3-4 parallel) | B2 success |
| D1-D3 (eval) | 4-8 | 4-8h | C grid complete |
| E (paper) | 0 | 2-4 days | All phases complete |

**Total worst-case**: ~75 GPU-hours, ~4-7 days wall-clock with 3-4 parallel jobs.

---

## 6. Risk Register

### Top 10 Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | TaSaR doesn't improve over supervised (pseudo-labels no better than ILP labels) | MEDIUM (40%) | HIGH (core hypothesis fails) | Test deeper; if still fails, paper shifts to "first neural VNE competitive with ILP at 1000x speed" |
| R2 | Variable-length SBS tree breaks TaSaR (crashes or produces wrong results) | MEDIUM (30%) | CRITICAL (blocker) | Run B1 first. Debug trie interaction. Likely fix: filter dead-end trajectories before best-leaf selection in `perform_tasar` line 326 |
| R3 | SIL training diverges (pseudo-labels change too fast) | MEDIUM (30%) | HIGH | Reduce LR to 1e-4, tighten gradient clipping to 0.5, use EMA policy for generation |
| R4 | CPU generation too slow (each epoch takes >4 hours) | HIGH (60%) | MEDIUM | Switch to GPU workers (B3). Or reduce `num_instances_to_generate` to 128 |
| R5 | Optimality gap already small (<2%) | MEDIUM (30%) | MEDIUM (SIL has little room) | Shift contribution to inference speed, generalization, and "first neural VNE" framing |
| R6 | GPU contention with other cluster users | HIGH (50%) | MEDIUM | Use MPS sharing + multiple partitions. gpuISIN is least contended |
| R7 | HiGHS times out on large test instances | HIGH (60%) for 60-100 nodes | LOW | For generalization, compare against heuristic baselines (greedy, shortest-path) instead of ILP |
| R8 | Candidate enumeration too expensive for SBS (hidden O(exponential) cost) | MEDIUM (30%) | MEDIUM | Cache candidate paths between transitions (identical resources -> identical candidates). Or prune earlier |
| R9 | Architecture bottleneck confirmed but scaling doesn't help | LOW (20%) | MEDIUM | The limitation is data representation, not parameters. Investigate feature engineering |
| R10 | Ray initialization failures on SLURM | LOW (10%) | MEDIUM | `main_train_cycle` already handles Ray init. Set `RAY_DEDUP_LOGS=0` (already in `gumbeldore_dataset.py`) |

### The "TaSaR Fails for VNE" Contingency Plan

**Definition of failure**: After Phase B (15 epochs of SIL at k=16,s=4), the validation objective is no better than the supervised baseline (-22.9). The generated pseudo-labels are consistently worse than the supervised model's own greedy output.

**Three-tiered response**:

**Tier 1 — Try alternatives within the codebase (1 week)**:
- Switch to `search_type = "wor"` (vanilla WOR without TaSaR replanning). The paper's WOR may work better for VNE's short sequences than TaSaR's commit-then-replan.
- Increase `num_instances_to_generate` to 1024 (more label diversity).
- Start from random weights instead of supervised pretrained (paper's approach).
- Test with lower LR (1e-4) and higher gradient clipping (0.5).

**Tier 2 — Change the problem formulation (2 weeks)**:
- Switch to the admission control regime (contended resources, rejection decisions). This gives SIL more room to improve because the optimal action is less obvious from the state alone.
- Use revenue-profit objective instead of min-cost. Profit-based labels have more variance, making SIL's improvement more detectable.
- Enable `enable_admission = True` and set contended resource ranges per `vne/config.py` inline notes (lines 15-23).

**Tier 3 — Pivot paper strategy (no additional compute)**:
- Publish as "Neural Virtual Network Embedding: A Supervised Learning Baseline" rather than "TaSaR for VNE."
- Key contribution: First open-source neural VNE framework, comprehensive benchmark, speed vs quality Pareto analysis.
- Negative result on SIL: "Why Self-Improvement Fails for VNE" — analysis of why pseudo-labels cannot beat ILP labels for this problem.

### Decision Tree at Each Major Gate

```
Gate A1-A4 (diagnostic complete):
├── Gap < 5% → Path: "near-optimal baseline, SIL is optional luxury"
│   └── Proceed to B1 with lower priority on SIL improvement
├── Gap 5-20% → Path: "room to improve, SIL primary route"
│   └── Proceed to B1 with full priority
└── Gap > 20% → Path: "fundamental architecture problem"
    └── Skip to A5 (architecture diagnostic)

Gate A5 (architecture tested):
├── 6+6 significantly better → adopt 6+6, re-run A1 gap
└── 6+6 same as 1+9 → architecture not the bottleneck, proceed to B1

Gate B1 (SIL feasibility):
├── Crashes → debug variable-length SBS interaction
├── Stable, no improvement → proceed to B2 (may converge later)
└── Stable, improves → proceed to C grid with confidence

Gate B2 (full SIL run):
├── Val improves >3% over supervised → proceed to C grid
├── Val within 3% of supervised → proceed to C grid but with contingency
└── Val degrades >5% from supervised → invoke Tier 1 contingency

Gate C2-C4 (grid checkpoint after 5 epochs):
├── 2+ configs show improvement → complete full grid
├── 1 config shows improvement → try WOR as alternative
└── 0 configs show improvement → invoke Tier 1/2 contingency
```

---

## 7. Paper Strategy

### Target Venues by Result Strength

| Result Tier | Primary Venue | Secondary Venue | Requirements |
|------------|--------------|-----------------|--------------|
| TaSaR improves >5% over supervised | NeurIPS/ICML workshop | AAAI/IJCAI | 3+ baselines, statistical significance, generalization |
| TaSaR matches supervised (within 3%) | INFOCOM | IEEE/ACM ToN | Speed plot, generalization, literature comparison |
| Supervised only (SIL skipped) | Networking workshop | Arxiv | Comprehensive benchmark, open-source release |
| Negative result on SIL | NeurIPS workshop on ML for Systems | Arxiv | Thorough ablation, community guidance |

### Narrative Arc by Result

**If TaSaR works**:
> "We present the first self-improved learning method for Virtual Network Embedding, extending the TaSaR framework (ECAI-2024) to this structurally novel problem domain. Unlike supervised learning, which is capped by the ILP solver's solution quality, TaSaR iteratively generates its own training data via stochastic beam search with a reconsider mechanism. On instances with 20-40 node substrates, our method achieves a 0.5% optimality gap — a 3x improvement over the supervised ceiling of 1.7% — while requiring only milliseconds per inference. TaSaR generalizes to substrates 3x larger than training, where the ILP solver becomes intractable."

**If TaSaR matches but doesn't beat**:
> "We present the first comprehensive neural framework for VNE, including instance generation, ILP labeling, supervised training, and self-improved learning. Our supervised model matches ILP quality within 2% while providing 1000x speedup at inference time. While SIL (TaSaR) does not significantly improve over supervised learning for the embed-all min-cost regime — because the ILP labels already represent the optimal policy — we show that TaSaR produces competitive solutions without requiring any solver interaction at training time, opening the door to scenarios where expert labels are unavailable."

**If SIL fails (negative result)**:
> "We provide the first systematic analysis of why supervised learning and self-improvement can fail for VNE. Our diagnostic framework reveals that the encoder bottleneck (1 vs 9 layer asymmetry) and the degenerate action space (many equally-good paths) combine to create an irreducible performance ceiling. We identify specific architectural and data interventions, and release an open-source benchmark for future work."

### Minimum Viable Paper (Regardless of Outcome)

**Required for any publishable output**:
1. Optimality gap measurement (A1) — the single most important number
2. Training curves for supervised and TaSaR
3. Pareto frontier: gap vs inference time
4. Generalization results (at least substrate size scaling)
5. Open-source release of all code and data

**Figure plan**:
- Fig 1: Training curves (val objective vs epoch) for supervised vs TaSaR
- Fig 2: Pareto frontier (gap vs inference time)
- Fig 3: Parameter sensitivity heatmap (k x s)
- Fig 4: Generalization bar chart (gap across substrate sizes)
- Table 1: Main results with all baselines
- Table 2: Ablation study

---

## 8. Implementation Details

### Exact Config Changes for First SIL Run

**Method A: Using env vars (preferred, no file modification)**:
```bash
export VNE_LEARNING_TYPE=gumbeldore
export VNE_NUM_GENERATE=64    # small for first test
export VNE_BEAM_WIDTH=8
export VNE_REPLAN_STEPS=2
export VNE_NUM_EPOCHS=1       # single epoch for feasibility check
export VNE_RUN_TAG=tasar_test
```

**Method B: In config.py (if env vars are insufficient)**:
```python
# vne/config.py, line 7
self.learning_type = "gumbeldore"

# line 137
self.gumbeldore_config["search_type"] = "tasar"

# line 139
self.gumbeldore_config["beam_width"] = 8

# line 155
self.gumbeldore_config["perform_first_round_deterministic"] = True

# line 159
self.gumbeldore_config["replan_steps"] = 2

# line 129
self.gumbeldore_config["num_instances_to_generate"] = 64

# Keep devices_for_workers = ["cpu"] for first test (line 127)
```

### Expected SLURM Commands

**Diagnostic run (Phase A1, gap computation on existing model)**:
```bash
# Load best supervised checkpoint, evaluate with gap logging
cd /mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider
./.venv/bin/python -c "
import pickle, numpy as np
from vne.config import VNEConfig
from vne_main import evaluate, get_network
import torch

config = VNEConfig()
device = torch.device('cuda')
network = get_network(config, device)
metrics, loggable = evaluate('Validation', config, network, config.validation_set_path, config.validation_custom_num_instances)
print(loggable)
"
```

**First SIL epoch (Phase B1)**:
```bash
sbatch --partition=gpuISIN --gres=mps:30 \
  --export=ALL,VNE_RUN_TAG=tasar_feasibility,\
VNE_LEARNING_TYPE=gumbeldore,VNE_NUM_GENERATE=64,\
VNE_BEAM_WIDTH=8,VNE_REPLAN_STEPS=2,VNE_NUM_EPOCHS=1,\
VNE_GPU_MEM_FRACTION=0.20 \
  scripts/vne_train.sbatch
```

**Full TaSaR run (Phase B2)**:
```bash
sbatch --partition=gpuISIN --gres=mps:40 \
  --export=ALL,VNE_RUN_TAG=tasar_k16s4,\
VNE_LEARNING_TYPE=gumbeldore,VNE_NUM_GENERATE=256,\
VNE_BEAM_WIDTH=16,VNE_REPLAN_STEPS=4,VNE_NUM_EPOCHS=15,\
VNE_GPU_MEM_FRACTION=0.25 \
  scripts/vne_train.sbatch
```

**Architecture scaling (Phase A5, symmetric 6+6)**:
```bash
sbatch --partition=gpuISIN --gres=mps:40 \
  --export=ALL,VNE_RUN_TAG=arch_sym66,\
VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6,\
VNE_NUM_EPOCHS=15,VNE_GPU_MEM_FRACTION=0.25 \
  scripts/vne_train.sbatch
```

**Medium architecture (Phase A6, dim=192, ReZero)**:
```bash
sbatch --partition=gpuISIN --gres=mps:50 \
  --export=ALL,VNE_RUN_TAG=arch_medium,\
VNE_EMBEDDING_DIM=192,VNE_HIDDEN_DIM=384,\
VNE_NUM_HEADS=12,VNE_NUM_ENCODER_LAYERS=6,\
VNE_NUM_DECODER_LAYERS=6,VNE_FF_DIM=768,\
VNE_NUM_EPOCHS=15,VNE_GPU_MEM_FRACTION=0.30 \
  scripts/vne_train.sbatch
```
Note: `use_rezero_transformer` env var not in `_apply_env_overrides`. Need to add:
```python
if env.get("VNE_USE_REZERO"):
    config.use_rezero_transformer = env["VNE_USE_REZERO"].lower() == "true"
```

### Monitoring and Logging Strategy

**Per-epoch log output** (logged to `logs/vne_train_<tag>_<jobid>.log`):
```
>> Epoch X. Avg loss: Y.YY
Train avg loss: Y.YY
Avg generated obj: Z.ZZ       (only for Gumbeldore)
{Validation beam width 1. Obj.: -AA.AA}
```

**Extended logging for SIL** (add to `evaluate()`):
```python
# Optimality gap per epoch
with open(to_evaluate_path, "rb") as f:
    all_instances = pickle.load(f)[:num_instances]
ilp_objs = np.array([inst.get("objective", np.nan) for inst in all_instances])
gap = float(np.mean(model_objs / ilp_objs))
gap_std = float(np.std(model_objs / ilp_objs))
# Log
logger.log_metrics({
    "opt_gap": gap,
    "opt_gap_std": gap_std,
    "ilp_mean": float(np.mean(ilp_objs)),
    "model_mean": float(np.mean(model_objs)),
}, step=epoch)
```

**Post-run analysis**:
```bash
./.venv/bin/python scripts/vne_plot_training.py --log logs/vne_train_tasar_k16s4_<jobid>.log
```

**Crash recovery note**: The Gumbeldore loop does not checkpoint generated data. If a SLURM job times out during generation (epoch 12 of 15), the entire epoch's generated data is lost. The `best_model.pt` is saved only after validation. Consider setting `--time=2-00:00:00` (48h) for SIL runs.

### How to Verify Correctness Before Scaling

**Checklist before the first SIL run**:
1. [ ] Supervised model loads and evaluates without error (gap computed)
2. [ ] `vne/trajectory.py:init_batch_from_instance_list` works with raw instances from the generator
3. [ ] `vne/trajectory.py:log_probability_fn` returns correct-shaped numpy arrays
4. [ ] `vne/trajectory.py:transition_fn` handles all edge cases (final link, last request, dead-end)
5. [ ] `IncrementalSBS` can be instantiated with VNE root nodes
6. [ ] `perform_tasar` returns one leaf per instance without crashing

**To test items 3-6 in isolation**:
```python
# From the vne_main.py directory
from vne.config import VNEConfig
from vne.instance_generator import make_instance
from vne.trajectory import Trajectory
from vne.network import VNEPolicyNetwork
from core.incremental_sbs import IncrementalSBS
import torch, copy

config = VNEConfig()
device = torch.device("cuda")
network = VNEPolicyNetwork(config, device)

# Generate a single instance
instance = make_instance(config)

# Initialize trajectory
roots = Trajectory.init_batch_from_instance_list([instance], network, device)
print(f"Roots: {len(roots)}, candidates: {roots[0].num_actions()}")

# Test log prob
log_probs = Trajectory.log_probability_fn(roots, network)
print(f"Log probs shape: {[lp.shape for lp in log_probs]}")

# Test transition
child, is_leaf = roots[0].transition_fn(0)
print(f"Transition: is_leaf={is_leaf}, candidates={child.num_actions()}, objective={child.objective}")

# Test IncrementalSBS
inc_sbs = IncrementalSBS(
    roots, 
    lambda trajs: Trajectory.log_probability_fn(trajs, network),
    lambda pairs: [t.transition_fn(a) for t, a in pairs],
    Trajectory.to_max_evaluation_fn
)
results = inc_sbs.perform_tasar(beam_width=4, deterministic=True, replan_steps=2)
print(f"TASAR results: {len(results)} leaves, obj={results[0][0].state.objective}")
```

This test script should be run BEFORE the first SIL SLURM submission to catch integration errors early.

---

## Appendix: Resolved Agent Disagreements

| Topic | Disagreement | Resolution |
|-------|-------------|------------|
| **Architecture asymmetry** | Developer: 1+9 is fine. Criticizer: 1+9 is the bottleneck. | A5 (6+6 test) resolves empirically. Test before SIL. |
| **Beam_width** | Developer: 8. Criticizer: 32-64. Investigator: measure first. | Start at 16, grid {16,32,64}. A2 measures floor. |
| **replan_steps** | Developer: 2. Criticizer: 5-20. Refiner: 2-8. | Grid over {2,4,8}. s=8 ~ 1/5 avg sequence length. |
| **num_instances_to_generate** | Developer: 256. Everyone else: 512-1024. | Start at 64 (feasibility), then 256, then 512 for production. |
| **ReZero** | Disabled in config. Criticizer/Counterfactual: enable. | Enable as part of architecture scaling (A6). |
| **SIL feasibility** | Criticizer: skeptical (optimal labels ceiling). Investigator: optimistic. Counterfactual: "truly can't beat optimal." | Run the experiment. Contingency plans ready for each outcome. |
| **Paper venue** | Refiner: NeurIPS. Counterfactual: safety net at networking venue. | Aspire to ML venue; if SIL doesn't work, pivot to networking venue. |
| **GPU generation** | All agree CPU is too slow. Resolution unclear on implementation. | B3: add MPS cap to workers. Implement before large grid. |

---

*This roadmap was synthesized from position papers by five specialist agents (Developer, Criticizer, Investigator, Refiner, CounterfactualMaster). All disagreements are resolved via explicit empirical tests with go/no-go gates rather than pre-judgment.*
