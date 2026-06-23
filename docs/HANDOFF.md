# VNE+TaSaR — Phase 2 Handoff

**Date**: 2026-06-23  
**Branch**: `yorgo-work` (commit `9be96cb`)  
**Phase 1**: ✅ Complete | **Phase 2**: 🔧 Code ready, needs GPU

---

## 1. What This Repo Is

TaSaR (ECAI-2024) applied to Virtual Network Embedding (VNE). Fork of [gumbeldore](https://github.com/grimmlab/gumbeldore).

| Directory | Purpose |
|---|---|
| `vne/` | Problem: config, trajectory MDP, BQ/LEHD networks, dataset loader, features |
| `core/` | Search (SBS, TaSaR, beam search) + shared training loop |
| `scripts/` | SLURM launch scripts |
| `model_checkpoints/vne/results/` | Training outputs |
| `data/vne/` | Pickled instances (ILP-labeled) |
| `vne/PROBLEM_FORMULATION.md` | Formal VNE problem definition |

**Current scale**: 60–80 substrate comm nodes, 10–20 virtual requests, line topology, embed-all, min-cost objective.

---

## 2. Phase 1 Results (Supervised Baselines) — DONE

5 architectures, 2.5k ILP-labeled instances, 30 epochs each:

| Architecture | Dim | Epochs | Best Gap% | Feas% | Checkpoint |
|---|---|---|---|---|---|
| **BQ** | 128 | 30 | **2.08%** test | 36.8% | `model_checkpoints/vne/results/phase1_bq128_42476/last_model.pt` |
| **LEHD** | 128 | 20 | **1.40%** val | 69.5% | `model_checkpoints/vne/results/phase1_lehd128_42487/last_model.pt` |
| BQ | 192 | 13 | — | — | OOM killed |
| LEHD | 192 | — | — | — | OOM killed |
| LEHD | 256 | — | — | — | OOM killed |

**Best two**: BQ-128 and LEHD-128 are the seed models for Phase 2.

---

## 3. Phase 2 — TaSaR Self-Improving Learning (SIL)

Take a Phase 1 checkpoint, generate instances via TaSaR beam search, train one epoch on self-generated labels, repeat.

**Target**: k×s grid search: `beam_width`(k) × `replan_steps`(s).

**Progress**:

| Job | Arch | k | s | Epochs | Result |
|---|---|---|---|---|---|
| 42727 | BQ-128 | 32 | 4 | 3 | Cancelled (OOM) |
| 42730 | LEHD-128 | 32 | 4 | 3 | Cancelled (OOM) |
| 42731 | BQ-128 | 32 | 4 | 3 | Cancelled (680s/it, no cap) |
| 42741 | BQ-128 | 4 | 2 | 3 | Epoch 1 trained, crashed on load (fixed) |
| **42846** | BQ-128 | 8 | 4 | 10 | **PENDING** (cluster full) |

**Recommended pilot**: k=8, s=4 for both BQ-128 and LEHD-128. k=32 was too slow; k=4 may be too narrow.

---

## 4. Critical Code Fixes (on this branch)

### 4.1 Candidate enumeration cap — `vne/trajectory.py:131`
```python
max_candidates: int = 500  # cap DFS path enumeration
```
60–80 node graphs produce 3,000+ candidate paths per decision step. Forward pass over all of them takes **680s/instance at k=32**. With cap=500: ~5× speedup. Early links that hit the cap just use the first 500 paths — minimal gap impact.

### 4.2 check_future_completion OFF — `vne/trajectory.py:130`
```python
check_future_completion: bool = False  # OFF: O(c^d) recursion
```
When True, every candidate triggers a recursive feasibility check. At 60–80 nodes this is exponential (100,000× slowdown). **Always keep OFF.**

### 4.3 3-layer path-completeness defense — NEW (2026-06-23)
TaSaR search can produce partially-assigned trajectories. Fixed at three layers:

1. **`vne_main.py:_result_has_complete_paths()`** — validates paths before writing pickle
2. **`vne/dataset.py:RandomVNEDataset.__init__`** — filters incomplete instances on load
3. **`vne/dataset.py:_chosen_path()`** — raises `ValueError` not `IndexError` for missing paths

### 4.4 Ray-free validation — `vne_main.py`
`validate()` and `test()` use `_evaluate_sequential()` — no Ray. Ray workers deadlock under MPS GPU co-location.

### 4.5 Env var overrides — `vne_main.py`
`VNE_LOAD_CHECKPOINT_FROM_PATH`, `VNE_SEARCH_TYPE`, `VNE_BEAM_WIDTH`, `VNE_REPLAN_STEPS`, `VNE_NUM_GENERATE`, etc.

---

## 5. How to Run Phase 2

### 5.1 Setup
```bash
cd step-and-reconsider
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**RTX 5060 note**: Blackwell (sm_120) needs PyTorch ≥2.6 + CUDA ≥12.6. The repo's `requirements.txt` targets CUDA 12.1. Override:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt  # skip torch if already installed
```

### 5.2 BQ-128 pilot (k=8, s=4, 10 epochs)
```bash
VNE_ARCHITECTURE=bq \
VNE_EMBEDDING_DIM=128 \
VNE_NUM_HEADS=8 \
VNE_FF_DIM=512 \
VNE_LEARNING_TYPE=gumbeldore \
VNE_SEARCH_TYPE=tasar \
VNE_NUM_EPOCHS=10 \
VNE_BEAM_WIDTH=8 \
VNE_REPLAN_STEPS=4 \
VNE_NUM_GENERATE=256 \
VNE_BATCH_SIZE=4 \
VNE_LR=2e-4 \
VNE_SEED=600 \
VNE_LOAD_CHECKPOINT_FROM_PATH=./model_checkpoints/vne/results/phase1_bq128_42476/last_model.pt \
VNE_RUN_TAG=phase2_bq128_k8_s4 \
python vne_main.py
```

### 5.3 LEHD-128 pilot (k=8, s=4, 10 epochs)
```bash
VNE_ARCHITECTURE=lehd \
VNE_EMBEDDING_DIM=128 \
VNE_NUM_HEADS=8 \
VNE_FF_DIM=512 \
VNE_NUM_ENCODER_LAYERS=6 \
VNE_NUM_DECODER_LAYERS=6 \
VNE_LEARNING_TYPE=gumbeldore \
VNE_SEARCH_TYPE=tasar \
VNE_NUM_EPOCHS=10 \
VNE_BEAM_WIDTH=8 \
VNE_REPLAN_STEPS=4 \
VNE_NUM_GENERATE=256 \
VNE_BATCH_SIZE=4 \
VNE_LR=2e-4 \
VNE_SEED=700 \
VNE_LOAD_CHECKPOINT_FROM_PATH=./model_checkpoints/vne/results/phase1_lehd128_42487/last_model.pt \
VNE_RUN_TAG=phase2_lehd128_k8_s4 \
python vne_main.py
```

### 5.4 Environment variables

| Variable | Purpose | Pilot value |
|---|---|---|
| `VNE_ARCHITECTURE` | `bq` or `lehd` | `bq` |
| `VNE_EMBEDDING_DIM` | Model dimension | `128` |
| `VNE_LEARNING_TYPE` | `supervised` or `gumbeldore` | `gumbeldore` |
| `VNE_SEARCH_TYPE` | `tasar` | `tasar` |
| `VNE_BEAM_WIDTH` | k (beam size) | `8` |
| `VNE_REPLAN_STEPS` | s (step size) | `4` |
| `VNE_NUM_EPOCHS` | Gumbeldore epochs | `10` |
| `VNE_NUM_GENERATE` | Instances per epoch | `256` |
| `VNE_BATCH_SIZE` | Training batch size | `4` (use `2` for 8GB VRAM) |
| `VNE_LR` | Learning rate | `2e-4` |
| `VNE_LOAD_CHECKPOINT_FROM_PATH` | Phase 1 model | See above |
| `VNE_RUN_TAG` | Results folder name | `phase2_bq128_k8_s4` |
| `VNE_NUM_ENCODER_LAYERS` | LEHD encoder depth | `6` |
| `VNE_NUM_DECODER_LAYERS` | LEHD decoder depth | `6` |

### 5.5 VRAM guidance for RTX 5060
- BQ-128 model: 1.8M params (~7 MB). Training base: ~27 MB.
- Main memory hog: attention over tokens (comm nodes + virtual nodes + candidates).
  - BQ: all tokens in one flat sequence → O(total²). With cap=500 candidates + 80 nodes + 20 virtuals ≈ 600 tokens. Attention ≈ 600² × 8 heads × 4 bytes ≈ 11.5 MB per layer, 9 layers ≈ 100 MB.
  - LEHD: separate encoder (substrate only) and decoder (context + virtuals + candidates). Decoder attention is over fewer tokens.
- **Recommendation**: start with `VNE_BATCH_SIZE=2`. If OOM, drop to `1`. For LEHD, `2` should be safe.
- Data generation (beam search) runs in inference mode (no gradients) — uses less VRAM than training.

---

## 6. What to Watch For

- **Data generation**: progress bar `0/256 → 256/256`. At k=8 with candidate cap, ~10–30s per instance. Total: ~1–2 hours for 256 instances.
- **"Skipped X infeasible, Y incomplete"**: printed after generation. A few are normal. >50% means the search is struggling (increase k or decrease substrate size).
- **Training loss**: should decrease across epochs.
- **Validation**: gap% should improve or hold steady. feas% should stay >50%.
- **Output**: `model_checkpoints/vne/results/<tag>/` — `last_model.pt`, `best_model.pt`, `log.txt`.

---

## 7. After Pilot Succeeds — Grid Search

Run for k ∈ {4,8,16} × s ∈ {2,4,8}. Record for each cell:

| Column | How to get it |
|---|---|
| Gap% | Test set evaluation (beam_width=1) |
| Feas% | From validation log |
| Time | `time/inst_ms` × 1024 instances ÷ 1000 = seconds |
| g(k,s) | k for BS, k×s for TaSaR (paper's compute-fair metric) |

Report format follows TaSaR paper Table 2.

---

## 8. Key Files

| File | Role |
|---|---|
| `vne/config.py` | `VNEConfig` — all hyperparameters, paths, ranges |
| `vne/trajectory.py` | VNE MDP: state, transitions, candidate enumeration, cost |
| `vne/dataset.py` | `RandomVNEDataset` — loads pickle, indexes decisions |
| `vne/bq_network.py` | BQ network: unified transformer over all tokens |
| `vne/network.py` | LEHD network: separate encoder/decoder |
| `vne/features.py` | `build_vne_state_input` — tensor dict from MDP state |
| `vne_main.py` | Glue: validation, test, data generation, env overrides |
| `core/train.py` | `main_train_cycle` — shared training loop |
| `core/incremental_sbs.py` | `perform_tasar` — the TaSaR search algorithm |
| `vne/PROBLEM_FORMULATION.md` | Formal problem definition |

---

## 9. Cluster Note

Job 42846 (BQ-128, k=8, s=4, 10 epochs) is queued on SLURM `gpu_HIGH` — PENDING until a GPU frees up. All nodes occupied by other users (no ETA). If you're reading this locally, ignore — just run it on your RTX 5060.
