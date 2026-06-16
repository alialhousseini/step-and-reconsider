# VNE+TaSaR — Colleague Quick-Start Guide

> **Audience:** The colleague running heuristics & metaheuristics baselines.
> **Last updated:** 2026-06-16

---

## 0. What This Repo Is

Implementation of "Take a Step and Reconsider" (TaSaR, ECAI-2024) applied to **Virtual Network Embedding (VNE)**. Forked from [gumbeldore](https://github.com/grimmlab/gumbeldore).

Full problem formulation: [`vne/PROBLEM_FORMULATION.md`](vne/PROBLEM_FORMULATION.md)

---

## 1. Recent Code Changes (June 2026)

### New Architecture: BQ (`vne/bq_network.py`)

We now have **two** policy network architectures:

| Architecture | File | Style | Params (dim=128) |
|-------------|------|-------|-------------------|
| **LEHD** | `vne/network.py` | Encoder-decoder (graph attention + cross-attention) | ~2.4M |
| **BQ** | `vne/bq_network.py` | Single unified transformer (all tokens in one stack) | ~1.8M |

Switch via `config.architecture = "bq"` or `"lehd"` in `vne/config.py` (or env var `VNE_ARCHITECTURE`).

Both accept the same state input format (`build_vne_state_input` in `vne/features.py`) and return `List[torch.Tensor]` of logits — they are drop-in replacements for each other.

### Batched Forward Pass (`vne/network.py`)

The LEHD network now has a vectorized `_forward_batch` method — processes multiple states in one GPU pass. Same logical output, ~4-8× faster than the per-instance forward.

### Config Updates (`vne/config.py`)

Key changes:
- `architecture = "lehd"` (new, switch to `"bq"` for BQ)
- `use_rezero_transformer = True` (was False, now paper-standard)
- `num_encoder_layers = 6`, `num_decoder_layers = 6` (symmetric, was 1+9)
- `num_transformer_blocks = 9` (new, for BQ architecture)
- Parameter ranges: `num_substrate_comm_nodes_range = (60, 80)`, `num_virtual_requests_range = (10, 20)` (scaled up from 20-40/2-6)
- `enable_admission = False` (embed-all regime)

### Evaluation Improvements (`vne_main.py`)

- `get_network()` is architecture-aware (reads `config.architecture`)
- Validation now logs: gap% to ILP optimum, feasibility%, per-instance inference time
- Wall-clock timing tracked in training functions
- New env vars: `VNE_ARCHITECTURE`, `VNE_VALIDATION_SET_PATH`, `VNE_TEST_SET_PATH`, `VNE_NUM_TRANSFORMER_BLOCKS`

### Solver Fix (`vne/validation_set_generator.py`)

Fixed `_concrete_solver` to not pass `None` values to Gurobi's `setParam` — enables local Gurobi dataset generation.

### New Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `vne_gen_local.py` | Generate ILP-labeled datasets locally (needs Gurobi license) |
| `vne_gen_datasets.sh` | Submit SLURM array jobs for cluster dataset generation |
| `vne_phase1_launch.sh` | Launch all 5 architecture baselines in parallel |
| `vne_train.sbatch` | Single GPU training run (MPS sharing) |
| `vne_gen_array.sbatch` | SLURM array for shard generation |
| `vne_gen_shard.py` | One shard of dataset instances |
| `vne_merge_shards.py` | Merge shards into final pickle + self-check |
| `vne_gap_analysis.py` | Analyze optimality gaps from training logs |
| `vne_grid_submit.sh` | Submit k×s grid search |
| `vne_grid_results.py` | Aggregate grid search results |
| `vne_plot_training.py` | Plot training curves |

---

## 2. Experimental Plan (Summary)

Full plan: [`docs/paper/VNE_EXPERIMENTAL_PLAN.md`](docs/paper/VNE_EXPERIMENTAL_PLAN.md)

### NOW (being executed)

**Phase 1 — Supervised Architecture Baselines:**
- 5 architectures: BQ-128, BQ-192, LEHD-128, LEHD-192, LEHD-256
- 10k ILP-labeled instances, 30 epochs each
- Train at 60-80n substrates, 10-20 requests, line topology
- Goal: establish which architecture learns best from expert labels

**Phase 2 — TaSaR k×s Grid:**
- 12 runs: k ∈ {16, 32, 64} × s ∈ {2, 4, 8, 16}
- 15 Gumbeldore epochs each
- Goal: find best search parameters for self-improved learning

**Phase 3 — Multi-beam Evaluation:**
- Beam ∈ {1, 4, 8, 16, 32, 64} on best models

**Phase 4 — Generalization:**
- Test at 2× (120-160n), 3× (180-240n), 4-5× (240-400n)

### SKIP / LATER (journal extensions)

| Item | Reason |
|------|--------|
| Top-p threshold ablation | Inference-only, not essential for core story |
| Admission control | Different problem → follow-up paper |
| Topology generalization (ER, mesh) | Novel but adds risk; test feasibility first |
| RL baselines (REINFORCE) | New training module needed |
| Greedy heuristic baselines | Quick but not needed for core comparison |

---

## 3. For Your Baselines

### The Problem

VNE with **static, embed-all, min-cost** objective:
- Substrate: directed line graph, 60-80 comm nodes, each with a comp node attached
- Virtual requests: 10-20 chain-style VNRs, each 2-5 processing nodes
- BW per edge: 4-12, CAP per comp link: 3-10
- Demands: 1-4 per link
- Objective: minimize total routing cost (processing path length × demand)

### Input/Output Format

Instances are Python dicts in pickles. See `vne/PROBLEM_FORMULATION.md` for the full schema. Quick reference:

```python
instance = {
    "substrate": {
        "num_comm_nodes": 70,
        "communication_edges": [(0,1), (1,2), ...],
        "communication_bandwidth": {(0,1): 7, ...},
        "compute_attachment": {0: 0, 1: 1, ...},
        "compute_capacity": {0: 5, 1: 8, ...},
    },
    "requests": [
        {
            "num_processing_nodes": 3,       # k (including F_0 and F_{k-1})
            "source_link_demand": 2,          # dem(S->F_0)
            "destination_link_demand": 3,     # dem(F_{k-1}->D)
            "processing_link_demands": [4, 3],  # dem(F_i->F_{i+1}) for each link
        },
        # ... more requests
    ],
    # ILP solution (if labeled):
    "f_placements": [[c0, c1, c2], ...],     # comm node for each F_i, per request
    "processing_paths": [[(c0,c3,c5), ...], ...],  # comm path for each link, per request
    "objective": -47.0,                       # ILP optimum (negative cost)
    "accepted": [True, True, ...],            # per-request admission
}
```

### Existing Datasets

- **`data/vne/vne_supervised_training_dataset_50k.pickle`** — 50k ILP-labeled instances at 20-40n, 2-6rq (use this NOW)
- **`data/vne/vne_validation_dataset_1k.pickle`** — 1k instances
- **`data/vne/vne_test_dataset_2k.pickle`** — 2k instances
- New 10k dataset at 60-80n, 10-20rq: in progress (generating via local Gurobi)

**Use the 50k dataset for now.** The new 10k dataset will replace it when ready.

### How to Run Training

```bash
# Supervised learning with LEHD-128 (default):
VNE_LEARNING_TYPE=supervised VNE_GPU_MEM_FRACTION=0.30 sbatch --partition=gpuISIN --gres=mps:30 scripts/vne_train.sbatch

# With BQ architecture:
VNE_ARCHITECTURE=bq VNE_LEARNING_TYPE=supervised VNE_GPU_MEM_FRACTION=0.30 sbatch --partition=gpuISIN --gres=mps:30 scripts/vne_train.sbatch

# Custom config overrides (see _apply_env_overrides in vne_main.py):
VNE_EMBEDDING_DIM=192 VNE_NUM_HEADS=12 VNE_FF_DIM=768 ...
```

### How to Evaluate

Load a checkpoint and run evaluation:
```python
from vne.config import VNEConfig
from vne_main import get_network, evaluate
config = VNEConfig()
config.test_set_path = "data/vne/vne_test_dataset_2k.pickle"
network = get_network(config, torch.device("cuda"))
network.load_state_dict(torch.load("path/to/best_model.pt")["model_weights"])
_, results = evaluate("Test", config, network, config.test_set_path)
# results includes: mean_obj, mean_gap_pct, feasibility_pct, time/inst_ms
```

### For Heuristic/Metaheuristic Baselines

Implement your solver as a function that takes an `instance` dict and returns:
```python
{
    "f_placements": [[c0, c1, ...], ...],      # per request
    "processing_paths": [[(c0,...,c3), ...], ...],  # per request
    "objective": float,                          # negative cost
}
```

Use `vne.validation_set_generator.run_self_check(dataset)` to verify constraint satisfaction.

---

## 4. Key Decisions Made

1. **Static, embed-all, no admission** — Clean paper analog to TSP/CVRP
2. **Both BQ and LEHD** — Replicating paper's architecture comparison
3. **Line topology only** (for now) — ER/mesh parked for journal
4. **k×s grid on best architecture only** — Not on all 5 architectures
5. **Generalization = size scaling only** — Same as paper (TSP N=100→200→500)
6. **Wall-clock = g(k,s) fairness + systematic timing** — More rigorous than paper
7. **RL/heuristics = journal extensions** — Core paper is SIL vs SL comparison

---

## 5. Quick Links

| Document | Content |
|----------|---------|
| [`vne/PROBLEM_FORMULATION.md`](vne/PROBLEM_FORMULATION.md) | VNE problem definition |
| [`docs/paper/VNE_EXPERIMENTAL_PLAN.md`](docs/paper/VNE_EXPERIMENTAL_PLAN.md) | Full experimental protocol |
| [`docs/paper/VNE_BASELINE_MATRIX.md`](docs/paper/VNE_BASELINE_MATRIX.md) | All baselines mapped |
| [`docs/paper/VNE_GENERALIZATION_AXES.md`](docs/paper/VNE_GENERALIZATION_AXES.md) | Generalization design |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Session state + cluster how-to |
| [`CLAUDE.md`](CLAUDE.md) | Repo architecture overview |
