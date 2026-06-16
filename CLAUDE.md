# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **⚡ ACTIVE VNE WORK — read [`docs/HANDOFF.md`](docs/HANDOFF.md) first.** It has the
> current state, environment/cluster how-to (uv venv, SLURM, MPS GPU sharing,
> HiGHS solver), the generated 50k/1k/2k datasets, the vectorized network, the
> A/B result (the "epoch-6 drop" is training *divergence*, not data scarcity),
> and the next step (enable gradient clipping + re-run). Repo now lives at
> `~/scratch/step-and-reconsider` (moved out of the shared `ISIN` folder).

## What this repo is

Implementation of "Take a Step and Reconsider" (TaSaR), an ECAI-2024 sequence-decoding method for self-improved neural combinatorial optimization (paper: 10.3233/FAIA240707). It is a fork of [gumbeldore](https://github.com/grimmlab/gumbeldore); the substantive new code lives in `core/incremental_sbs.py::IncrementalSBS.perform_tasar`. Everything else is gumbeldore scaffolding that was inherited and is mostly unchanged.

Supported problem classes: TSP, CVRP, JSSP, Gomoku, and **VNE (Virtual Network Embedding)** — the VNE extension under `vne/` is now **fully implemented** (config, instance generator, ILP label generator, trajectory, LEHD-style policy network with a batched forward, dataset/replay, and `vne_main.py`). See `docs/HANDOFF.md` and `vne/PROBLEM_FORMULATION.md`.

## Commands

Dependencies (Python ≥3.10 expected, given Docker base of `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`):
```
pip install -r requirements.txt
```

Reproduce paper greedy results (loads checkpoints from `model_checkpoints/`, runs evaluation on data in `data/`):
```
python reproduce.py
```
Before running, edit the top of `reproduce.py`: `CUDA_VISIBLE_DEVICES`, `devices_for_eval_workers` (one entry per parallel worker — `"cpu"` or `"cuda:N"`), and `beams_with_batch_sizes`. Defaults are 4 CPU workers, batch size 4 — slow but works without a GPU.

Train a single problem (each `*_main.py` is the entrypoint; settings come from the matching `*/config.py`):
```
python tsp_main.py
python cvrp_main.py
python jssp_main.py
python gomoku_main.py
```
There is no test framework, linter, or CI configured. To change hyperparameters, edit the `<Problem>Config` class — there are no CLI flags.

Docker build (CUDA 12.1):
```
docker build -t step-and-reconsider .
```

## Architecture

### Layering

- `core/` — problem-agnostic search and training:
  - `stochastic_beam_search.py` — base SBS over a trajectory tree.
  - `incremental_sbs.py` — `IncrementalSBS` class. Holds two search loops:
    - `perform_tasar(...)` — the contribution of this repo; commit to the best solution for `replan_steps` actions, then sample alternatives without replacement.
    - `perform_incremental_sbs(...)` — original gumbeldore method, kept for comparison.
  - `beam_search.py` — deterministic beam search.
  - `gumbeldore_dataset.py` — Ray-based parallel data generation. `GumbeldoreDataset` distributes problem instances across workers (one per entry in `devices_for_workers`), each worker runs SBS/TaSaR, results are pickled and reused as the next epoch's training data.
  - `train.py::main_train_cycle` — the single training loop used by every problem. Supports `learning_type = "supervised"` or `"gumbeldore"`; the latter alternates dataset generation and one training epoch.
  - `abstracts.py` — `BaseTrajectory` (the interface every problem must implement: `init_batch_from_instance_list`, `log_probability_fn`, `transition_fn`, `to_max_evaluation_fn`, `num_actions`) and the `Config` type alias.

- Per-problem packages (`tsp/`, `cvrp/`, `jssp/`, `gomoku/`) each contain:
  - `config.py` — single `<Problem>Config` class; this is the only place to tune hyperparameters.
  - `trajectory.py` — `BaseTrajectory` subclass: the problem's MDP, state, action mask, and objective.
  - `dataset.py` — torch `Dataset` that yields training batches from pickled solutions.
  - `network.py` / `bq_network.py` / `lehd_network.py` — policy networks (TSP and CVRP each ship both BQ and LEHD architectures; pick via `config.architecture`).

- `*_main.py` (root) — thin glue per problem. Each wires together the problem-specific `get_network`, `train_for_one_epoch_*`, `validate`, `test`, dataset, and trajectory class, and calls `core.train.main_train_cycle`. They are nearly identical — when changing the training loop, expect to mirror edits across all four.

- `modules/` — shared NN building blocks (`rztx.py` for the ReZero transformer used by BQ networks; `alphazero/` for the Gomoku policy network).

- `logger.py` — file + optional MLflow logging.

### The gumbeldore_config dict

All search behavior is driven by `config.gumbeldore_config` (dict on every `<Problem>Config`). The key switch is `search_type`:
- `"tasar"` — uses `perform_tasar`. Relevant params: `beam_width`, `replan_steps` (step size `s`), `min_nucleus_top_p` (constant Top-p), `perform_first_round_deterministic`. Most other params (`num_rounds`, `advantage_constant`, `min_max_normalize_advantage`, `expected_value_use_simple_mean`, `use_pure_outcomes`, `normalize_advantage_by_visit_count`) are **ignored**.
- `"wor"`, `"gumbeldore"`, `"theory_gumbeldore"` — original gumbeldore variants using `perform_incremental_sbs`.
- `"beam_search"` — deterministic beam search, set internally by the evaluator for non-Gumbeldore eval.

See the README for the full parameter description.

### Data and checkpoints

- `data/` — pickled instances per problem class. Format documented in `data/README.md` (instance dicts with `inst`/`tour`/`sol` for TSP, `nodes`/`demands`/`tour_*` for CVRP, `proc_times`/`ops_machines`/`job_seq` for JSSP). The supervised training pickle shipped under `data/tsp/` is a 10k-instance sample; the paper's full 1M-instance training set is not in the repo.
- `model_checkpoints/{tsp,cvrp,jssp}/...` — pretrained weights `reproduce.py` loads. JSSP has a single checkpoint; TSP and CVRP have separate `bq/` and `lehd/` subdirectories.
- `results/<timestamp>/` — created per training run; contains `last_model.pt` and `best_model.pt`.

## Conventions to know

- Ray is initialized in `main_train_cycle` and `reproduce.py` with `num_gpus` derived from the distinct non-CPU entries in `devices_for_*_workers`. `CUDA_VISIBLE_DEVICES` is set on the config object because Ray has trouble auto-detecting multiple GPUs.
- `core/incremental_sbs.py` calls `sys.setrecursionlimit(10000)` because policy updates recurse over the trie.
- Adding a new problem means: a `BaseTrajectory` subclass, a `<Problem>Config`, a dataset, a policy network, and a `<problem>_main.py` that wires them into `main_train_cycle`. The four existing `*_main.py` files are the template.
- MLflow logging is off by default (`log_to_mlflow = False`); the placeholder credentials in each `config.py` are not real and should not be committed if filled in.
- Trajectories' `to_max_evaluation_fn` returns an objective to **maximize** (e.g., negative tour length for routing problems).

## Project subagents (.claude/agents/)

Three project-scoped Claude Code subagents are committed at `.claude/agents/`:

- **`vne-surveyor`** — read-only per-scope auditor (Sonnet). Dispatched by the orchestrator; rarely invoked directly.
- **`code-reviewer`** — reviews a diff or file set against repo conventions (Sonnet). Use when you have changes ready for a second-pair-of-eyes pass.
- **`consistency-orchestrator`** — coordinator that dispatches `vne-surveyor` instances in parallel and synthesizes a consolidated audit (Opus). Use for broad consistency checks, especially `vne/` vs `vne/PROBLEM_FORMULATION.md` vs the `tsp/cvrp/jssp` reference baselines.

Design and rationale: `docs/superpowers/specs/2026-05-18-vne-agent-orchestration-design.md`.

