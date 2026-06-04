# Repository Architecture Memory

Last updated: 2026-05-22

This note is a durable map for future Codex work in this repository. It inherits
the existing `.claude/` guidance, then corrects it against the current
filesystem.

## Mission

The repository implements "Take a Step and Reconsider" (TaSaR) for neural
combinatorial optimization. It is built on a Gumbeldore-style framework where
problem-specific code exposes a trajectory/search interface and the shared
`core/` package handles stochastic beam search, data generation, and training.

Supported inherited problem families are TSP, CVRP, JSSP, and experimental
Gomoku. The strategic extension point is `vne/`, which adapts the framework to
Virtual Network Embedding.

## Directory Map

- `.claude/`: project agent definitions and guidance from previous Claude Code
  work. These are useful context, but stale references are possible.
- `core/`: problem-agnostic search, generation, and training.
- `tsp/`, `cvrp/`, `jssp/`, `gomoku/`: reference problem packages.
- `modules/`: shared neural network building blocks, mainly ReZero
  Transformer layers and a LightZero-style AlphaZero stack.
- `data/`: pickled datasets and generated training artifacts.
- `model_checkpoints/`: pretrained checkpoints for TSP, CVRP, and JSSP.
- `vne/`: VNE formulation docs, toy/general data generation, trajectory,
  dataset, and candidate scorer.
- root `*_main.py`: per-problem adapters into `core.train.main_train_cycle`.
- `reproduce.py`: reproduction/evaluation script for TSP, CVRP, and JSSP.

## Core Contract

Every searchable problem is represented by a `BaseTrajectory` subclass from
`core/abstracts.py`.

Required methods:

- `init_batch_from_instance_list(instances, network, device)`: convert raw
  instances into root trajectories. Existing problems may pre-encode static
  tensors with the network here.
- `log_probability_fn(trajectories, network, to_numpy)`: return next-action
  log probabilities. Generic SBS expects a list of NumPy arrays; optimized beam
  search expects a rectangular torch tensor when `to_numpy=False`.
- `transition_fn(action)`: return `(new_trajectory, is_finished)` without
  mutating the original trajectory.
- `to_max_evaluation_fn()`: terminal objective to maximize. Minimization
  problems return negative cost.
- `num_actions()`: number of currently feasible actions.

Important sign convention: search maximizes `to_max_evaluation_fn`, while
validation metrics returned to `main_train_cycle` are minimized.

## Training Flow

The lifecycle is centralized in `core/train.py::main_train_cycle`.

1. Config object is created by the root entrypoint.
2. Ray is initialized from distinct non-CPU eval devices.
3. Logger, NumPy seed, and torch seed are set.
4. Network is built with `get_network(config, device)`.
5. Optional checkpoint state is loaded.
6. For each epoch:
   - supervised mode reuses one dataloader;
   - gumbeldore mode generates a search dataset, then trains on it;
   - metrics are logged;
   - validation runs;
   - `last_model.pt` is saved;
   - `best_model.pt` is updated when the validation metric decreases.
7. If `test_set_path` is present, the best model is loaded and tested.
8. Ray is shut down.

`validation_every_n_epochs` exists in configs but is not currently used by the
loop; validation happens every epoch.

## Search And Generation

`core/gumbeldore_dataset.py::GumbeldoreDataset` is the shared generation and
evaluation harness. Each problem supplies:

- `generate_instances`
- `trajectory_cls`
- `get_network`
- `beam_leaves_to_result`
- `process_search_results`

Generation creates a Ray job pool, starts one worker per configured device,
turns raw instances into root trajectories, runs search, converts leaves into
problem-specific solution records, and writes or summarizes the result.

Search modes:

- `beam_search`: deterministic search. TSP/CVRP can use optimized rectangular
  beam search; JSSP/Gomoku/VNE use generic deterministic SBS because their
  action spaces or depths can vary. VNE is explicitly hard-coded into this
  generic path in `core/gumbeldore_dataset.py`.
- `tasar`: calls `IncrementalSBS.perform_tasar`, which samples alternatives,
  follows the best current leaf for `replan_steps`, and returns one best leaf
  per root.
- `wor`, `gumbeldore`, `theory_gumbeldore`: original incremental SBS variants
  using rounds and optional advantage updates.

Under `search_type = "tasar"`, the meaningful search settings are
`beam_width`, `replan_steps`, `min_nucleus_top_p`, and
`perform_first_round_deterministic`. Several older Gumbeldore parameters remain
in configs but are not read by `perform_tasar`.

## Reference Problem Patterns

All mature problem packages have:

- `<problem>/config.py`
- `<problem>/trajectory.py`
- `<problem>/dataset.py`
- one or more network modules
- root `<problem>_main.py`

Main scripts usually define:

- `get_network`
- `generate_instances`
- `beam_leaves_to_result`
- `save_search_results_to_dataset`
- `evaluate`
- `validate`
- `test`
- `get_gumbeldore_dataloader`
- `get_supervised_dataloader`
- `train_with_dataloader`
- one epoch function for supervised and/or gumbeldore training

Datasets usually return an entire minibatch from `Dataset.__getitem__`; the
outer `DataLoader` uses `batch_size=1`, and the training loop removes the extra
dimension. VNE is different: `RandomVNEDataset` returns one replay example, and
`vne_main.py::collate_vne_batch` keeps variable-length structured state dicts
in a list so `VNEConfig.batch_size_training` acts as the true minibatch size.

### TSP

- Instances: `inst` coordinates, optional `tour` and `sol`.
- Action: choose next city from remaining nodes.
- Objective: minimize tour length, exposed as negative length to search.
- Networks: BQ and LEHD.

### CVRP

- Instances: depot/customer coordinates, demands, capacity, route flags.
- Action: choose customer plus direct/via-depot flag.
- Objective: minimize route length, exposed as negative length to search.
- Networks: BQ and LEHD.

### JSSP

- Instances: processing times, operation machines, machine attention mask,
  job sequence, optional makespan.
- Action: choose which job's next operation to schedule.
- Objective: minimize makespan, exposed as negative makespan to search.
- Network: JobTransformer.

### Gomoku

- Experimental. Uses AlphaZero-style policy/value model.
- The checked-out tree lacks the referenced `gomoku.env` package, so this
  package is currently not a reliable reproduction baseline.

### VNE

- Current runtime is a static multi-request chain VNE scaffold.
- Root adapter: `vne_main.py`, following the same callback pattern as
  `tsp_main.py`, `cvrp_main.py`, and `jssp_main.py`. It auto-ensures missing
  validation and supervised-training pickles before training.
- Instance generator: `vne/instance_generator.py`, config-driven line
  substrate size, request count, and resource/demand ranges.
- Trajectory: `vne/trajectory.py`, one action chooses a complete substrate path
  for the current `(request_idx, processing_link_idx)` decision and terminates
  after all links of all requests are embedded.
- Dataset replay: `vne/dataset.py`, one solved instance becomes one
  next-action example per request/link decision.
- Network: `vne/network.py`, an LEHD-style policy over structured state
  features from `vne/features.py`.
- Validation generator: `vne/validation_set_generator.py`, a broader PuLP ILP
  generator that can create solved chain instances beyond what the runtime
  currently consumes.

## Data And Checkpoints

`data/README.md` documents TSP, CVRP, and JSSP schemas. It does not yet
document VNE. VNE schema is currently documented by `vne/` code and markdowns.

Available checkpoint layout:

- `model_checkpoints/tsp/{bq,lehd}/checkpoint.pt`
- `model_checkpoints/cvrp/{bq,lehd}/checkpoint.pt`
- `model_checkpoints/jssp/checkpoint.pt`

There is no pretrained VNE checkpoint under `model_checkpoints/`. A local toy
VNE run exists under `results/2026-05-22--11-43-42/`.

Canonical VNE dataset paths:

- `data/vne/vne_validation_dataset.pickle`
- `data/vne/vne_supervised_training_dataset.pickle`
- `data/vne/vne_gumbeldore_training_dataset.pickle`
- `data/vne/vne_test_dataset.pickle` (optional, not auto-generated)

Older VNE toy/general pickles may exist in previous checkouts. The active VNE
entrypoint now auto-creates missing validation and supervised training pickles
from `vne/validation_set_generator.py` when needed.

Expected checkpoint keys at call sites:

- `model_weights`
- `best_model_weights`
- `optimizer_state`
- `epochs_trained`
- `validation_opt_gap`

## Runtime Notes

`requirements.txt` is unpinned and lists `torch`, `numpy`, `ray`, `tqdm`,
`mlflow`, `boto3`, `easydict`, and `pulp`.

The Dockerfile uses `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`, installs
`tmux`, and installs requirements.

Python is not available as `python` in the current Windows shell session. The
command resolves to the Microsoft Store app alias. The local `.venv` exists but
its `pyvenv.cfg` also points at the Windows Store alias, so `.venv/Scripts/python.exe`
cannot launch in this session.

Git metadata is also incomplete: `.git/` exists, but `git status` reports that
the directory is not a Git repository. The `.git` directory lacks normal files
such as `HEAD` and `config`.

## Cross-Cutting Traps

- `transition_fn` must copy mutable state. Search caches and revisits branches.
- Candidate ordering must match across trajectory search, dataset replay, and
  target-index construction.
- Empty feasible action sets should be prevented or explicitly handled.
- Networks must expose `.device`, `get_weights()`, and normal
  `load_state_dict` compatibility for Ray workers.
- Python `random` is not seeded in `main_train_cycle`; code that uses
  `random` directly needs separate seeding for exact reproducibility.
- `vne/trajectory.py::Trajectory` implements the `BaseTrajectory` method
  surface but does not explicitly subclass `BaseTrajectory`.
- `validation_every_n_epochs` is present in configs but not honored by
  `main_train_cycle`; validation runs every epoch.
- `gomoku/` is not a dependable template for VNE because it references missing
  environment code and diverges from the routing/scheduling packages.
