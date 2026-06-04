# Working Rules And Reproduction Notes

Last updated: 2026-05-22

This note captures practical rules for continuing the project without losing
time to known repo conventions.

## First Checks In A New Session

1. Read `CLAUDE.md` and `.claude/agents/*.md`.
2. Read these Codex notes:
   - `docs/codex/REPO_ARCHITECTURE.md`
   - `docs/codex/VNE_TECHNICAL_MEMORY.md`
   - `docs/codex/WORKING_RULES_AND_REPRODUCTION.md`
3. Check `git status` before edits.
4. Inspect the target files directly before changing them.
5. For VNE changes, compare against `vne/PROBLEM_FORMULATION.md` and
   `vne/vne_README.md`.

Current repository caveat: `.git/` exists but is incomplete in this checkout;
`git status --short` reports "not a git repository". Do not rely on git diff
or git status until the repository metadata is restored.

## Environment Setup

Expected setup when Python is available:

```bash
pip install -r requirements.txt
```

The current Windows shell session does not have a usable `python` command. It
resolves to the Microsoft Store app alias. `py -3.12` also fails in this
sandbox. The local `.venv` exists, but `.venv/pyvenv.cfg` points to the same
Windows Store alias, so `.venv/Scripts/python.exe` cannot launch here.

Before running training, generation, pickle inspection, or import checks,
restore a real Python interpreter for the virtual environment or use the
provided Docker image.

Docker:

```bash
docker build -t step-and-reconsider .
```

The image is CUDA-oriented and starts from
`pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`.

## Reproduction Commands

Reproduce inherited greedy results:

```bash
python reproduce.py
```

Before running, adjust the top of `reproduce.py`:

- `CUDA_VISIBLE_DEVICES`
- `devices_for_eval_workers`
- `beams_with_batch_sizes`

`reproduce.py` covers TSP, CVRP, and JSSP only. It does not evaluate VNE.

Train individual problems:

```bash
python tsp_main.py
python cvrp_main.py
python jssp_main.py
python gomoku_main.py
python vne_main.py
```

Hyperparameters are edited in the matching `<problem>/config.py`; there are no
normal CLI flags for training.

## VNE Data Generation

The VNE validation generator is CLI-capable:

```bash
python -m vne.validation_set_generator --num-instances 64 --seed 1234 --out data/vne/vne_val.pickle --self-check
```

Problem size, topology, and demand/resource ranges are read from `VNEConfig`.
Change `vne/config.py` first, then run the generator.

`vne_main.py` now auto-ensures the canonical validation pickle before training
and auto-ensures the canonical supervised training pickle only in supervised
mode. `VNEConfig.validation_num_instances` controls the validation count, while
`VNEConfig.supervised_training_num_instances` controls the supervised training
count. Use separate generation seeds:
`VNEConfig.validation_generation_seed` for validation and
`VNEConfig.supervised_training_generation_seed` for supervised training.
Manual CLI generation is still useful for custom validation/test files.

Canonical VNE dataset paths:

- `data/vne/vne_validation_dataset.pickle`
- `data/vne/vne_supervised_training_dataset.pickle`
- `data/vne/vne_gumbeldore_training_dataset.pickle`
- `data/vne/vne_test_dataset.pickle` (optional, not auto-generated)

Legacy VNE datasets may still exist in older checkouts, but new reusable VNE
datasets should live under `data/vne/`.

## Experiment Hygiene

- Keep legacy one-request VNE as a small fast smoke test.
- Keep generated validation/supervised datasets aligned with the current
  `VNEConfig`.
- Keep generated datasets under `data/vne/` when they are reusable.
- Keep search-generated temporary training data separate from static validation
  and test pickles.
- Record seed, topology, chain length, demand ranges, objective settings, and
  solver settings in filenames or sidecar notes.
- Keep `data/README.md` in sync once VNE schemas stabilize; it currently
  documents only TSP, CVRP, and JSSP.

## Sign Conventions

- Trajectory objective returned to search: maximize.
- Routing/scheduling costs should be negated for `to_max_evaluation_fn`.
- Validation metric returned to `main_train_cycle`: minimize.
- For VNE toy code today: `objective = -cost`, and validation returns
  `-mean_obj`, effectively mean cost.

## Useful Project Agents

The `.claude/agents/` roster documents prior agent workflows:

- `vne-surveyor`: read-only per-scope VNE audit.
- `code-reviewer`: read-only diff or file-set review.
- `consistency-orchestrator`: coordinates VNE consistency audits.

In this Codex environment, use subagents only when explicitly allowed by the
user or when a future instruction grants that permission.

## Known Gaps To Respect

- No tests, no linter, no CI are configured.
- `gomoku/` references missing `gomoku.env` files in this checkout.
- `core/abstracts.py` now includes `VNEConfig`, but the shared core still
  imports problem configs directly; adding more problem classes will require
  touching core typing/import surfaces.
- `validation_every_n_epochs` is not honored by the shared training loop.
- Python `random` is not seeded by `main_train_cycle`.
- `requirements.txt` is unpinned, so environment drift is likely.
- VNE has no pretrained checkpoint.
- `vne/trajectory.py::Trajectory` is duck-typed but does not subclass
  `BaseTrajectory`.
- `VNEConfig.batch_size_training` is the VNE replay dataloader minibatch size.
  `vne_main.py::collate_vne_batch` keeps variable-length candidate sets as a
  list and the training loop averages cross-entropy across examples.
- `VNEConfig.custom_num_batches = None` trains over every VNE replay decision.
  If a capped value is used, `RandomVNEDataset` samples across the full decision
  pool rather than taking a deterministic prefix.
- `CLAUDE.md` is stale about VNE: the tree now has VNE code and data, not docs
  only.

## Latest Local VNE Result

`results/2026-05-22--11-43-42/` is a toy VNE CPU run:

- 10 epochs
- TaSaR search, beam width 8, `replan_steps = 1`
- generated 256 toy instances per epoch
- validation beam-1 objective stayed at `-2.625`
- test beam-1 objective was `-2.4140625`

Use it as a smoke-test reference only.

## Preferred Validation Style For New VNE Work

When Python is available, use layered smoke checks:

1. Import check for modified modules.
2. Generate a tiny toy dataset with `--self-check`.
3. Run a tiny VNE evaluation with `num_epochs = 0` or a small training run.
4. Generate a tiny chain dataset and verify candidate replay.
5. Compare one or more chain objectives against the ILP solution.

If a change touches `core/`, also run at least one inherited problem in a tiny
configuration or reproduction eval, because all problem packages share that
surface.
