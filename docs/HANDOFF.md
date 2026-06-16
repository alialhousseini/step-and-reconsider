# VNE Work — Session Handoff

**Read this first to continue the VNE work.** It captures the full state as of the
last session so a new chat can pick up seamlessly.

Repo now lives at **`/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider`**
(also `~/scratch/step-and-reconsider`). It was moved out of the shared
`/mnt/beegfs/scratch/ISIN` folder at the cluster owner's request. The sibling
hierarchical-VNE project moved to `~/scratch/XVNE_new`.

---

## 0. TL;DR — where we are and what's next

- **Goal:** the student's VNE supervised training plateaued and validation
  "dropped around epoch 6". Hypothesis was *too little data + the free CBC solver*.
- **What we did:** built a reproducible GPU env, a multi-solver label generator,
  a **50k** solved training set (+1k val, +2k test), **vectorized the network
  forward (4.4× faster, numerically identical)**, and ran a clean **1k-vs-50k A/B**.
- **Key finding (important):** more data did **not** fix the epoch-6 drop. The
  "drop" is **optimization instability / divergence**, not data scarcity. The 50k
  run's *training loss explodes at epoch 7–9* (1.3 → 8.97) and validation craters
  (−23 → −140) before noisily recovering; the 1k run is stable but plateaus at
  val ≈ −23. Root cause: `lr=1e-3` with **no gradient clipping, no LR decay** —
  more-diverse data just exposes it harder.
- **DONE (2026-06-13): gradient clipping fixed the divergence.** Set
  `optimizer["gradient_clipping"] = 1.0` (lr/decay unchanged, to isolate the effect)
  and re-ran the 1k-vs-50k A/B (jobs 40019 `clip1k`, 40020 `clip50k`). The 50k run's
  epoch 7–9 explosion is **gone**: train loss stayed in a tight 1.28–1.43 band for all
  15 epochs (vs 8.97 before), validation held ~−23 throughout (vs −140 before). See §4.
- **But "more data" did NOT win.** Once both are stable they plateau at the *same*
  objective — best val clip1k **−22.91** (e5) vs clip50k **−22.97** (e2); test obj
  clip1k **−25.594** vs clip50k **−25.631**. 50× the data bought no held-out gain.
  So **training stability was the real lever; data quantity is not** for this
  objective/architecture. This updates the prior expectation that 50k would beat 1k.
- **Next step:** the easy stability win is banked. To move the −23/−25 plateau itself
  needs a different lever, not more data — candidates: report/optimize the **optimality
  gap** vs the ILP labels the val/test pickles carry (§5 step 2), revisit the LEHD
  policy capacity/architecture, or LR schedule + longer training. See §5.

---

## 1. Environment & how to run (cluster-specific)

- **Python env:** `./.venv` — a **uv-managed standalone CPython 3.11** (lives under
  `~/.local/share/uv/python`, so it resolves on *all* cluster nodes; the system
  python is only 3.9 and differs between login/compute nodes). Pinned in
  `requirements-gpu.txt` (torch 2.5.1+cu121, ray, pulp, gurobipy, highspy, …).
  Always call `./.venv/bin/python` directly. To rebuild: `uv venv --python-preference
  only-managed --python 3.11 .venv` then `uv pip install -r requirements-gpu.txt`
  (torch needs `--index-url https://download.pytorch.org/whl/cu121`).
- **Cluster = SLURM**, no conda/modules for python/cuda. Submit via `sbatch`.
  - GPU partitions: `gpuISIN` (gnode05 A30, gnode06 2×L40S — **our group, least
    contended, prefer this**), `gpu`/`gpu_HIGH` (V100/A100, busier). CPU: `compute`.
  - **Be a good GPU citizen: share via MPS**, don't grab whole cards. Request
    `--gres=mps:N` (N = % of a GPU, e.g. 30–40) instead of `--gres=gpu:1`, and the
    training entrypoint caps torch memory via `VNE_GPU_MEM_FRACTION` (default 0.15).
    Compute nodes have outbound internet; a stronger GPU (A100) is a valid upgrade
    now that the forward is batched (it was overhead-bound before).
- **Scripts** (`scripts/`):
  - `vne_train.sbatch` — GPU training via MPS. Env knobs: `VNE_RUN_TAG`,
    `VNE_CUSTOM_NUM_INSTANCES` (subset train size for the A/B baseline),
    `VNE_NUM_EPOCHS`, `VNE_GPU_MEM_FRACTION`, `VNE_TRAINING_SET_PATH`,
    `VNE_RESULTS_PATH`. (`vne_main.py::_apply_env_overrides` reads these.)
  - `vne_gen_shard.py` + `vne_gen_array.sbatch` + `vne_merge_shards.py` —
    parallel dataset generation as a SLURM array (shard → merge → manifest +
    self-check). Solver-agnostic.
  - `vne_plot_training.py` — parse a run log into per-epoch train-loss vs
    val-objective CSV/PNG (`--log logs/vne_train_<tag>_<jobid>.log`).

Example A/B launch (both share one A30 via MPS):
```
sbatch --partition=gpuISIN --gres=mps:40 \
  --export=ALL,VNE_RUN_TAG=base1k,VNE_CUSTOM_NUM_INSTANCES=1000,VNE_GPU_MEM_FRACTION=0.25 \
  scripts/vne_train.sbatch
sbatch --partition=gpuISIN --gres=mps:40 \
  --export=ALL,VNE_RUN_TAG=scaled50k,VNE_GPU_MEM_FRACTION=0.25 \
  scripts/vne_train.sbatch
```

---

## 2. Datasets (generated this session, in `data/vne/`)

All in the **original embed-all + min-cost regime** (substrate 20–40 nodes,
2–6 requests; admission OFF), labels = exact ILP optima via **HiGHS**:

- `vne_supervised_training_dataset_50k.pickle` — 50,000 instances (~501k replay
  decisions). ~50× the prior 1,024-instance set.
- `vne_validation_dataset_1k.pickle` — 1,000.
- `vne_test_dataset_2k.pickle` — 2,000.
- Each has a `*.manifest.json` (counts, seeds, distributions). 0 duplicates,
  self-check passed. Disjoint seed bands (train 1e6+, val 2e6+, test 3e6+).
- `config.py` points `training_set_path` / `validation_set_path` / `test_set_path`
  at these. The old small pickles are untouched for reference.

---

## 3. What changed in the code (vs commit `594e733`)

- **`vne/network.py`** — `VNEPolicyNetwork.forward` is now **batched** (`_forward_batch`)
  with **candidate-count bucketing** (`candidate_token_budget`, default 8000) to
  cap padding memory. Numerically identical to the old per-example path
  (`_forward_one`, kept), ~4.4× faster, single host→device copy. The encoder/decoder
  have no positional encoding (pure set attention), which is *why* block-padding +
  masks is exact.
- **`vne/validation_set_generator.py`** — multi-solver layer
  (`_concrete_solver` / `_solve_with_fallback`, `solver ∈ {auto,highs,gurobi,cplex,cbc}`,
  HiGHS default), `solver_kwargs_from_config` (single source of truth),
  **keep-best-feasible-on-timeout** (accept incumbent → no timeout bias), and two
  objective modes: `lex` (max-acceptance → min-cost, big-M tight-W single MILP) and
  `profit` (revenue − cost). **Admission is OFF in the active config.**
- **`vne/dataset.py`** — replay is **rejection-safe** (`_accepted_for`,
  `_has_nested_paths` fix); skips rejected requests. (Matters only when admission is on.)
- **`vne/config.py`** — active regime restored to embed-all min-cost; GPU training
  defaults (`training_device="cuda"`, eval workers on GPU, beam-1 validation,
  `validation_custom_num_instances=128`, `custom_num_batches=("absolute",10000)`,
  `num_epochs=15`). Admission/solver knobs documented inline.
- **`vne_main.py`** — `_apply_env_overrides` (env-driven A/B), GPU memory cap.
- **`scripts/`, `requirements-gpu.txt`** — new (see §1).
- Nothing committed yet (`git status` shows the diff). Consider committing on a
  branch before the next experiments.

---

## 4. The A/B results in detail (`artifacts/`)

### 4a. Original un-clipped A/B (`artifacts/vne_base1k.csv|png`, `vne_scaled50k.csv|png`)

| epoch | base1k (1k data) train→val | scaled50k (50k data) train→val |
|---|---|---|
| 1–6 | 1.59→… , val stable ≈ −23 | 1.53→1.32 , val stable ≈ −23 |
| 7 | stable ≈ −23 | train **1.57**, val **−44** |
| 9 | stable ≈ −23 | train **8.97 💥**, val **−140** |
| 10–15 | stable ≈ −23 | noisy recovery to ≈ −23 |

Interpretation: divergence (not overfitting). Root cause `lr=1e-3` + no clipping.

### 4b. Clipped re-run, `gradient_clipping=1.0` (2026-06-13, `artifacts/vne_clip1k.csv|png`, `vne_clip50k.csv|png`)

Jobs 40019 (`clip1k`) / 40020 (`clip50k`), shared A30 via MPS, ~7.7 min/epoch.

| epoch | clip1k (1k) train→val | clip50k (50k) train→val |
|---|---|---|
| 1 | 1.66 → −23.49 | 1.58 → −23.49 |
| 5 | 1.34 → **−22.91** (best) | 1.32 → −23.23 |
| 7 | 1.28 → −23.10 | 1.29 → −23.15  *(was train 1.57/val −44)* |
| 9 | 1.29 → −23.30 | 1.30 → −23.56  *(was train 8.97/val −140)* |
| 11 | **2.66** → −22.95  *(own transient bump, clipping held val)* | 1.28 → −23.04 |
| 15 | 1.32 → −23.02 | 1.29 → −23.23 |
| **best val** | **−22.91** (e5) | **−22.97** (e2) |
| **test obj (2k)** | **−25.594** | **−25.631** |

Two findings:
1. **Clipping eliminated the divergence.** clip50k's train loss never left 1.28–1.43;
   the epoch 7–9 explosion (→8.97, val −140) is gone, val held ~−23 all 15 epochs.
2. **Data quantity is not the lever.** Both saturate at the *same* val (~−22.9) and
   test (~−25.6); 50× data gave no held-out gain. clip1k even had its own transient
   train-loss bump at e11 (→2.66) but clipping kept its val at −22.95. So *training
   stability* mattered; *more data* did not.

---

## 5. Recommended next steps

1. **DONE — stabilize training.** `optimizer["gradient_clipping"] = 1.0` is now set in
   `vne/config.py`; it eliminated the divergence (§4b). lr/decay left unchanged so the
   A/B isolated the clipping effect — `lr=3e-4` / `decay_factor=0.9` remain available if
   a future run wants to also damp the residual train-loss wobble.
2. **Report optimality gap (do this next — it's now the main lever).** The val/test
   pickles carry the ILP `objective`, so log `model_cost / ilp_cost` per epoch. Both
   runs plateau at the same mean obj (~−23 val / −25.6 test); the open question is
   *how far that is from the ILP optimum* and whether the gap (not the mean) separates
   1k vs 50k. This is the highest-value next experiment — more data is ruled out (§4b),
   so the plateau itself must be attacked (gap reporting, then policy capacity / LR
   schedule / longer training).
3. **Bigger/faster GPU** (A100) now that the forward is batched — straightforward.
4. **Further forward speedups** if needed: the candidate bucketing is the main
   lever; could also batch the candidate-feature build.

---

## 6. Parked: admission-control / contended-VNE (separate project)

A richer direction was explored then parked to keep the data-scaling experiment
clean. To resume: in `config.py` set the **contended ranges** (requests (8,16),
bandwidth/capacity (3,6), demands (2,4)), `enable_admission=True`,
`validation_objective="lex"` (max-acceptance→min-cost) or `"profit"`. Then
generation rejects ~0–44% of requests (packing-limited), which **requires making
`vne/trajectory.py` acceptance-aware** (a real reject/skip action — it currently
embeds every request and dead-ends on infeasible ones). `dataset.py` already
tolerates rejected requests. See the inline notes in `config.py`.

Gurobi note: the academic **named-user** license is node-locked to a workstation and
**won't run on the cluster**; use **HiGHS** (free, installed) or get an **academic
WLS** license. Pip-CPLEX is size-capped; full CPLEX needs the IBM installer.

---

## 7. How to resume in a fresh chat

1. `cd ~/scratch/step-and-reconsider` and start Claude Code there.
2. Say: *"Read docs/HANDOFF.md and continue the VNE work — start with §5 step 1
   (gradient clipping + re-run the A/B)."*
3. Sanity check the env: `./.venv/bin/python -c "import torch; print(torch.__version__)"`.
