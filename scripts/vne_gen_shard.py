"""Generate one shard of solved VNE instances (for SLURM array data generation).

Each array task produces an independent, deterministic shard by deriving its RNG
seed from a base offset plus the SLURM array task id, then writes a pickle that
the merge step concatenates. Reuses the existing generator/solver verbatim
(`vne.validation_set_generator.make_validation_dataset` + `run_self_check`), so
labels are exactly the ILP optima (Gurobi-preferred, CBC fallback).

Usage (env or flags; flags win):
    python scripts/vne_gen_shard.py \
        --split train --num-per-shard 1000 --seed-base 1000000 \
        --task-id $SLURM_ARRAY_TASK_ID --solver auto --threads 4 \
        --time-limit 60 --out-dir data/vne/shards
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vne.config import VNEConfig
from vne.validation_set_generator import make_validation_dataset, run_self_check, solver_kwargs_from_config


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else default


def main() -> None:
    config = VNEConfig()
    p = argparse.ArgumentParser(description="Generate one VNE data shard")
    p.add_argument("--split", default=os.environ.get("VNE_SPLIT", "train"),
                   help="Logical split name, used only in the output filename.")
    p.add_argument("--num-per-shard", type=int,
                   default=_env_int("VNE_NUM_PER_SHARD", 1000))
    p.add_argument("--seed-base", type=int,
                   default=_env_int("VNE_SEED_BASE", 1_000_000),
                   help="Shard seed = seed_base + task_id. Keep splits in disjoint bands.")
    p.add_argument("--task-id", type=int,
                   default=_env_int("SLURM_ARRAY_TASK_ID", 0))
    p.add_argument("--solver", choices=["auto", "highs", "gurobi", "cplex", "cbc"],
                   default=os.environ.get("VNE_SOLVER", getattr(config, "validation_solver", "highs")))
    p.add_argument("--threads", type=int,
                   default=_env_int("VNE_THREADS", _env_int("SLURM_CPUS_PER_TASK", 0)))
    p.add_argument("--time-limit", type=int,
                   default=_env_int("VNE_TIME_LIMIT", config.validation_solver_time_limit))
    p.add_argument("--mip-gap", type=float,
                   default=float(os.environ.get("VNE_MIP_GAP", getattr(config, "validation_solver_mip_gap", 0.0))))
    p.add_argument("--out-dir", default=os.environ.get("VNE_OUT_DIR", "data/vne/shards"))
    args = p.parse_args()

    seed = args.seed_base + args.task_id
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}_shard{args.task_id:05d}_seed{seed}.pickle")

    print(f">> shard {args.task_id}: split={args.split} n={args.num_per_shard} seed={seed} "
          f"solver={args.solver} threads={args.threads} tl={args.time_limit}s -> {out_path}",
          flush=True)
    t0 = time.time()
    # Admission/revenue/cost semantics come from the config; per-task knobs
    # (backend, threads, time limit, gap) override.
    solver_kwargs = solver_kwargs_from_config(
        config,
        solver=args.solver,
        threads=args.threads,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
    )
    dataset = make_validation_dataset(
        args.num_per_shard,
        config,
        with_solutions=True,
        solver_kwargs=solver_kwargs,
        seed=seed,
        progress=False,
    )
    run_self_check(dataset)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)
    dt = time.time() - t0
    print(f">> shard {args.task_id}: wrote {len(dataset)} instances in {dt:.1f}s "
          f"({dt / max(len(dataset),1):.2f}s/inst)", flush=True)


if __name__ == "__main__":
    main()
