#!/usr/bin/env python3
"""
VNE Dataset Generator — Local Gurobi Edition
============================================
Generates ILP-labeled VNE datasets for the TaSaR+ VNE paper using a local
Gurobi installation with a full (non-restricted) license.

Target scale: 60-80 substrate communication nodes, 10-20 virtual requests.

PREREQUISITES (install on your local machine):
    pip install pulp gurobipy tqdm

USAGE (run from the repo root):
    # Generate all 3 datasets (train + val + test):
    python scripts/vne_gen_local.py --all

    # Or generate one split at a time:
    python scripts/vne_gen_local.py --split train --num-instances 10000 --out data/vne/vne_supervised_training_dataset_10k.pickle
    python scripts/vne_gen_local.py --split val   --num-instances 1000  --out data/vne/vne_validation_dataset_1k.pickle
    python scripts/vne_gen_local.py --split test  --num-instances 2000  --out data/vne/vne_test_dataset_2k.pickle

    # Resume from checkpoint if interrupted:
    python scripts/vne_gen_local.py --split train --num-instances 10000 --out data/vne/... --resume-from data/vne/vne_train_checkpoint.pickle

WHAT TO SEND BACK:
    After generation, send these 3 pickle files back to the cluster:
      data/vne/vne_supervised_training_dataset_10k.pickle  (~500 MB)
      data/vne/vne_validation_dataset_1k.pickle             (~50 MB)
      data/vne/vne_test_dataset_2k.pickle                   (~100 MB)
    Place them at the same paths on the cluster.

ESTIMATED RUNTIME (Gurobi full license, 8-core CPU):
    Training:   10,000 instances × ~5s  ≈ 14 hours
    Validation:  1,000 instances × ~5s  ≈ 1.5 hours
    Test:        2,000 instances × ~5s  ≈ 3 hours
    Total (--all):                      ≈ 18 hours
    (Actual times scale with your CPU — Gurobi is typically 10-50× faster than HiGHS)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# --- Path setup: ensure we can import from the repo ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vne.config import VNEConfig
from vne.validation_set_generator import (
    generate_instance,
    make_validation_dataset,
    run_self_check,
    solver_kwargs_from_config,
)


def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def save_checkpoint(
    dataset: List[Dict],
    out_path: str,
    split: str,
    seed: int,
    solver: str,
    elapsed: float,
) -> str:
    """Save a checkpoint pickle that can be resumed from."""
    checkpoint_path = out_path.replace(".pickle", "_checkpoint.pickle")
    meta = {
        "split": split,
        "seed": seed,
        "solver": solver,
        "num_completed": len(dataset),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"meta": meta, "dataset": dataset}, f)
    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> tuple[List[Dict], dict]:
    """Load a checkpoint. Returns (dataset_so_far, meta_dict)."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    return data["dataset"], data["meta"]


def check_gurobi() -> bool:
    """Verify that Gurobi is available with a full license."""
    try:
        import pulp

        solver = pulp.GUROBI(msg=False, timeLimit=10)
        if not solver.available():
            print("ERROR: Gurobi solver not available via PuLP.")
            print("       Make sure 'gurobipy' is installed and licensed.")
            return False
        print("Gurobi solver detected and available.")
        return True
    except Exception as e:
        print(f"ERROR: Cannot initialize Gurobi: {e}")
        print("       Install with: pip install gurobipy")
        print("       Also ensure your Gurobi license is activated.")
        return False


def preview_config(config: VNEConfig) -> None:
    """Print a summary of the generation configuration."""
    n_lo, n_hi = config.num_substrate_comm_nodes_range
    r_lo, r_hi = config.num_virtual_requests_range
    v_lo, v_hi = config.num_virtual_nodes_range
    bw_lo, bw_hi = config.substrate_communication_bandwidth_range
    cap_lo, cap_hi = config.substrate_compute_capacity_range
    print("─" * 60)
    print("Configuration (from vne/config.py)")
    print("─" * 60)
    print(f"  Substrate nodes:     {n_lo}–{n_hi}")
    print(f"  Virtual requests:    {r_lo}–{r_hi}")
    print(f"  VNR nodes/request:   {v_lo}–{v_hi}")
    print(f"  Topology:            {config.substrate_topology}")
    print(f"  BW per edge:         {bw_lo}–{bw_hi}")
    print(f"  CAP per comp link:   {cap_lo}–{cap_hi}")
    print(f"  Compute attach prob: {config.substrate_compute_attach_probability}")
    print(f"  Admission control:   {config.enable_admission}")
    print(f"  Objective:           {config.validation_objective}")
    print("─" * 60)


def generate_one_split(
    *,
    split: str,
    num_instances: int,
    out_path: str,
    seed: int,
    solver: str,
    threads: int,
    time_limit: int,
    mip_gap: float,
    checkpoint_every: int,
    resume_from: Optional[str],
) -> List[Dict]:
    """Generate one dataset split with checkpointing and progress display."""

    config = VNEConfig()
    solver_kwargs = solver_kwargs_from_config(
        config,
        solver=solver,
        threads=threads,
        time_limit_s=time_limit,
        mip_gap=mip_gap,
    )

    # Resume or start fresh
    dataset: List[Dict] = []
    start_idx = 0
    if resume_from and os.path.exists(resume_from):
        dataset, meta = load_checkpoint(resume_from)
        start_idx = meta["num_completed"]
        print(f"Resumed from checkpoint: {start_idx} instances already generated.")
        print(f"  Elapsed so far: {format_duration(meta['elapsed_seconds'])}")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    remaining = num_instances - len(dataset)
    if remaining <= 0:
        print(f"Already have {len(dataset)} instances — nothing to generate.")
        return dataset

    print(f"Generating {remaining} instances (total target: {num_instances})...")
    print(f"  Solver: {solver} | Threads: {threads} | Time limit: {time_limit}s | MIP gap: {mip_gap}")
    print()

    # We generate one at a time (not using make_validation_dataset's batch)
    # so we can checkpoint and show detailed progress.
    rng = random.Random(seed)
    # Advance RNG past already-generated instances
    for _ in range(start_idx):
        random.Random(seed + _)  # dummy advance (approximate)

    total_elapsed = 0.0
    instance_times: List[float] = []
    t_start = time.time()

    for i in range(start_idx, num_instances):
        instance_seed = seed + i
        rng_instance = random.Random(instance_seed)

        t0 = time.time()
        resample_count = 0
        max_resample = 50  # generous resample budget

        while True:
            try:
                instance = generate_instance(
                    config,
                    rng_instance,
                    with_solutions=True,
                    solver_kwargs=solver_kwargs,
                    max_resample=max_resample - resample_count,
                )
                break
            except RuntimeError:
                resample_count += 1
                if resample_count >= max_resample:
                    raise RuntimeError(
                        f"Could not generate feasible instance after {max_resample} resamples "
                        f"at instance index {i}. Try widening resource ranges or enabling admission."
                    )
                # Create a new RNG for the resample (otherwise we get the same instance)
                rng_instance = random.Random(instance_seed + resample_count * 10000)

        dt = time.time() - t0
        instance_times.append(dt)
        total_elapsed += dt
        dataset.append(instance)

        # Progress display
        n_done = i + 1
        n_remain = num_instances - n_done
        avg_time = total_elapsed / (n_done - start_idx)
        eta = avg_time * n_remain
        pct = n_done / num_instances * 100

        sub = instance["substrate"]
        reqs = instance.get("requests", [])
        obj = instance.get("objective", float("nan"))

        print(
            f"  [{n_done:5d}/{num_instances} {pct:5.1f}%]  "
            f"{sub['num_comm_nodes']:2d}n {len(reqs):2d}rq  "
            f"obj={obj:8.1f}  "
            f"{dt:5.1f}s  "
            f"avg={avg_time:.1f}s  "
            f"ETA={format_duration(eta)}  "
            f"elapsed={format_duration(time.time() - t_start)}",
            flush=True,
        )

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            ckpt = save_checkpoint(
                dataset, out_path, split, seed, solver, time.time() - t_start
            )
            print(f"  ── checkpoint ({i + 1} instances) → {ckpt}", flush=True)

    total_time = time.time() - t_start
    print()
    print(f"Generation complete: {len(dataset)} instances in {format_duration(total_time)}")
    if instance_times:
        print(f"  Instance times: min={min(instance_times):.1f}s  max={max(instance_times):.1f}s  mean={sum(instance_times)/len(instance_times):.1f}s")
    print()

    # Self-check
    print("Running self-check on generated embeddings...")
    try:
        run_self_check(dataset)
        print("  Self-check PASSED.")
    except AssertionError as e:
        print(f"  Self-check FAILED: {e}")
        print("  The dataset may have constraint violations. Check the ILP formulation.")
        raise
    print()

    # Save final pickle
    print(f"Saving to {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    # Save manifest
    sizes = [inst["substrate"]["num_comm_nodes"] for inst in dataset]
    nreq = [len(inst.get("requests", [])) for inst in dataset]
    total_decisions = sum(
        sum(r["num_processing_nodes"] - 1 for r in inst.get("requests", []))
        for inst in dataset
    )
    manifest = {
        "out": out_path,
        "split": split,
        "num_instances": len(dataset),
        "total_replay_decisions": total_decisions,
        "substrate_nodes": {
            "min": min(sizes), "max": max(sizes), "mean": sum(sizes) / len(sizes)
        },
        "requests_per_instance": {
            "min": min(nreq), "max": max(nreq), "mean": sum(nreq) / len(nreq)
        },
        "instance_time_s": {
            "min": min(instance_times) if instance_times else 0,
            "max": max(instance_times) if instance_times else 0,
            "mean": sum(instance_times) / len(instance_times) if instance_times else 0,
        },
        "total_time_s": total_time,
        "solver": solver,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
    }
    manifest_path = out_path + ".manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Clean up checkpoint
    checkpoint_path = out_path.replace(".pickle", "_checkpoint.pickle")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Manifest:  {manifest_path}")
    print(f"  Decisions: {total_decisions:,} (avg {total_decisions/len(dataset):.1f}/instance)")
    print()

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VNE Dataset Generator — Local Gurobi Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Generate everything (train 10k + val 1k + test 2k):
  python scripts/vne_gen_local.py --all

  # Training set only:
  python scripts/vne_gen_local.py --split train --num-instances 10000 --out data/vne/vne_supervised_training_dataset_10k.pickle

  # Resume interrupted run:
  python scripts/vne_gen_local.py --split train --num-instances 10000 --out data/vne/... --resume-from data/vne/vne_train_checkpoint.pickle
        """,
    )
    parser.add_argument("--all", action="store_true", help="Generate all 3 datasets (train+val+test)")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Which split to generate")
    parser.add_argument("--num-instances", type=int, help="Number of instances to generate")
    parser.add_argument("--out", type=str, help="Output pickle path")
    parser.add_argument("--seed", type=int, default=1000000, help="Base seed (default: 1000000)")
    parser.add_argument("--solver", default="gurobi", choices=["gurobi", "highs", "cplex", "cbc"])
    parser.add_argument("--threads", type=int, default=0, help="Solver threads, 0=auto (default: 0)")
    parser.add_argument("--time-limit", type=int, default=120, help="Per-instance solver time limit in seconds (default: 120)")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="MIP optimality gap, 0.0=exact (default: 0.0)")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save checkpoint every N instances (default: 500)")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint pickle")
    args = parser.parse_args()

    # --- Validation ---
    if not args.all and (not args.split or not args.num_instances or not args.out):
        parser.error("Either --all or (--split + --num-instances + --out) is required")

    if not check_gurobi():
        sys.exit(1)

    config = VNEConfig()
    preview_config(config)

    # --- Define the jobs ---
    if args.all:
        jobs = [
            {
                "split": "train",
                "num_instances": 10000,
                "out": "data/vne/vne_supervised_training_dataset_10k.pickle",
                "seed": 1000000,
            },
            {
                "split": "val",
                "num_instances": 1000,
                "out": "data/vne/vne_validation_dataset_1k.pickle",
                "seed": 2000000,
            },
            {
                "split": "test",
                "num_instances": 2000,
                "out": "data/vne/vne_test_dataset_2k.pickle",
                "seed": 3000000,
            },
        ]
    else:
        jobs = [
            {
                "split": args.split,
                "num_instances": args.num_instances,
                "out": args.out,
                "seed": args.seed,
            }
        ]

    # --- Run ---
    total_start = time.time()
    all_success = True

    for job_idx, job in enumerate(jobs):
        print()
        print("=" * 60)
        print(f" JOB {job_idx + 1}/{len(jobs)}: {job['split'].upper()} — {job['num_instances']:,} instances")
        print("=" * 60)

        if os.path.exists(job["out"]) and not args.resume_from:
            print(f"Output already exists: {job['out']}")
            print("Skipping (delete the file or use --resume-from to regenerate).")
            print()
            continue

        try:
            generate_one_split(
                split=job["split"],
                num_instances=job["num_instances"],
                out_path=job["out"],
                seed=job["seed"],
                solver=args.solver,
                threads=args.threads,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                checkpoint_every=args.checkpoint_every,
                resume_from=args.resume_from if job_idx == 0 else None,
            )
        except KeyboardInterrupt:
            print()
            print("Interrupted. Checkpoint saved — resume with:")
            ckpt = job["out"].replace(".pickle", "_checkpoint.pickle")
            print(f"  python scripts/vne_gen_local.py --split {job['split']} --num-instances {job['num_instances']} --out {job['out']} --resume-from {ckpt}")
            all_success = False
            break
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_success = False
            break

    total_time = time.time() - total_start
    print()
    print("=" * 60)
    if all_success:
        print(f" ALL DONE — {format_duration(total_time)}")
        print()
        print("Copy these files back to the cluster at the same paths:")
        for job in jobs:
            print(f"  {job['out']}")
            print(f"  {job['out']}.manifest.json")
        print()
        print("Then on the cluster, launch Phase 1 training:")
        print("  bash scripts/vne_phase1_launch.sh")
    else:
        print(" GENERATION INCOMPLETE — check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
