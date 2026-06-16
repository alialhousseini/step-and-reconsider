#!/usr/bin/env python3
"""Collect and analyze results from the VNE Gumbeldore/TaSaR grid search.

Reads all grid job logs, extracts per-epoch validation objectives, best
val per run, test scores, and printing a summary table + best config.

Usage: python scripts/vne_grid_results.py
"""
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("VNE_REPO", "/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider"))
LOGS_DIR = REPO / "logs"
RESULTS_DIR = REPO / "model_checkpoints" / "vne" / "results"

# Configs from the grid: beam_width x replan_steps
GRID = [
    (8, 2), (8, 4), (8, 8),
    (16, 2), (16, 4), (16, 8),
    (32, 2), (32, 4), (32, 8),
    (64, 2), (64, 4), (64, 8),
]


def extract_from_slurm_log(job_id: int) -> dict | None:
    """Parse a SLURM .out log to extract epoch-by-epoch validation and test results."""
    log_path = LOGS_DIR / f"vne_train_{job_id}.out"
    if not log_path.exists():
        # Try glob for tag-based logs
        candidates = list(LOGS_DIR.glob(f"vne_train_*_{job_id}.out"))
        if not candidates:
            return None
        log_path = candidates[0]

    with open(log_path) as f:
        content = f.read()

    result = {
        "job_id": job_id,
        "epochs": [],
        "best_val": float("inf"),
        "best_epoch": None,
        "test_obj": None,
        "completed": False,
    }

    # Extract epoch results
    current_epoch = None
    current_train_loss = None
    for line in content.split("\n"):
        # Match: >> Epoch N. Avg loss: X.XXX
        epoch_match = re.search(r">> Epoch (\d+)\. Avg loss: ([\d.]+)", line)
        if epoch_match:
            if current_epoch is not None:
                result["epochs"].append({
                    "epoch": current_epoch,
                    "train_loss": current_train_loss,
                    "val_obj": None,
                })
            current_epoch = int(epoch_match.group(1))
            current_train_loss = float(epoch_match.group(2))
            continue

        # Match: {'Validation beam width 1. Obj.': -XX.XX}
        val_match = re.search(r"Validation beam width 1\. Obj\.'[:\s]*(-?[\d.]+)", line)
        if val_match and current_epoch is not None:
            val_obj = float(val_match.group(1))
            result["epochs"].append({
                "epoch": current_epoch,
                "train_loss": current_train_loss,
                "val_obj": val_obj,
            })
            if val_obj > result["best_val"]:  # less negative = better
                result["best_val"] = val_obj
                result["best_epoch"] = current_epoch
            current_epoch = None
            continue

        # Match: {'Test beam width 1. Obj.': -XX.XX}
        test_match = re.search(r"Test beam width 1\. Obj\.'[:\s]*(-?[\d.]+)", line)
        if test_match:
            result["test_obj"] = float(test_match.group(1))

        # Check completion
        if "Done. results:" in line:
            result["completed"] = True

    if current_epoch is not None:
        result["epochs"].append({
            "epoch": current_epoch,
            "train_loss": current_train_loss,
            "val_obj": None,
        })

    return result if result["epochs"] else None


def extract_from_results_dir(job_id: int) -> dict | None:
    """Parse the training log.txt from the results directory for a job."""
    # Find the results directory matching this job_id
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and str(job_id) in d.name:
            log_path = d / "log.txt"
            if not log_path.exists():
                continue
            try:
                with open(log_path) as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception:
                continue

            result = {
                "job_id": job_id,
                "run_tag": d.name,
                "epochs": [],
                "best_val": float("inf"),
                "best_epoch": None,
                "test_obj": None,
                "completed": False,
            }

            for line in lines:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "Train avg loss" in entry and "epoch" in entry:
                    epoch = entry["epoch"]
                    train_loss = entry["Train avg loss"]
                    result["epochs"].append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_obj": None,
                    })

                if "Validation beam width 1. Obj." in entry:
                    val_obj = entry["Validation beam width 1. Obj."]
                    if result["epochs"]:
                        result["epochs"][-1]["val_obj"] = val_obj
                    if val_obj > result["best_val"]:
                        result["best_val"] = val_obj
                        result["best_epoch"] = result["epochs"][-1]["epoch"] if result["epochs"] else None

                if "Test beam width 1. Obj." in entry:
                    result["test_obj"] = entry["Test beam width 1. Obj."]
                    result["completed"] = True

            return result if result["epochs"] else None
    return None


def map_jobs_to_configs():
    """Find which job ID corresponds to which (beam_width, replan_steps) pair."""
    mapping = {}
    for d in RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        match = re.search(r"gumbeldore_(?:fix_)?b(\d+)_s(\d+)_(\d+)", d.name)
        if match:
            bw = int(match.group(1))
            rs = int(match.group(2))
            job_id = int(match.group(3))
            mapping[(bw, rs)] = job_id
    return mapping


def main():
    mapping = map_jobs_to_configs()
    if not mapping:
        print("No grid results found. Looking for SLURM logs...")
        # Try SLURM logs
        for bw, rs in GRID:
            print(f"  b{bw}_s{rs}: no results found")

    print("=" * 72)
    print("VNE TaSaR Grid Results")
    print("=" * 72)
    print(f"{'BW':<6} {'RS':<6} {'Job':<8} {'Best Val':<12} {'Best Ep':<8} {'Test':<12} {'Status'}")
    print("-" * 72)

    rows = []
    for bw, rs in GRID:
        job_id = mapping.get((bw, rs))
        if not job_id:
            rows.append((bw, rs, "?", None, None, None, "no data"))
            continue

        # Try results dir first (more reliable), then SLURM log
        data = extract_from_results_dir(job_id) or extract_from_slurm_log(job_id)
        if not data:
            rows.append((bw, rs, job_id, None, None, None, "no log"))
            continue

        status = "DONE" if data["completed"] else f"{len(data['epochs'])}/10 ep"
        rows.append((
            bw, rs, job_id,
            data["best_val"],
            data["best_epoch"],
            data["test_obj"],
            status,
        ))

    # Sort by beam_width, replan_steps
    rows.sort(key=lambda r: (r[0], r[1]))

    for bw, rs, job_id, best_val, best_ep, test_obj, status in rows:
        print(f"{bw:<6} {rs:<6} {str(job_id):<8} "
              f"{best_val if best_val is not None else '---':<12} "
              f"{best_ep if best_ep is not None else '---':<8} "
              f"{test_obj if test_obj is not None else '---':<12} "
              f"{status}")

    print("-" * 72)

    # Find best config
    completed = [r for r in rows if r[3] is not None]
    if completed:
        best = max(completed, key=lambda r: r[3])
        print(f"\nBest config: beam_width={best[0]}, replan_steps={best[1]} "
              f"(val={best[3]}, epoch={best[4]})")

        # Compare to baselines
        print(f"\nBaselines:")
        print(f"  Supervised (2M, 50k ILP): best val -23.04")
        print(f"  Pilot (b16_s4, 32 inst):  best val -23.00")

        if best[3] is not None and best[3] > -23.00:
            print(f"\n*** Grid BEAT supervised baseline! ***")
        elif best[3] is not None:
            gap = abs(best[3] - (-23.00))
            print(f"\nGrid best {best[3]:.2f} vs pilot -23.00: Δ={gap:.2f}")

    # Show runs still in progress
    in_progress = [r for r in rows if "ep" in str(r[6])]
    if in_progress:
        print(f"\n{len(in_progress)} run(s) still in progress.")


if __name__ == "__main__":
    main()
