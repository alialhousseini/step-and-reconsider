#!/usr/bin/env python3
"""Analyze VNE training results: compute optimality gap vs ILP labels.

Usage:
    python scripts/vne_gap_analysis.py --log logs/vne_train_<tag>_<jobid>.log
    python scripts/vne_gap_analysis.py --results model_checkpoints/vne/results/<dir>
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict


def parse_training_log(log_path: str) -> list[dict]:
    """Extract per-epoch train loss and validation objective from a training log."""
    epochs = []
    with open(log_path) as f:
        for line in f:
            m = re.match(r">> Epoch (\d+)\. Avg loss: ([\d.]+)", line)
            if m:
                epochs.append({"epoch": int(m.group(1)), "train_loss": float(m.group(2))})
                continue
            m = re.search(r"'Validation beam width 1\. Obj\.': (-?[\d.]+)", line)
            if m and epochs:
                epochs[-1]["val_obj"] = float(m.group(1))
            if "Got new best model" in line and epochs:
                epochs[-1]["is_best"] = True
    return epochs


def compute_ilp_stats(test_pickle_path: str) -> dict:
    """Compute ILP optimum statistics from a test pickle."""
    with open(test_pickle_path, "rb") as f:
        data = pickle.load(f)
    objectives = [inst["objective"] for inst in data if "objective" in inst]
    import numpy as np
    arr = np.array(objectives)
    return {
        "count": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_gap(model_obj: float, ilp_mean_obj: float) -> dict:
    """Compute optimality gap statistics.

    Both model_obj and ilp_mean_obj are NEGATIVE (maximization convention).
    Less negative = lower cost = better.
    model_obj is typically more negative (worse) than ilp_mean_obj.
    """
    model_cost = -model_obj  # Convert to positive cost
    ilp_cost = -ilp_mean_obj  # Convert to positive cost
    abs_gap = model_cost - ilp_cost  # Positive = model is worse
    rel_gap_pct = (abs_gap / ilp_cost) * 100.0 if ilp_cost > 0 else 0.0
    return {
        "model_obj": model_obj,
        "ilp_obj": ilp_mean_obj,
        "model_cost": model_cost,
        "ilp_cost": ilp_cost,
        "abs_gap": abs_gap,
        "rel_gap_pct": rel_gap_pct,
    }


def main():
    p = argparse.ArgumentParser(description="VNE optimality gap analysis")
    p.add_argument("--log", help="Path to training log file")
    p.add_argument("--results", help="Path to results directory")
    p.add_argument("--test-pickle",
                   default="data/vne/vne_test_dataset_2k.pickle",
                   help="Path to test pickle with ILP labels")
    p.add_argument("--val-pickle",
                   default="data/vne/vne_validation_dataset_1k.pickle",
                   help="Path to validation pickle with ILP labels")
    args = p.parse_args()

    if args.log:
        epochs = parse_training_log(args.log)
    elif args.results:
        # Find log in results directory
        log_candidates = [
            f for f in os.listdir(args.results) if f.endswith(".log")
        ]
        if not log_candidates:
            # Search parent logs directory
            tag = os.path.basename(args.results)
            log_dir = os.path.join(os.path.dirname(os.path.dirname(args.results)), "logs")
            # Actually, logs are in repo_root/logs/
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(repo_root, "logs")
            # Try to find matching log
            for f in sorted(os.listdir(log_dir)):
                if tag in f and f.endswith(".log"):
                    log_candidates.append(os.path.join(log_dir, f))
        if log_candidates:
            epochs = parse_training_log(log_candidates[-1])
        else:
            print("No log found for results dir:", args.results)
            sys.exit(1)
    else:
        print("Need --log or --results")
        sys.exit(1)

    if not epochs:
        print("No epochs found in log.")
        sys.exit(1)

    # Compute ILP stats
    print("=== ILP Optimum Statistics ===")
    val_ilp = compute_ilp_stats(args.val_pickle)
    test_ilp = compute_ilp_stats(args.test_pickle)
    print(f"Validation ILP: mean={val_ilp['mean']:.4f}  std={val_ilp['std']:.4f}  "
          f"n={val_ilp['count']}  range=[{val_ilp['min']:.2f}, {val_ilp['max']:.2f}]")
    print(f"Test ILP:       mean={test_ilp['mean']:.4f}  std={test_ilp['std']:.4f}  "
          f"n={test_ilp['count']}  range=[{test_ilp['min']:.2f}, {test_ilp['max']:.2f}]")
    print()

    # Per-epoch gap
    print("=== Per-Epoch Results ===")
    print(f"{'Epoch':>5s}  {'TrainLoss':>9s}  {'ValObj':>9s}  {'AbsGap':>7s}  "
          f"{'RelGap%':>7s}  {'Best?':>5s}")
    print("-" * 65)
    for ep in epochs:
        if "val_obj" not in ep:
            continue
        gap = compute_gap(ep["val_obj"], val_ilp["mean"])
        best = "BEST" if ep.get("is_best") else ""
        print(f"{ep['epoch']:5d}  {ep['train_loss']:9.4f}  {ep['val_obj']:9.4f}  "
              f"{gap['abs_gap']:7.4f}  {gap['rel_gap_pct']:6.2f}%  {best:>5s}")

    print()
    best_ep = min((e for e in epochs if "val_obj" in e), key=lambda e: -e["val_obj"])
    best_gap = compute_gap(best_ep["val_obj"], val_ilp["mean"])
    print(f"Best validation: epoch {best_ep['epoch']}  obj={best_ep['val_obj']:.4f}  "
          f"gap={best_gap['rel_gap_pct']:.2f}%")


if __name__ == "__main__":
    main()
