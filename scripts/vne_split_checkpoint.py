#!/usr/bin/env python3
"""
Split the 4k Gurobi checkpoint into train/val/test datasets at 60-80n scale.

Usage:
    python scripts/vne_split_checkpoint.py
    python scripts/vne_split_checkpoint.py --train 3000 --val 500 --test 500
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split checkpoint into train/val/test")
    parser.add_argument("--checkpoint", default="vne_supervised_training_dataset_10k_checkpoint.pickle")
    parser.add_argument("--train", type=int, default=2500)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--out-dir", default="data/vne")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    out_dir = PROJECT_ROOT / args.out_dir

    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    instances = data["dataset"]
    meta = data["meta"]
    total = len(instances)

    print(f"  {total} instances | solver={meta['solver']} | seed={meta['seed']}")
    print(f"  Elapsed during generation: {meta['elapsed_seconds'] / 3600:.1f}h")

    requested = args.train + args.val + args.test
    if requested > total:
        print(f"ERROR: requested {requested} instances but only {total} available")
        sys.exit(1)

    splits = {
        "train": (0, args.train),
        "val": (args.train, args.train + args.val),
        "test": (args.train + args.val, args.train + args.val + args.test),
    }

    names = {
        "train": "vne_supervised_training_dataset_4k.pickle",
        "val": "vne_validation_dataset_1k.pickle",
        "test": "vne_test_dataset_2k.pickle",
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    all_passed = True

    for split_name, (start, end) in splits.items():
        split_data = instances[start:end]
        out_path = out_dir / names[split_name]

        # Stats
        sizes = [d["substrate"]["num_comm_nodes"] for d in split_data]
        nreq = [len(d.get("requests", [])) for d in split_data]
        objs = [d.get("objective", float("nan")) for d in split_data]
        total_dec = sum(
            sum(r["num_processing_nodes"] - 1 for r in d.get("requests", []))
            for d in split_data
        )

        print(f"\n  {split_name.upper()}: {len(split_data)} instances "
              f"(indices {start}–{end - 1})")
        print(f"    Nodes:     {min(sizes)}–{max(sizes)}, mean={sum(sizes) / len(sizes):.1f}")
        print(f"    Requests:  {min(nreq)}–{max(nreq)}, mean={sum(nreq) / len(nreq):.1f}")
        print(f"    Objective: {min(objs):.1f} … {max(objs):.1f}, mean={sum(objs) / len(objs):.1f}")
        print(f"    Decisions: {total_dec} ({total_dec / len(split_data):.1f}/inst)")

        # Verify ILP labels
        missing = sum(
            1 for d in split_data
            if "f_placements" not in d or d.get("objective") is None
        )
        if missing:
            print(f"    ⚠  {missing} instances missing ILP labels!")
            all_passed = False

        if args.dry_run:
            continue

        # Save pickle
        with open(out_path, "wb") as f:
            pickle.dump(split_data, f)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"    → {out_path} ({size_mb:.1f} MB)")

        # Manifest
        manifest = {
            "out": names[split_name],
            "split": split_name,
            "num_instances": len(split_data),
            "total_replay_decisions": total_dec,
            "substrate_nodes": {
                "min": min(sizes), "max": max(sizes), "mean": sum(sizes) / len(sizes),
            },
            "requests_per_instance": {
                "min": min(nreq), "max": max(nreq), "mean": sum(nreq) / len(nreq),
            },
            "objective": {
                "min": min(objs), "max": max(objs), "mean": sum(objs) / len(objs),
            },
            "source": "split from 4k Gurobi checkpoint (60-80n, 10-20rq)",
            "generated_at": datetime.now().isoformat(),
        }
        manifest_path = str(out_path) + ".manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"    → {manifest_path}")

    # Self-check
    if not args.dry_run:
        print(f"\n  Self-check...")
        from vne.validation_set_generator import run_self_check

        for split_name, (start, end) in splits.items():
            split_data = instances[start:end]
            try:
                run_self_check(split_data)
                print(f"    {split_name}: ✓ PASSED")
            except AssertionError as e:
                print(f"    {split_name}: ✗ FAILED — {e}")
                all_passed = False

    print()
    if args.dry_run:
        print("  (dry-run, no files written)")
    elif all_passed:
        print("  ALL DONE — all splits passed self-check ✓")
    else:
        print("  DONE with warnings — see above ⚠")


if __name__ == "__main__":
    main()
