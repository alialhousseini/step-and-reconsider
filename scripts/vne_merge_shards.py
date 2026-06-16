"""Merge VNE data shards into one solved pickle, with integrity checks.

Concatenates all shards matching a glob, asserts no duplicate instances across
shards (different RNG seeds should already guarantee this), re-runs the embedding
self-check on the union, writes the final pickle plus a JSON manifest describing
counts, seeds, and the size/request distribution.

Usage:
    python scripts/vne_merge_shards.py \
        --shard-glob 'data/vne/shards/train_shard*.pickle' \
        --out data/vne/vne_supervised_training_dataset_50k.pickle
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vne.validation_set_generator import run_self_check


def _instance_key(inst: dict) -> tuple:
    """A hashable fingerprint of the raw problem (substrate + requests)."""
    sub = inst["substrate"]
    reqs = inst.get("requests") or [inst.get("request")]
    return (
        sub["num_comm_nodes"],
        tuple(sorted(sub["communication_bandwidth"].items())),
        tuple(sorted(sub["compute_capacity"].items())),
        tuple(sorted((c, p) for c, p in sub.get("compute_attachment", {}).items())),
        tuple(
            (r["num_processing_nodes"], r["source_link_demand"],
             r["destination_link_demand"], tuple(r["processing_link_demands"]))
            for r in reqs
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Merge VNE shards")
    p.add_argument("--shard-glob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-duplicates", action="store_true",
                   help="Do not fail on duplicate instances across shards (still reported).")
    p.add_argument("--skip-self-check", action="store_true")
    args = p.parse_args()

    shard_paths = sorted(glob.glob(args.shard_glob))
    if not shard_paths:
        raise SystemExit(f"No shards matched: {args.shard_glob}")
    print(f"Merging {len(shard_paths)} shards", flush=True)

    merged: list = []
    seen: dict = {}
    duplicates = 0
    for sp in shard_paths:
        with open(sp, "rb") as f:
            shard = pickle.load(f)
        for inst in shard:
            key = _instance_key(inst)
            if key in seen:
                duplicates += 1
                if not args.allow_duplicates:
                    continue
            seen[key] = True
            merged.append(inst)
        print(f"  {os.path.basename(sp)}: +{len(shard)} (running total {len(merged)})", flush=True)

    if duplicates:
        msg = f"Found {duplicates} duplicate instances across shards"
        if not args.allow_duplicates:
            print(f"WARNING: {msg}; dropped them (use --allow-duplicates to keep).", flush=True)
        else:
            print(f"WARNING: {msg}; kept (allow-duplicates).", flush=True)

    if not args.skip_self_check:
        run_self_check(merged)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(merged, f)

    # Manifest with distributions for sanity.
    sizes = [i["substrate"]["num_comm_nodes"] for i in merged]
    nreq = [len(i.get("requests") or [i.get("request")]) for i in merged]
    decisions = sum(
        sum(r["num_processing_nodes"] - 1 for r in (i.get("requests") or [i.get("request")]))
        for i in merged
    )
    manifest = {
        "out": args.out,
        "num_shards": len(shard_paths),
        "num_instances": len(merged),
        "duplicates_detected": duplicates,
        "total_replay_decisions": decisions,
        "substrate_nodes": {"min": min(sizes), "max": max(sizes), "mean": sum(sizes) / len(sizes)},
        "requests_per_instance": {"min": min(nreq), "max": max(nreq), "mean": sum(nreq) / len(nreq)},
        "shards": [os.path.basename(s) for s in shard_paths],
    }
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f">> wrote {len(merged)} instances to {args.out}", flush=True)


if __name__ == "__main__":
    main()
