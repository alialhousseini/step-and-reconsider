"""Parse a VNE training run's log into a per-epoch train-loss vs. validation curve.

Reads the run log (the file `scripts/vne_train.sbatch` tees, or model_checkpoints/
vne/results/<ts>/log.txt) and extracts, per epoch, the average train loss and the
validation objective, so we can see exactly where/whether validation turns
(the "epoch-6 drop"). Writes a CSV and, if matplotlib is present, a PNG.

Usage:
    python scripts/vne_plot_training.py --log logs/vne_train_<jobid>.log [--out-prefix artifacts/run]
"""
from __future__ import annotations

import argparse
import csv
import os
import re

LOSS_RE = re.compile(r">> Epoch (\d+)\.\s*Avg loss:\s*([-\d.eE+]+)")
# Validation line is the printed loggable dict, e.g.
#   {'Validation beam width 1. Obj.': -21.5, 'Validation beam width 4. Obj.': -20.8}
VAL_OBJ_RE = re.compile(r"Validation beam width (\d+)\. Obj\.':\s*([-\d.eE+]+)")


def parse(log_path: str):
    epochs = []
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()

    cur = None
    for line in lines:
        m = LOSS_RE.search(line)
        if m:
            if cur is not None:
                epochs.append(cur)
            cur = {"epoch": int(m.group(1)), "train_loss": float(m.group(2))}
            continue
        for bw, obj in VAL_OBJ_RE.findall(line):
            if cur is not None:
                cur[f"val_obj_beam{bw}"] = float(obj)
    if cur is not None:
        epochs.append(cur)
    return epochs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out-prefix", default="artifacts/vne_run")
    args = p.parse_args()

    epochs = parse(args.log)
    if not epochs:
        raise SystemExit(f"No epochs parsed from {args.log} (run may not have started training yet).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    cols = sorted({k for e in epochs for k in e})
    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss"] + [c for c in cols if c.startswith("val_")])
        w.writeheader()
        for e in epochs:
            w.writerow({k: e.get(k, "") for k in w.fieldnames})
    print(f"wrote {csv_path} ({len(epochs)} epochs)")

    # Console summary: find where validation best occurs (objective is maximize-style,
    # printed as negative cost -> higher (less negative) is better).
    beam_key = next((c for c in cols if c.startswith("val_obj_beam4")), None) or \
               next((c for c in cols if c.startswith("val_obj_beam")), None)
    if beam_key:
        vals = [(e["epoch"], e.get(beam_key)) for e in epochs if e.get(beam_key) is not None]
        if vals:
            best = max(vals, key=lambda t: t[1])
            print(f"best {beam_key} = {best[1]} at epoch {best[0]} (of {vals[-1][0]})")
            print("epoch / train_loss / " + beam_key)
            for e in epochs:
                print(f"  {e['epoch']:>3}  {e.get('train_loss'):>10}  {e.get(beam_key)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        xs = [e["epoch"] for e in epochs]
        ax1.plot(xs, [e.get("train_loss") for e in epochs], "b-o", label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="b")
        ax2 = ax1.twinx()
        for c in cols:
            if c.startswith("val_obj_beam"):
                ax2.plot(xs, [e.get(c) for e in epochs], "-s", label=c)
        ax2.set_ylabel("val objective (higher=better)")
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=8)
        png = args.out_prefix + ".png"
        fig.tight_layout(); fig.savefig(png, dpi=120)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(matplotlib plot skipped: {e})")


if __name__ == "__main__":
    main()
