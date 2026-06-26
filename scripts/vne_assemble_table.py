"""Assemble the final paper-style VNE Phase 2 table from per-checkpoint eval JSONs.

Each input JSON (from vne_eval_table.py) holds per-instance results per config for
ONE checkpoint. This script selects the table rows across checkpoints, computes a
GLOBAL F* (instances feasible under EVERY selected row) so gap% is orderable, and
emits the markdown table + caption.

Rows are defined below in ROW_PLAN as (display_label, json_path, config_key).
"""
import sys, os, json
import numpy as np

# Edit/extend as needed; config_key must match a key in that JSON's per_instance.
SCR = sys.argv[1] if len(sys.argv) > 1 else "."
G = os.environ.get("EVAL_GROUP", "BQ")  # architecture group label (BQ / LEHD)
ROW_PLAN = [
    (f"{G} SL, greedy",        f"{SCR}/eval_sl.json",    "greedy"),
    (f"GD SIL ({G}), greedy",  f"{SCR}/eval_gdsil.json", "greedy"),
    (f"Ours ({G}), greedy",    f"{SCR}/eval_tasar.json", "greedy"),
    (f"Ours ({G}), bs k=16",   f"{SCR}/eval_tasar.json", "bs k=16"),
    (f"Ours ({G}), k=8 s=4",   f"{SCR}/eval_tasar.json", "TaSaR k=8 s=4"),
    (f"Ours ({G}), k=16 s=4",  f"{SCR}/eval_tasar.json", "TaSaR k=16 s=4"),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    caches = {}
    rows_pi = []   # (label, per_inst_list, eval_row_meta)
    for label, path, key in ROW_PLAN:
        if path not in caches:
            caches[path] = load(path)
        data = caches[path]
        pi = data["per_instance"][key]
        meta = next((r for r in data["rows"] if r["method"] == key), {})
        rows_pi.append((label, pi, meta))

    N = len(rows_pi[0][1])
    assert all(len(pi) == N for _, pi, _ in rows_pi), "row length mismatch (different N / order)"

    # global F* = feasible under EVERY row
    fstar = [i for i in range(N) if all(pi[i]["feasible"] for _, pi, _ in rows_pi)]

    def gap_on(idx, pi):
        gs = []
        for i in idx:
            p = pi[i]
            if p["feasible"] and p["ilp"] not in (None, 0):
                gs.append((p["ilp"] - p["obj"]) / abs(p["ilp"]) * 100.0)
        return float(np.mean(gs)) if gs else float("nan")

    out_rows = []
    for label, pi, meta in rows_pi:
        out_rows.append({
            "method": label,
            "feas_pct": sum(p["feasible"] for p in pi) / N * 100.0,
            "gap_Fstar": gap_on(fstar, pi),
            "mean_g": float(np.mean([p["g"] for p in pi])),
            "wall_s": meta.get("wall_s", float("nan")),
            "ms_inst": meta.get("ms_per_inst", float("nan")),
        })

    best_feas = max(r["feas_pct"] for r in out_rows)
    valid_gaps = [r["gap_Fstar"] for r in out_rows if r["gap_Fstar"] == r["gap_Fstar"]]
    best_gap = min(valid_gaps) if valid_gaps else None

    print(f"\n=== VNE PHASE 2 RESULTS TABLE ({G}) ===  N={N}  |F*|={len(fstar)}\n")
    print("| Method | Feas% | Gap%(F*) | g(k,s) | Time set(s) | ms/inst |")
    print("|---|---|---|---|---|---|")
    for r in out_rows:
        fs = f"**{r['feas_pct']:.1f}**" if r["feas_pct"] == best_feas else f"{r['feas_pct']:.1f}"
        g = r["gap_Fstar"]
        gs = "n/a" if g != g else (f"**{g:.3f}**" if g == best_gap else f"{g:.3f}")
        print(f"| {r['method']} | {fs} | {gs} | {r['mean_g']:.0f} | {r['wall_s']:.0f} | {r['ms_inst']:.0f} |")

    with open(f"{SCR}/final_table.json", "w") as f:
        json.dump({"N": N, "Fstar": len(fstar), "rows": out_rows}, f, indent=2)
    print(f"\n|F*| = {len(fstar)} of {N} instances feasible under all rows.")
    print(f"saved -> {SCR}/final_table.json")


if __name__ == "__main__":
    main()
