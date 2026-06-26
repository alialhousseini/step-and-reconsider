"""Feasibility-vs-scale table + figure (Phase 4 extended, no labels needed).

Reads per-(scale, model) eval JSONs produced by vne_eval_table.py on unlabeled
larger-substrate sets and reports greedy/beam-16 feasibility% vs substrate scale.

Usage: python scripts/vne_genscale_table.py <SCRATCH_DIR>
Expects files: genscale_<model>_<nodes>.json with configs greedy, bs k=16.
"""
import sys, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = sys.argv[1]
MODELS = ["bq", "lehd"]
SCALES = [80, 120, 160, 240, 320]   # nodes
BASE = 70.0  # mid of trained 60-80 range, for the x-axis "x trained" label

def feas(path, key):
    d = json.load(open(path))
    pi = d["per_instance"][key]
    return sum(p["feasible"] for p in pi) / len(pi) * 100.0

rows = {}  # (model, nodes) -> {greedy, bs16}
for m in MODELS:
    for n in SCALES:
        p = f"{SCR}/genscale_{m}_{n}.json"
        if os.path.exists(p):
            rows[(m, n)] = {"greedy": feas(p, "greedy"), "bs16": feas(p, "bs k=16")}

print("\n=== Feasibility vs substrate scale (Ours models, unlabeled, no solver) ===")
print("| nodes | ~x trained | BQ greedy | BQ bs16 | LEHD greedy | LEHD bs16 |")
print("|-------|------------|-----------|---------|-------------|-----------|")
for n in SCALES:
    bq = rows.get(("bq", n), {}); le = rows.get(("lehd", n), {})
    def f(d, k): return f"{d[k]:.1f}" if k in d else "-"
    print(f"| {n} | {n/BASE:.1f}x | {f(bq,'greedy')} | {f(bq,'bs16')} | {f(le,'greedy')} | {f(le,'bs16')} |")

# figure: feas% vs nodes
fig, ax = plt.subplots(figsize=(7, 4.5))
styles = {("bq","greedy"):"C0--o", ("bq","bs16"):"C0-o", ("lehd","greedy"):"C1--s", ("lehd","bs16"):"C1-s"}
for (m, meth), st in styles.items():
    xs = [n for n in SCALES if (m, n) in rows]
    ys = [rows[(m, n)][meth] for n in xs]
    if xs:
        ax.plot(xs, ys, st, label=f"{m.upper()} {('greedy' if meth=='greedy' else 'beam-16')}")
ax.axvspan(60, 80, color="gray", alpha=0.15, label="trained range (60-80)")
ax.set_xlabel("substrate nodes"); ax.set_ylabel("Feasibility% (N per scale)")
ax.set_title("Generalization: feasibility vs substrate scale (no retraining)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout()
out = "artifacts/vne_phase4_feas_vs_scale.png"
fig.savefig(out, dpi=120)
print(f"\nsaved figure -> {out}")
