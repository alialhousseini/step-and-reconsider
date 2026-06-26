"""Phase 3 — Multi-beam evaluation table + gap-vs-beam figure.

Beam-width sweep (deterministic beam search) for the Phase 2 'Ours' models only
(BQ TaSaR p=0.9, LEHD TaSaR p=0.9 — no new models, no supervised baselines).

REUSES beam=1 (greedy) and beam=16 from the Phase 2 eval JSONs; the missing
widths {4,8,32,64} come from the Phase 3 eval JSONs. For each model, gap% is
averaged over that model's all-beam-feasible set (fixed denominator => the
gap-vs-beam curve is orderable); feas% is over the full N.

Usage: python scripts/vne_phase3_table.py <SCRATCH_DIR>
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = sys.argv[1]

# model -> [(beam, json_path, config_key)]
MODELS = {
    "BQ Ours": [
        (1,  f"{SCR}/eval_tasar.json",      "greedy"),
        (4,  f"{SCR}/eval_p3_bq.json",      "bs k=4"),
        (8,  f"{SCR}/eval_p3_bq.json",      "bs k=8"),
        (16, f"{SCR}/eval_tasar.json",      "bs k=16"),
        (32, f"{SCR}/eval_p3_bq.json",      "bs k=32"),
        (64, f"{SCR}/eval_p3_bq.json",      "bs k=64"),
    ],
    "LEHD Ours": [
        (1,  f"{SCR}/lehd/eval_tasar.json", "greedy"),
        (4,  f"{SCR}/eval_p3_lehd.json",    "bs k=4"),
        (8,  f"{SCR}/eval_p3_lehd.json",    "bs k=8"),
        (16, f"{SCR}/lehd/eval_tasar.json", "bs k=16"),
        (32, f"{SCR}/eval_p3_lehd.json",    "bs k=32"),
        (64, f"{SCR}/eval_p3_lehd.json",    "bs k=64"),
    ],
}

_cache = {}
def load(p):
    if p not in _cache:
        _cache[p] = json.load(open(p))
    return _cache[p]

def gap(idx, pi):
    g = [(pi[i]["ilp"] - pi[i]["obj"]) / abs(pi[i]["ilp"]) * 100
         for i in idx if pi[i]["feasible"] and pi[i]["ilp"] not in (None, 0)]
    return float(np.mean(g)) if g else float("nan")

def ms_of(path, key):
    for r in load(path)["rows"]:
        if r["method"] == key:
            return r["ms_per_inst"]
    return float("nan")

results = {}
for model, plan in MODELS.items():
    PIs = {b: load(p)["per_instance"][k] for b, p, k in plan}
    N = len(next(iter(PIs.values())))
    allfeas = [i for i in range(N) if all(PIs[b][i]["feasible"] for b in PIs)]
    rows = []
    for b, p, k in plan:
        pi = PIs[b]
        rows.append({"beam": b,
                     "feas": sum(x["feasible"] for x in pi) / N * 100.0,
                     "gap": gap(allfeas, pi),
                     "ms": ms_of(p, k)})
    results[model] = {"rows": rows, "Nfeas": len(allfeas), "N": N}

# ---- Table 3 ----
print("\n=== TABLE 3: Multi-beam evaluation (deterministic beam search) ===")
for model, d in results.items():
    print(f"\n{model}  (N={d['N']}, gap over all-beam-feasible |set|={d['Nfeas']})")
    print("| beam | Feas% | Gap% | ms/inst |")
    print("|------|-------|------|---------|")
    for r in d["rows"]:
        print(f"| {r['beam']:>4} | {r['feas']:.1f} | {r['gap']:.3f} | {r['ms']:.0f} |")

# ---- Figure: gap & feas vs beam width ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for model, d in results.items():
    bs = [r["beam"] for r in d["rows"]]
    ax1.plot(bs, [r["gap"] for r in d["rows"]], marker="o", label=model)
    ax2.plot(bs, [r["feas"] for r in d["rows"]], marker="o", label=model)
ax1.set_xscale("log", base=2); ax1.set_xticks(bs); ax1.set_xticklabels(bs)
ax2.set_xscale("log", base=2); ax2.set_xticks(bs); ax2.set_xticklabels(bs)
ax1.set_title("Gap vs beam width"); ax1.set_xlabel("beam width"); ax1.set_ylabel("Gap% (all-beam-feasible set)"); ax1.legend(); ax1.grid(alpha=.3)
ax2.set_title("Feasibility vs beam width"); ax2.set_xlabel("beam width"); ax2.set_ylabel("Feas% (full test set)"); ax2.legend(); ax2.grid(alpha=.3)
fig.tight_layout()
out = "artifacts/vne_phase3_beam_scaling.png"
fig.savefig(out, dpi=120)
print(f"\nsaved figure -> {out}")
with open(f"{SCR}/phase3_table.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"saved -> {SCR}/phase3_table.json")
