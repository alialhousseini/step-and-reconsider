"""Assemble a beam-search comparison table (bs8 / bs16) for one architecture group.

Consumes the eval_out/<group>/{eval_sl,eval_gdsil,eval_tasar}.json files produced by
vne_eval_table.py with EVAL_PLAN="bs:8,bs:16". Compares SL / GD-SIL / Ours all under
DETERMINISTIC beam search at matched budgets. gap% is averaged over the group's global
F* (instances feasible under every row at both budgets) so the numbers are orderable.

Usage: python scripts/vne_assemble_bs.py eval_out/bq BQ
"""
import sys, json
import numpy as np

GROUP_DIR = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "BQ"
ROWS = [("SL", "eval_sl.json"), ("GD-SIL", "eval_gdsil.json"), ("Ours", "eval_tasar.json")]
BUDGETS = ["bs k=8", "bs k=16"]

data = {name: json.load(open(f"{GROUP_DIR}/{fn}")) for name, fn in ROWS}
pi = {(name, b): data[name]["per_instance"][b] for name, _ in ROWS for b in BUDGETS}
N = len(next(iter(pi.values())))
Fstar = [i for i in range(N) if all(pi[key][i]["feasible"] for key in pi)]

def feas(p):
    return sum(x["feasible"] for x in p) / N * 100.0

def gap(p, idx):
    g = [(p[i]["ilp"] - p[i]["obj"]) / abs(p[i]["ilp"]) * 100.0
         for i in idx if p[i]["feasible"] and p[i]["ilp"] not in (None, 0)]
    return float(np.mean(g)) if g else float("nan")

print(f"### {LABEL} — beam-search comparison  (N={N}, |F*|={len(Fstar)})")
print("| Method | bs8 Feas% | bs8 Gap%(F*) | bs16 Feas% | bs16 Gap%(F*) |")
print("|--------|-----------|--------------|------------|---------------|")
for name, _ in ROWS:
    p8, p16 = pi[(name, "bs k=8")], pi[(name, "bs k=16")]
    lbl = name + (" *(diverged)*" if name == "GD-SIL" and feas(p16) < 70 else "")
    def cell(v): return "nan" if v != v else f"{v:.2f}"
    print(f"| {lbl} | {feas(p8):.1f} | {cell(gap(p8, Fstar))} | {feas(p16):.1f} | {cell(gap(p16, Fstar))} |")
