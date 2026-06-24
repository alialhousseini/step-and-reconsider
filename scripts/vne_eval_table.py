"""Phase 2 results table for VNE — paper-aligned (TaSaR Table-1 analogue).

Per the paper-alignment review, this evaluator:
  * runs TaSaR with a real Top-p (default 0.9), NOT p=1.0 (which is WOR-SBS);
  * collects PER-INSTANCE results so gap% can be averaged over F* = the set of
    instances feasible under EVERY method (fixed denominator => orderable),
    instead of each method's own feasible subset (non-comparable);
  * reports the compute budget two ways: measured wall-clock (dispute-proof)
    and node transitions g(k,s) (paper Eq. 3), where smaller s => MORE compute;
  * excludes instances with ilp_obj == 0 from the gap mean (counts them).

Budgets evaluated: greedy (g=l), BS k (g=k*l), TaSaR (k,s) (g=k*(t*l-(s*t^2-s*t)/2), t=ceil(l/s)).

Usage:
    python scripts/vne_eval_table.py CKPT ARCH N_INSTANCES [TOPP]
Env: EVAL_TEST_PATH, EVAL_OUT, EVAL_LABEL (group label for the table).
"""
import sys, os, json, time, copy, math, pickle

import numpy as np
import torch

from vne.config import VNEConfig
from vne.trajectory import Trajectory as VNETrajectory, _requests
from vne_main import get_network
import core.stochastic_beam_search as sbs
from core.incremental_sbs import IncrementalSBS


def build_config(arch):
    cfg = VNEConfig()
    cfg.architecture = arch
    cfg.embedding_dim = 128
    cfg.latent_dimension = 128
    cfg.num_attention_heads = 8
    cfg.feedforward_dimension = 512
    return cfg


def load_network(cfg, ckpt_path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_network(cfg, dev)
    ck = torch.load(ckpt_path, map_location=dev)
    net.load_state_dict(ck["model_weights"])
    net.to(net.device)
    net.eval()
    return net


def solution_length(inst):
    """l = number of virtual-link assignment decisions to fully solve the instance."""
    return sum(r["num_processing_nodes"] - 1 for r in _requests(inst))


def node_transitions(mode, k, s, l):
    """Paper Eq. 3 compute budget. greedy=BS with k=1."""
    if mode == "bs":
        return k * l
    t = math.ceil(l / s)
    return k * (t * l - (s * t * t - s * t) / 2.0)


@torch.no_grad()
def run_config(cfg, net, instances, mode, k, s, top_p):
    device = net.device

    def child_log_prob(trajs):
        return VNETrajectory.log_probability_fn(trajectories=trajs, network=net, to_numpy=True)

    def child_transition(pairs):
        return [t.transition_fn(a) for t, a in pairs]

    eval_batch = 8 if len(instances) >= 128 else 4
    per_inst = []  # (feasible, model_obj, ilp_obj, l, g)
    t0 = time.time()
    for start in range(0, len(instances), eval_batch):
        batch = instances[start:start + eval_batch]
        roots = VNETrajectory.init_batch_from_instance_list(
            instances=[copy.deepcopy(b) for b in batch], network=net, device=device)
        if mode == "bs":
            leaves = sbs.stochastic_beam_search(
                child_log_probability_fn=child_log_prob, child_transition_fn=child_transition,
                root_states=roots, beam_width=k, deterministic=True)
        else:
            inc = IncrementalSBS(roots, child_log_prob, child_transition,
                                 VNETrajectory.to_max_evaluation_fn, memory_aggressive=False)
            leaves = inc.perform_tasar(beam_width=k, deterministic=False,
                                       nucleus_top_p=top_p, replan_steps=s)
        for i, inst in enumerate(batch):
            best = max((lf.state for lf in leaves[i]), key=VNETrajectory.to_max_evaluation_fn)
            obj = VNETrajectory.to_max_evaluation_fn(best)
            l = solution_length(inst)
            per_inst.append({
                "feasible": bool(obj > float("-inf")),
                "obj": float(obj) if obj > float("-inf") else None,
                "ilp": inst.get("objective"),
                "l": l,
                "g": node_transitions(mode, k, s, l),
            })
    wall = time.time() - t0
    return per_inst, wall


def main():
    ckpt = sys.argv[1]
    arch = sys.argv[2] if len(sys.argv) > 2 else "bq"
    n_inst = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    top_p = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
    test_path = os.environ.get("EVAL_TEST_PATH", "data/vne/vne_test_dataset_2k.pickle")
    label = os.environ.get("EVAL_LABEL", arch.upper())

    cfg = build_config(arch)
    net = load_network(cfg, ckpt)
    with open(test_path, "rb") as f:
        instances = pickle.load(f)[:n_inst]
    N = len(instances)

    # EVAL_PLAN env: comma-separated configs, e.g. "greedy,bs:16,tasar:8:4,tasar:16:4".
    # Default = full set.
    def parse_plan(spec):
        out = []
        for tok in spec.split(","):
            tok = tok.strip()
            if tok == "greedy":
                out.append(("greedy", "bs", 1, None))
            elif tok.startswith("bs:"):
                k = int(tok.split(":")[1]); out.append((f"bs k={k}", "bs", k, None))
            elif tok.startswith("tasar:"):
                _, k, s = tok.split(":"); out.append((f"TaSaR k={k} s={s}", "tasar", int(k), int(s)))
        return out

    default_plan = "greedy,bs:8,bs:16,tasar:8:2,tasar:8:4,tasar:16:4"
    plan = parse_plan(os.environ.get("EVAL_PLAN", default_plan))

    results = {}
    for name, mode, k, s in plan:
        print(f"[eval] {name} (n={N}, top_p={top_p if mode=='tasar' else '-'}) ...", flush=True)
        per_inst, wall = run_config(cfg, net, instances, mode, k, s, top_p)
        results[name] = {"per_inst": per_inst, "wall_s": wall, "mode": mode, "k": k, "s": s}
        feas = sum(p["feasible"] for p in per_inst) / N * 100.0
        print(f"    -> feas={feas:.1f}%  wall={wall:.0f}s  ({wall/N*1000:.0f} ms/inst)", flush=True)

    # F* = instances feasible under EVERY method (fixed denominator for gap%)
    fstar = [i for i in range(N)
             if all(results[name]["per_inst"][i]["feasible"] for name in results)]

    def gap_on(indices, per_inst):
        gaps = []
        for i in indices:
            p = per_inst[i]
            if p["feasible"] and p["ilp"] not in (None, 0):
                gaps.append((p["ilp"] - p["obj"]) / abs(p["ilp"]) * 100.0)
        return float(np.mean(gaps)) if gaps else float("nan")

    rows = []
    for name, mode, k, s in plan:
        r = results[name]
        pis = r["per_inst"]
        feas = sum(p["feasible"] for p in pis) / N * 100.0
        rows.append({
            "method": name,
            "feas_pct": feas,
            "gap_Fstar_pct": gap_on(fstar, pis),     # fixed denominator -> orderable
            "gap_own_pct": gap_on(range(N), pis),    # over own feasible set (reference)
            "mean_g": float(np.mean([p["g"] for p in pis])),
            "wall_s": r["wall_s"],
            "ms_per_inst": r["wall_s"] / N * 1000.0,
        })

    # markdown table; bold feasibility-first, gap on F*
    print(f"\n\n=== VNE Phase 2 results: group {label} ===")
    print(f"ckpt={ckpt}  N={N}  |F*|={len(fstar)}  top_p={top_p}")
    print("\n| Method | Feas% | Gap%(F*) | g(k,s) | Time set(s) | ms/inst |")
    print("|---|---|---|---|---|---|")
    best_feas = max(r["feas_pct"] for r in rows)
    best_gap = min((r["gap_Fstar_pct"] for r in rows if r["gap_Fstar_pct"] == r["gap_Fstar_pct"]), default=None)
    for r in rows:
        f = f"**{r['feas_pct']:.1f}**" if r["feas_pct"] == best_feas else f"{r['feas_pct']:.1f}"
        g = r["gap_Fstar_pct"]
        gs = "nan" if g != g else (f"**{g:.3f}**" if g == best_gap else f"{g:.3f}")
        print(f"| {r['method']} | {f} | {gs} | {r['mean_g']:.0f} | {r['wall_s']:.0f} | {r['ms_per_inst']:.0f} |")

    out = os.environ.get("EVAL_OUT", "eval_table.json")
    with open(out, "w") as f:
        json.dump({"ckpt": ckpt, "label": label, "N": N, "Fstar_size": len(fstar),
                   "top_p": top_p, "rows": rows,
                   "per_instance": {k: results[k]["per_inst"] for k in results}}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
