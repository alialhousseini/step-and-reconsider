"""Parallel VNE instance generator + ILP labeling (for Phase 4 generalization sets).

Generates instances at a chosen substrate-node scale and labels them with the ILP
optimum via Gurobi (single-threaded per worker; parallelism is across instances).

Usage:
    python scripts/vne_gen_parallel.py --nodes 120 160 --num 40 \
        --out data/vne/vne_gen_2x.pickle --time-limit 120 --mip-gap 0.0 --workers 8
"""
import argparse, os, sys, time, random, pickle
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne.config import VNEConfig
from vne.validation_set_generator import generate_instance, solver_kwargs_from_config, save_dataset

# globals set per worker via initializer
_G = {}

def _init(nlo, nhi, vlo, vhi, rlo, rhi, solver, time_limit, mip_gap, threads):
    cfg = VNEConfig()
    cfg.num_substrate_comm_nodes_range = (nlo, nhi)
    cfg.num_virtual_nodes_range = (vlo, vhi)
    cfg.num_virtual_requests_range = (rlo, rhi)
    _G["cfg"] = cfg
    _G["sk"] = solver_kwargs_from_config(cfg, solver=solver, time_limit_s=time_limit,
                                         mip_gap=mip_gap, threads=threads)

def _gen_one(args):
    idx, seed = args
    cfg, sk = _G["cfg"], _G["sk"]
    t0 = time.time()
    try:
        rng = random.Random(seed + idx)
        inst = generate_instance(cfg, rng, with_solutions=True, solver_kwargs=sk, max_resample=50)
        return (idx, inst, time.time() - t0, None)
    except Exception as e:  # infeasible-after-resamples / solver error
        return (idx, None, time.time() - t0, f"{type(e).__name__}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, nargs=2, required=True, metavar=("LO", "HI"))
    ap.add_argument("--vnodes", type=int, nargs=2, default=[2, 5])
    ap.add_argument("--reqs", type=int, nargs=2, default=[10, 20])
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--mip-gap", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4, help="Gurobi threads per worker")
    ap.add_argument("--seed", type=int, default=4000)
    args = ap.parse_args()

    print(f"[gen] nodes={args.nodes} reqs={args.reqs} num={args.num} solver={args.solver} "
          f"tlim={args.time_limit}s gap={args.mip_gap} workers={args.workers} -> {args.out}", flush=True)
    init_args = (args.nodes[0], args.nodes[1], args.vnodes[0], args.vnodes[1],
                 args.reqs[0], args.reqs[1], args.solver, args.time_limit, args.mip_gap, args.threads)

    t0 = time.time()
    dataset, times, fails = [], [], 0
    with mp.Pool(args.workers, initializer=_init, initargs=init_args) as pool:
        for idx, inst, dt, err in pool.imap_unordered(_gen_one, [(i, args.seed) for i in range(args.num)]):
            if inst is not None:
                dataset.append(inst); times.append(dt)
            else:
                fails += 1
            done = len(dataset) + fails
            avg = sum(times) / max(len(times), 1)
            print(f"  [{done}/{args.num}] kept={len(dataset)} fail={fails} "
                  f"last={dt:.1f}s avg_solve={avg:.1f}s elapsed={time.time()-t0:.0f}s", flush=True)

    save_dataset(args.out, dataset)
    print(f"[gen] DONE {len(dataset)}/{args.num} labeled in {time.time()-t0:.0f}s "
          f"(avg {sum(times)/max(len(times),1):.1f}s/inst) -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
