"""Generate UNLABELED VNE instances at a chosen substrate scale (no ILP solver).

For feasibility-generalization tests: feas% only needs the model + a feasibility-
possible instance, not an ILP label. make_instance() ensures a feasible embedding
exists (fast greedy check), so feas% is a fair test. Requests keep the trained
10-20 range; only substrate node count scales.

Usage: python scripts/vne_gen_unlabeled.py --nodes 240 --num 16 --out path.pickle
"""
import argparse, os, sys, time, pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vne.config import VNEConfig
from vne.instance_generator import make_instance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--reqs", type=int, nargs=2, default=[10, 20])
    ap.add_argument("--vnodes", type=int, nargs=2, default=[2, 5])
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = VNEConfig()
    cfg.num_substrate_comm_nodes_range = (args.nodes, args.nodes)
    cfg.num_virtual_requests_range = (args.reqs[0], args.reqs[1])
    cfg.num_virtual_nodes_range = (args.vnodes[0], args.vnodes[1])

    t0 = time.time()
    insts = [make_instance(cfg) for _ in range(args.num)]
    pickle.dump(insts, open(args.out, "wb"))
    print(f"[gen-unlabeled] {args.num}x {args.nodes}-node instances in {time.time()-t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
