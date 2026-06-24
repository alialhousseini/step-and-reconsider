"""Milestone-0 harness for the VNE hot-path optimisation work.

Two jobs:
  * fingerprint  -- deterministic greedy rollout on a FIXED instance; emits a
                    behavioural fingerprint (per-step chosen paths, candidate
                    counts, objective). Used as the EQUIVALENCE GATE: optimised
                    code (FIX A/C/D-safe) must reproduce this byte-for-byte.
  * profile      -- cProfile a real perform_tasar(k,s) run on the same instance;
                    prints wall-clock + top cumulative functions. Used to rank
                    hotspots and measure speedup.

Scale is fixed and moderate (40 comm nodes, 8 requests) so iteration is fast.
Equivalence is scale-independent; the hotspot ranking is representative.

Usage:
    python scripts/vne_hotpath_harness.py fingerprint  out.json
    python scripts/vne_hotpath_harness.py profile
"""
import sys, os, json, random, copy, time, cProfile, pstats, io

import numpy as np
import torch

from vne.config import VNEConfig
from vne.instance_generator import make_instance
from vne.trajectory import Trajectory
from vne_main import get_network
import core.stochastic_beam_search as sbs
from core.incremental_sbs import IncrementalSBS

CKPT = "./model_checkpoints/vne/results/phase1_bq128_42476/last_model.pt"
SEED = 12345


def build_config():
    cfg = VNEConfig()
    cfg.architecture = "bq"
    cfg.embedding_dim = 128
    cfg.latent_dimension = 128
    cfg.num_attention_heads = 8
    cfg.feedforward_dimension = 512
    # fixed scale for fast, deterministic iteration; override via env for real scale
    nodes = int(os.environ.get("HARNESS_NODES", "40"))
    reqs = int(os.environ.get("HARNESS_REQS", "8"))
    cfg.num_substrate_comm_nodes_range = (nodes, nodes)
    cfg.num_virtual_requests_range = (reqs, reqs)
    cfg.num_virtual_nodes_range = (2, 5)
    return cfg


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_fixed_instance(cfg):
    seed_everything()
    return make_instance(cfg)


def load_network(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_network(cfg, dev)
    ck = torch.load(CKPT, map_location=dev)
    net.load_state_dict(ck["model_weights"])
    net.to(net.device)
    net.eval()
    return net


@torch.no_grad()
def greedy_rollout(instance, net):
    """Fully deterministic: argmax of the policy each step. Returns a fingerprint."""
    traj = Trajectory.init_batch_from_instance_list([copy.deepcopy(instance)], net, net.device)[0]
    steps = []
    step0_candidates = [list(p) for p in traj.action_candidates]
    guard = 0
    while traj.num_actions() > 0 and guard < 10000:
        logps = Trajectory.log_probability_fn([traj], net, to_numpy=True)[0]
        a = int(np.argmax(logps))
        chosen = traj.action_candidates[a]
        steps.append({
            "req": traj.current_request_idx,
            "link": traj.current_link_idx,
            "n_cand": traj.num_actions(),
            "chosen": list(chosen),
        })
        traj, finished = traj.transition_fn(a)
        guard += 1
        if finished:
            break
    return {
        "objective": round(float(traj.objective), 6),
        "n_steps": len(steps),
        "step0_n_candidates": len(step0_candidates),
        "step0_candidates": step0_candidates,
        "steps": steps,
    }


def cmd_fingerprint(out_path):
    cfg = build_config()
    instance = make_fixed_instance(cfg)
    net = load_network(cfg)
    fp = greedy_rollout(instance, net)
    with open(out_path, "w") as f:
        json.dump(fp, f)
    print(f"fingerprint: objective={fp['objective']} n_steps={fp['n_steps']} "
          f"step0_candidates={fp['step0_n_candidates']} -> {out_path}")


def cmd_profile():
    cfg = build_config()
    instance = make_fixed_instance(cfg)
    net = load_network(cfg)

    def child_log_probability_fn(trajs):
        return Trajectory.log_probability_fn(trajectories=trajs, network=net, to_numpy=True)

    def child_transition_fn(pairs):
        return [t.transition_fn(a) for t, a in pairs]

    def run_once():
        roots = Trajectory.init_batch_from_instance_list([copy.deepcopy(instance)], net, net.device)
        inc = IncrementalSBS(roots, child_log_probability_fn, child_transition_fn,
                             Trajectory.to_max_evaluation_fn, memory_aggressive=False)
        return inc.perform_tasar(beam_width=8, deterministic=False, nucleus_top_p=1.0, replan_steps=4)

    # warm up CUDA / caches once (not timed)
    seed_everything(); run_once()

    seed_everything()
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run_once()
    pr.disable()
    wall = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(f"\n=== perform_tasar(k=8,s=4) wall-clock: {wall:.2f}s for ONE instance ===\n")
    # also tottime ranking (self time) - best for spotting pure-python hotspots
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(15)
    print("--- TOP BY SELF TIME (tottime) ---")
    print(s2.getvalue())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "profile"
    if mode == "fingerprint":
        cmd_fingerprint(sys.argv[2] if len(sys.argv) > 2 else "fingerprint.json")
    else:
        cmd_profile()
