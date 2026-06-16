"""Validation-set pickle generator for the extended VNE problem.

Schema follows `vne/PROBLEM_FORMULATION.md` (the source of truth):

- Virtual nodes carry NO information. All demand lives on links.
- A virtual chain is `S -> F_0 -> F_1 -> ... -> F_{k-1} -> D` with three
  link types:
    * one source link (S -> F_0)            -- placed on a comp link at F_0
    * (k-1) processing links (F_i -> F_{i+1}) -- placed on directed comm paths
    * one destination link (F_{k-1} -> D)   -- placed on a comp link at F_{k-1}
- Comp links are paired (C_i <-> P_i share one physical resource), so a
    reservation deducts from both directions at once.
- One problem instance contains one substrate and a sampled number of static
  virtual requests sharing the same substrate resources.

CLI entry point:

    python -m vne.validation_set_generator --num-instances 64 \
        --seed 1234 --out data/vne/vne_val.pickle
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import pulp
from tqdm import tqdm

from vne.config import VNEConfig

EdgeInt = Tuple[int, int]


# Preference order used by the "auto" choice: fast/free backends first, then any
# licensed commercial backend, then the always-available CBC fallback. Reordered
# so generation runs anywhere (no license) yet uses a faster solver when present.
_AUTO_ORDER = ["highs", "gurobi", "cplex", "cbc"]


def _concrete_solver(name: str, time_limit_s: int, threads: int, mip_gap: float):
    """Construct a PuLP solver backend by name, or return None if unavailable.

    `name` in {"highs", "gurobi", "cplex", "cbc"}.
      - highs : open-source (highspy); no license, runs on every node.
      - gurobi: gurobipy bindings; needs a license covering the model size.
      - cplex : IBM CPLEX via `cplex` binary on PATH (full/academic install).
      - cbc   : bundled fallback, always available.
    `threads=0` and `mip_gap=0.0` let the backend use its defaults / prove
    optimality. A small `mip_gap` (e.g. 0.01) trades a tiny optimality margin
    for large speedups on symmetric instances.
    """
    # Build kwargs dict without None values — PuLP passes them directly to
    # the solver's setParam, and Gurobi rejects None for typed parameters.
    kwargs: dict = {"msg": False, "timeLimit": time_limit_s}
    if threads and threads > 0:
        kwargs["threads"] = threads
    if mip_gap and mip_gap > 0:
        kwargs["gapRel"] = mip_gap

    if name == "highs":
        try:
            solver = pulp.HiGHS(**kwargs)
        except Exception:
            return None
        return solver if solver.available() else None
    if name == "gurobi":
        try:
            kwargs["msg"] = 0  # Gurobi uses int, not bool
            solver = pulp.GUROBI(**kwargs)
        except Exception:
            return None
        return solver if solver.available() else None
    if name == "cplex":
        try:
            solver = pulp.CPLEX_CMD(**kwargs)
        except Exception:
            return None
        return solver if solver.available() else None
    if name == "cbc":
        return pulp.PULP_CBC_CMD(**kwargs)
    raise ValueError(f"Unknown solver backend: {name}")


def _solve_with_fallback(prob, *, solver: str, time_limit_s: int, threads: int, mip_gap: float = 0.0):
    """Solve `prob`, honoring `solver` in {"auto", "highs", "gurobi", "cplex", "cbc"}.

    "auto" tries fast/free backends first (HiGHS), then any licensed commercial
    backend, then CBC. A backend that is unavailable, unlicensed, or rejects the
    model (e.g. size-limited license) is skipped. Every backend returns the exact
    ILP optimum when it reaches "Optimal" (or within `mip_gap` if set), so the
    fallback does not degrade label quality.
    """
    order = {
        "auto": _AUTO_ORDER,
        "highs": ["highs"],
        "gurobi": ["gurobi"],
        "cplex": ["cplex"],
        "cbc": ["cbc"],
    }.get(solver)
    if order is None:
        raise ValueError(f"Unknown solver choice: {solver!r} (use auto|highs|gurobi|cplex|cbc)")

    last_error: Exception | None = None
    for name in order:
        backend = _concrete_solver(name, time_limit_s, threads, mip_gap)
        if backend is None:
            continue
        try:
            return prob.solve(backend)
        except Exception as exc:  # e.g. GurobiError: model too large for license
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No usable MILP solver available (tried: %s)" % ", ".join(order))


def _requests(instance: Dict) -> List[Dict]:
    if "requests" in instance:
        return instance["requests"]
    return [instance["request"]]


def generate_substrate(
    num_comm_nodes: int,
    edge_prob: float,
    attach_prob: float,
    bw_range: Tuple[int, int],
    cap_range: Tuple[int, int],
    rng: random.Random,
    *,
    topology: str,
) -> Dict:
    """Sample a substrate network G(V=C u P, E=E_comm u E_comp).

    `topology="er"`   - directed Erdős-Rényi over the comm nodes.
    `topology="line"` - deterministic line graph (i, i+1).

    `compute_attachment` maps comm-node id -> comp-node id.
    """
    if num_comm_nodes < 2:
        raise ValueError("num_comm_nodes must be >= 2")
    if topology == "line":
        edges: List[EdgeInt] = [(i, i + 1) for i in range(num_comm_nodes - 1)]
    elif topology == "er":
        edges = [
            (i, j)
            for i in range(num_comm_nodes)
            for j in range(num_comm_nodes)
            if i != j and rng.random() < edge_prob
        ]
    else:
        raise ValueError(f"Unknown topology: {topology}")

    bandwidth = {edge: rng.randint(bw_range[0], bw_range[1]) for edge in edges}

    attachment: Dict[int, int] = {}
    capacity: Dict[int, int] = {}
    for c in range(num_comm_nodes):
        if rng.random() < attach_prob:
            comp_id = c
            attachment[c] = comp_id
            capacity[comp_id] = rng.randint(cap_range[0], cap_range[1])

    return {
        "num_comm_nodes": num_comm_nodes,
        "communication_edges": edges,
        "communication_bandwidth": bandwidth,
        "compute_attachment": attachment,
        "compute_capacity": capacity,
    }


def generate_request(
    num_requests_range: Tuple[int, int],
    num_virtual_nodes_range: Tuple[int, int],
    capdem_range: Tuple[int, int],
    bwdem_range: Tuple[int, int],
    rng: random.Random,
) -> List[Dict]:
    """Sample static chain virtual requests from config-driven size ranges.

    Each request has k processing nodes (F_0..F_{k-1}) and exactly one source
    link plus one destination link. The number of requests is sampled once per
    instance; each request's k is sampled independently.
    """
    request_low, request_high = num_requests_range
    node_low, node_high = num_virtual_nodes_range
    if request_low < 1 or request_high < request_low:
        raise ValueError("num_virtual_requests_range must be ordered and have lower bound >= 1")
    if node_low < 2 or node_high < node_low:
        raise ValueError("num_virtual_nodes_range must be ordered and have lower bound >= 2")

    requests: List[Dict] = []
    for _ in range(rng.randint(request_low, request_high)):
        num_processing_nodes = rng.randint(node_low, node_high)
        chain_length = num_processing_nodes - 1
        requests.append(
            {
                "num_processing_nodes": num_processing_nodes,
                "source_link_demand": rng.randint(capdem_range[0], capdem_range[1]),
                "destination_link_demand": rng.randint(capdem_range[0], capdem_range[1]),
                "processing_link_demands": [
                    rng.randint(bwdem_range[0], bwdem_range[1])
                    for _ in range(chain_length)
                ],
            }
        )
    return requests


def _request_total_demand(source_dem: int, dest_dem: int, pl_dems: List[int]) -> int:
    """Total link demand of a request: source + destination + all processing demands."""
    return source_dem + dest_dem + sum(pl_dems)


def _build_and_solve_ilp(
    instance: Dict,
    *,
    revenue: float,
    cost_comm_per_unit: float,
    cost_comp_per_unit: float,
    time_limit_s: int,
    solver: str = "auto",
    threads: int = 0,
    mip_gap: float = 0.0,
    enable_admission: bool = True,
    revenue_per_demand_unit: float = 1.0,
    objective_mode: str = "lex",
) -> Tuple[
    Optional[List[List[int]]],
    Optional[List[List[List[int]]]],
    Optional[float],
    Optional[List[bool]],
]:
    """Build the MILP from PROBLEM_FORMULATION.md and solve it.

    Returns nested (f_placements, processing_paths, objective, accepted) on
    success, or (None, None, None, None) if infeasible or the solver did not
    reach optimality. The outer solution dimension is request index.

    Admission control (``enable_admission=True``, the default and the model in
    vne_README.md:357): a per-request binary ``accept[r]`` gates placement, so
    the solver may reject a request whose embedding is unprofitable or
    impossible. The objective becomes ``sum_r revenue_r * accept[r] - costs``,
    with ``revenue_r = revenue + revenue_per_demand_unit * total_demand(r)``,
    so revenue scales with the resources a request asks for (the standard
    Yu-et-al. revenue model). A rejected request reserves nothing and gets an
    empty placement/path list. With ``enable_admission=False`` every request is
    forced (``sum_c f == 1``) and an unplaceable request makes the instance
    infeasible (the legacy behaviour).

    Comp reservations apply only at F_0 (source link) and F_{k-1} (destination
    link). Intermediate processing nodes F_1..F_{k-2} are placed at comm nodes
    (so the chained processing paths line up) but consume no comp resource.
    All virtual requests share the same substrate bandwidth and compute
    capacity constraints.
    """
    substrate = instance["substrate"]
    requests = _requests(instance)

    num_comm: int = substrate["num_comm_nodes"]
    edges: List[EdgeInt] = list(substrate["communication_edges"])
    bw: Dict[EdgeInt, int] = substrate["communication_bandwidth"]
    cap: Dict[int, int] = substrate["compute_capacity"]
    attach: Dict[int, int] = substrate["compute_attachment"]

    num_requests = len(requests)
    ks: List[int] = [request["num_processing_nodes"] for request in requests]
    source_dems: List[int] = [request["source_link_demand"] for request in requests]
    dest_dems: List[int] = [request["destination_link_demand"] for request in requests]
    pl_dems: List[List[int]] = [
        list(request["processing_link_demands"])
        for request in requests
    ]
    for r, k in enumerate(ks):
        assert len(pl_dems[r]) == k - 1, "processing_link_demands length must be k-1"

    prob = pulp.LpProblem("VNE", pulp.LpMaximize)

    # f[r, i, c] = 1 iff request r's processing node F_i is placed at comm node c.
    # F_0 requires an attached comp node with capacity >= source_dem.
    # F_{k-1} requires an attached comp node with capacity >= dest_dem.
    # Intermediate F's can sit on any comm node (no comp reservation).
    f: Dict[Tuple[int, int, int], Optional[pulp.LpVariable]] = {}
    for r in range(num_requests):
        k = ks[r]
        for i in range(k):
            for c in range(num_comm):
                allowed = True
                if i == 0:
                    allowed = c in attach and cap.get(attach[c], 0) >= source_dems[r]
                if i == k - 1:
                    allowed = allowed and (
                        c in attach and cap.get(attach[c], 0) >= dest_dems[r]
                    )
                f[r, i, c] = (
                    pulp.LpVariable(f"f_{r}_{i}_{c}", cat="Binary")
                    if allowed
                    else None
                )

    # accept[r] = 1 iff request r is embedded. Without admission it is the
    # constant 1, reducing the placement rule to the legacy "place everything".
    accept: Dict[int, object] = {}
    for r in range(num_requests):
        accept[r] = (
            pulp.LpVariable(f"accept_{r}", cat="Binary") if enable_admission else 1
        )

    # Place every node of an accepted request exactly once: sum_c f == accept[r].
    for r in range(num_requests):
        for i in range(ks[r]):
            terms = [f[r, i, c] for c in range(num_comm) if f[r, i, c] is not None]
            if not terms:
                if enable_admission:
                    # No legal host for this node -> the request cannot be accepted.
                    prob += accept[r] == 0, f"reject_{r}_{i}"
                    continue
                return None, None, None, None
            prob += pulp.lpSum(terms) == accept[r], f"place_{r}_{i}"

    # Comp-capacity at attached comm nodes: source_dem at F_0, dest_dem at F_{k-1}.
    # (Both directions of the comp link share one physical resource per PROBLEM_FORMULATION.)
    for c, comp_id in attach.items():
        terms: List = []
        for r in range(num_requests):
            if f[r, 0, c] is not None:
                terms.append(source_dems[r] * f[r, 0, c])
            if f[r, ks[r] - 1, c] is not None:
                terms.append(dest_dems[r] * f[r, ks[r] - 1, c])
        if terms:
            prob += pulp.lpSum(terms) <= cap[comp_id], f"cap_{c}"

    # y[r, ell, e] = 1 iff request r's processing link ell uses comm edge e.
    y: Dict[Tuple[int, int, EdgeInt], pulp.LpVariable] = {}
    for r in range(num_requests):
        for ell in range(ks[r] - 1):
            for e in edges:
                y[r, ell, e] = pulp.LpVariable(
                    f"y_{r}_{ell}_{e[0]}_{e[1]}", cat="Binary"
                )

    out_edges = {c: [e for e in edges if e[0] == c] for c in range(num_comm)}
    in_edges = {c: [e for e in edges if e[1] == c] for c in range(num_comm)}

    # Flow conservation per processing link.
    # The directed path of link ell goes from F_ell's location to F_{ell+1}'s location.
    for r in range(num_requests):
        for ell in range(ks[r] - 1):
            for c in range(num_comm):
                outflow = pulp.lpSum(y[r, ell, e] for e in out_edges[c])
                inflow = pulp.lpSum(y[r, ell, e] for e in in_edges[c])
                src = f[r, ell, c] if f[r, ell, c] is not None else 0
                dst = f[r, ell + 1, c] if f[r, ell + 1, c] is not None else 0
                prob += outflow - inflow == src - dst, f"flow_{r}_{ell}_{c}"
                # Path length >= 1 (a processing link must follow at least one comm hop).
                if f[r, ell, c] is not None and f[r, ell + 1, c] is not None:
                    prob += (
                        f[r, ell, c] + f[r, ell + 1, c] <= 1
                    ), f"distinct_{r}_{ell}_{c}"

    # Aggregate bandwidth on each comm edge.
    for e in edges:
        terms = [
            pl_dems[r][ell] * y[r, ell, e]
            for r in range(num_requests)
            for ell in range(ks[r] - 1)
        ]
        if terms:
            prob += pulp.lpSum(terms) <= bw[e], f"bw_{e[0]}_{e[1]}"

    cost_comm_term = pulp.lpSum(
        cost_comm_per_unit * pl_dems[r][ell] * y[r, ell, e]
        for r in range(num_requests)
        for ell in range(ks[r] - 1)
        for e in edges
    )
    # Comp cost: source_dem + dest_dem reserved exactly once per accepted request.
    cost_comp_term = pulp.lpSum(
        cost_comp_per_unit * (source_dems[r] + dest_dems[r]) * accept[r]
        for r in range(num_requests)
    )
    cost_term = cost_comm_term + cost_comp_term
    accept_term = pulp.lpSum(accept[r] for r in range(num_requests))

    if objective_mode == "lex":
        # Lexicographic "maximize acceptance, then minimize cost" as a SINGLE
        # MILP via a big-M acceptance weight W chosen just above the maximum
        # achievable routing cost, so accepting one more request always beats any
        # cost saving (exact lexicographic order). A per-instance tight W (vs a
        # fixed huge constant) keeps the LP relaxation tight and the solve fast,
        # and the cost term inside the same objective guides the search far better
        # than a separate two-stage acceptance solve.
        #   acceptance stays exact under mip_gap as long as 1/max_accept > mip_gap
        #   (one fewer acceptance costs ~W, i.e. > mip_gap of the objective);
        #   the gap is absorbed by the (secondary) routing cost, which is fine for
        #   training labels.
        max_cost_bound = sum(
            pl_dems[r][ell] * (num_comm - 1) * cost_comm_per_unit
            for r in range(num_requests)
            for ell in range(ks[r] - 1)
        ) + sum(
            (source_dems[r] + dest_dems[r]) * cost_comp_per_unit
            for r in range(num_requests)
        )
        big_w = float(max_cost_bound) + 1.0
        prob.sense = pulp.LpMaximize
        prob.setObjective(big_w * accept_term - cost_term)
    elif objective_mode == "profit":
        # Single revenue-minus-cost objective: the solver drops unprofitable
        # requests (revenue scales with demand).
        revenue_term = pulp.lpSum(
            (
                revenue
                + revenue_per_demand_unit
                * _request_total_demand(source_dems[r], dest_dems[r], pl_dems[r])
            )
            * accept[r]
            for r in range(num_requests)
        )
        prob.sense = pulp.LpMaximize
        prob.setObjective(revenue_term - cost_term)
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode!r} (use 'lex' or 'profit')")

    status = _solve_with_fallback(prob, solver=solver, time_limit_s=time_limit_s, threads=threads, mip_gap=mip_gap)

    # Accept a proven-optimal solution OR the best feasible incumbent found
    # before a timeout. Keeping the incumbent means hard instances are never
    # silently discarded and resampled, so there is no timeout-selection bias
    # (the labels for those few are within the solver's gap of optimal).
    feasible_sol_states = (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible)
    if getattr(prob, "sol_status", None) not in feasible_sol_states:
        # Fall back to the classic status for backends that don't set sol_status.
        if pulp.LpStatus[status] != "Optimal":
            return None, None, None, None

    accepted: List[bool] = [
        (not enable_admission) or (pulp.value(accept[r]) > 0.5)
        for r in range(num_requests)
    ]

    all_f_placements: List[List[int]] = []
    all_processing_paths: List[List[List[int]]] = []
    for r in range(num_requests):
        if not accepted[r]:
            # Rejected request: no placement, no path, no reservation.
            all_f_placements.append([])
            all_processing_paths.append([])
            continue

        f_placements: List[int] = []
        for i in range(ks[r]):
            chosen_c = next(
                c for c in range(num_comm)
                if f[r, i, c] is not None and pulp.value(f[r, i, c]) > 0.5
            )
            f_placements.append(chosen_c)

        processing_paths: List[List[int]] = []
        for ell in range(ks[r] - 1):
            used: List[EdgeInt] = [
                e for e in edges
                if pulp.value(y[r, ell, e]) > 0.5
            ]
            start = f_placements[ell]
            end = f_placements[ell + 1]
            outgoing = {e[0]: e for e in used}
            path = [start]
            cursor = start
            while cursor != end:
                if cursor not in outgoing:
                    return None, None, None, None
                nxt = outgoing[cursor][1]
                path.append(nxt)
                cursor = nxt
            processing_paths.append(path)

        all_f_placements.append(f_placements)
        all_processing_paths.append(processing_paths)

    if objective_mode == "profit":
        # Maximize-style profit (revenue - cost).
        objective_value = float(pulp.value(prob.objective))
    else:
        # "lex": store -cost so the convention stays "higher is better" (matches
        # Trajectory.to_max_evaluation_fn, which also returns -cost). The big-M
        # acceptance term is excluded from the stored scalar; acceptance is
        # reported separately via `accepted`.
        objective_value = -float(pulp.value(cost_term))
    return all_f_placements, all_processing_paths, objective_value, accepted


def solve_instance_ilp(
    instance: Dict,
    *,
    time_limit_s: int = 30,
    revenue: Optional[float] = None,
    cost_comm_per_unit: float = 1.0,
    cost_comp_per_unit: float = 0.0,
    solver: str = "auto",
    threads: int = 0,
    mip_gap: float = 0.0,
    enable_admission: bool = True,
    revenue_per_demand_unit: float = 1.0,
    objective_mode: str = "lex",
) -> Dict:
    """Solve the embedding ILP and write the solution fields in place.

    Adds:
        - `f_placements`      : list[list[int]], one placement list per request
                                (empty for a rejected request)
        - `processing_paths`  : list[list[list[int]]], one path-list per request
                                (empty for a rejected request)
        - `accepted`          : list[bool], whether each request was embedded
        - `objective`         : ILP optimum (revenue of accepted - costs)
        - `chosen_path`       : tuple[int, int] alias, only for legacy one-request k=2

    With `enable_admission=True` (default) the solver chooses which requests to
    embed to maximize revenue minus cost; `revenue` here is a flat per-request
    term added on top of the demand-proportional `revenue_per_demand_unit`.
    """
    if revenue is None:
        revenue = 0.0
    f_placements, processing_paths, objective, accepted = _build_and_solve_ilp(
        instance,
        revenue=revenue,
        cost_comm_per_unit=cost_comm_per_unit,
        cost_comp_per_unit=cost_comp_per_unit,
        time_limit_s=time_limit_s,
        solver=solver,
        threads=threads,
        mip_gap=mip_gap,
        objective_mode=objective_mode,
        enable_admission=enable_admission,
        revenue_per_demand_unit=revenue_per_demand_unit,
    )
    if f_placements is None:
        raise RuntimeError("ILP did not find an optimal solution (infeasible or timed out)")

    instance["f_placements"] = [list(placements) for placements in f_placements]
    instance["processing_paths"] = [
        [tuple(path) for path in request_paths]
        for request_paths in processing_paths
    ]
    instance["accepted"] = list(accepted)
    instance["objective"] = objective
    instance.pop("chosen_path", None)
    requests = _requests(instance)
    # Single-link back-compat: a processing link's (start, end) pair.
    if (
        len(requests) == 1
        and requests[0]["num_processing_nodes"] == 2
        and accepted[0]
    ):
        instance["chosen_path"] = (f_placements[0][0], f_placements[0][1])
    return instance


def solver_kwargs_from_config(config: VNEConfig, **overrides) -> Dict:
    """Build the `solve_instance_ilp` kwargs from a VNEConfig (single source of truth).

    Threads the MILP backend choice and the admission/revenue/cost model from the
    config so every generation entry point (vne_main preflight, the SLURM shard
    script, the CLI) produces labels with identical semantics. `overrides` win,
    e.g. to force a backend or time limit per SLURM task.
    """
    kwargs = {
        "time_limit_s": config.validation_solver_time_limit,
        "solver": getattr(config, "validation_solver", "highs"),
        "threads": getattr(config, "validation_solver_threads", 0),
        "mip_gap": getattr(config, "validation_solver_mip_gap", 0.0),
        "objective_mode": getattr(config, "validation_objective", "lex"),
        "enable_admission": getattr(config, "enable_admission", True),
        "revenue": getattr(config, "validation_revenue_per_request", 0.0),
        "revenue_per_demand_unit": getattr(config, "validation_revenue_per_demand_unit", 1.0),
        "cost_comm_per_unit": getattr(config, "validation_cost_comm_per_unit", 1.0),
        "cost_comp_per_unit": getattr(config, "validation_cost_comp_per_unit", 0.0),
    }
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def generate_instance(
    config: VNEConfig,
    rng: random.Random,
    *,
    with_solutions: bool,
    solver_kwargs: Optional[Dict] = None,
    max_resample: int = 50,
) -> Dict:
    """Resample one substrate plus static VN requests until feasible if solved."""
    solver_kwargs = solver_kwargs or {}
    substrate_low, substrate_high = config.num_substrate_comm_nodes_range
    if substrate_low < 2 or substrate_high < substrate_low:
        raise ValueError("num_substrate_comm_nodes_range must be ordered and have lower bound >= 2")
    for _ in range(max_resample):
        instance = {
            "substrate": generate_substrate(
                num_comm_nodes=rng.randint(substrate_low, substrate_high),
                edge_prob=config.substrate_edge_probability,
                attach_prob=config.substrate_compute_attach_probability,
                bw_range=config.substrate_communication_bandwidth_range,
                cap_range=config.substrate_compute_capacity_range,
                rng=rng,
                topology=config.substrate_topology,
            ),
            "requests": generate_request(
                num_requests_range=config.num_virtual_requests_range,
                num_virtual_nodes_range=config.num_virtual_nodes_range,
                capdem_range=config.virtual_compute_demand_range,
                bwdem_range=config.virtual_communication_demand_range,
                rng=rng,
            ),
        }
        if not with_solutions:
            return instance
        try:
            solved = solve_instance_ilp(instance, **solver_kwargs)
        except RuntimeError:
            continue
        # With admission the ILP is always feasible (it can reject everything);
        # skip instances where nothing was embedded so labels are non-trivial.
        if not any(solved.get("accepted", [True])):
            continue
        return solved
    raise RuntimeError(
        f"Could not generate a feasible instance after {max_resample} resamples; "
        "consider widening capacity/bandwidth ranges or lowering demands."
    )


def make_validation_dataset(
    num_instances: int,
    config: VNEConfig,
    *,
    with_solutions: bool = True,
    solver_kwargs: Optional[Dict] = None,
    seed: int = 0,
    progress: bool = True,
) -> List[Dict]:
    rng = random.Random(seed)
    iterator = range(num_instances)
    if progress:
        iterator = tqdm(iterator, desc="generate", total=num_instances)
    dataset: List[Dict] = []
    for _ in iterator:
        dataset.append(
            generate_instance(
                config,
                rng,
                with_solutions=with_solutions,
                solver_kwargs=solver_kwargs,
            )
        )
    return dataset


def save_dataset(path: str, dataset: List[Dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


def _verify_solution(instance: Dict) -> None:
    """Replay the embedding against PROBLEM_FORMULATION constraints."""
    substrate = instance["substrate"]
    requests = _requests(instance)
    bw = dict(substrate["communication_bandwidth"])
    cap = dict(substrate["compute_capacity"])
    attach = substrate["compute_attachment"]
    edges_set = set(substrate["communication_edges"])

    f_placements_all = instance["f_placements"]
    processing_paths_all = instance["processing_paths"]
    if f_placements_all and isinstance(f_placements_all[0], int):
        f_placements_all = [f_placements_all]
    if processing_paths_all and processing_paths_all[0] and isinstance(processing_paths_all[0][0], int):
        processing_paths_all = [processing_paths_all]

    if len(f_placements_all) != len(requests):
        raise AssertionError("f_placements request count does not match requests")
    if len(processing_paths_all) != len(requests):
        raise AssertionError("processing_paths request count does not match requests")

    accepted = instance.get("accepted")
    if accepted is None:
        accepted = [True] * len(requests)
    if len(accepted) != len(requests):
        raise AssertionError("accepted length does not match requests")

    for request_idx, request in enumerate(requests):
        f_placements: List[int] = list(f_placements_all[request_idx])
        processing_paths: List[Tuple[int, ...]] = list(processing_paths_all[request_idx])

        if not accepted[request_idx]:
            # A rejected request reserves nothing and carries no placement/path.
            if f_placements or processing_paths:
                raise AssertionError(
                    f"request {request_idx}: rejected but has a placement/path"
                )
            continue

        k = request["num_processing_nodes"]
        pl_dems = list(request["processing_link_demands"])

        if len(f_placements) != k:
            raise AssertionError(f"request {request_idx}: f_placements length {len(f_placements)} != k {k}")
        if len(processing_paths) != k - 1:
            raise AssertionError(
                f"request {request_idx}: processing_paths length {len(processing_paths)} != k-1 {k-1}"
            )

        # F_0 and F_{k-1} must sit on attached comm nodes.
        if f_placements[0] not in attach:
            raise AssertionError(f"request {request_idx}: F_0 placed at unattached comm node {f_placements[0]}")
        if f_placements[-1] not in attach:
            raise AssertionError(
                f"request {request_idx}: F_{k-1} placed at unattached comm node {f_placements[-1]}"
            )

        # Comp reservations: source_dem at F_0's comp link, dest_dem at F_{k-1}'s.
        cap[attach[f_placements[0]]] -= request["source_link_demand"]
        if cap[attach[f_placements[0]]] < 0:
            raise AssertionError(
                f"request {request_idx}: source-link reservation exceeds capacity at comp {attach[f_placements[0]]}"
            )
        cap[attach[f_placements[-1]]] -= request["destination_link_demand"]
        if cap[attach[f_placements[-1]]] < 0:
            raise AssertionError(
                f"request {request_idx}: destination-link reservation exceeds capacity at comp {attach[f_placements[-1]]}"
            )

        # Processing paths: chain coupling + comm-link direction + bandwidth.
        for ell, path in enumerate(processing_paths):
            if path[0] != f_placements[ell] or path[-1] != f_placements[ell + 1]:
                raise AssertionError(
                    f"request {request_idx}: processing path {ell} endpoints don't match F_{ell}, F_{ell+1}"
                )
            if len(path) < 2:
                raise AssertionError(
                    f"request {request_idx}: processing path {ell} has length < 1 hop (F_{ell} == F_{ell+1})"
                )
            for a, b in zip(path[:-1], path[1:]):
                if (a, b) not in edges_set:
                    raise AssertionError(f"request {request_idx}: path uses non-existent comm edge ({a},{b})")
                bw[(a, b)] -= pl_dems[ell]
                if bw[(a, b)] < 0:
                    raise AssertionError(f"request {request_idx}: bandwidth exceeded on edge ({a},{b})")


def run_self_check(dataset: List[Dict]) -> None:
    """Verify each instance's embedding."""
    checked_solutions = 0
    for i, inst in enumerate(dataset):
        if "f_placements" not in inst:
            continue
        _verify_solution(inst)
        checked_solutions += 1
    print(
        f"self-check OK: {len(dataset)} instances loaded, "
        f"{checked_solutions} solutions verified"
    )


def main() -> None:
    config = VNEConfig()
    p = argparse.ArgumentParser(description="VNE validation pickle generator")
    p.add_argument("--num-instances", type=int, default=config.validation_num_instances)
    p.add_argument("--seed", type=int, default=config.validation_generation_seed)
    p.set_defaults(with_solutions=config.validation_with_solutions)
    p.add_argument("--with-solutions", dest="with_solutions", action="store_true")
    p.add_argument("--no-solutions", dest="with_solutions", action="store_false")
    p.add_argument("--solver-time-limit", type=int, default=config.validation_solver_time_limit)
    p.add_argument("--solver", choices=["auto", "highs", "gurobi", "cplex", "cbc"], default=getattr(config, "validation_solver", "highs"))
    p.add_argument("--threads", type=int, default=getattr(config, "validation_solver_threads", 0))
    p.add_argument("--mip-gap", type=float, default=getattr(config, "validation_solver_mip_gap", 0.0))
    p.add_argument("--objective", choices=["lex", "profit"], default=getattr(config, "validation_objective", "lex"))
    p.set_defaults(enable_admission=getattr(config, "enable_admission", True))
    p.add_argument("--admission", dest="enable_admission", action="store_true",
                   help="let the ILP reject unprofitable/unplaceable requests")
    p.add_argument("--no-admission", dest="enable_admission", action="store_false",
                   help="force every request to be embedded (legacy behaviour)")
    p.add_argument("--revenue", type=float, default=getattr(config, "validation_revenue_per_request", 0.0))
    p.add_argument("--revenue-per-demand-unit", type=float,
                   default=getattr(config, "validation_revenue_per_demand_unit", 1.0))
    p.add_argument("--cost-comm-per-unit", type=float,
                   default=getattr(config, "validation_cost_comm_per_unit", 1.0))
    p.add_argument("--cost-comp-per-unit", type=float,
                   default=getattr(config, "validation_cost_comp_per_unit", 0.0))
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--out", type=str, default=config.validation_output_path)
    args = p.parse_args()

    n_low, n_high = config.num_substrate_comm_nodes_range
    rq_low, rq_high = config.num_virtual_requests_range
    vn_low, vn_high = config.num_virtual_nodes_range
    out = args.out or (
        f"data/vne/vne_n{n_low}-{n_high}_rq{rq_low}-{rq_high}"
        f"_vn{vn_low}-{vn_high}_{args.num_instances}_seed{args.seed}.pickle"
    )

    print(f"Generating {args.num_instances} VNE instances -> {out}")
    dataset = make_validation_dataset(
        args.num_instances,
        config,
        with_solutions=args.with_solutions,
        solver_kwargs={
            "time_limit_s": args.solver_time_limit,
            "solver": args.solver,
            "threads": args.threads,
            "mip_gap": args.mip_gap,
            "objective_mode": args.objective,
            "enable_admission": args.enable_admission,
            "revenue": args.revenue,
            "revenue_per_demand_unit": args.revenue_per_demand_unit,
            "cost_comm_per_unit": args.cost_comm_per_unit,
            "cost_comp_per_unit": args.cost_comp_per_unit,
        },
        seed=args.seed,
    )
    save_dataset(out, dataset)
    print(f"Saved {len(dataset)} instances to {out}")

    if args.self_check:
        run_self_check(dataset)


if __name__ == "__main__":
    main()
