"""Shared solver loop for VNE heuristic baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from vne.trajectory import Trajectory

Path = Tuple[int, ...]
Selector = Callable[[Dict, List[Path], Dict, Dict, Dict, int, int, bool], Path]


def requests_from_instance(instance: Dict) -> List[Dict]:
    if "requests" in instance:
        return instance["requests"]
    return [instance["request"]]


def compute_attachment(substrate: Dict) -> Dict[int, int]:
    attach = substrate.get("compute_attachment")
    if attach is None:
        return {node: node for node in range(substrate["num_comm_nodes"])}
    return dict(attach)


@dataclass
class HeuristicSolution:
    feasible: bool
    objective: float
    processing_paths: List[List[Path]] = field(default_factory=list)
    f_placements: List[List[int]] = field(default_factory=list)

    @property
    def cost(self) -> float:
        if not self.feasible or not math.isfinite(self.objective):
            return float("inf")
        return -self.objective


def solve_instance(
    instance: Dict,
    selector: Selector,
    *,
    check_future_completion: bool = False,
) -> HeuristicSolution:
    """Build one full embedding using the provided path-selection heuristic."""

    substrate = instance["substrate"]
    requests = requests_from_instance(instance)
    attach = compute_attachment(substrate)
    residual_bandwidth = dict(substrate["communication_bandwidth"])
    residual_compute = dict(substrate["compute_capacity"])
    processing_paths: List[List[Path]] = [[] for _ in requests]
    f_placements: List[List[int]] = [[] for _ in requests]
    objective = 0.0

    for request_idx, request in enumerate(requests):
        forced_start = None
        for link_idx in range(request["num_processing_nodes"] - 1):
            candidates = Trajectory._candidate_paths(
                instance=instance,
                residual_bandwidth=residual_bandwidth,
                residual_compute=residual_compute,
                compute_attachment=attach,
                request_idx=request_idx,
                link_idx=link_idx,
                forced_start=forced_start,
                check_future_completion=check_future_completion,
            )
            if not candidates:
                return HeuristicSolution(feasible=False, objective=float("-inf"))

            path = selector(
                instance,
                candidates,
                residual_bandwidth,
                residual_compute,
                attach,
                request_idx,
                link_idx,
                check_future_completion,
            )
            residual_bandwidth, residual_compute = Trajectory._apply_path(
                instance,
                residual_bandwidth,
                residual_compute,
                attach,
                request_idx,
                link_idx,
                path,
            )
            if link_idx == 0:
                f_placements[request_idx].append(path[0])
            f_placements[request_idx].append(path[-1])
            processing_paths[request_idx].append(path)
            objective -= Trajectory.compute_cost(path, request["processing_link_demands"][link_idx])
            forced_start = path[-1]

    return HeuristicSolution(
        feasible=True,
        objective=float(objective),
        processing_paths=processing_paths,
        f_placements=f_placements,
    )


def cheapest_path(candidates: List[Path], demand: int) -> Path:
    """Cost-greedy fallback used by multiple heuristics."""

    return min(
        candidates,
        key=lambda path: (Trajectory.compute_cost(path, demand), len(path), path[0], path[-1], path),
    )
