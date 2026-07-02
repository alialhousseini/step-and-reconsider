"""Balanced greedy resource-aware path selection."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vne.trajectory import Trajectory
from vne_heuristic.solver import Path, cheapest_path, requests_from_instance


def select_path(
    instance: Dict,
    candidates: List[Path],
    residual_bandwidth: Dict[Tuple[int, int], int],
    residual_compute: Dict[int, int],
    compute_attachment: Dict[int, int],
    request_idx: int,
    link_idx: int,
    check_future_completion: bool,
) -> Path:
    """Choose a low-cost path that preserves bottleneck resources.

    score = (1 / cost) * min_resource_ratio_after_applying
    """

    request = requests_from_instance(instance)[request_idx]
    demand = request["processing_link_demands"][link_idx]
    best_path = cheapest_path(candidates, demand)
    best_score = float("-inf")

    for path in candidates:
        cost = Trajectory.compute_cost(path, demand)
        next_bandwidth, next_compute = Trajectory._apply_path(
            instance,
            residual_bandwidth,
            residual_compute,
            compute_attachment,
            request_idx,
            link_idx,
            path,
        )
        score = _min_resource_ratio(instance, next_bandwidth, next_compute) / max(cost, 1e-9)
        if score > best_score or (
            score == best_score and Trajectory.compute_cost(path, demand) < Trajectory.compute_cost(best_path, demand)
        ):
            best_score = score
            best_path = path
    return best_path


def _min_resource_ratio(
    instance: Dict,
    residual_bandwidth: Dict[Tuple[int, int], int],
    residual_compute: Dict[int, int],
) -> float:
    substrate = instance["substrate"]
    ratios = []
    for edge, original in substrate["communication_bandwidth"].items():
        if original > 0:
            ratios.append(residual_bandwidth.get(edge, 0) / original)
    for node, original in substrate["compute_capacity"].items():
        if original > 0:
            ratios.append(residual_compute.get(node, 0) / original)
    if not ratios:
        return 0.0
    return min(ratios)

if __name__ == "__main__":
    print("This module defines the balanced_greedy selector. Run `python -m vne_heuristic.evaluator` to evaluate it.")


