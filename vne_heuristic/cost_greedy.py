"""Cost-greedy path selection — always picks the cheapest feasible path."""

from __future__ import annotations

from typing import Dict, List, Tuple

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
    """Choose the substrate path with lowest cost for the current link.

    This is the simplest possible heuristic: pure greed on immediate cost,
    blind to future resource depletion. It serves as the floor baseline.
    """

    request = requests_from_instance(instance)[request_idx]
    demand = request["processing_link_demands"][link_idx]
    return cheapest_path(candidates, demand)
