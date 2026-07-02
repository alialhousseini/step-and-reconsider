"""One-step feasibility lookahead path selection."""

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
    """Choose the path that leaves many options for the next virtual link.

    score = (1 / cost) * num_feasible_next_link_candidates
    If all candidates leave no next-link options, fall back to cost-greedy.
    """

    request = requests_from_instance(instance)[request_idx]
    demand = request["processing_link_demands"][link_idx]
    fallback = cheapest_path(candidates, demand)
    best_path = fallback
    best_score = float("-inf")
    any_future_options = False

    for path in candidates:
        next_bandwidth, next_compute = Trajectory._apply_path(
            instance,
            residual_bandwidth,
            residual_compute,
            compute_attachment,
            request_idx,
            link_idx,
            path,
        )
        next_count = _next_candidate_count(
            instance,
            next_bandwidth,
            next_compute,
            compute_attachment,
            request_idx,
            link_idx,
            path,
        )
        any_future_options = any_future_options or next_count > 0
        score = next_count / max(Trajectory.compute_cost(path, demand), 1e-9)
        if score > best_score or (
            score == best_score and Trajectory.compute_cost(path, demand) < Trajectory.compute_cost(best_path, demand)
        ):
            best_score = score
            best_path = path

    return best_path if any_future_options else fallback


def _next_candidate_count(
    instance: Dict,
    residual_bandwidth: Dict[Tuple[int, int], int],
    residual_compute: Dict[int, int],
    compute_attachment: Dict[int, int],
    request_idx: int,
    link_idx: int,
    chosen_path: Path,
) -> int:
    next_request_idx, next_link_idx = Trajectory._next_position(instance, request_idx, link_idx)
    if next_request_idx is None:
        return 1
    forced_start = chosen_path[-1] if next_request_idx == request_idx else None
    return len(
        Trajectory._candidate_paths(
            instance=instance,
            residual_bandwidth=residual_bandwidth,
            residual_compute=residual_compute,
            compute_attachment=compute_attachment,
            request_idx=next_request_idx,
            link_idx=next_link_idx,
            forced_start=forced_start,
            check_future_completion=False,
        )
    )

if __name__ == "__main__":
    print("This module defines the one_step_lookahead selector. Run `python -m vne_heuristic.evaluator` to evaluate it.")


