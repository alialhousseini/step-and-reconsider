"""Ant Colony Optimization solver for VNE instances.

The solver piggybacks on `vne.trajectory.Trajectory` for candidate generation,
resource updates, and the objective convention. This keeps ACO aligned with the
supervised and Gumbeldore routes without modifying `vne/`.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from vne.trajectory import Trajectory
from vne_aco.config import ACOParameters

Path = Tuple[int, ...]
Edge = Tuple[int, int]


def _requests(instance: Dict) -> List[Dict]:
    if "requests" in instance:
        return instance["requests"]
    return [instance["request"]]


def _compute_attachment(substrate: Dict) -> Dict[int, int]:
    attach = substrate.get("compute_attachment")
    if attach is None:
        return {node: node for node in range(substrate["num_comm_nodes"])}
    return dict(attach)


@dataclass
class ACOSolution:
    feasible: bool
    objective: float
    processing_paths: List[List[Path]] = field(default_factory=list)
    f_placements: List[List[int]] = field(default_factory=list)
    best_iteration: int | None = None

    @property
    def cost(self) -> float:
        if not self.feasible or not math.isfinite(self.objective):
            return float("inf")
        return -self.objective


class ACOSolver:
    """Solve one VNE instance using ant colony search."""

    def __init__(self, parameters: ACOParameters, seed: int = 0):
        parameters.validate()
        self.parameters = parameters
        self.rng = random.Random(seed)

    def solve(self, instance: Dict) -> ACOSolution:
        pheromone = {
            tuple(edge): self.parameters.initial_pheromone
            for edge in instance["substrate"]["communication_edges"]
        }
        global_best: ACOSolution | None = None

        for iteration in range(self.parameters.num_iterations):
            solutions = [self._construct_solution(instance, pheromone) for _ in range(self.parameters.num_ants)]
            feasible_solutions = [solution for solution in solutions if solution.feasible]
            if feasible_solutions:
                iteration_best = max(feasible_solutions, key=lambda solution: solution.objective)
                if global_best is None or iteration_best.objective > global_best.objective:
                    global_best = copy.deepcopy(iteration_best)
                    global_best.best_iteration = iteration
            self._update_pheromone(pheromone, feasible_solutions, global_best)

        if global_best is None:
            return ACOSolution(feasible=False, objective=float("-inf"))
        return global_best

    def _construct_solution(self, instance: Dict, pheromone: Dict[Edge, float]) -> ACOSolution:
        substrate = instance["substrate"]
        requests = _requests(instance)
        attach = _compute_attachment(substrate)
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
                    check_future_completion=self.parameters.check_future_completion,
                )
                if not candidates:
                    return ACOSolution(feasible=False, objective=float("-inf"))

                path = self._choose_path(candidates, request, link_idx, pheromone)
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

        return ACOSolution(
            feasible=True,
            objective=float(objective),
            processing_paths=processing_paths,
            f_placements=f_placements,
        )

    def _choose_path(
        self,
        candidates: List[Path],
        request: Dict,
        link_idx: int,
        pheromone: Dict[Edge, float],
    ) -> Path:
        scores = [
            self._path_score(path, request["processing_link_demands"][link_idx], pheromone)
            for path in candidates
        ]
        if self.rng.random() < self.parameters.exploitation_probability:
            return candidates[max(range(len(candidates)), key=lambda idx: scores[idx])]

        total = sum(scores)
        if total <= 0 or not math.isfinite(total):
            return self.rng.choice(candidates)
        threshold = self.rng.random() * total
        running = 0.0
        for path, score in zip(candidates, scores):
            running += score
            if running >= threshold:
                return path
        return candidates[-1]

    def _path_score(self, path: Path, demand: int, pheromone: Dict[Edge, float]) -> float:
        edges = Trajectory._path_edges(path)
        avg_pheromone = sum(pheromone.get(edge, self.parameters.min_pheromone) for edge in edges) / len(edges)
        cost = Trajectory.compute_cost(path, demand)
        heuristic = 1.0 / max(cost, 1e-9)
        return (avg_pheromone ** self.parameters.alpha) * (heuristic ** self.parameters.beta)

    def _update_pheromone(
        self,
        pheromone: Dict[Edge, float],
        feasible_solutions: List[ACOSolution],
        global_best: ACOSolution | None,
    ) -> None:
        evaporation_factor = 1.0 - self.parameters.evaporation_rate
        for edge in list(pheromone):
            pheromone[edge] = max(self.parameters.min_pheromone, pheromone[edge] * evaporation_factor)

        elite = sorted(feasible_solutions, key=lambda solution: solution.objective, reverse=True)[
            : self.parameters.elite_ants
        ]
        for solution in elite:
            self._deposit(pheromone, solution, weight=1.0)
        if global_best is not None and self.parameters.global_best_weight > 0:
            self._deposit(pheromone, global_best, weight=self.parameters.global_best_weight)

    def _deposit(self, pheromone: Dict[Edge, float], solution: ACOSolution, weight: float) -> None:
        if not solution.feasible:
            return
        deposit = weight / max(solution.cost, 1e-9)
        for edge in self._solution_edges(solution):
            pheromone[edge] = pheromone.get(edge, self.parameters.min_pheromone) + deposit

    @staticmethod
    def _solution_edges(solution: ACOSolution) -> Iterable[Edge]:
        for request_paths in solution.processing_paths:
            for path in request_paths:
                yield from Trajectory._path_edges(path)
