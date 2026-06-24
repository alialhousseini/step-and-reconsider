"""Tuning and validation utilities for the VNE ACO baseline."""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import asdict
from typing import Dict, List, Sequence

from vne_aco.config import ACOParameters
from vne_aco.solver import ACOSolution, ACOSolver


def load_instances(path: str, limit: int | None) -> List[Dict]:
    with open(path, "rb") as f:
        instances = pickle.load(f)
    return instances if limit is None else instances[:limit]


def ensure_embed_all_instances(instances: Sequence[Dict]) -> None:
    """Fail clearly if a dataset contains admission-control labels."""

    for idx, instance in enumerate(instances):
        accepted = instance.get("accepted")
        if accepted is not None and not all(accepted):
            raise ValueError(
                f"Instance {idx} contains rejected requests. "
                "The ACO baseline currently supports embed-all VNE only."
            )


def evaluate_parameters(
    instances: Sequence[Dict],
    parameters: ACOParameters,
    seed: int,
    progress: bool = False,
) -> Dict:
    start = time.time()
    solutions = []
    solved_aco_objectives = []
    solved_ilp_objectives = []

    for idx, instance in enumerate(instances):
        solution = ACOSolver(parameters, seed=seed + idx).solve(instance)
        solutions.append(solution)
        if progress:
            status = "feasible" if solution.feasible else "infeasible"
            parts = [f"[{idx + 1}/{len(instances)}] {status}: objective={solution.objective}"]
            if solution.feasible and instance.get("objective") is not None:
                solved_aco_objectives.append(solution.objective)
                ilp_objective = float(instance["objective"])
                solved_ilp_objectives.append(ilp_objective)
                parts.append(f"ilp={ilp_objective}")
                parts.append(f"avg_aco_solved={_mean(solved_aco_objectives)}")
                parts.append(f"avg_ilp_solved={_mean(solved_ilp_objectives)}")
            print(" | ".join(parts))

    return summarize(instances, solutions, time.time() - start, parameters)


def tune_parameters(
    instances: Sequence[Dict],
    candidate_parameters: Sequence[ACOParameters],
    seed: int,
    progress: bool = False,
) -> tuple[ACOParameters, List[Dict]]:
    if not candidate_parameters:
        raise ValueError("candidate_parameters cannot be empty")

    best_parameters = candidate_parameters[0]
    best_rank = None
    results = []
    for idx, parameters in enumerate(candidate_parameters):
        if progress:
            print(f">> Tuning ACO candidate {idx + 1}/{len(candidate_parameters)}: {asdict(parameters)}")
        metrics = evaluate_parameters(instances, parameters, seed=seed + 100_000 * idx)
        results.append(metrics)
        rank = _rank(metrics)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_parameters = parameters
    return best_parameters, results


def summarize(
    instances: Sequence[Dict],
    solutions: Sequence[ACOSolution],
    elapsed_s: float,
    parameters: ACOParameters,
) -> Dict:
    feasible = [solution for solution in solutions if solution.feasible]
    objectives = [solution.objective for solution in feasible]
    all_ilp_objectives = [
        float(instance["objective"])
        for instance in instances
        if instance.get("objective") is not None
    ]
    solved_ilp_objectives = []
    gaps = []

    for instance, solution in zip(instances, solutions):
        if not solution.feasible or instance.get("objective") is None:
            continue
        ilp_objective = float(instance["objective"])
        solved_ilp_objectives.append(ilp_objective)
        if ilp_objective == 0:
            gaps.append(0.0)
        else:
            gaps.append((ilp_objective - solution.objective) / abs(ilp_objective) * 100.0)

    num_instances = len(instances)
    return {
        "parameters": asdict(parameters),
        "num_instances": num_instances,
        "num_feasible": len(feasible),
        "feasibility_pct": (len(feasible) / max(num_instances, 1)) * 100.0,
        "aco_avg_objective": _mean(objectives),
        "aco_best_objective": max(objectives) if objectives else float("-inf"),
        "aco_worst_objective": min(objectives) if objectives else float("-inf"),
        "num_feasible_with_ilp": len(solved_ilp_objectives),
        "ilp_avg_objective": _mean(solved_ilp_objectives),
        "ilp_best_objective": max(solved_ilp_objectives) if solved_ilp_objectives else float("nan"),
        "ilp_worst_objective": min(solved_ilp_objectives) if solved_ilp_objectives else float("nan"),
        "ilp_avg_objective_all_instances": _mean(all_ilp_objectives),
        "gap_pct": _mean(gaps),
        "runtime_s": elapsed_s,
        "runtime_per_instance_ms": elapsed_s / max(num_instances, 1) * 1000.0,
    }


def _rank(metrics: Dict) -> tuple[float, float, float]:
    objective = metrics["aco_avg_objective"]
    gap = metrics["gap_pct"]
    return (
        metrics["feasibility_pct"],
        -gap if math.isfinite(gap) else float("-inf"),
        objective if math.isfinite(objective) else float("-inf"),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))
