"""Evaluate deterministic VNE heuristic baselines."""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vne_heuristic import balanced_greedy, cost_greedy, one_step_lookahead
from vne_heuristic.config import HeuristicConfig
from vne_heuristic.solver import HeuristicSolution, solve_instance


SELECTORS = {
    "cost_greedy": cost_greedy.select_path,
    "balanced_greedy": balanced_greedy.select_path,
    "one_step_lookahead": one_step_lookahead.select_path,
}


def load_instances(path: str, limit: int | None) -> List[Dict]:
    with open(path, "rb") as f:
        instances = pickle.load(f)
    return instances if limit is None else instances[:limit]


def ensure_embed_all_instances(instances: Sequence[Dict]) -> None:
    for idx, instance in enumerate(instances):
        accepted = instance.get("accepted")
        if accepted is not None and not all(accepted):
            raise ValueError(
                f"Instance {idx} contains rejected requests. "
                "The heuristic baselines currently support embed-all VNE only."
            )


def evaluate_heuristic(
    instances: Sequence[Dict],
    heuristic_name: str,
    config: HeuristicConfig,
) -> Dict:
    if heuristic_name not in SELECTORS:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    selector = SELECTORS[heuristic_name]
    start = time.time()
    solutions = []
    solved_objectives = []
    solved_ilp_objectives = []

    for idx, instance in enumerate(instances):
        solution = solve_instance(
            instance,
            selector,
            check_future_completion=config.check_future_completion,
        )
        solutions.append(solution)
        if config.progress:
            status = "feasible" if solution.feasible else "infeasible"
            parts = [f"[{idx + 1}/{len(instances)}] {heuristic_name} {status}: objective={solution.objective}"]
            if solution.feasible and instance.get("objective") is not None:
                solved_objectives.append(solution.objective)
                ilp_objective = float(instance["objective"])
                solved_ilp_objectives.append(ilp_objective)
                parts.append(f"ilp={ilp_objective}")
                parts.append(f"avg_heuristic_solved={_mean(solved_objectives)}")
                parts.append(f"avg_ilp_solved={_mean(solved_ilp_objectives)}")
            print(" | ".join(parts))

    return summarize(instances, solutions, time.time() - start, heuristic_name)


def summarize(
    instances: Sequence[Dict],
    solutions: Sequence[HeuristicSolution],
    elapsed_s: float,
    heuristic_name: str,
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
        "heuristic": heuristic_name,
        "num_instances": num_instances,
        "num_feasible": len(feasible),
        "feasibility_pct": (len(feasible) / max(num_instances, 1)) * 100.0,
        "heuristic_avg_objective": _mean(objectives),
        "heuristic_best_objective": max(objectives) if objectives else float("-inf"),
        "heuristic_worst_objective": min(objectives) if objectives else float("-inf"),
        "num_feasible_with_ilp": len(solved_ilp_objectives),
        "ilp_avg_objective": _mean(solved_ilp_objectives),
        "ilp_best_objective": max(solved_ilp_objectives) if solved_ilp_objectives else float("nan"),
        "ilp_worst_objective": min(solved_ilp_objectives) if solved_ilp_objectives else float("nan"),
        "ilp_avg_objective_all_instances": _mean(all_ilp_objectives),
        "gap_pct": _mean(gaps),
        "runtime_s": elapsed_s,
        "runtime_per_instance_ms": elapsed_s / max(num_instances, 1) * 1000.0,
    }


def write_markdown_table(
    all_results: Dict[str, Dict],
    file_path: str,
    *,
    title: str = "Heuristic Baseline Results",
    dataset_info: str = "",
) -> None:
    """Write heuristic evaluation metrics as a combined markdown table."""

    lines = [
        f"# {title}",
        "",
    ]
    if dataset_info:
        lines.append(dataset_info)
        lines.append("")

    lines.extend([
        "## Comparison Table",
        "",
        "| Heuristic | Feas% | Gap% | ms/inst |",
        "|-----------|-------|------|---------|",
    ])

    for heuristic_name, metrics in all_results.items():
        label = {
            "cost_greedy": "Cost-Greedy",
            "balanced_greedy": "Balanced Greedy",
            "one_step_lookahead": "1-Step Lookahead",
        }.get(heuristic_name, heuristic_name)
        lines.append(
            f"| {label} | {metrics['feasibility_pct']:.2f} | {metrics['gap_pct']:.4f} | {metrics['runtime_per_instance_ms']:.2f} |"
        )

    lines.append("")

    # Detailed sections per heuristic
    for heuristic_name, metrics in all_results.items():
        label = {
            "cost_greedy": "Cost-Greedy",
            "balanced_greedy": "Balanced Greedy",
            "one_step_lookahead": "1-Step Lookahead",
        }.get(heuristic_name, heuristic_name)
        lines.extend([
            f"## {label} — Detail",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Instances | {metrics['num_instances']} |",
            f"| Feasible | {metrics['num_feasible']} / {metrics['num_instances']} ({metrics['feasibility_pct']:.2f}%) |",
            f"| Feasible with ILP label | {metrics['num_feasible_with_ilp']} |",
            f"| Heuristic avg objective | {metrics['heuristic_avg_objective']:.4f} |",
            f"| Heuristic best objective | {metrics['heuristic_best_objective']:.4f} |",
            f"| Heuristic worst objective | {metrics['heuristic_worst_objective']:.4f} |",
            f"| ILP avg objective (solved) | {metrics['ilp_avg_objective']:.4f} |",
            f"| ILP avg objective (all) | {metrics['ilp_avg_objective_all_instances']:.4f} |",
            f"| Gap % | {metrics['gap_pct']:.4f}% |",
            f"| Runtime (s) | {metrics['runtime_s']:.2f} |",
            f"| Runtime per instance (ms) | {metrics['runtime_per_instance_ms']:.2f} |",
            "",
        ])

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    config = HeuristicConfig()
    instances = load_instances(config.validation_dataset_path, config.validation_num_instances)
    ensure_embed_all_instances(instances)
    results = {
        heuristic_name: evaluate_heuristic(instances, heuristic_name, config)
        for heuristic_name in config.heuristics
    }
    print(json.dumps(results, indent=2, sort_keys=True))

    md_path = os.path.join(PROJECT_ROOT, "docs", "HEURISTIC_RESULTS.md")
    write_markdown_table(
        results,
        md_path,
        title="Heuristic Baseline Results",
        dataset_info=(
            f"Dataset: `{config.validation_dataset_path}` "
            f"(first {config.validation_num_instances} instances)."
        ),
    )
    print(f"\nMarkdown table written to: {md_path}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()



