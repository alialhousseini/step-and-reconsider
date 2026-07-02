"""Run the VNE ACO baseline.

Flow:
1. load a small solved supervised subset;
2. tune ACO parameters on that subset;
3. freeze the best parameters;
4. solve the validation dataset and print comparison metrics.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vne_aco.config import ACOConfig, ACOParameters
from vne_aco.evaluator import (
    ensure_embed_all_instances,
    evaluate_parameters,
    load_instances,
    tune_parameters,
    write_markdown_table,
)


def main() -> None:
    config = ACOConfig()

    tuning_instances = load_instances(
        config.supervised_tuning_dataset_path,
        config.tuning_num_instances,
        offset=config.tuning_offset,
    )
    validation_instances = load_instances(
        config.validation_dataset_path,
        config.validation_num_instances,
    )
    ensure_embed_all_instances(tuning_instances)
    ensure_embed_all_instances(validation_instances)

    # Tune with reduced search budget (16 ants × 30 iterations per grid entry).
    best_tuned, tuning_results = tune_parameters(
        tuning_instances,
        config.candidate_parameters(),
        seed=config.seed,
        progress=True,
    )

    # Evaluate with the full base search budget, keeping only the best
    # alpha/beta/evaporation from tuning.
    eval_parameters: ACOParameters = replace(
        config.base_parameters,
        alpha=best_tuned.alpha,
        beta=best_tuned.beta,
        evaporation_rate=best_tuned.evaporation_rate,
    )
    print(f"\nTuned: alpha={best_tuned.alpha}, beta={best_tuned.beta}, evap={best_tuned.evaporation_rate}")
    print(f"Eval budget: {eval_parameters.num_ants} ants x {eval_parameters.num_iterations} iter = {eval_parameters.num_ants * eval_parameters.num_iterations} attempts\n")

    validation_metrics = evaluate_parameters(
        validation_instances,
        eval_parameters,
        seed=config.seed + 1_000_000,
        progress=True,
    )

    print(json.dumps(
        {
            "best_parameters": asdict(best_tuned),
            "eval_parameters": asdict(eval_parameters),
            "tuning_results": tuning_results,
            "validation": validation_metrics,
        },
        indent=2,
        sort_keys=True,
    ))

    md_path = os.path.join(PROJECT_ROOT, "docs", "ACO_RESULTS.md")
    write_markdown_table(
        validation_metrics,
        md_path,
        title="ACO Metaheuristic Results",
        dataset_info=(
            f"Dataset: `{config.validation_dataset_path}` "
            f"(first {config.validation_num_instances} instances).  "
            f"Tuned on `{config.supervised_tuning_dataset_path}` "
            f"(indices {config.tuning_offset}–{config.tuning_offset + config.tuning_num_instances - 1}).  "
            f"Eval budget: {eval_parameters.num_ants} ants × {eval_parameters.num_iterations} iter."
        ),
    )
    print(f"\nMarkdown table written to: {md_path}")


if __name__ == "__main__":
    main()
