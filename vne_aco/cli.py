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
from dataclasses import asdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vne_aco.config import ACOConfig
from vne_aco.evaluator import (
    ensure_embed_all_instances,
    evaluate_parameters,
    load_instances,
    tune_parameters,
)


def main() -> None:
    config = ACOConfig()

    tuning_instances = load_instances(
        config.supervised_tuning_dataset_path,
        config.tuning_num_instances,
    )
    validation_instances = load_instances(
        config.validation_dataset_path,
        config.validation_num_instances,
    )
    ensure_embed_all_instances(tuning_instances)
    ensure_embed_all_instances(validation_instances)

    best_parameters, tuning_results = tune_parameters(
        tuning_instances,
        config.candidate_parameters(),
        seed=config.seed,
        progress=True,
    )
    validation_metrics = evaluate_parameters(
        validation_instances,
        best_parameters,
        seed=config.seed + 1_000_000,
        progress=True,
    )

    print(json.dumps(
        {
            "best_parameters": asdict(best_parameters),
            "tuning_results": tuning_results,
            "validation": validation_metrics,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
