"""Configuration for the VNE Ant Colony Optimization baseline.

ACO does not train neural weights. It only tunes search hyperparameters on a
small solved supervised subset, then solves the validation set with the chosen
fixed parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class ACOParameters:
    """Search parameters used by one ACO solve run."""

    num_ants: int = 32
    num_iterations: int = 100
    alpha: float = 1.0
    beta: float = 3.0
    evaporation_rate: float = 0.25
    elite_ants: int = 4
    exploitation_probability: float = 0.10
    global_best_weight: float = 2.0
    initial_pheromone: float = 1.0
    min_pheromone: float = 1e-6
    check_future_completion: bool = False

    def validate(self) -> None:
        if self.num_ants < 1:
            raise ValueError("num_ants must be >= 1")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if not 0 <= self.evaporation_rate < 1:
            raise ValueError("evaporation_rate must be in [0, 1)")
        if self.elite_ants < 1:
            raise ValueError("elite_ants must be >= 1")
        if not 0 <= self.exploitation_probability <= 1:
            raise ValueError("exploitation_probability must be in [0, 1]")
        if self.global_best_weight < 0:
            raise ValueError("global_best_weight must be >= 0")
        if self.initial_pheromone <= 0:
            raise ValueError("initial_pheromone must be > 0")
        if self.min_pheromone <= 0:
            raise ValueError("min_pheromone must be > 0")


class ACOConfig:
    """Single source of truth for the standalone ACO baseline."""

    def __init__(self):
        # Dataset paths. Tuning uses the same test dataset as evaluation, but a
        # non-overlapping slice (offset=256) to avoid data leakage.
        self.supervised_tuning_dataset_path = "./data/vne/vne_test_dataset_2k.pickle"
        self.validation_dataset_path = "./data/vne/vne_test_dataset_2k.pickle"

        # Tuning instances are drawn from indices [offset : offset+num].
        # Evaluation always uses the first N instances.
        self.tuning_num_instances = 20
        self.tuning_offset = 256
        self.validation_num_instances = 256

        # Reproducibility seed for ant choices and tuning runs.
        self.seed = 0

        # ACO assumes the current embed-all VNE formulation. Admission/rejection
        # is not enabled here because the current VNE trajectory also embeds all
        # requests in order.
        self.enable_admission = False

        # Base/default parameters. They are also used when run_tuning=False.
        self.base_parameters = ACOParameters()

        # Tuning grid covering a wider range of alpha, beta, evaporation_rate.
        # Lower beta values (less greedy toward short paths) are included because
        # on large instances the pure cost heuristic often leads to dead ends.
        # Tuning grid. Each entry overrides base_parameters. num_ants and
        # num_iterations are kept low for fast tuning; after the best
        # alpha/beta/evaporation is found, eval runs with the full base budget.
        self.tuning_grid: Tuple[dict, ...] = (
            {"num_ants": 16, "num_iterations": 30, "alpha": 1.0, "beta": 1.0, "evaporation_rate": 0.15},
            {"num_ants": 16, "num_iterations": 30, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.20},
            {"num_ants": 16, "num_iterations": 30, "alpha": 1.5, "beta": 1.5, "evaporation_rate": 0.25},
            {"num_ants": 16, "num_iterations": 30, "alpha": 2.0, "beta": 1.0, "evaporation_rate": 0.15},
            {"num_ants": 16, "num_iterations": 30, "alpha": 2.0, "beta": 2.0, "evaporation_rate": 0.25},
        )

    def candidate_parameters(self) -> list[ACOParameters]:
        return [replace(self.base_parameters, **overrides) for overrides in self.tuning_grid]
