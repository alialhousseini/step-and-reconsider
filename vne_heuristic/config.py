"""Configuration for standalone VNE heuristic baselines.

The heuristic package reads VNE pickles and imports VNE trajectory helpers, but
it does not modify `vne/`, `vne_main.py`, supervised training, or Gumbeldore.
"""

from __future__ import annotations


class HeuristicConfig:
    """Single source of truth for the heuristic comparison route."""

    def __init__(self):
        # Dataset paths. The supervised dataset is kept here for future tuning
        # needs, but the current two heuristics are deterministic and do not use
        # tuning.
        self.supervised_tuning_dataset_path = "./data/vne/vne_supervised_training_dataset.pickle"
        self.validation_dataset_path = "./data/vne/vne_validation_dataset_1k.pickle"

        # Dataset subset sizes.
        self.tuning_num_instances = 10
        self.validation_num_instances = 128

        # Heuristics to evaluate. Valid names are "balanced_greedy" and
        # "one_step_lookahead".
        self.heuristics = ("balanced_greedy", "one_step_lookahead")

        # Candidate generation mode. False means each heuristic only sees paths
        # feasible for the current virtual link. This keeps the baseline honest:
        # it does not use the full recursive future-completion filter.
        self.check_future_completion = False

        # Admission/rejection is not supported here because the current VNE
        # trajectory also embeds all requests in order.
        self.enable_admission = False

        # Print per-instance progress while evaluating.
        self.progress = True
