# VNE Heuristic Baselines

This directory contains deterministic heuristic baselines for the same VNE
instances used by supervised, Gumbeldore, and ACO.

The heuristics piggyback on `vne.trajectory.Trajectory` for:

- feasible candidate substrate paths;
- residual bandwidth and compute updates;
- the objective convention.

They do not modify `vne/`.

## Run

Edit paths and subset sizes in `vne_heuristic/config.py`, then run:

```bash
python -m vne_heuristic.evaluator
```

Direct execution also works:

```bash
python vne_heuristic/evaluator.py
```

## Objective

The objective matches current VNE:

```text
cost = sum(virtual bandwidth demand * substrate path hop count)
objective = -cost
```

Higher objective is better.

## Heuristics

### Balanced Greedy

Scores each candidate path as:

```text
score = (1 / cost) * min_resource_ratio_after_applying
```

This keeps the low-cost preference but penalizes choices that drain a substrate
edge or compute node into a bottleneck.

### One-Step Feasibility Lookahead

Scores each candidate path as:

```text
score = (1 / cost) * num_feasible_next_link_candidates
```

If every candidate leaves zero next-link options, it falls back to the cheapest
current path.

## Constraint

The package currently supports embed-all instances only:

```text
enable_admission = False
```

Admission/rejection should be added only after the main VNE trajectory supports
acceptance-aware actions.


