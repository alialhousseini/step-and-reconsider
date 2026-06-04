# VNE Development Strategy

Last updated: 2026-06-02

This is the working CTO note for turning the current VNE scaffold into a
credible network-control experiment inside the TaSaR codebase.

## Current Position

The repository already has the generic TaSaR machinery needed for VNE:

- `core/train.py` handles training, validation, checkpoints, and logging.
- `core/gumbeldore_dataset.py` handles Ray-based search data generation.
- `core/incremental_sbs.py::IncrementalSBS.perform_tasar` is the main TaSaR
  search method.
- `vne_main.py` already wires VNE into the same callback pattern used by TSP,
  CVRP, and JSSP.

The current VNE implementation is a static multi-request chain scaffold:

- config-driven directed line substrate with sampled size for random
  Gumbeldore generation;
- one substrate per problem instance;
- a sampled number of static virtual requests per substrate;
- each virtual request samples its own chain size from the config range;
- one action chooses one complete directed substrate path for one
  `(request_idx, processing_link_idx)` decision;
- trajectory terminates after all links of all requests are embedded;
- objective is negative bandwidth-weighted hop cost;
- policy is an LEHD-style encoder/decoder over structured VNE state features.

The validation generator and runtime now agree on chain-style data:

- `vne/validation_set_generator.py` can generate line or ER substrates;
- it solves all static virtual requests jointly with PuLP;
- it emits nested `f_placements`, nested `processing_paths`, and `objective`;
- `vne/dataset.py` replays one training example per actual request/link
  decision.
- `vne_main.py` auto-generates the canonical validation pickle before
  training if it is missing, and auto-generates the canonical supervised
  training pickle only when `learning_type="supervised"`.

## Strategic Target

Build VNE in three explicit tiers.

1. Legacy one-request VNE
   - Keep this as a compatibility smoke test for old `data/vne/vne_toy_*.pickle`
     files.
   - It proves the callback stack, Ray generation, TaSaR search, and replay
     path still work.

2. Static multi-request chain VNE
   - This is the current implementation tier.
   - Substrate size, request count, and each request's virtual-node count are
     sampled from inclusive config ranges.
   - One TaSaR action embeds one processing link `F_i -> F_{i+1}` by choosing a
     complete directed substrate communication path for a specific request.
   - The state tracks current request index, current link index, residual
     bandwidth, residual compute, placements, selected paths, accumulated
     objective, and feasibility.
   - Dataset replay reconstructs the actual variable-length request/link prefix
     and trains the next path action.

3. General graph VNE
   - Original virtual graph `R(M, K)` with arbitrary communication links.
   - Requires a decision-order policy or a fixed ordering heuristic over `K`.
   - Requires a graph-aware policy representation.

## Next Engineering Milestone

The next milestone should be "make the multi-request chain experiment
scientifically useful".

Acceptance criteria:

- regenerate supervised/validation/test datasets for the configured size
  ranges;
- report ILP gap or objective difference when validation data has ILP labels;
- define the objective model beyond negative bandwidth-weighted hop count;
- decide candidate path limits for dense ER substrates;
- run controlled experiments across substrate size, request count, and demand
  load.

## Design Decisions To Lock

1. Compute semantics for intermediate processing nodes.

   The current ILP reserves compute only for `F_0` and `F_{k-1}` through source
   and destination links. The rigorous formulation suggests every original
   virtual node has compute demand after augmentation. Pick one and make code,
   docs, and ILP agree.

2. Objective model.

   Toy mode uses `-(hops * bandwidth_demand)`. Full VNE should use an explicit
   revenue-minus-cost form, with communication cost, computation cost, and
   infeasible/rejection behavior documented.

3. Candidate path set.

   Decide whether to enumerate all simple paths, bounded-hop simple paths,
   k-shortest paths, or feasible paths up to a cutoff. This choice controls
   both runtime and learning difficulty.

4. Virtual-link order.

   Chain mode has natural order. General graph mode needs either a fixed
   heuristic order or an additional learned decision.

## Implementation Warnings

- Keep action ordering stable across trajectory search, dataset replay, and
  target-index construction.
- Do not silently reinterpret a solved multi-request instance as a flat
  single-request path list.
- Preserve the search sign convention: `to_max_evaluation_fn` returns a value
  to maximize; `main_train_cycle` validation metric is minimized.
- Do not put dataset generation in `vne/__init__.py`; startup preflight belongs
  in `vne_main.py`.
- VNE replay uses a custom collate that keeps variable-length LEHD state dicts
  as a list and averages loss over the examples in each minibatch.
- Dense directed substrates can create many simple paths; keep smoke tests
  small or add a candidate path cutoff before scaling.

## Experiment Roadmap

1. Generate fresh multi-request validation/test pickles for the chosen config.
2. Compare search trajectory objective against ILP objective on a tiny dataset.
3. Run legacy replay, multi-request line, and multi-request ER smoke tests.
4. Run a short Gumbeldore training cycle and inspect generated datasets.
5. Add objective-gap logging.
6. Use the LEHD policy as the graph-aware baseline before larger experiments.
