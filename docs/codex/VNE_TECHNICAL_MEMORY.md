# VNE Technical Memory

Last updated: 2026-06-02

This note is the working technical memory for the VNE extension. The current
repo now has a config-driven multi-request chain implementation, plus broader
formulation docs for future arbitrary-graph VNE work. Future work should avoid
mixing those targets silently.

## Canonical Formulation From Docs

The strongest formulation is in `vne/vne_README.md`, with supporting prose in
`vne/PROBLEM_FORMULATION.md`.

Substrate:

- Directed graph `G(V, E)`.
- Communication nodes `C` and computational nodes `P`, with `V = C union P`.
- Communication links `E_comm` carry bandwidth.
- Computational links `E_comp` attach communication nodes to computational
  nodes and carry compute capacity.
- Paired computational directions represent one physical compute resource.

Virtual request:

- Original virtual graph `R(M, K)`.
- Original virtual nodes have compute demands.
- Original virtual communication links have bandwidth demands.
- In the augmented view, node compute demands become computational links, so
  all resource quantities live on links.

TaSaR decision:

- One action embeds one original virtual communication link `k = (u, v)`.
- The action is a complete directed substrate communication path.
- Path endpoints induce placements for `u` and `v`.
- Endpoint placements induce compute reservations.
- The full sequence is one path choice per original virtual communication link.

Open design choice from docs: the order of original virtual communication links
`K` may be fixed heuristically or learned indirectly.

## Current Implementation

The active code implements a static multi-request chain VNE case:

- `vne/config.py`: single source of truth for VNE problem size, topology,
  virtual-request count range, per-request virtual-node range, demand ranges,
  resource ranges, validation generation, and training settings.
- `vne/instance_generator.py`: config-driven directed line substrate for
  Gumbeldore/search-generated training data; each problem instance has one
  sampled-size substrate plus a sampled number of static VN requests sharing
  the same residual resources. Each request samples its own chain size.
- `vne/trajectory.py`: `BaseTrajectory` subclass with one action per
  `(request_idx, processing_link_idx)`. Actions are explicit substrate
  communication-node paths, residual resources are updated after each action,
  and terminality happens after all links of all requests are embedded.
- `vne/network.py`: LEHD-style policy over structured state features from
  `vne/features.py`.
- `vne/dataset.py`: replay dataset rebuilds the same candidate sets as
  `Trajectory`, using nested `processing_paths[request_idx][link_idx]` for
  solved data and `chosen_path` only as a legacy single-request alias. Replay
  length is the total number of actual processing links across all requests.
  Supervised replay skips recursive future-completion filtering when rebuilding
  current candidates; the saved expert path provides the future feasibility
  guarantee. Search and validation still use future-completion filtering.
- `vne/validation_set_generator.py`: config-driven validation generator and
  joint ILP solver for line or ER chain-style instances. The old CLI toy mode
  has been removed.
- `vne_main.py`: root adapter into `main_train_cycle`, `GumbeldoreDataset`,
  validation, test, and training. It now auto-ensures the canonical validation
  pickle before training and auto-ensures the supervised training pickle only
  for `learning_type="supervised"`. Search results are saved as
  `processing_paths`, `f_placements`, and `objective`.

The atomic action means:

1. choose one explicit substrate communication path for request `r`, processing
   link `ell`;
2. place/continue that request's chain so path endpoints match `F_ell` and
   `F_{ell+1}`;
3. reserve source compute only on link 0;
4. reserve destination compute only on the final processing link;
5. reserve processing bandwidth on every communication edge in the chosen path;
6. score objective as negative bandwidth-weighted hop cost.

## VNE Data Schema In Use

Raw VNE instances use:

- `instance["substrate"]`
  - `num_comm_nodes`
  - `communication_edges`
  - `communication_bandwidth`
  - `compute_attachment`
  - `compute_capacity`
- `instance["requests"]`: list of static VN requests
  - each request has:
    - `num_processing_nodes`
    - `source_link_demand`
    - `destination_link_demand`
    - `processing_link_demands`

Legacy single-request pickles with `instance["request"]` are still readable
only when they satisfy the active config ranges; new generators write
`instance["requests"]`.

Solved validation/general data may add:

- `f_placements`: nested list, one placement list per request.
- `processing_paths`: nested list, one path list per request.
- `objective`: ILP objective.
- `chosen_path`: back-compat alias only for one-request, one-link cases.

Generated gumbeldore data adds at least:

- nested `processing_paths`
- nested `f_placements`
- `objective`
- `chosen_path` only for one-request, one-link back-compat

Canonical Task 4 dataset paths:

- `data/vne/vne_validation_dataset.pickle`: validation set, auto-generated by
  `vne_main.py` if missing. Count comes from
  `VNEConfig.validation_num_instances`; generation seed comes from
  `VNEConfig.validation_generation_seed`.
- `data/vne/vne_supervised_training_dataset.pickle`: solved supervised
  training set, auto-generated by `vne_main.py` only in supervised mode if
  missing. Count comes from `VNEConfig.supervised_training_num_instances`;
  generation seed comes from `VNEConfig.supervised_training_generation_seed`.
- `data/vne/vne_gumbeldore_training_dataset.pickle`: search-generated
  Gumbeldore training set.
- `data/vne/vne_test_dataset.pickle`: optional test set path. Test generation
  is not automatic; if this file is missing, `vne_main.py` skips test
  evaluation with a warning.

Legacy dataset inventory:

- `data/vne/vne_toy_train.pickle`: legacy single-request supervised/reference
  training set.
- `data/vne/vne_toy_val.pickle`: legacy single-request validation set.
- `data/vne/vne_toy_test.pickle`: legacy single-request test set.
- `data/vne/vne_general_n8_k3_64_seed1.pickle`: solved general chain reference
  set with 8 communication nodes and chain length 3.
- `data/vne_toy_gumbeldore_dataset.pickle`: legacy search-generated
  single-request training set from an earlier local VNE run.

## Remaining Constraints

1. Gumbeldore random instance generation is line-only.

   `vne/instance_generator.py` deliberately raises for non-line topology.
   `validation_set_generator.py`, `Trajectory`, and `RandomVNEDataset` can
   handle directed graph paths from solved ER chain data, but search-generated
   random training instances are still line-topology only.

2. Compute-demand semantics are mixed.

   The rigorous formulation gives every original virtual node a compute demand.
   The current chain/ILP implementation reserves compute only for source and
   destination links; intermediate processing nodes consume no compute.

3. Objective is placeholder.

   Runtime objective is negative bandwidth-weighted hop count. The formulation
   expects revenue minus communication and computation costs.

4. Policy network is now LEHD-style.

   `vne/network.py` uses a light transformer encoder over substrate node,
   substrate edge, and virtual-link tokens, then a heavier decoder over global,
   current-link, and candidate-path tokens. The action remains one feasible
   substrate path per current virtual link.

5. Training batch semantics now use a custom collate function.

   `RandomVNEDataset` returns one structured state example per `__getitem__`.
   `vne_main.py::collate_vne_batch` batches these variable-length state dicts
   as a list, so `VNEConfig.batch_size_training` is now the actual dataloader
   minibatch size for VNE replay training.

   With `VNEConfig.custom_num_batches = None`, VNE exposes every replay
   decision in the solved training pickle. If `custom_num_batches` caps the
   epoch length, `RandomVNEDataset` samples from the full decision index instead
   of repeatedly taking the first prefix of instances.

6. Validation metric sign depends on objective convention.

   Current VNE stores `objective = -cost`, and `evaluate` returns
   `-mean_obj` to `main_train_cycle`, which is minimized. This is consistent
   only while `mean_obj` is a maximize-style value. If future code stores
   positive cost, this must change.

## Latest Local VNE Smoke Run

`results/2026-05-22--11-43-42/` contains a 10-epoch CPU toy VNE run with:

- `best_model.pt`
- `last_model.pt`
- `log.txt`

The run used `learning_type = "gumbeldore"`, `search_type = "tasar"`,
`beam_width = 8`, `replan_steps = 1`, and generated 256 toy instances per
epoch. The validation beam-1 objective stayed at `-2.625` for all epochs; test
beam-1 objective was `-2.4140625`. Treat this as a smoke result only, not a
scientific benchmark.

## Recommended Strategic Target

Use the rigorous path-action formulation as the long-term target. The current
practical milestone is static multi-request chain VNE with config-driven
generation and replay; the next larger target is dynamic arrivals or arbitrary
`R(M, K)` virtual graphs.

This gives three tiers:

1. Legacy single-request line VNE: keep as a fast smoke check.
2. Static variable-size multi-request chain VNE: sampled request counts and
   per-request chain sizes sharing one sampled-size substrate, directed
   substrate paths, ILP-backed labels.
3. General virtual-graph VNE: arbitrary original virtual communication graph
   with a chosen edge ordering policy.

## Implementation Roadmap

1. Align objective.

   Decide and document cost terms:
   communication cost, compute cost, optional revenue, and rejection/infeasible
   behavior. Keep `to_max_evaluation_fn` as maximize-value and validation as
   minimize-metric.

2. Upgrade policy representation.

   Short term: candidate scorer with richer path features and residual
   features. Medium term: graph/request encoder using shared Transformer
   building blocks or a graph neural architecture.

3. Extend random training generation beyond line topology.

   Either support ER in `instance_generator.py` with a feasibility guard, or
   route random search generation through the validation generator when
   topology is not line.

4. Add guardrails.

   Raise clear errors for empty candidate sets, unsupported topology, missing
   solution fields, invalid processing path endpoints, and objective/schema
   mismatch.

## Regression Criteria

- Legacy single-request VNE still replays on `data/vne/vne_toy_*.pickle`.
- Variable-size multi-request chain data from `vne.validation_set_generator` can
  be consumed by trajectory, dataset, validation, and test paths.
- A solved multi-request instance replays into one training example per
  `(request_idx, processing_link_idx)`.
- `vne/features.py` is the single source for model inputs used by both
  supervised replay and trajectory search.
- Search terminality happens only after all processing links of all requests
  are embedded.
- Residual bandwidth and compute updates match the formulation and ILP checker.
- Validation reports both mean objective and, when ILP objective exists, a gap
  or comparable metric.
- Unsupported general-graph data fails with a useful message until implemented.

## Open Questions

- Should intermediate processing nodes in chain VNE consume compute, or only
  source/destination link demands? The rigorous formulation suggests every
  original virtual node has compute demand; the current ILP only charges the
  endpoints.
- Should virtual-link ordering be fixed by chain order first, then generalized
  later, or should ordering become part of the action space?
- What revenue model should be used for accepted requests, and should rejected
  embeddings be part of the learning problem?
- Should substrate paths be simple paths only, bounded-length paths, or all
  feasible paths up to a configurable cutoff?
