# VNE Code Walkthrough


## Big Picture

The current VNE runtime is a config-driven multi-request chain embedding scaffold:

- the substrate is a directed communication line whose size comes from
  `VNEConfig.num_substrate_comm_nodes_range` for random Gumbeldore generation;
- computational attachments are generated according to `VNEConfig`;
- the number of static virtual requests and each request's chain size are
  sampled from `VNEConfig` ranges;
- one action means choosing one complete substrate path for the current
  request's current processing link.

With the default config, each instance samples one substrate size, one request
count, and then an independent virtual-node count for each request. Larger
sampled virtual requests add more chained path decisions; larger sampled
request counts add more static requests sharing the same substrate.

## 1. Startup Flow

Execution starts in `vne_main.py`.

When you run:

the following happens:

1. a `VNEConfig` object is created;
2. `vne_main.py` ensures the canonical validation dataset exists;
3. if `learning_type == "supervised"`, `vne_main.py` also ensures the
   canonical supervised training dataset exists;
4. `main_train_cycle(...)` from the shared training code is called;
5. VNE-specific callbacks are passed into the shared engine.

Those callbacks include:

- `get_network`
- `generate_instances`
- `validate`
- `test`
- `train_for_one_epoch_gumbeldore`
- `train_for_one_epoch_supervised`

So `vne_main.py` is the adapter between the generic training/search framework and the VNE-specific code.

## 2. Configuration

Configuration lives in `vne/config.py`.

`VNEConfig` defines:

- the learning mode;
- the problem size, request count, and demand/resource ranges;
- optimizer settings;
- Gumbeldore/TASAR search settings;
- dataset paths for supervised and generated data.

Important fields in the current scaffold are:

- `learning_type = "gumbeldore"`
- `num_substrate_comm_nodes_range = (10, 20)`
- `num_virtual_requests_range = (2, 6)`
- `num_virtual_nodes_range = (2, 5)`
- `substrate_topology = "line"`
- `substrate_compute_attach_probability = 1.0`
- `substrate_edge_probability = 0.4`
- `substrate_communication_bandwidth_range = (4, 12)`
- `substrate_compute_capacity_range = (3, 10)`
- `virtual_compute_demand_range = (1, 4)`
- `virtual_communication_demand_range = (1, 4)`
- `validation_num_instances = 120`
- `validation_generation_seed = 1`
- `supervised_training_num_instances = 4096*2`
- `supervised_training_generation_seed = 0`
- `custom_num_batches = None`
- `gumbeldore_config["num_instances_to_generate"] = 256`
- `gumbeldore_config["destination_path"] = "./data/vne/vne_gumbeldore_training_dataset.pickle"`
- `gumbeldore_config["search_type"] = "tasar"`
- `gumbeldore_config["beam_width"] = 8`
- `gumbeldore_config["replan_steps"] = 1`
- `training_set_path = "./data/vne/vne_supervised_training_dataset.pickle"`
- `validation_set_path = "./data/vne/vne_validation_dataset.pickle"`
- `test_set_path = "./data/vne/vne_test_dataset.pickle"`

`vne_main.py` generates missing validation data before either training route,
using `validation_num_instances` and `validation_generation_seed`. It generates
missing supervised training data only for the supervised route, using
`supervised_training_num_instances` and `supervised_training_generation_seed`.
Set `custom_num_batches = None` to train on every replay decision in the solved
training pickle. If a capped `custom_num_batches` value is provided, VNE samples
replay decisions from the full decision pool rather than taking a prefix.
The test dataset is not auto-generated; if it is missing, test evaluation is
skipped with a warning.

Search keeps transitioning until every processing link of every sampled request
has been embedded, even when different requests in the same instance have
different numbers of processing nodes.

## 3. Shared Training Loop

The real training loop is in `core/train.py`, not in the VNE folder.

For VNE, the loop behaves like this:

1. initialize Ray;
2. build the VNE policy network;
3. create the optimizer and scheduler;
4. for each epoch:
   - generate data with search if using Gumbeldore;
   - train on that data with cross-entropy;
   - validate the current model;
   - save checkpoints;
5. optionally run test evaluation at the end.

So VNE is using the same shared training framework as the other problems.

## 4. Instance Generation

Instances are created in `vne/instance_generator.py`.

This file defines functions such as:

- `make_substrate_instance(config)`
- `make_virtual_request(config)`
- `make_instance(config)`
- `make_dataset(config, num_instances)`
- helper functions for line-path feasibility

### 4.1 Substrate

The substrate communication graph is a line with size sampled from
`VNEConfig.num_substrate_comm_nodes_range`. A sampled four-node substrate would
look like:

```text
C0 -> C1 -> C2 -> C3
```

Communication nodes receive computational attachments according to `VNEConfig`.
`instance_generator.py` currently supports random generation for
`substrate_topology = "line"`; the validation generator also supports ER
topologies.

The file generates values from `VNEConfig`:

- communication bandwidth for line edges;
- computational capacity at communication locations.

### 4.2 Virtual Requests

The virtual request count, size, and demands come from `VNEConfig`. Each
instance samples a request count from `num_virtual_requests_range`, and each
request samples its own size from `num_virtual_nodes_range`, so one instance can
look like:

```text
request 0: F0 -> F1
request 1: F0 -> F1 -> F2
request 2: F0 -> F1 -> F2 -> F3
```

For each request, the file generates:

- a source-link compute demand for `F_0`;
- a destination-link compute demand for the final processing node;
- one bandwidth demand per processing link.

For a request with `num_processing_nodes = 3`, the processing chain has two
path decisions:

```text
F0 -> F1 -> F2
```

### 4.3 Candidate Paths

For line-generated instances, the candidate action space is all contiguous
directed substrate paths. With four communication nodes:

- `(0, 1)`
- `(0, 1, 2)`
- `(0, 1, 2, 3)`
- `(1, 2)`
- `(1, 2, 3)`
- `(2, 3)`

Each tuple means:

- start at the first substrate communication node;
- end at the final substrate communication node;
- use each consecutive edge in the tuple.

For solved validation data with general directed edges, candidates are explicit
simple directed paths built from `substrate["communication_edges"]`.

### 4.4 Feasibility

An instance must have at least one feasible embedding for all static requests
on the shared substrate.

A path is feasible only if:

- the first path start has enough compute for the source-link demand;
- the final path end has enough compute for the destination-link demand;
- each communication edge in every path has enough bandwidth for that
  processing-link demand;
- later processing paths start where the previous processing path ended.
- all requests share the same residual bandwidth and compute resources.

The generator keeps resampling until every request in the instance can be
embedded together.

## 5. What One Action Means

This is the key modeling choice.

One action is not:

- one substrate hop;
- one separate node placement;
- one compute reservation alone.

Instead, one action is one complete substrate path choice for the current
request's current processing link.

For example, if the chosen path is `(1, 2, 3)`, this means:

- place the current processing-link source at substrate node `1`;
- place the current processing-link destination at substrate node `3`;
- reserve source compute at node `1` only if this is the first processing link;
- reserve destination compute at node `3` only if this is the final processing link;
- reserve bandwidth on communication edges `(1,2)` and `(2,3)`.

So the endpoint placement and the communication reservation happen together in one atomic decision.

## 6. Trajectory State

The search state is defined in `vne/trajectory.py`.

The `Trajectory` dataclass stores:

- the original instance;
- residual bandwidth on substrate communication edges;
- residual compute on substrate nodes;
- the list of feasible action candidates;
- the current request index;
- the current processing-link index;
- the nested selected `processing_paths`;
- the nested induced `f_placements`;
- the final objective.

### 6.1 Initializing a Trajectory

`init_batch_from_instance_list(...)` does the following for each instance:

1. read the substrate and virtual requests;
2. enumerate all candidate paths for request 0, processing link 0;
3. keep only candidates that can complete all remaining links and requests;
4. create a fresh `Trajectory` holding:
   - residual bandwidth copied from the instance;
   - residual compute copied from the instance;
   - the feasible action list.

So the root trajectory is an unsolved multi-request VNE instance plus its
current path candidates.

## 7. LEHD State Features

`vne/features.py` builds the structured input used by both supervised replay and
trajectory search. For each decision it returns:

1. substrate node features;
2. substrate edge features;
3. virtual-link features for all requests;
4. candidate-path features;
5. padded candidate node and edge indices with masks;
6. the index of the current virtual link.

The candidate-path row still includes the original 9 summary features:

1. start node index
2. end node index
3. path length in hops
4. bottleneck bandwidth along the path
5. residual compute at the start node
6. residual compute at the end node
7. source-link demand, nonzero only for the first processing link
8. destination-link demand, nonzero only for the final processing link
9. bandwidth demand of the current processing link

So if an instance has `N` feasible path candidates, the network receives one
structured state dict with `N` candidate rows plus substrate node, substrate
edge, and virtual-link tensors.

## 8. Policy Network

The model lives in `vne/network.py`.

The scaffold uses a custom LEHD-style `VNEPolicyNetwork`:

1. encode substrate node, substrate edge, and virtual-link tokens;
2. pool encoded node/edge representations along each candidate path;
3. decode global state, current virtual link, and candidate path tokens;
4. output one scalar score for each candidate path.

During a forward pass:

- each candidate path gets one logit;
- the candidate list for a trajectory becomes a list of action scores.

This is intentionally simple. It is only meant to validate the VNE interface and training flow before introducing a more graph-aware encoder.

## 9. Search-Time Log Probabilities

The function `Trajectory.log_probability_fn(...)` connects the VNE network to the shared search code.

It does the following:

1. build candidate feature tensors for each trajectory;
2. call the policy network;
3. apply `log_softmax` over the candidate logits;
4. return log-probabilities to the shared search engine.

The shared search code does not need to know it is solving VNE. It only sees:

- a list of action log-probabilities;
- a transition function;
- an evaluation function.

## 10. Transition Function

`Trajectory.transition_fn(...)` applies one chosen action.

The method:

1. copies the current trajectory;
2. reads the chosen path from the action index;
3. subtracts source-link compute from the path start only on the first link;
4. subtracts destination-link compute from the path end only on the final link;
5. subtracts bandwidth demand from each communication edge in the path;
6. appends the path and induced placement;
7. computes the objective;
8. advances to the next processing link, or to the next request when the
   current request is complete;
9. returns terminal only after every processing link of every request has been
   embedded.

## 11. Objective

The current scaffold uses a simple objective:

```text
cost = path_hops * bandwidth_demand
objective = -cost
```

This means:

- shorter paths are preferred;
- larger bandwidth-weighted paths are more expensive;
- the search still maximizes because the cost is negated.

This is only a placeholder. A fuller VNE version would likely use a more explicit revenue-minus-cost objective.

## 12. Gumbeldore Data Generation

In `vne_main.py`, the function `get_gumbeldore_dataloader(...)` creates a `GumbeldoreDataset`.

That shared component handles:

1. generating raw VNE instances;
2. sending them to Ray workers;
3. building `Trajectory` objects in each worker;
4. running the configured search method, such as TASAR;
5. collecting finished results;
6. saving them as a generated dataset.

### 12.1 Result Packaging

The VNE-specific function `beam_leaves_to_result(...)` converts finished trajectories into stored results.

In the current scaffold, it returns:

- `processing_paths`
- `f_placements`
- `objective`
- `chosen_path` only when there is one request and one processing link

Then `save_search_results_to_dataset(...)` copies those values back into each original instance and writes the resulting list to a pickle file.

So a generated training example contains:

- the substrate;
- the virtual requests;
- the chosen solution paths and processing-node placements;
- the resulting objective.

## 13. Supervised Replay Dataset

Replay logic lives in `vne/dataset.py`.

The class `RandomVNEDataset` turns solved instances into next-action training examples.

For each stored instance:

1. choose the `(request_idx, processing_link_idx)` position implied by the
   solved instance;
2. replay all earlier `processing_paths` in the stored request/link order to
   reconstruct residual resources;
3. rebuild the feasible candidate set for the current request/link decision;
4. read the saved path from nested `processing_paths`, or `chosen_path` for
   legacy one-request data;
5. find the index of that path inside the candidate list;
6. return:
   - `state`
   - `next_action_idx`

With variable-size multi-request chain data, the dataset cycles through one
replay example per actual `(request_idx, processing_link_idx)` in the solved
instance.

This mirrors the pattern used by the other problem folders: save a full solution, then replay it into next-step supervision.
Replay candidate construction checks only current local feasibility; it does
not recursively prove future completion for every candidate, because the saved
expert solution already proves the replay prefix can be completed.

## 14. Actual Training Step

Training happens in `train_with_dataloader(...)` inside `vne_main.py`.

For each batch:

1. load up to `batch_size_training` replayed training examples;
2. keep each variable-length structured VNE state as one list item;
3. send the list of state dicts to the LEHD network;
4. receive one logits vector over candidate paths per replay example;
5. compare each logits vector against its correct candidate index;
6. average `CrossEntropyLoss` across the examples;
7. backpropagate;
8. update the network parameters.

So the learning rule is still standard next-action supervised learning, but the
policy now uses a light encoder and heavier decoder instead of the old MLP path
scorer.

That is true in both training modes:

- in `supervised` mode, labels come from an existing solved dataset;
- in `gumbeldore` mode, labels come from search-generated pseudo-expert solutions.

## 15. Validation and Test

Validation and test are handled in `evaluate(...)` inside `vne_main.py`.

That function:

1. loads stored validation or test instances;
2. runs search to choose a path for each instance;
3. collects the returned objectives;
4. reports the mean objective.

At this stage, evaluation only logs the mean objective. It does not yet compute optimality gaps, because the default data does not include a separate ground-truth optimum field.
