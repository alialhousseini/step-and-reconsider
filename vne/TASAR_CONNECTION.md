# VNE and TaSaR Connection

This file is a first draft describing how the extended VNE formulation could be connected to the Take a Step and Reconsider (TaSaR) pipeline. It is intentionally preliminary and should be reviewed after the problem and mathematical formulations are finalized.

## Goal

TaSaR expects a solution to be represented as a sequence of decisions:

```text
a = (a_1, a_2, ..., a_T)
```

For VNE, the main challenge is to define each action a_t so that:

- the sequence has a clear order,
- each action has a well-defined meaning,
- feasibility can be checked incrementally,
- the representation is compatible with both computational-resource allocation and communication-path routing.

## Proposed Decision Order

For a chain virtual network:

```text
S -> F_1 -> F_2 -> ... -> F_n -> D
```

a possible allocation sequence is:

```text
a = (
  source_attachment,
  path(F_1 -> F_2),
  path(F_2 -> F_3),
  ...,
  path(F_{n-1} -> F_n),
  destination_attachment
)
```

where:

- `source_attachment` selects the computational link used by S -> F_1.
- `path(F_i -> F_j)` selects a directed substrate communication path for the processing link F_i -> F_j.
- `destination_attachment` selects the computational link used by F_n -> D.

## Alternative Fine-Grained Decision Order

Instead of selecting a full path in one action, TaSaR may use edge-level actions:

```text
a = (
  source_attachment,
  edge_1_for(F_1 -> F_2),
  edge_2_for(F_1 -> F_2),
  stop_path,
  edge_1_for(F_2 -> F_3),
  ...
  destination_attachment
)
```

This representation may be easier for a step-by-step model, but it requires explicit stopping actions and stronger incremental feasibility checks.

## State Representation

At each TaSaR step, the state should include:

- the residual capacity of each computational link,
- the residual bandwidth of each communication link,
- the virtual link currently being allocated,
- the partial directed path if the current virtual link is a processing link,
- the current communication node reached by the partial path,
- a mask of feasible next actions.

## Feasibility Checks

A candidate action is feasible only if:

- it selects a compatible substrate resource,
- the selected substrate resource has enough residual capacity or bandwidth,
- directed communication edges are followed in their correct direction,
- the action does not break the required source-to-destination structure of the current virtual link,
- after completing an allocation step, residual resources remain non-negative.

For computational links, both directions of the same computational attachment are updated together.

## Reconsideration Step

In TaSaR, the model may first propose a partial or complete action sequence and then reconsider it. For VNE, reconsideration could be used to:

- replace an infeasible substrate path,
- choose a cheaper path with enough residual bandwidth,
- move source or destination attachment to a computational link with more residual capacity,
- reject the virtual request if no feasible allocation exists.

## Open Design Questions

- Should a path be selected as one action, or should each substrate communication edge be selected as one action?
- Should the model explicitly place processing nodes F_i on computational nodes P_j, or is the current link-level representation sufficient?
- Should infeasible actions be masked completely, or allowed and penalized?
- Should the sequence always end with a complete embedding, or should request rejection be a valid terminal action?
