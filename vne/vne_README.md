# Rigorous Formulation of Extended VNE for TaSAR


## 1. Purpose

The goal is to define a version of VNE that can be used inside the TaSAR pipeline, where a solution must be represented as a sequence

```text
a = (a1, a2, ..., aT)
```

of discrete decisions.

The classical VNE formulation is not suitable for this directly because it separates node placement from edge routing. The extended formulation below moves the resource-bearing quantities to the link level and makes each decision correspond to the embedding of one original virtual communication link together with the computation reservations induced by its two endpoints.

## 2. Base Notation

### 2.1 Substrate network

Let the substrate network be a directed graph

```text
G(V, E)
```

where:

- `C` is the set of communication nodes,
- `P` is the set of computational nodes,
- `V = C union P`,
- `E_comp` is the set of directed computational links between communication and computational nodes,
- `E_comm` is the set of directed communication links between communication nodes,
- `E = E_comp union E_comm`.

This matches the notation used in the uploaded drafts.

We assume:

- each computational node in `P` is attached to one communication node in `C`,
- each communication node has at most one attached computational node,
- if computational node `p in P` is attached to communication node `c in C`, then both directed links `(p, c)` and `(c, p)` exist in `E_comp`,
- these two directed computational links represent the same physical computational resource.

For each substrate communication link `e in Ecomm`, define:

```text
BWe
```

as the available bandwidth on `e`.

For each substrate computational attachment, define:

```text
CAPe,    for e in Ecomp
```

with the understanding that if `e` and `reverse(e)` are the two directions of the same attachment, then they share the same physical computation resource and are updated together.

### 2.2 Original virtual network

Let the original virtual request be a directed graph

```text
R(M, K)
```

where:

- `M` is the set of original virtual nodes,
- `K` is the set of original virtual communication links.

For each virtual node `m in M`, define its computation demand:

```text
capdem_m
```

For each original virtual communication link `k in K`, define its bandwidth demand:

```text
bwdem_k
```

This is the cleanest generalization of the earlier drafts. The earlier chain notation using `S`, `D`, and `F` is a special case of this more general graph notation.

## 3. Why the Classical Formulation Is Not Enough

In the classical VNE problem:

- virtual node placement and virtual edge routing are two different kinds of decisions,
- there are many equivalent orderings of partial decisions,
- it is difficult to represent the solution as one clean decision sequence for TaSAR.

The extended formulation fixes this by shifting the node demands to the link level.

## 4. Augmented Virtual Network

### 4.1 Construction

For each original virtual node `m in M`, introduce one auxiliary virtual node, denoted

```text
a(m)
```

and one virtual computational link

```text
k_comp(m)
```

that attaches `a(m)` to `m`.

The original virtual communication links in `K` remain unchanged.

Thus the augmented virtual graph can be written as

```text
R_aug(M_aug, K_aug)
```

where:

- `M_aug = M union A`,
- `A = {a(m) : m in M}`,
- `K_aug = K union K_comp`,
- `K_comp = {k_comp(m) : m in M}`.

### 4.2 Demands in the augmented virtual graph

For each original virtual node `m in M`, the computation demand `capdem_m` is moved to the computational link `k_comp(m)`.

For each original virtual communication link `k in K`, the bandwidth demand `bwdem_k` remains on `k`.

So after augmentation:

- all computation demands live on links in `K_comp`,
- all communication demands live on links in `K`.

### 4.3 Example

The original virtual fragment

```text
V1 --- B1 --- V2
```

becomes

```text
A1 --- V1 --- B1 --- V2 --- A2
```

where:

- `A1 --- V1` is the computational link carrying the original node demand of `V1`,
- `B1` remains the original virtual communication link and keeps its bandwidth demand,
- `V2 --- A2` is the computational link carrying the original node demand of `V2`.

This transformation is applied to the whole virtual graph.

## 5. Augmented Substrate Network

The substrate is already written in augmented form through the partition

```
V = C union P
```

and

```
E = E_comp union E_comm.
```

Its interpretation is:

- communication resources live on the communication links `E_comm`,
- computation resources live on the computational links `E_comp`,
- the computational links attached to a communication node determine how much computation can be reserved at that communication-node location.

### 5.1 Example

A substrate fragment may look like:

```text
S1 --- B1 --- S2 --- B2 --- S3
|             |             |
A1            A2            A3
```

Here:

- `S1`, `S2`, `S3` are substrate communication nodes,
- `B1`, `B2` are substrate communication links,
- `A1---S1`, `A2---S2`, `A3---S3` are substrate computation links.

## 6. Fundamental Decision Unit

This is the key point of the agreed formulation.

Let

```text
k = (u, v) in K
```

be one original virtual communication link in the original virtual graph `R(M, K)`.

Then the computational links induced by its endpoints are:

- `k_comp(u)`,
- `k_comp(v)`.

The embedding of `k` is defined as one atomic decision:

- choose one substrate communication path for `k`,
- which automatically fixes where `u` and `v` are placed,
- and therefore automatically fixes where `k_comp(u)` and `k_comp(v)` are reserved in the substrate.

So the decision unit is not:

- embed `k_comp(u)`,
- then embed `k`,
- then embed `k_comp(v)`.

Instead, the three belong to one atomic embedding operation centered on the original virtual communication link `k`.

## 7. Substrate Path Representation

For `k = (u, v) in K`, let one candidate substrate path be

```text
P_k = (c_0, c_1, ..., c_L)
```

where:

- `c_0, c_1, ..., c_L` are nodes in `C`,
- for every `ell = 0, ..., L-1`, the directed communication link

```text
(c_ell, c_{ell+1}) in E_comm.
```

Then:

- `u` is placed at communication node `c_0`,
- `v` is placed at communication node `c_L`.

This placement is not chosen separately. It is induced by the chosen communication path `P_k`.

## 8. Induced Computational Reservations

Let

```text
att(c)
```

denote the substrate computational link associated with communication node `c`, if such an attachment exists.

If the candidate path for `k = (u, v)` is

```text
P_k = (c_0, ..., c_L),
```

then the path induces the following endpoint computational reservations:

- reserve the demand of `u`, namely `capdem_u`, on `att(c_0)`,
- reserve the demand of `v`, namely `capdem_v`, on `att(c_L)`.

So once `P_k` is chosen, the reservations of `k_comp(u)` and `k_comp(v)` are fixed automatically.

## 9. Feasibility of a Candidate Path

A candidate substrate path `P_k = (c_0, ..., c_L)` for `k = (u, v) in K` is feasible if and only if all of the following hold.

### 9.1 Path validity

For every `ell = 0, ..., L-1`,

```text
(c_ell, c_{ell+1}) in E_comm.
```

### 9.2 Direction

The path follows the direction of the substrate communication links exactly.

### 9.3 Communication bandwidth

For every communication link `e` used in `P_k`,

```text
BW_e >= bwdem_k.
```

### 9.4 Left endpoint computation

The computational attachment at `c_0` exists and has enough remaining computation resource:

```text
CAP_{att(c_0)} >= capdem_u.
```

### 9.5 Right endpoint computation

The computational attachment at `c_L` exists and has enough remaining computation resource:

```text
CAP_{att(c_L)} >= capdem_v.
```

### 9.6 Residual non-negativity after reservation

After applying the induced reservations, all residual substrate resources remain non-negative.

If no feasible path exists for `k`, then `k` cannot be embedded in the current residual substrate state.

## 10. Residual Resource Update

Suppose `k = (u, v) in K` is embedded by choosing path

```text
P_k = (c_0, ..., c_L).
```

Then the substrate state is updated immediately.

### 10.1 Communication update

For each communication link `e` used by `P_k`,

```text
BW_e := BW_e - bwdem_k.
```

### 10.2 Left endpoint computation update

```text
CAP_{att(c_0)} := CAP_{att(c_0)} - capdem_u.
```

### 10.3 Right endpoint computation update

```text
CAP_{att(c_L)} := CAP_{att(c_L)} - capdem_v.
```

### 10.4 Shared-direction computational links

If `att(c)` is represented by two directed computational links in `E_comp`, then both directions are updated together because they represent the same physical computation resource.

## 11. Objective

The uploaded mathematical draft used a profit objective. We keep that form.

Let:

- `revenue(R)` be the revenue obtained if the request `R` is accepted,
- `cost_comm(e, k)` be the cost of reserving communication link `e in E_comm` for original virtual communication link `k in K`,
- `cost_comp(e, m)` be the cost of reserving computational link `e in E_comp` for original virtual node `m in M`.

Then for a complete embedding of `R`, a natural objective is:

```text
maximize
  revenue(R)
  - sum over used communication reservations
  - sum over used computational reservations.
```

More explicitly, if `P_k` is the chosen substrate communication path for each `k = (u, v) in K`, then:

```text
maximize
  revenue(R)
  - sum_{k in K} sum_{e in P_k} cost_comm(e, k)
  - sum_{k=(u,v) in K} cost_comp(att(start(P_k)), u)
  - sum_{k=(u,v) in K} cost_comp(att(end(P_k)), v).
```

The exact cost model can be refined later, but the important point is that communication cost and endpoint computation cost are both determined by the same atomic path choice.

## 12. Sequence Representation for TaSAR

TaSAR requires a sequence

```text
a = (a_1, a_2, ..., a_T).
```

In this formulation:

- each `a_t` corresponds to one original virtual communication link,
- each `a_t` is a choice of one feasible substrate communication path,
- the induced endpoint computational reservations are part of the same decision.

So one decision is:

```text
a_t = P_{k_t},
```

where `k_t in K` is the current original virtual communication link being embedded.

This means the sequence is not:

```text
node placement, edge routing, node placement, ...
```

and it is not:

```text
one substrate hop, another substrate hop, STOP.
```

Instead, it is:

```text
one complete substrate path choice per original virtual communication link.
```

## 13. Ordering of Decisions

One open question remains:

- in what order should the original virtual communication links in `K` be embedded?

Let

```text
sigma = (k_1, k_2, ..., k_|K|)
```

be an ordering of the original virtual communication links.

Then the TaSAR sequence takes the form:

```text
a = (P_{k_1}, P_{k_2}, ..., P_{k_|K|}).
```

The ordering `sigma` is still a design choice. It may be fixed heuristically or learned indirectly through the implementation.

## 14. Candidate Set View

Because each decision is a complete path choice, the natural implementation view is candidate-based.

For the current original virtual communication link `k = (u, v)`, construct a candidate set

```text
Cand(k)
```

where each element of `Cand(k)` is a feasible substrate communication path `P_k`.

Each candidate path implicitly defines:

- the placement of `u`,
- the placement of `v`,
- the reservation of the endpoint computational links,
- the communication reservations along the path,
- the resulting residual slack and cost.

The policy network should then rank the candidates in `Cand(k)`.


## 15. Final Summary

The final agreed formulation is:

- The original virtual graph is `R(M, K)`.
- Each original virtual node `m in M` has computation demand `capdem_m`.
- Each original virtual communication link `k in K` has bandwidth demand `bwdem_k`.
- The virtual graph is augmented so node demands become computational links.
- The substrate graph `G(V, E)` is already written in augmented form with `E = E_comp union E_comm`.
- Embedding one original virtual communication link `k = (u, v)` means choosing one substrate communication path `P_k`.
- Once `P_k` is chosen, `u` and `v` are fixed at the endpoints of `P_k`.
- Therefore the endpoint computational reservations are fixed automatically.
- Communication and computation resources are both subtracted immediately from the substrate.
- The TaSAR decision sequence consists of one complete substrate path choice per original virtual communication link.

That is the formulation we should use before moving on to the graph network.
