# Mathematical Formulation

This file provides a first mathematical formulation of the extended Virtual Network Embedding (VNE) problem described in `PROBLEM_FORMULATION.md`. The goal is to make the objects, decision variables, constraints, and objective explicit enough to check the intended behavior before implementation.

## Sets

Let the substrate network be a directed graph G(V, E), where:

- C is the set of communication nodes.
- P is the set of computational nodes.
- V = C \union P.
- E_comp is the set of directed computational links between communication and computational nodes.
- E_comm is the set of directed communication links between communication nodes.
- E = E_comp \union E_comm.

Let the virtual network request be a directed graph R(M, K), where:

- S is the single source node.
- D is the single destination node.
- F is the ordered set of processing nodes.
- M = {S} \union F \union {D}.
- K_src contains the single source link S -> F_1.
- K_proc contains the directed processing links F_i -> F_j.
- K_dst contains the single destination link F_z -> D.
- K = K_src \union K_proc \union K_dst.

For the current project version, the virtual network is assumed to be a chain:

```text
S -> F_1 -> F_2 -> ... -> F_n -> D
```

## Parameters

For substrate resources:

- CAP_e: available capacity on computational link e \in E_comp.
- BW_e: available bandwidth on communication link e \in E_comm.

For virtual demands:

- dem_k: demand of virtual link k \in K.

For economics:

- revenue(R): revenue obtained if virtual request R is accepted.
- cost_comp(e, k): cost of assigning source or destination demand k to computational link e.
- cost_comm(e, k): cost of routing processing demand k through communication link e.

## Decision Variables

Computational-link allocation:

- x_{k,e} \in {0,1}: equals 1 if source or destination virtual link k \in K_src \union K_dst is allocated on computational link e \in E_comp.

Communication-link allocation:

- y_{k,e} \in {0,1}: equals 1 if processing virtual link k \in K_proc uses communication link e \in E_comm.

Acceptance:

- z \in {0,1}: equals 1 if the virtual network request is accepted, and 0 otherwise.

## Constraints

### Source and Destination Allocation

Each source and destination link must be allocated to exactly one compatible computational link if the request is accepted:

```text
sum_{e \in E_comp} x_{k,e} = z,    for all k \in K_src \union K_dst
```

### Computational Capacity

The total demand allocated to each computational link cannot exceed its available capacity:

```text
sum_{k \in K_src \union K_dst} dem_k x_{k,e} <= CAP_e,    for all e \in E_comp
```

If e and reverse(e) are the two directions of the same computational attachment, then they represent the same resource and must be updated together:

```text
residual_CAP_e = residual_CAP_reverse(e)
```

### Processing-Link Flow Conservation

Each processing link k = F_i -> F_j must be routed as a directed path through communication links. Let src(k) be the communication node associated with the computational placement of F_i, and dst(k) be the communication node associated with the computational placement of F_j.

For each communication node c \in C:

```text
sum_{e outgoing from c} y_{k,e} - sum_{e incoming to c} y_{k,e}
  =  1   if c = src(k)
  = -1   if c = dst(k)
  =  0   otherwise
```

This enforces that processing demands follow the direction of substrate communication links exactly.

### Communication Bandwidth

The total processing demand routed through each communication link cannot exceed its available bandwidth:

```text
sum_{k \in K_proc} dem_k y_{k,e} <= BW_e,    for all e \in E_comm
```

### Compatibility

Source and destination links can only be allocated on computational links:

```text
x_{k,e} is defined only for k \in K_src \union K_dst and e \in E_comp
```

Processing links can only be routed on communication links:

```text
y_{k,e} is defined only for k \in K_proc and e \in E_comm
```

## Objective

A simple objective is to maximize profit:

```text
maximize
  z * revenue(R)
  - sum_{k \in K_src \union K_dst} sum_{e \in E_comp} cost_comp(e,k) x_{k,e}
  - sum_{k \in K_proc} sum_{e \in E_comm} cost_comm(e,k) y_{k,e}
```

Another possible objective is to maximize the revenue-to-cost ratio, but this is nonlinear and may require reformulation or approximation. For a first MILP formulation, the profit objective is simpler and easier to validate.

## Notes To Review

- This draft treats source and destination demands as allocations on computational links.
- This draft treats processing demands as directed paths on communication links.
- The exact relation between virtual processing nodes F_i and computational nodes P_j still needs to be finalized if the implementation should explicitly model function placement.
- The acceptance variable z allows request rejection. If the single request must always be embedded, fix z = 1.
