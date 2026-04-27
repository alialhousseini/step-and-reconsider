# Problem Formulation

In this file, we would like to describe the problem of Virtual Network Embedding (VNE) extended to be used in the format of Take a Step and Reconsider (TaSaR). This folder, like other folders (e.g., `tsp/`, `jssp/`, and `cvrp/`) will contain the necessary and essential files to train models on this problem. However, we first provide here a clear description and explanation of the VNE.

## Problem Definition
The Virtual Network Embedding problem has been widely used and been defined as follows:
- We have a substrate network graph (SN): A directed graph made of a set of nodes and edges
- Substrate nodes: Nodes carrying capacities
- Substrate edges: Edges between nodes carrying some bandwidth
- A Virtual Network (VN): A directed graph made of a set of nodes and edges
- Virtual Nodes: Nodes carrying some capacity demands (they demand this amount of capacity)
- Virtual edges: Edges requiring some amount of bandwidth

The goal is to find a one-to-one placement (of nodes) and routing (of edges) that respects some specific constraints such as:
1. Total capacity used over a specific node is less than available
2. Same applies for bandwidth of edges
3. Virtual edges, can be allocated over a path on the substrate network
4. The goal is to maximize revenue over cost.

## Why this famous formulation fails here?
This formulation fails because we would like to generate a tuple `a=(a1,a2,a3,...)` such that this tuple represents a possible feasible solution that we would like to train our model on. By using the current approach, the search space is large, and many equivalent solutions may exist under different tuple orderings. Also, we cannot have some elements in the tuple related to nodes, and others to edges.

## Solution: Extended VNE
In this section we describe the extended VNE that we would like to implement. It extends VNE by changing the structure of the problem.

### Definition of elements
We define two graphs: G(V,E) (substrate network) and R(M,K) (virtual network), both graphs are directed.

#### Substrate network G
The substrate network is defined by a set of nodes V and a set of interconnections (edges) E.
The set of nodes V contains **two types** of nodes: computational nodes P, and communication nodes C. Thus, V = C \union P.
Similarly, we define **two types** of edges: computational links C -> P (or, P -> C) and communication links (C -> C). This means that a connection between a computational node P_i and a communication node C_j is represented by a computational link (and vice versa). Whereas, a connection between two communication nodes C_j and C_k is represented by a communication link.

As a summary: The substrate network G contains two types of nodes, and two types of edges. This can be seen as an augmented version of the original substrate in the VNE.

**NOTE**: For each communication node, we have at most a single computational node connected to it.

**NOTE**: If a computational node P_i is attached to a communication node C_i, then both directed computational links P_i -> C_i and C_i -> P_i exist. During allocation, both directions represent the same computational resource, and we always deduct the allocated capacity from both directions at the same time.

```
    P_i <---> C_i ---> C_j <--- C_k <---> P_k
               |
              C_z --> C_t
                       |
                      P_t
```
In this example, we see that we always have a bidirectional connection between the communication node and its computational node. However, between communication nodes, it is not necessary to have this.

> Now we will move to define the attributes of each object defined so far in this subsection

In this formulation, nodes, regardless of their type, DO NOT carry any kind of information; all information is defined over the link-level. 

For each computational link: we denote by CAP(P_i -> C_i) = CAP(C_i -> P_i) the capacity that this link holds.
*Side note*: We originally have an initial capacity, and later, we reserve from it when a virtual network routes its demand through this link.

For each communication link: we denote by BW(C_i -> C_j) the bandwidth that this link holds.
*Side note*: We originally have an initial bandwidth, and later, we reserve from it when a virtual network routes its demand through this link.

#### Virtual Network R
The virtual network is defined by a set of nodes M and a set of interconnections (edges) K.
The set of nodes M contains three types of nodes: source node S, processing nodes F, and destination node D, then M = S \union F \union D.

Similarly, we define **three types** of edges: source link S -> F_i which connects the source node of the virtual network to a processing node; a processing link (F_i -> F_j) that connects a processing function to another; and a destination link (F_z -> D) that connects a processing node to a destination node.

As a summary: The virtual network R contains three types of nodes, and three types of edges. This can be seen as an augmented version of the original virtual network in the VNE.

**NOTE**: For each virtual network R, we have a single source node S (and thus, a single source link) and a single destination node D (and thus, a single destination link).

**NOTE**: A processing demand from F_i -> F_j must follow the direction of the substrate communication links exactly.

```
  S --(1)--> F_i --(2)--> F_j --(3)--> F_k --(4)--> D
```
where, 
(1) is a source link
(2) and (3) are processing links
and (4) is a destination link

In this example, we see that in the current version of this project, we will assume only virtual networks of the chain-style, i.e, a chain of linear consecutive nodes.

> Now we will move to define the attributes of each object defined so far in this subsection

In this formulation, nodes, regardless of their type, DO NOT carry any kind of information; all information is defined over the link-level. 

For each source link and destination link: we denote by dem(S -> F_i), and similarly, dem(F_x -> D), the demands of source and destination links respectively.
*Side note*: During the allocation process, source and destination links will be placed on computational links of the substrate network.

For each processing link: we denote by dem(F_i -> F_j) the bandwidth **required** by this link.
*Side note*: During the allocation process, processing links will be placed on communication links of the substrate network.

### Scope of this document
This document focuses on a single allocation between one substrate network and one virtual network. In the complete VNE setting, requests may arrive online, and the residual capacities of the substrate network change after each accepted allocation. Here, we describe only one allocation step to clarify how the extended VNE representation works.

### An example of an allocation process
Assume the following substrate network:

```
  P_1 <---> C_1 ---> C_2 <---> P_2
                \       |
                 v      v
                  C_3 -> C_4 <---> P_4
```

The computational links have the following capacities:

- CAP(P_1 -> C_1) = CAP(C_1 -> P_1) = 10
- CAP(P_2 -> C_2) = CAP(C_2 -> P_2) = 8
- CAP(P_4 -> C_4) = CAP(C_4 -> P_4) = 12

The communication links have the following bandwidths:

- BW(C_1 -> C_2) = 7
- BW(C_1 -> C_3) = 5
- BW(C_2 -> C_4) = 6
- BW(C_3 -> C_4) = 4

Assume the following virtual network:

```
  S --3--> F_1 --4--> F_2 --5--> D
```

where:

- dem(S -> F_1) = 3
- dem(F_1 -> F_2) = 4
- dem(F_2 -> D) = 5

One feasible allocation is:

- Place the source link S -> F_1 on the computational link C_1 -> P_1. Since computational links are paired, reserve 3 units from both C_1 -> P_1 and P_1 -> C_1.
- Place the processing link F_1 -> F_2 on the directed communication path C_1 -> C_2 -> C_4. Reserve 4 units from C_1 -> C_2 and C_2 -> C_4.
- Place the destination link F_2 -> D on the computational link C_4 -> P_4. Since computational links are paired, reserve 5 units from both C_4 -> P_4 and P_4 -> C_4.

After this allocation, the residual capacities and bandwidths are:

- CAP(P_1 -> C_1) = CAP(C_1 -> P_1) = 7
- CAP(P_4 -> C_4) = CAP(C_4 -> P_4) = 7
- BW(C_1 -> C_2) = 3
- BW(C_2 -> C_4) = 2

This allocation is feasible because all source, processing, and destination demands are placed on compatible substrate links, all residual resources remain non-negative, and the processing demand follows the direction of the substrate communication links.


### Conclusion
