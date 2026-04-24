# Problem Formulation

In this file, we would like to describe the problem of Virtual Network Embedding (VNE) extended to be used in the format of Take a Step and Reconsider (TaSaR). This folder, like other folder (e.g., `tsp/` , `jssp/`, and `cvrp/`) will contain the necessary and essential files to train models on this problem. However, we provide first here a clear description and explanation of the VNE.

## Problem Definition
The Virtual Network Embedding problem has been widely used and been defined as follows:
- We have a substrate network graph (SN): A directed graph made of a set of nodes and edges
- Substrate nodes: Nodes carrying capacities
- Substrate edges: Edges between nodes carrying some bandwidth
- A Virtual Network (VN): A directed graph made of a set of nodes and edges
- Virtual Nodes: Nodes carrying some capacity demands (they demand this amount of capacity)
- Virtual edges: Edges requiring some amount of bandwdith

The goal is to find a one-to-one placement (of nodes) and routing (of edges) such that respecting some specific constraints such as:
1. Total capacity used over a specific node is less than available
2. Same applies for bandwidth of edges
3. Virtual edges, can be allocated over a path on the substrate network
4. The goal is to maximize revenue over cost.

## Why this famous formulation fails here?
This formulation fails because we would like to generate a tuple `a=(a1,a2,a3,...)` such that this tuple represents a possible feasible solution that we would like to train our model on. By using the current approach, the search space is large, the tuple are permutationally-invariant. And we cannot have some elements in the tuple related to nodes, and others to edges.

## Solution: Extended VNE
In this section we describe the extended VNE that we would like to implement. Its extends VNE by changing the strcuture of the problem

### Definition of elements
We define two graphs: G(V,E) (substrate network) and R(M,K) (virutal network), both graphs are directed.

#### Substrate network G
The substrate network is defined by a set of nodes V and a set of interconnections (edges) E.
The set of nodes V contains two types of nodes: computational nodes P, and communication nodes C. Thus, V = C \union P
Similarly, we define two types of edges: computational links C -> P (or, P -> C) and communication links (C -> C). This means that a connection between a computational node P_i and a communication node C_j is represented by a computational link (and vice versa). Whereas, a connection between two communication nodes C_j and C_k is represented by a communication link.

As a summary: The substrate network G, contains two type of nodes, and two type of edges. This can be seen as an augmented version of the original substrate in the VNE.

**NOTE**: For each communication node, we have at most a single computational node connected to it.

```
    P_i <---> C_i ---> C_j <--- C_k <---> P_k
               |
              C_z --> C_t
                       |
                      P_t
```
In this example, we see that we always have bidirectional connection between the communication node and its computational node. However, between communication nodes, its is not necessary to have this.

> Now we will move to define the attributes of each object defined so far in this subsection

In this formulation, nodes, regardless of their type, they DO NOT carry any kind of infromation, all information are defined over the link-level. 

For each computational link: we denote by CAP(P_i -> C_i) = CAP(C_i -> P_i) as the capacity that this link holds
*Side note*: We originally, have an initial capacity, and later, we reserve from it when a virtual network routes its demand through this link

For each communication link: we denote by BW(C_i -> C_j) as the bandwidth that this link holds
*Side note*: We originally, have an initial bandwidth, and later, we reserve from it when a virtual network routes its demand through this link

#### Virtual Network R
The vritual network is defined by a set of nodes M and a set of interconnections (edges) K.
The set of nodes M contains three types of nodes: source node S, processing nodes F, and destination node D, then M = S \union F \unioon D.

Similarly, we define three types of edges: source link S -> F_i which connects the source node of the virtual network to a processing node; And processing link (F_i -> F_j) that connects a processing function to another; and a destination link (F_z -> D) that connects a processing node to a destination node.

As a summary: The virtual network R, contains three types of nodes, and three types of edges. This can be seen as an augmented version of the original virutal in the VNE.

**NOTE**: For each virutal network R, we have a single source node S (and thus, a single source link) and a single destination node D (and thus, a single destination link)

```
  S --(1)--> F_i --(2)--> F_j --(3)--> F_k --(4)--> D
```
where, 
(1) Is a source link
(2) (3) and (4) are processing links
and (4) is destination link

In this example, we see that in the current version of this project, we will assume only virtual networks of the chain-style, i.e, a chain of linear consecutive nodes.

> Now we will move to define the attributes of each object defined so far in this subsection

In this formulation, nodes, regardless of their type, they DO NOT carry any kind of infromation, all information are defined over the link-level. 

For each source link, and destination link: we denote by dem(S -> F_i), and similarly, dem(F_x -> D) the demands of source and destination links respectively.
*Side note*: During the allocation process, source and destination links will be placed on computational links of the substrate network.

For each processing link: we denote by dem(F_i -> F_j) as the bandwidth **required** that this link is demanding.
*Side note*: During the allocation process, processing links will be placed on communication links of the substrate network.

### An example of a allocation process 
- Appearence of source and destination on the SUBSTRATE


### Conclusion