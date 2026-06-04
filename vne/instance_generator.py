"""Substrate/request generator aligned with `vne/PROBLEM_FORMULATION.md`.

Substrate: line of communication nodes with config-driven size range,
computational attachments, and resource ranges.

Request: each virtual network is a chain `S -> F_0 -> ... -> F_k -> D`.
Demands live on links only: a source-link demand, a destination-link demand,
and processing-link demands. One problem instance contains one substrate and
a sampled number of static virtual requests sharing its resources.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple


def make_substrate_instance(config) -> Dict:
    low, high = config.num_substrate_comm_nodes_range
    if low < 2 or high < low:
        raise ValueError("num_substrate_comm_nodes_range must be ordered and have lower bound >= 2")
    num_comm_nodes = random.randint(low, high)
    if config.substrate_topology != "line":
        raise ValueError("vne.instance_generator currently supports only line topology")
    communication_edges = [
        (node, node + 1)
        for node in range(num_comm_nodes - 1)
    ]
    communication_bandwidth = {
        edge: random.randint(*config.substrate_communication_bandwidth_range)
        for edge in communication_edges
    }
    compute_attachment = {
        c: c
        for c in range(num_comm_nodes)
        if random.random() < config.substrate_compute_attach_probability
    }
    compute_capacity = {
        comp_id: random.randint(*config.substrate_compute_capacity_range)
        for comp_id in compute_attachment.values()
    }
    return {
        "num_comm_nodes": num_comm_nodes,
        "communication_edges": communication_edges,
        "communication_bandwidth": communication_bandwidth,
        "compute_attachment": compute_attachment,
        "compute_capacity": compute_capacity,
    }


def make_virtual_request(config) -> List[Dict]:
    request_low, request_high = config.num_virtual_requests_range
    node_low, node_high = config.num_virtual_nodes_range
    if request_low < 1 or request_high < request_low:
        raise ValueError("num_virtual_requests_range must be ordered and have lower bound >= 1")
    if node_low < 2 or node_high < node_low:
        raise ValueError("num_virtual_nodes_range must be ordered and have lower bound >= 2")

    requests = []
    for _ in range(random.randint(request_low, request_high)):
        num_processing_nodes = random.randint(node_low, node_high)
        chain_length = num_processing_nodes - 1
        requests.append(
            {
                "num_processing_nodes": num_processing_nodes,
                "source_link_demand": random.randint(*config.virtual_compute_demand_range),
                "destination_link_demand": random.randint(*config.virtual_compute_demand_range),
                "processing_link_demands": [
                    random.randint(*config.virtual_communication_demand_range)
                    for _ in range(chain_length)
                ],
            }
        )
    return requests


def path_to_edges(path: Tuple[int, int]) -> List[Tuple[int, int]]:
    start, end = path
    return [(node, node + 1) for node in range(start, end)]


def _comp_capacity_at(substrate: Dict, residual_compute: Dict[int, int], comm_node: int) -> int:
    attach = substrate.get("compute_attachment")
    if attach is None:
        return residual_compute.get(comm_node, 0)
    comp_id = attach.get(comm_node)
    if comp_id is None:
        return 0
    return residual_compute.get(comp_id, 0)


def _has_feasible_embedding(instance: Dict) -> bool:
    substrate = instance["substrate"]
    requests = instance["requests"]
    residual_bandwidth = dict(substrate["communication_bandwidth"])
    residual_compute = dict(substrate["compute_capacity"])
    attach = substrate.get("compute_attachment") or {}

    def search(request_idx: int, link_idx: int, start_node: int | None) -> bool:
        request = requests[request_idx]
        chain_length = request["num_processing_nodes"] - 1
        source_dem = request["source_link_demand"] if link_idx == 0 else 0
        dest_dem = request["destination_link_demand"] if link_idx == chain_length - 1 else 0
        proc_dem = request["processing_link_demands"][link_idx]
        starts = [start_node] if start_node is not None else range(substrate["num_comm_nodes"])
        for start in starts:
            if source_dem and _comp_capacity_at(substrate, residual_compute, start) < source_dem:
                continue
            for end in range(start + 1, substrate["num_comm_nodes"]):
                if dest_dem and _comp_capacity_at(substrate, residual_compute, end) < dest_dem:
                    continue
                edges = path_to_edges((start, end))
                if any(residual_bandwidth[edge] < proc_dem for edge in edges):
                    continue
                if source_dem:
                    residual_compute[attach[start]] -= source_dem
                if dest_dem:
                    residual_compute[attach[end]] -= dest_dem
                for edge in edges:
                    residual_bandwidth[edge] -= proc_dem
                if link_idx + 1 < chain_length:
                    feasible = search(request_idx, link_idx + 1, end)
                elif request_idx + 1 < len(requests):
                    feasible = search(request_idx + 1, 0, None)
                else:
                    feasible = True
                if feasible:
                    return True
                for edge in edges:
                    residual_bandwidth[edge] += proc_dem
                if dest_dem:
                    residual_compute[attach[end]] += dest_dem
                if source_dem:
                    residual_compute[attach[start]] += source_dem
        return False

    return bool(attach) and search(0, 0, None)


def make_instance(config) -> Dict:
    while True:
        instance = {
            "substrate": make_substrate_instance(config),
            "requests": make_virtual_request(config),
        }
        if _has_feasible_embedding(instance):
            return instance


def make_dataset(config, num_instances: int) -> List[Dict]:
    return [make_instance(config) for _ in range(num_instances)]
