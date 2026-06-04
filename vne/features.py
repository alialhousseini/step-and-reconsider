from __future__ import annotations

from typing import Dict, List, Tuple

import torch

Path = Tuple[int, ...]
Edge = Tuple[int, int]

NODE_FEATURE_DIM = 8
EDGE_FEATURE_DIM = 4
VIRTUAL_FEATURE_DIM = 9
CANDIDATE_FEATURE_DIM = 9


def requests_from_instance(instance: Dict) -> List[Dict]:
    if "requests" in instance:
        return instance["requests"]
    return [instance["request"]]


def path_edges(path: Path) -> List[Edge]:
    return list(zip(path[:-1], path[1:]))


def comp_at(
    compute_attachment: Dict[int, int],
    residual_compute: Dict[int, int],
    comm_node: int,
) -> int:
    comp_id = compute_attachment.get(comm_node)
    if comp_id is None:
        return 0
    return residual_compute.get(comp_id, 0)


def _safe_denominator(value: float) -> float:
    return max(float(value), 1.0)


def _normalizers(instance: Dict) -> Dict[str, float]:
    substrate = instance["substrate"]
    requests = requests_from_instance(instance)
    bandwidth_values = list(substrate["communication_bandwidth"].values())
    compute_values = list(substrate["compute_capacity"].values())
    proc_demands = [
        demand
        for request in requests
        for demand in request["processing_link_demands"]
    ]
    compute_demands = [
        demand
        for request in requests
        for demand in (
            request["source_link_demand"],
            request["destination_link_demand"],
        )
    ]
    return {
        "node_id": _safe_denominator(substrate["num_comm_nodes"] - 1),
        "edge_index": _safe_denominator(len(substrate["communication_edges"]) - 1),
        "bandwidth": _safe_denominator(max(bandwidth_values) if bandwidth_values else 1),
        "compute": _safe_denominator(max(compute_values) if compute_values else 1),
        "proc_demand": _safe_denominator(max(proc_demands) if proc_demands else 1),
        "compute_demand": _safe_denominator(max(compute_demands) if compute_demands else 1),
        "request_idx": _safe_denominator(len(requests) - 1),
        "link_idx": _safe_denominator(
            max(request["num_processing_nodes"] - 2 for request in requests)
            if requests
            else 1
        ),
        "chain_length": _safe_denominator(
            max(request["num_processing_nodes"] - 1 for request in requests)
            if requests
            else 1
        ),
    }


def _build_node_features(
    instance: Dict,
    residual_bandwidth: Dict[Edge, int],
    residual_compute: Dict[int, int],
    compute_attachment: Dict[int, int],
    norm: Dict[str, float],
) -> torch.Tensor:
    substrate = instance["substrate"]
    num_nodes = substrate["num_comm_nodes"]
    original_compute = substrate["compute_capacity"]
    edges = list(substrate["communication_edges"])
    in_edges = {node: [] for node in range(num_nodes)}
    out_edges = {node: [] for node in range(num_nodes)}
    for edge in edges:
        out_edges[edge[0]].append(edge)
        in_edges[edge[1]].append(edge)

    max_degree = _safe_denominator(max(len(edges), 1))
    rows = []
    for node in range(num_nodes):
        comp_id = compute_attachment.get(node)
        has_attachment = 1.0 if comp_id is not None else 0.0
        residual_comp = residual_compute.get(comp_id, 0) if comp_id is not None else 0
        original_comp = original_compute.get(comp_id, 0) if comp_id is not None else 0
        outgoing_bw = sum(residual_bandwidth.get(edge, 0) for edge in out_edges[node])
        incoming_bw = sum(residual_bandwidth.get(edge, 0) for edge in in_edges[node])
        rows.append(
            [
                float(node) / norm["node_id"],
                float(residual_comp) / norm["compute"],
                float(original_comp) / norm["compute"],
                has_attachment,
                float(outgoing_bw) / (norm["bandwidth"] * max_degree),
                float(incoming_bw) / (norm["bandwidth"] * max_degree),
                float(len(out_edges[node])) / max_degree,
                float(len(in_edges[node])) / max_degree,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _build_edge_features(
    instance: Dict,
    residual_bandwidth: Dict[Edge, int],
    norm: Dict[str, float],
) -> torch.Tensor:
    substrate = instance["substrate"]
    rows = []
    for start, end in substrate["communication_edges"]:
        original_bw = substrate["communication_bandwidth"][(start, end)]
        residual_bw = residual_bandwidth.get((start, end), 0)
        rows.append(
            [
                float(start) / norm["node_id"],
                float(end) / norm["node_id"],
                float(residual_bw) / norm["bandwidth"],
                float(original_bw) / norm["bandwidth"],
            ]
        )
    if not rows:
        return torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _virtual_status(
    request_idx: int,
    link_idx: int,
    current_request_idx: int,
    current_link_idx: int,
) -> Tuple[float, float, float]:
    if request_idx < current_request_idx or (
        request_idx == current_request_idx and link_idx < current_link_idx
    ):
        return 1.0, 0.0, 0.0
    if request_idx == current_request_idx and link_idx == current_link_idx:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def _build_virtual_features(
    instance: Dict,
    current_request_idx: int,
    current_link_idx: int,
    norm: Dict[str, float],
) -> Tuple[torch.Tensor, int]:
    rows = []
    current_virtual_index = 0
    flat_idx = 0
    for request_idx, request in enumerate(requests_from_instance(instance)):
        chain_length = request["num_processing_nodes"] - 1
        for link_idx, proc_dem in enumerate(request["processing_link_demands"]):
            source_dem = request["source_link_demand"] if link_idx == 0 else 0
            dest_dem = (
                request["destination_link_demand"]
                if link_idx == chain_length - 1
                else 0
            )
            is_past, is_current, is_future = _virtual_status(
                request_idx,
                link_idx,
                current_request_idx,
                current_link_idx,
            )
            if is_current:
                current_virtual_index = flat_idx
            rows.append(
                [
                    float(request_idx) / norm["request_idx"],
                    float(link_idx) / norm["link_idx"],
                    float(proc_dem) / norm["proc_demand"],
                    float(source_dem) / norm["compute_demand"],
                    float(dest_dem) / norm["compute_demand"],
                    is_past,
                    is_current,
                    is_future,
                    float(chain_length) / norm["chain_length"],
                ]
            )
            flat_idx += 1
    return torch.tensor(rows, dtype=torch.float32), current_virtual_index


def _build_candidate_features(
    instance: Dict,
    candidates: List[Path],
    request_idx: int,
    link_idx: int,
    residual_bandwidth: Dict[Edge, int],
    residual_compute: Dict[int, int],
    compute_attachment: Dict[int, int],
    norm: Dict[str, float],
) -> torch.Tensor:
    request = requests_from_instance(instance)[request_idx]
    chain_length = request["num_processing_nodes"] - 1
    source_dem = request["source_link_demand"] if link_idx == 0 else 0
    dest_dem = (
        request["destination_link_demand"]
        if link_idx == chain_length - 1
        else 0
    )
    proc_dem = request["processing_link_demands"][link_idx]
    rows = []
    for path in candidates:
        start, end = path[0], path[-1]
        edges = path_edges(path)
        path_bw = min(residual_bandwidth[edge] for edge in edges)
        rows.append(
            [
                float(start) / norm["node_id"],
                float(end) / norm["node_id"],
                float(len(edges)) / _safe_denominator(instance["substrate"]["num_comm_nodes"] - 1),
                float(path_bw) / norm["bandwidth"],
                float(comp_at(compute_attachment, residual_compute, start)) / norm["compute"],
                float(comp_at(compute_attachment, residual_compute, end)) / norm["compute"],
                float(source_dem) / norm["compute_demand"],
                float(dest_dem) / norm["compute_demand"],
                float(proc_dem) / norm["proc_demand"],
            ]
        )
    if not rows:
        return torch.zeros((0, CANDIDATE_FEATURE_DIM), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _build_candidate_indices(
    instance: Dict,
    candidates: List[Path],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_to_idx = {
        edge: idx
        for idx, edge in enumerate(instance["substrate"]["communication_edges"])
    }
    max_path_nodes = max((len(path) for path in candidates), default=1)
    max_path_edges = max((len(path) - 1 for path in candidates), default=1)
    node_indices = torch.zeros((len(candidates), max_path_nodes), dtype=torch.long)
    node_mask = torch.zeros((len(candidates), max_path_nodes), dtype=torch.bool)
    edge_indices = torch.zeros((len(candidates), max_path_edges), dtype=torch.long)
    edge_mask = torch.zeros((len(candidates), max_path_edges), dtype=torch.bool)

    for candidate_idx, path in enumerate(candidates):
        for pos, node in enumerate(path):
            node_indices[candidate_idx, pos] = int(node)
            node_mask[candidate_idx, pos] = True
        for pos, edge in enumerate(path_edges(path)):
            edge_indices[candidate_idx, pos] = edge_to_idx[edge]
            edge_mask[candidate_idx, pos] = True

    return node_indices, node_mask, edge_indices, edge_mask


def build_vne_state_input(
    *,
    instance: Dict,
    candidates: List[Path],
    request_idx: int,
    link_idx: int,
    residual_bandwidth: Dict[Edge, int],
    residual_compute: Dict[int, int],
    compute_attachment: Dict[int, int],
) -> Dict[str, torch.Tensor]:
    norm = _normalizers(instance)
    virtual_features, current_virtual_index = _build_virtual_features(
        instance,
        request_idx,
        link_idx,
        norm,
    )
    (
        candidate_node_indices,
        candidate_node_mask,
        candidate_edge_indices,
        candidate_edge_mask,
    ) = _build_candidate_indices(instance, candidates)
    return {
        "node_features": _build_node_features(
            instance,
            residual_bandwidth,
            residual_compute,
            compute_attachment,
            norm,
        ),
        "edge_features": _build_edge_features(instance, residual_bandwidth, norm),
        "virtual_features": virtual_features,
        "candidate_features": _build_candidate_features(
            instance,
            candidates,
            request_idx,
            link_idx,
            residual_bandwidth,
            residual_compute,
            compute_attachment,
            norm,
        ),
        "candidate_node_indices": candidate_node_indices,
        "candidate_node_mask": candidate_node_mask,
        "candidate_edge_indices": candidate_edge_indices,
        "candidate_edge_mask": candidate_edge_mask,
        "current_virtual_index": torch.tensor(current_virtual_index, dtype=torch.long),
    }
