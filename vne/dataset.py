"""VNE supervised replay dataset aligned with `vne/PROBLEM_FORMULATION.md`.

Each pickled instance carries:
    requests[request_idx]["source_link_demand"], ["destination_link_demand"],
    requests[request_idx]["processing_link_demands"]
    substrate["compute_attachment"], substrate["compute_capacity"]
    instance["processing_paths"]            -- one path-list per request
        OR instance["chosen_path"]          -- single-link back-compat alias
"""
from __future__ import annotations

import pickle
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from vne.config import VNEConfig
from vne.features import build_vne_state_input
from vne.trajectory import Trajectory

CandidatePath = Tuple[int, ...]


def _requests(instance: Dict) -> List[Dict]:
    if "requests" in instance:
        return instance["requests"]
    return [instance["request"]]


def _attach_for(substrate: Dict) -> Dict[int, int]:
    attach = substrate.get("compute_attachment")
    if attach is None:
        return {c: c for c in range(substrate["num_comm_nodes"])}
    return attach


def _has_nested_paths(paths: List) -> bool:
    return bool(paths) and bool(paths[0]) and isinstance(paths[0][0], (list, tuple))


def _chosen_path(instance: Dict, config: VNEConfig, request_idx: int, link_idx: int) -> CandidatePath:
    if "processing_paths" in instance:
        paths = instance["processing_paths"]
        if _has_nested_paths(paths):
            return tuple(int(node) for node in paths[request_idx][link_idx])
        if request_idx == 0:
            return tuple(int(node) for node in paths[link_idx])
    if "chosen_path" in instance and request_idx == 0 and link_idx == 0:
        start, end = tuple(int(node) for node in instance["chosen_path"])
        if config.substrate_topology == "line":
            return tuple(range(start, end + 1))
        return (start, end)
    raise ValueError("Solved VNE instance is missing processing_paths for the configured virtual requests.")


def _validate_inclusive_range(value_range: Tuple[int, int], name: str, minimum: int) -> None:
    low, high = value_range
    if low < minimum or high < low:
        raise ValueError(f"{name} must be ordered and have lower bound >= {minimum}.")


def _value_in_range(value: int, value_range: Tuple[int, int]) -> bool:
    low, high = value_range
    return low <= value <= high


def _validate_instance_size(instance: Dict, config: VNEConfig) -> None:
    substrate = instance["substrate"]
    requests = _requests(instance)
    if not _value_in_range(substrate["num_comm_nodes"], config.num_substrate_comm_nodes_range):
        raise ValueError("VNE instance substrate size is outside VNEConfig.num_substrate_comm_nodes_range.")
    if not _value_in_range(len(requests), config.num_virtual_requests_range):
        raise ValueError("VNE instance request count is outside VNEConfig.num_virtual_requests_range.")
    for request in requests:
        if not _value_in_range(request["num_processing_nodes"], config.num_virtual_nodes_range):
            raise ValueError("VNE instance virtual size is outside VNEConfig.num_virtual_nodes_range.")


def candidate_paths_from_instance(
    instance: Dict,
    config: VNEConfig,
    request_idx: int,
    link_idx: int,
) -> Tuple[List[CandidatePath], Dict[Tuple[int, int], int], Dict[int, int]]:
    substrate = instance["substrate"]
    requests = _requests(instance)
    residual_bandwidth = dict(substrate["communication_bandwidth"])
    residual_compute = dict(substrate["compute_capacity"])
    attach = _attach_for(substrate)
    _validate_instance_size(instance, config)

    for prefix_request_idx in range(request_idx + 1):
        prefix_chain_length = requests[prefix_request_idx]["num_processing_nodes"] - 1
        limit = link_idx if prefix_request_idx == request_idx else prefix_chain_length
        for prefix_link_idx in range(limit):
            prefix_path = _chosen_path(instance, config, prefix_request_idx, prefix_link_idx)
            residual_bandwidth, residual_compute = Trajectory._apply_path(
                instance,
                residual_bandwidth,
                residual_compute,
                attach,
                prefix_request_idx,
                prefix_link_idx,
                prefix_path,
            )

    forced_start = (
        _chosen_path(instance, config, request_idx, link_idx - 1)[-1]
        if link_idx > 0
        else None
    )

    candidates = Trajectory._candidate_paths(
        instance=instance,
        residual_bandwidth=residual_bandwidth,
        residual_compute=residual_compute,
        compute_attachment=attach,
        request_idx=request_idx,
        link_idx=link_idx,
        forced_start=forced_start,
        check_future_completion=False,
    )
    return candidates, residual_bandwidth, residual_compute


class RandomVNEDataset(Dataset):
    def __init__(
        self,
        config: VNEConfig,
        expert_pickle_file: str,
        custom_num_instances=None,
        custom_num_batches=None,
    ):
        self.config = config
        _validate_inclusive_range(config.num_substrate_comm_nodes_range, "num_substrate_comm_nodes_range", 2)
        _validate_inclusive_range(config.num_virtual_requests_range, "num_virtual_requests_range", 1)
        _validate_inclusive_range(config.num_virtual_nodes_range, "num_virtual_nodes_range", 2)
        with open(expert_pickle_file, "rb") as f:
            self.instances = pickle.load(f)
        if custom_num_instances is not None:
            self.instances = self.instances[:custom_num_instances]

        self.decision_index = []
        for instance_idx, instance in enumerate(self.instances):
            requests = _requests(instance)
            _validate_instance_size(instance, config)
            for request_idx, request in enumerate(requests):
                for link_idx in range(request["num_processing_nodes"] - 1):
                    self.decision_index.append((instance_idx, request_idx, link_idx))
        if not self.decision_index:
            raise ValueError("VNE dataset has no replay decisions.")

        if custom_num_batches is None:
            self.length = len(self.decision_index)
            self.sample_random_decisions = False
        else:
            mode, value = custom_num_batches
            if mode == "absolute":
                self.length = int(value)
            elif mode == "multiplier":
                self.length = int(value) * len(self.decision_index)
            else:
                raise ValueError(f"Unknown custom_num_batches mode: {mode}")
            self.sample_random_decisions = True

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.sample_random_decisions:
            instance_idx, request_idx, link_idx = random.choice(self.decision_index)
        else:
            instance_idx, request_idx, link_idx = self.decision_index[idx]
        instance = self.instances[instance_idx]
        candidates, residual_bandwidth, residual_compute = candidate_paths_from_instance(
            instance,
            self.config,
            request_idx,
            link_idx,
        )
        chosen_path = _chosen_path(instance, self.config, request_idx, link_idx)
        target_index = candidates.index(chosen_path)
        return {
            "state": build_vne_state_input(
                instance=instance,
                candidates=candidates,
                request_idx=request_idx,
                link_idx=link_idx,
                residual_bandwidth=residual_bandwidth,
                residual_compute=residual_compute,
                compute_attachment=_attach_for(instance["substrate"]),
            ),
            "next_action_idx": torch.tensor(target_index, dtype=torch.long),
        }
