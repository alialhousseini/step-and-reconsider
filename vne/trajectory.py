"""VNE trajectory aligned with `vne/PROBLEM_FORMULATION.md`.

Schema in use:
    instance["requests"]                  -- static VN requests sharing one substrate
    request["source_link_demand"]         -- dem(S -> F_0), comp resource at F_0
    request["destination_link_demand"]    -- dem(F_{k-1} -> D), comp resource at F_{k-1}
    request["processing_link_demands"]    -- dem(F_i -> F_{i+1}), bandwidth along each path
    substrate["compute_attachment"][c]    -- comm node c -> attached comp node id
    substrate["compute_capacity"][p]      -- comp node id -> capacity (paired both directions)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from core.abstracts import BaseTrajectory
from vne.features import build_vne_state_input

Path = Tuple[int, ...]


def _requests(instance: Dict) -> List[Dict]:
    if "requests" in instance:
        return instance["requests"]
    return [instance["request"]]


def _comp_at(attach: Dict[int, int], residual_compute: Dict[int, int], comm_node: int) -> int:
    if attach is None:
        return residual_compute.get(comm_node, 0)
    comp_id = attach.get(comm_node)
    if comp_id is None:
        return 0
    return residual_compute.get(comp_id, 0)


@dataclass
class Trajectory(BaseTrajectory):
    instance: Dict
    residual_bandwidth: Dict[Tuple[int, int], int]
    residual_compute: Dict[int, int]
    compute_attachment: Dict[int, int]
    action_candidates: List[Path]
    current_request_idx: int = 0
    current_link_idx: int = 0
    processing_paths: List[List[Path]] = field(default_factory=list)
    f_placements: List[List[int]] = field(default_factory=list)
    chosen_path: Tuple[int, int] | None = None
    objective: float = 0.0

    @staticmethod
    def init_batch_from_instance_list(instances: List[Dict], network, device):
        trajectories = []
        for instance in instances:
            substrate = instance["substrate"]
            attach = substrate.get("compute_attachment") or {
                c: c for c in range(substrate["num_comm_nodes"])
            }
            residual_bandwidth = copy.deepcopy(substrate["communication_bandwidth"])
            residual_compute = copy.deepcopy(substrate["compute_capacity"])
            trajectories.append(
                Trajectory(
                    instance=instance,
                    residual_bandwidth=residual_bandwidth,
                    residual_compute=residual_compute,
                    compute_attachment=dict(attach),
                    action_candidates=Trajectory._candidate_paths(
                        instance=instance,
                        residual_bandwidth=residual_bandwidth,
                        residual_compute=residual_compute,
                        compute_attachment=dict(attach),
                        request_idx=0,
                        link_idx=0,
                        forced_start=None,
                    ),
                )
            )
        return trajectories

    @staticmethod
    def _path_edges(path: Path) -> List[Tuple[int, int]]:
        return list(zip(path[:-1], path[1:]))

    @staticmethod
    def _next_position(instance: Dict, request_idx: int, link_idx: int) -> Tuple[int | None, int | None]:
        requests = _requests(instance)
        chain_length = requests[request_idx]["num_processing_nodes"] - 1
        if link_idx + 1 < chain_length:
            return request_idx, link_idx + 1
        if request_idx + 1 < len(requests):
            return request_idx + 1, 0
        return None, None

    @staticmethod
    def _apply_path(
        instance: Dict,
        residual_bandwidth: Dict[Tuple[int, int], int],
        residual_compute: Dict[int, int],
        compute_attachment: Dict[int, int],
        request_idx: int,
        link_idx: int,
        path: Path,
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
        request = _requests(instance)[request_idx]
        chain_length = request["num_processing_nodes"] - 1
        proc_dem = request["processing_link_demands"][link_idx]
        new_bandwidth = copy.deepcopy(residual_bandwidth)
        new_compute = copy.deepcopy(residual_compute)
        if link_idx == 0:
            new_compute[compute_attachment[path[0]]] -= request["source_link_demand"]
        if link_idx == chain_length - 1:
            new_compute[compute_attachment[path[-1]]] -= request["destination_link_demand"]
        for edge in Trajectory._path_edges(path):
            new_bandwidth[edge] -= proc_dem
        return new_bandwidth, new_compute

    @staticmethod
    def _candidate_paths(
        instance: Dict,
        residual_bandwidth: Dict[Tuple[int, int], int],
        residual_compute: Dict[int, int],
        compute_attachment: Dict[int, int],
        request_idx: int,
        link_idx: int,
        forced_start: int | None,
        first_only: bool = False,
        check_future_completion: bool = False,  # OFF: exponential recursion at 60-80n scale
    ) -> List[Path]:
        substrate = instance["substrate"]
        request = _requests(instance)[request_idx]
        chain_length = request["num_processing_nodes"] - 1
        source_dem = request["source_link_demand"] if link_idx == 0 else 0
        dest_dem = request["destination_link_demand"] if link_idx == chain_length - 1 else 0
        proc_dem = request["processing_link_demands"][link_idx]

        adjacency = {node: [] for node in range(substrate["num_comm_nodes"])}
        for start, end in substrate["communication_edges"]:
            adjacency[start].append(end)

        candidates: List[Path] = []
        start_nodes = [forced_start] if forced_start is not None else range(substrate["num_comm_nodes"])
        for start in start_nodes:
            if _comp_at(compute_attachment, residual_compute, start) < source_dem:
                continue
            stack: List[Tuple[int, Path]] = [(start, (start,))]
            while stack:
                node, path = stack.pop()
                for nxt in adjacency.get(node, []):
                    if nxt in path:
                        continue
                    next_path = path + (nxt,)
                    edge = (node, nxt)
                    if residual_bandwidth.get(edge, 0) < proc_dem:
                        continue
                    if _comp_at(compute_attachment, residual_compute, nxt) >= dest_dem:
                        candidates.append(next_path)
                    stack.append((nxt, next_path))
        if not check_future_completion:
            return candidates[:1] if first_only else candidates

        next_request_idx, next_link_idx = Trajectory._next_position(instance, request_idx, link_idx)
        if next_request_idx is None:
            return candidates[:1] if first_only else candidates

        completable_candidates: List[Path] = []
        for path in candidates:
            next_bandwidth, next_compute = Trajectory._apply_path(
                instance,
                residual_bandwidth,
                residual_compute,
                compute_attachment,
                request_idx,
                link_idx,
                path,
            )
            if Trajectory._candidate_paths(
                instance=instance,
                residual_bandwidth=next_bandwidth,
                residual_compute=next_compute,
                compute_attachment=compute_attachment,
                request_idx=next_request_idx,
                link_idx=next_link_idx,
                forced_start=path[-1] if next_request_idx == request_idx else None,
                first_only=True,
            ):
                if first_only:
                    return [path]
                completable_candidates.append(path)
        return completable_candidates

    @staticmethod
    def log_probability_fn(trajectories: List["Trajectory"], network, to_numpy=True):
        logits_batch = network(
            [
                build_vne_state_input(
                    instance=trajectory.instance,
                    candidates=trajectory.action_candidates,
                    request_idx=trajectory.current_request_idx,
                    link_idx=trajectory.current_link_idx,
                    residual_bandwidth=trajectory.residual_bandwidth,
                    residual_compute=trajectory.residual_compute,
                    compute_attachment=trajectory.compute_attachment,
                )
                for trajectory in trajectories
            ]
        )
        output = []
        for logits in logits_batch:
            if logits.numel() == 0:
                log_probs = torch.empty((0,), dtype=torch.float32)
            else:
                log_probs = torch.log_softmax(logits, dim=0)
            output.append(log_probs.detach().cpu().numpy() if to_numpy else log_probs)
        return output

    def transition_fn(self, action_index: int):
        new_trajectory = copy.deepcopy(self)
        chosen_path = new_trajectory.action_candidates[action_index]
        request_idx = new_trajectory.current_request_idx
        request = _requests(new_trajectory.instance)[request_idx]
        link_idx = new_trajectory.current_link_idx

        proc_dem = request["processing_link_demands"][link_idx]

        start, end = chosen_path[0], chosen_path[-1]
        attach = new_trajectory.compute_attachment
        new_bandwidth, new_compute = self._apply_path(
            new_trajectory.instance,
            new_trajectory.residual_bandwidth,
            new_trajectory.residual_compute,
            attach,
            request_idx,
            link_idx,
            chosen_path,
        )
        new_trajectory.residual_bandwidth = new_bandwidth
        new_trajectory.residual_compute = new_compute

        while len(new_trajectory.processing_paths) <= request_idx:
            new_trajectory.processing_paths.append([])
        while len(new_trajectory.f_placements) <= request_idx:
            new_trajectory.f_placements.append([])
        if link_idx == 0:
            new_trajectory.f_placements[request_idx].append(start)
        new_trajectory.f_placements[request_idx].append(end)
        new_trajectory.processing_paths[request_idx].append(chosen_path)
        new_trajectory.chosen_path = (start, end)
        new_trajectory.objective -= self.compute_cost(chosen_path, proc_dem)

        next_request_idx, next_link_idx = self._next_position(
            new_trajectory.instance,
            request_idx,
            link_idx,
        )
        is_finished = next_request_idx is None
        if not is_finished:
            new_trajectory.current_request_idx = next_request_idx
            new_trajectory.current_link_idx = next_link_idx
            new_trajectory.action_candidates = self._candidate_paths(
                instance=new_trajectory.instance,
                residual_bandwidth=new_trajectory.residual_bandwidth,
                residual_compute=new_trajectory.residual_compute,
                compute_attachment=new_trajectory.compute_attachment,
                request_idx=next_request_idx,
                link_idx=next_link_idx,
                forced_start=end if next_request_idx == request_idx else None,
            )
            if not new_trajectory.action_candidates:
                new_trajectory.objective = float("-inf")
                is_finished = True
        else:
            if len(_requests(new_trajectory.instance)) == 1 and len(new_trajectory.processing_paths[0]) == 1:
                new_trajectory.chosen_path = (
                    new_trajectory.processing_paths[0][0][0],
                    new_trajectory.processing_paths[0][0][-1],
                )
            new_trajectory.action_candidates = []

        return new_trajectory, is_finished

    @staticmethod
    def compute_cost(path: Path, proc_dem: int) -> float:
        return float((len(path) - 1) * proc_dem)

    @staticmethod
    def to_max_evaluation_fn(trajectory: "Trajectory") -> float:
        return float(trajectory.objective)

    def num_actions(self) -> int:
        return len(self.action_candidates)
