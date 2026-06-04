import copy
import os
import pickle
import sys
import time
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.gumbeldore_dataset import GumbeldoreDataset
from core.train import main_train_cycle
from vne.config import VNEConfig
from vne.dataset import RandomVNEDataset
from vne.instance_generator import make_dataset
from vne.network import VNEPolicyNetwork
from vne.trajectory import Trajectory as VNETrajectory
from vne.validation_set_generator import make_validation_dataset, run_self_check, save_dataset


def collate_vne_batch(batch):
    return {
        "state": [item["state"] for item in batch],
        "next_action_idx": torch.stack([item["next_action_idx"] for item in batch]),
    }


def get_network(config: VNEConfig, device: torch.device) -> VNEPolicyNetwork:
    return VNEPolicyNetwork(config, device)


def generate_instances(config: VNEConfig):
    if config.gumbeldore_config["active_search"] is None:
        instances = make_dataset(config, config.gumbeldore_config["num_instances_to_generate"])
    else:
        with open(config.gumbeldore_config["active_search"], "rb") as f:
            instances = pickle.load(f)
    return (
        instances,
        config.gumbeldore_config["batch_size_per_worker"],
        config.gumbeldore_config["batch_size_per_cpu_worker"],
    )


def ensure_solved_dataset(
    config: VNEConfig,
    path: str,
    purpose: str,
    num_instances: int,
    seed: int,
) -> None:
    if os.path.exists(path):
        print(f"Using existing {purpose} dataset: {path}")
        return

    print(f"Generating missing {purpose} dataset: {path}")
    dataset = make_validation_dataset(
        num_instances,
        config,
        with_solutions=True,
        solver_kwargs={"time_limit_s": config.validation_solver_time_limit},
        seed=seed,
    )
    run_self_check(dataset)
    save_dataset(path, dataset)
    print(f"Saved {purpose} dataset to {path}")


def ensure_required_datasets(config: VNEConfig) -> None:
    ensure_solved_dataset(
        config,
        config.validation_set_path,
        "validation",
        config.validation_num_instances,
        config.validation_generation_seed,
    )
    if config.learning_type == "supervised":
        ensure_solved_dataset(
            config,
            config.training_set_path,
            "supervised training",
            config.supervised_training_num_instances,
            config.supervised_training_generation_seed,
        )
    if config.test_set_path is not None and not os.path.exists(config.test_set_path):
        print(
            f"Test dataset not found: {config.test_set_path}. "
            "Skipping test evaluation; automatic test-set generation is not part of Task 4."
        )
        config.test_set_path = None


def beam_leaves_to_result(trajectories: List[VNETrajectory]):
    best = max(trajectories, key=lambda x: x.to_max_evaluation_fn(x))
    result = {
        "processing_paths": [
            [list(path) for path in request_paths]
            for request_paths in best.processing_paths
        ],
        "f_placements": [
            list(placements)
            for placements in best.f_placements
        ],
        "objective": float(best.objective),
    }
    if len(best.processing_paths) == 1 and len(best.processing_paths[0]) == 1:
        path = best.processing_paths[0][0]
        result["chosen_path"] = [path[0], path[-1]]
    return result


def save_search_results_to_dataset(destination_path: str, problem_instances, results, append_to_dataset):
    dataset = []
    for instance, result in zip(problem_instances, results):
        item = copy.deepcopy(instance)
        item.update(result)
        dataset.append(item)

    if append_to_dataset:
        with open(destination_path, "rb") as f:
            old_items = pickle.load(f)
        dataset = old_items + dataset

    with open(destination_path, "wb") as f:
        pickle.dump(dataset, f)

    return float(np.mean([item["objective"] for item in dataset]))


def evaluate(eval_type: str, config: VNEConfig, network: VNEPolicyNetwork, to_evaluate_path: str, num_instances: Optional[int] = None):
    def load_instances(conf):
        with open(to_evaluate_path, "rb") as f:
            instances = pickle.load(f)
        if num_instances is not None:
            instances = instances[:num_instances]
        return (
            instances,
            conf.gumbeldore_config["batch_size_per_worker"],
            conf.gumbeldore_config["batch_size_per_cpu_worker"],
        )

    def process_search_results(destination_path: str, problem_instances, results, append_to_dataset):
        objectives = np.array([result["objective"] for result in results], dtype=float)
        return {
            "mean_obj": float(objectives.mean()),
        }

    if not config.gumbeldore_eval:
        loggable = {}
        metric = None
        for beam_width, batch_size in config.beams_with_batch_sizes.items():
            _config = copy.deepcopy(config)
            _config.gumbeldore_config["search_type"] = "beam_search"
            _config.gumbeldore_config["beam_width"] = beam_width
            _config.gumbeldore_config["devices_for_workers"] = _config.devices_for_eval_workers
            _config.gumbeldore_config["batch_size_per_worker"] = batch_size
            _config.gumbeldore_config["batch_size_per_cpu_worker"] = batch_size
            results = GumbeldoreDataset(
                config=_config,
                trajectory_cls=VNETrajectory,
                generate_instances_fn=load_instances,
                get_network_fn=get_network,
                beam_leaves_to_result_fn=beam_leaves_to_result,
                process_search_results_fn=process_search_results,
            ).generate_dataset(copy.deepcopy(network.get_weights()), False)
            loggable[f"{eval_type} beam width {beam_width}. Obj."] = float(results["mean_obj"])
            if beam_width == config.validation_relevant_beam_width:
                metric = -results["mean_obj"]
        return metric, loggable

    results = GumbeldoreDataset(
        config=config,
        trajectory_cls=VNETrajectory,
        generate_instances_fn=load_instances,
        get_network_fn=get_network,
        beam_leaves_to_result_fn=beam_leaves_to_result,
        process_search_results_fn=process_search_results,
    ).generate_dataset(copy.deepcopy(network.get_weights()), False)
    return -results["mean_obj"], {f"{eval_type} Gumbeldore. Obj.": results["mean_obj"]}


def validate(config: VNEConfig, network: VNEPolicyNetwork):
    return evaluate("Validation", config, network, config.validation_set_path, config.validation_custom_num_instances)


def test(config: VNEConfig, network: VNEPolicyNetwork):
    _, loggable = evaluate("Test", config, network, config.test_set_path, None)
    return loggable


def get_gumbeldore_dataloader(config: VNEConfig, network_weights: dict, append_to_dataset: bool):
    dataset_generator = GumbeldoreDataset(
        config=config,
        trajectory_cls=VNETrajectory,
        generate_instances_fn=generate_instances,
        get_network_fn=get_network,
        beam_leaves_to_result_fn=beam_leaves_to_result,
        process_search_results_fn=save_search_results_to_dataset,
    )
    mean_generated_obj = dataset_generator.generate_dataset(network_weights, append_to_dataset)
    time.sleep(1)

    dataset = RandomVNEDataset(
        config=config,
        expert_pickle_file=config.gumbeldore_config["destination_path"],
        custom_num_instances=config.custom_num_instances,
        custom_num_batches=config.custom_num_batches,
    )
    return (
        DataLoader(
            dataset,
            batch_size=config.batch_size_training,
            shuffle=True,
            num_workers=config.num_dataloader_workers,
            pin_memory=config.training_device != "cpu" and torch.cuda.is_available(),
            persistent_workers=config.num_dataloader_workers > 0,
            collate_fn=collate_vne_batch,
        ),
        float(mean_generated_obj),
    )


def get_supervised_dataloader(config: VNEConfig) -> DataLoader:
    dataset = RandomVNEDataset(
        config=config,
        expert_pickle_file=config.training_set_path,
        custom_num_instances=config.custom_num_instances,
        custom_num_batches=config.custom_num_batches,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size_training,
        shuffle=True,
        num_workers=config.num_dataloader_workers,
        pin_memory=config.training_device != "cpu" and torch.cuda.is_available(),
        persistent_workers=config.num_dataloader_workers > 0,
        collate_fn=collate_vne_batch,
    )


def train_with_dataloader(config: VNEConfig, dataloader: DataLoader, network: VNEPolicyNetwork, optimizer: torch.optim.Optimizer):
    network.train()
    accumulated_loss = 0.0
    progress_bar = tqdm(range(len(dataloader)))
    data_iter = iter(dataloader)
    loss_fn = CrossEntropyLoss(reduction="mean")

    for _ in progress_bar:
        data = next(data_iter)
        next_action_idx = data["next_action_idx"].to(network.device)

        logits_batch = network(data["state"])
        losses = [
            loss_fn(logits.unsqueeze(0), target.unsqueeze(0))
            for logits, target in zip(logits_batch, next_action_idx)
        ]
        loss = torch.stack(losses).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()
        accumulated_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": loss.item()})

    return accumulated_loss / len(dataloader)


def train_for_one_epoch_gumbeldore(config: VNEConfig, network: VNEPolicyNetwork, network_weights: dict, optimizer: torch.optim.Optimizer, append_to_dataset: bool) -> Tuple[float, dict]:
    dataloader, mean_generated_obj = get_gumbeldore_dataloader(config, network_weights, append_to_dataset)
    avg_loss = train_with_dataloader(config, dataloader, network, optimizer)
    return avg_loss, {"Avg generated obj": float(mean_generated_obj)}


def train_for_one_epoch_supervised(config: VNEConfig, network: VNEPolicyNetwork, optimizer: torch.optim.Optimizer, dataloader: DataLoader):
    return train_with_dataloader(config, dataloader, network, optimizer)


if __name__ == "__main__":
    print(">> VNE <<")
    config = VNEConfig()
    ensure_required_datasets(config)
    main_train_cycle(
        learning_type=config.learning_type,
        config=config,
        get_network_fn=get_network,
        validation_fn=validate,
        test_fn=test,
        get_supervised_dataloader=get_supervised_dataloader,
        train_for_one_epoch_supervised_fn=train_for_one_epoch_supervised,
        train_for_one_epoch_gumbeldore_fn=train_for_one_epoch_gumbeldore,
    )
