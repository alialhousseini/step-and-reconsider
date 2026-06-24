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
import core.stochastic_beam_search as sbs
from core.train import main_train_cycle
from vne.config import VNEConfig
from vne.dataset import RandomVNEDataset
from vne.instance_generator import make_dataset
from vne.network import VNEPolicyNetwork
from vne.bq_network import BQPolicyNetwork
from vne.trajectory import Trajectory as VNETrajectory
from vne.validation_set_generator import make_validation_dataset, run_self_check, save_dataset, solver_kwargs_from_config


def collate_vne_batch(batch):
    return {
        "state": [item["state"] for item in batch],
        "next_action_idx": torch.stack([item["next_action_idx"] for item in batch]),
    }


def get_network(config: VNEConfig, device: torch.device):
    """Return the appropriate policy network for the configured architecture."""
    arch = getattr(config, "architecture", "lehd")
    if arch == "bq":
        return BQPolicyNetwork(config, device)
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
        solver_kwargs=solver_kwargs_from_config(config),
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


def _result_has_complete_paths(instance: dict, result: dict) -> bool:
    """Verify every accepted request has a path for every link."""
    paths = result.get("processing_paths", [])
    requests = instance.get("requests", [instance.get("request")] if "request" in instance else [])
    accepted = instance.get("accepted", [True] * len(requests))
    if not paths:
        return False
    # Detect nested vs flat
    nested = any(isinstance(p, list) and p and isinstance(p[0], (list, tuple)) for p in paths)
    for req_idx, req in enumerate(requests):
        if not accepted[req_idx]:
            continue
        chain_len = req["num_processing_nodes"] - 1
        if nested:
            if req_idx >= len(paths) or len(paths[req_idx]) < chain_len:
                return False
        else:
            if req_idx > 0:
                return False  # flat paths only support single-request
            if len(paths) < chain_len:
                return False
    return True


def save_search_results_to_dataset(destination_path: str, problem_instances, results, append_to_dataset):
    dataset = []
    skipped_infeasible = 0
    skipped_incomplete = 0
    for instance, result in zip(problem_instances, results):
        if result.get("objective", float("-inf")) <= float("-inf"):
            skipped_infeasible += 1
            continue  # infeasible — don't add to training set
        if not _result_has_complete_paths(instance, result):
            skipped_incomplete += 1
            continue  # partially assigned — can't be used as training target
        item = copy.deepcopy(instance)
        item.update(result)
        dataset.append(item)
    if skipped_infeasible or skipped_incomplete:
        print(f"  Skipped {skipped_infeasible} infeasible, {skipped_incomplete} incomplete / {len(results)} results.")

    if not dataset:
        raise RuntimeError(
            "All generated instances were infeasible. "
            "The search cannot find valid embeddings. "
            "Try a larger beam_width or smaller problem instances."
        )

    if append_to_dataset:
        with open(destination_path, "rb") as f:
            old_items = pickle.load(f)
        dataset = old_items + dataset

    with open(destination_path, "wb") as f:
        pickle.dump(dataset, f)

    return float(np.mean([item["objective"] for item in dataset]))


def _evaluate_sequential(
    eval_type: str,
    config: VNEConfig,
    network: VNEPolicyNetwork,
    to_evaluate_path: str,
    num_instances: Optional[int] = None,
    beam_width: int = 1,
):
    """Ray-free sequential evaluation (no GumbeldoreDataset, no Ray workers).

    Used for supervised-training per-epoch validation where the only goal is
    greedy (beam_width=1) evaluation on a small subset.  Eliminates Ray as a
    failure mode under MPS co-location — multiple SLURM jobs sharing one GPU
    each auto-start their own Ray cluster, and those clusters deadlock.
    """
    with open(to_evaluate_path, "rb") as f:
        instances = pickle.load(f)
    if num_instances is not None:
        instances = instances[:num_instances]
    total_inst = len(instances)

    device = torch.device(config.training_device)
    network.eval()

    # --- Callbacks matching async_sbs_worker in gumbeldore_dataset.py ---
    def _child_log_probability_fn(trajectories):
        return VNETrajectory.log_probability_fn(
            trajectories=trajectories, network=network, to_numpy=True
        )

    def _child_transition_fn(trajectory_action_pairs):
        return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]

    # Small-batch processing: 4-8 instances per beam_search call so the
    # network's batched forward is exercised without the Ray machinery.
    eval_batch = 8 if total_inst >= 128 else 4
    t0 = time.time()
    objectives = []
    ilp_objectives = []
    feasible_count = 0

    with torch.no_grad():
        for start in tqdm(range(0, total_inst, eval_batch), desc=f"{eval_type} seq"):
            batch_insts = instances[start: start + eval_batch]
            roots = VNETrajectory.init_batch_from_instance_list(
                instances=[copy.deepcopy(inst) for inst in batch_insts],
                network=network, device=device,
            )
            beam_leaves_batch = sbs.stochastic_beam_search(
                child_log_probability_fn=_child_log_probability_fn,
                child_transition_fn=_child_transition_fn,
                root_states=roots,
                beam_width=beam_width,
                deterministic=True,
            )
            for i, inst in enumerate(batch_insts):
                best_state = beam_leaves_batch[i][0].state
                obj = VNETrajectory.to_max_evaluation_fn(best_state)
                objectives.append(obj)

                ilp_obj = inst.get("objective")
                if ilp_obj is not None:
                    ilp_objectives.append(ilp_obj)
                if obj > float("-inf"):
                    feasible_count += 1

    t_eval = time.time() - t0

    objectives = np.array(objectives, dtype=float)
    gaps_pct = []
    for model_obj, ilp_obj in zip(objectives, ilp_objectives):
        if model_obj > float("-inf") and ilp_obj != 0:
            gaps_pct.append((ilp_obj - model_obj) / abs(ilp_obj) * 100.0)
        elif model_obj > float("-inf"):
            gaps_pct.append(0.0)

    n = len(objectives)
    label = f"{eval_type} beam width {beam_width}"
    loggable = {
        f"{label}. Obj.": float(objectives.mean()),
        f"{label} gap%": float(np.mean(gaps_pct)) if gaps_pct else float("nan"),
        f"{label} feas%": feasible_count / n * 100.0 if n else 0.0,
        f"{label} time/inst_ms": (t_eval / max(total_inst, 1)) * 1000.0,
    }
    # Best-model selection metric (LOWER is better), lexicographic to match
    # config.validation_objective="lex": feasibility first, then mean gap over
    # feasible instances. The mean objective alone is unusable for selection —
    # a single infeasible instance makes it -inf every epoch, so the previous
    # `-objectives.mean()` was always +inf and no best_model.pt was ever saved.
    feas_frac = feasible_count / n if n else 0.0
    mean_gap = float(np.mean(gaps_pct)) if gaps_pct else 1000.0
    metric = (1.0 - feas_frac) * 1000.0 + mean_gap

    return metric, loggable


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
        # Compute per-instance gap to ILP optimum and feasibility rate
        ilp_objectives = []
        gaps_pct = []
        feasible_count = 0
        for inst, res in zip(problem_instances, results):
            ilp_obj = inst.get("objective")
            if ilp_obj is not None:
                ilp_objectives.append(ilp_obj)
                model_obj = res["objective"]
                if model_obj > float("-inf") and ilp_obj != 0:
                    gap_pct = (ilp_obj - model_obj) / abs(ilp_obj) * 100.0
                    gaps_pct.append(gap_pct)
                    feasible_count += 1
                elif model_obj > float("-inf"):
                    feasible_count += 1
                    gaps_pct.append(0.0)
        n = len(results)
        return {
            "mean_obj": float(objectives.mean()),
            "mean_ilp_obj": float(np.mean(ilp_objectives)) if ilp_objectives else 0.0,
            "mean_gap_pct": float(np.mean(gaps_pct)) if gaps_pct else float("nan"),
            "feasibility_pct": feasible_count / n * 100.0 if n else 0.0,
        }

    if not config.gumbeldore_eval:
        # Load instances once to know count and avoid redundant I/O per beam width
        instances_raw, _, _ = load_instances(config)
        total_inst = len(instances_raw)
        loggable = {}
        metric = None
        for beam_width, batch_size in config.beams_with_batch_sizes.items():
            _config = copy.deepcopy(config)
            _config.gumbeldore_config["search_type"] = "beam_search"
            _config.gumbeldore_config["beam_width"] = beam_width
            _config.gumbeldore_config["devices_for_workers"] = _config.devices_for_eval_workers
            _config.gumbeldore_config["batch_size_per_worker"] = batch_size
            _config.gumbeldore_config["batch_size_per_cpu_worker"] = batch_size
            t0 = time.time()
            results = GumbeldoreDataset(
                config=_config,
                trajectory_cls=VNETrajectory,
                generate_instances_fn=load_instances,
                get_network_fn=get_network,
                beam_leaves_to_result_fn=beam_leaves_to_result,
                process_search_results_fn=process_search_results,
            ).generate_dataset(copy.deepcopy(network.get_weights()), False)
            t_eval = time.time() - t0
            loggable[f"{eval_type} beam width {beam_width}. Obj."] = float(results["mean_obj"])
            loggable[f"{eval_type} beam-{beam_width} gap%"] = float(results["mean_gap_pct"])
            loggable[f"{eval_type} beam-{beam_width} feas%"] = float(results["feasibility_pct"])
            loggable[f"{eval_type} beam-{beam_width} time/inst_ms"] = (t_eval / max(total_inst, 1)) * 1000.0
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
    # Always use Ray-free sequential evaluation.  Ray-based GumbeldoreDataset
    # workers deadlock under MPS co-location, and the per-epoch validation only
    # needs greedy (beam_width=1) on a small subset — a few minutes sequentially.
    return _evaluate_sequential(
        "Validation", config, network,
        config.validation_set_path, config.validation_custom_num_instances,
        beam_width=config.validation_relevant_beam_width,
    )


def test(config: VNEConfig, network: VNEPolicyNetwork):
    # Ray-free sequential — same rationale as validate().
    _, loggable = _evaluate_sequential(
        "Test", config, network,
        config.test_set_path, None,
        beam_width=config.validation_relevant_beam_width,
    )
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
    t0 = time.time()

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

    t_train = time.time() - t0
    return accumulated_loss / len(dataloader), t_train


def train_for_one_epoch_gumbeldore(config: VNEConfig, network: VNEPolicyNetwork, network_weights: dict, optimizer: torch.optim.Optimizer, append_to_dataset: bool) -> Tuple[float, dict]:
    t0 = time.time()
    dataloader, mean_generated_obj = get_gumbeldore_dataloader(config, network_weights, append_to_dataset)
    t_gen = time.time() - t0
    avg_loss, t_train = train_with_dataloader(config, dataloader, network, optimizer)
    return avg_loss, {"Avg generated obj": float(mean_generated_obj), "t_gen_s": t_gen, "t_train_s": t_train}


def train_for_one_epoch_supervised(config: VNEConfig, network: VNEPolicyNetwork, optimizer: torch.optim.Optimizer, dataloader: DataLoader):
    avg_loss, _t_train = train_with_dataloader(config, dataloader, network, optimizer)
    return avg_loss


def _apply_env_overrides(config: VNEConfig) -> None:
    """Lightweight experiment overrides via env vars (for A/B runs without
    editing config.py), plus a polite GPU memory cap so we share the MPS GPU."""
    env = os.environ
    if env.get("VNE_TRAINING_SET_PATH"):
        config.training_set_path = env["VNE_TRAINING_SET_PATH"]
    if env.get("VNE_VALIDATION_SET_PATH"):
        config.validation_set_path = env["VNE_VALIDATION_SET_PATH"]
    if env.get("VNE_TEST_SET_PATH"):
        config.test_set_path = env["VNE_TEST_SET_PATH"]
    if env.get("VNE_CUSTOM_NUM_INSTANCES"):
        config.custom_num_instances = int(env["VNE_CUSTOM_NUM_INSTANCES"])
    if env.get("VNE_NUM_EPOCHS"):
        config.num_epochs = int(env["VNE_NUM_EPOCHS"])
    if env.get("VNE_RESULTS_PATH"):
        config.results_path = env["VNE_RESULTS_PATH"]
    if env.get("VNE_EMBEDDING_DIM"):
        config.embedding_dim = int(env["VNE_EMBEDDING_DIM"])
        config.latent_dimension = config.embedding_dim
    if env.get("VNE_HIDDEN_DIM"):
        config.hidden_dim = int(env["VNE_HIDDEN_DIM"])
    if env.get("VNE_NUM_HEADS"):
        config.num_attention_heads = int(env["VNE_NUM_HEADS"])
    if env.get("VNE_NUM_DECODER_LAYERS"):
        config.num_decoder_layers = int(env["VNE_NUM_DECODER_LAYERS"])
    if env.get("VNE_NUM_ENCODER_LAYERS"):
        config.num_encoder_layers = int(env["VNE_NUM_ENCODER_LAYERS"])
    if env.get("VNE_NUM_TRANSFORMER_BLOCKS"):
        config.num_transformer_blocks = int(env["VNE_NUM_TRANSFORMER_BLOCKS"])
    if env.get("VNE_FF_DIM"):
        config.feedforward_dimension = int(env["VNE_FF_DIM"])
    if env.get("VNE_LR"):
        config.optimizer["lr"] = float(env["VNE_LR"])
    if env.get("VNE_LEARNING_TYPE"):
        config.learning_type = env["VNE_LEARNING_TYPE"]
    if env.get("VNE_ARCHITECTURE"):
        config.architecture = env["VNE_ARCHITECTURE"]
    if env.get("VNE_SEED"):
        config.seed = int(env["VNE_SEED"])
    if env.get("VNE_LOAD_CHECKPOINT_FROM_PATH"):
        config.load_checkpoint_from_path = env["VNE_LOAD_CHECKPOINT_FROM_PATH"]
        config.reset_best_validation = True  # fresh start for new training paradigm
    if env.get("VNE_BEAM_WIDTH"):
        config.gumbeldore_config["beam_width"] = int(env["VNE_BEAM_WIDTH"])
    if env.get("VNE_SEARCH_TYPE"):
        config.gumbeldore_config["search_type"] = env["VNE_SEARCH_TYPE"]
    if env.get("VNE_REPLAN_STEPS"):
        config.gumbeldore_config["replan_steps"] = int(env["VNE_REPLAN_STEPS"])
    if env.get("VNE_NUM_GENERATE"):
        config.gumbeldore_config["num_instances_to_generate"] = int(env["VNE_NUM_GENERATE"])
    if env.get("VNE_NUM_CPU_WORKERS"):
        config.gumbeldore_config["devices_for_workers"] = ["cpu"] * int(env["VNE_NUM_CPU_WORKERS"])
    if env.get("VNE_CPU_BATCH_SIZE"):
        config.gumbeldore_config["batch_size_per_cpu_worker"] = int(env["VNE_CPU_BATCH_SIZE"])
    if env.get("VNE_BATCH_SIZE"):
        config.batch_size_training = int(env["VNE_BATCH_SIZE"])
    if env.get("VNE_EVAL_WORKERS"):
        config.devices_for_eval_workers = ["cuda:0"] * int(env["VNE_EVAL_WORKERS"])
    if env.get("VNE_GEN_DEVICE"):
        # Comma-separated list of devices for Gumbeldore generation, e.g. "cuda:0" or "cuda:0,cuda:0,cuda:0"
        config.gumbeldore_config["devices_for_workers"] = env["VNE_GEN_DEVICE"].split(",")
        config.gumbeldore_config["batch_size_per_worker"] = int(env.get("VNE_GEN_BATCH_SIZE", "32"))
        config.gumbeldore_config["batch_size_per_cpu_worker"] = int(env.get("VNE_GEN_BATCH_SIZE", "32"))
    frac = env.get("VNE_GPU_MEM_FRACTION")
    if frac and str(config.training_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)
        print(f"Capped GPU memory fraction to {frac} (shared MPS GPU).")


if __name__ == "__main__":
    print(">> VNE <<")
    config = VNEConfig()
    _apply_env_overrides(config)
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
