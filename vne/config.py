import datetime
import os


class VNEConfig:
    def __init__(self):
        self.learning_type = "supervised" #gumbeldore or supervised

        # Problem generation.
        self.num_substrate_comm_nodes_range = (20, 40)
        self.num_virtual_requests_range = (2, 6)
        self.num_virtual_nodes_range = (2, 5)
        self.substrate_topology = "line"

        self.substrate_edge_probability = 0.4
        self.substrate_compute_attach_probability = 1.0
        self.substrate_communication_bandwidth_range = (4, 12)
        self.substrate_compute_capacity_range = (3, 10)
        self.virtual_compute_demand_range = (1, 4)
        self.virtual_communication_demand_range = (1, 4)

        # Validation-set generation.
        self.validation_num_instances = 120
        self.validation_generation_seed = 1
        self.validation_with_solutions = True
        self.validation_solver_time_limit = 30
        self.validation_output_path = None

        # Supervised training-set generation.
        self.supervised_training_num_instances = 1024
        self.supervised_training_generation_seed = 0

        # Network.
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.dropout = 0.0
        self.latent_dimension = self.embedding_dim
        self.feedforward_dimension = 4 * self.embedding_dim
        self.num_attention_heads = 4
        self.num_encoder_layers = 1
        self.num_decoder_layers = 6
        self.use_rezero_transformer = False

        self.load_checkpoint_from_path = None
        self.load_optimizer_state = True
        self.reset_best_validation = False

        # Training.
        self.seed = 42
        self.num_dataloader_workers = 1
        self.CUDA_VISIBLE_DEVICES = "0"
        self.training_device = "cpu"
        self.num_epochs = 10
        self.validation_every_n_epochs = 1
        self.batch_size_training = 128

        self.optimizer = {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "gradient_clipping": 0.0,
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1.0,
            },
        }

        # Gumbeldore training
        self.gumbeldore_config = {
            # Optional pickle of existing problem instances to search over. None means generate fresh instances.
            "active_search": None,
            # During Gumbeldore data generation, use the best validation model seen so far when available.
            "use_best_model_for_generation": True,
            # If generation does not improve the best model, append new generated samples instead of replacing the file.
            "append_if_not_new_best": False,
            # Devices used by Ray workers for Gumbeldore/search data generation.
            "devices_for_workers": ["cpu"],
            # Number of fresh VNE problem instances generated per Gumbeldore data-generation round.
            "num_instances_to_generate": 256,
            # Pickle path where Gumbeldore-generated solved training samples are saved.
            "destination_path": "./data/vne/vne_gumbeldore_training_dataset.pickle",
            # Per-worker search batch size for non-CPU worker devices.
            "batch_size_per_worker": 32,
            # Per-worker search batch size when the worker device is CPU.
            "batch_size_per_cpu_worker": 32,
            # Search algorithm used to generate training targets; "tasar" means Take a Step and Reconsider.
            "search_type": "tasar",
            # Number of partial solutions retained/explored during search.
            "beam_width": 8,
            # Number of stochastic beam search rounds for search modes that use repeated rounds.
            "num_rounds": 4,
            # Whether to pin worker processes to CPU cores.
            "pin_workers_to_core": False,
            # Exploration bonus constant used by advantage-based Gumbeldore variants.
            "advantage_constant": 0.3,
            # Whether to min-max normalize estimated advantages before updating priorities.
            "min_max_normalize_advantage": False,
            # Whether expected values use a simple mean instead of the default estimator.
            "expected_value_use_simple_mean": False,
            # Whether to use raw rollout outcomes directly instead of advantage-adjusted values.
            "use_pure_outcomes": False,
            # Whether to divide advantage estimates by visit counts.
            "normalize_advantage_by_visit_count": False,
            # Whether the first search round should be deterministic.
            "perform_first_round_deterministic": False,
            # Minimum nucleus top-p threshold used by TASAR action filtering.
            "min_nucleus_top_p": 1.0,
            # Number of decisions taken before TASAR replans.
            "replan_steps": 2,
        }

        # Evaluation.
        self.gumbeldore_eval = False
        self.devices_for_eval_workers = ["cpu"] * 2
        self.beams_with_batch_sizes = {
            1: 64,
            4: 32,
        }
        self.validation_relevant_beam_width = 4


        self.training_set_path = "./data/vne/vne_supervised_training_dataset.pickle"
        self.custom_num_instances = None
        self.custom_num_batches = None
        self.validation_set_path = "./data/vne/vne_validation_dataset.pickle"
        self.validation_custom_num_instances = None
        self.test_set_path = "./data/vne/vne_test_dataset.pickle"

        self.results_path = os.path.join(
            "./model_checkpoints/vne/results",
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
        )
        self.log_to_file = True
        self.log_to_mlflow = False
        self.mlflow_server_uri = "<mlflow_server_uri>"
