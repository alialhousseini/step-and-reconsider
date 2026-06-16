import datetime
import os


class VNEConfig:
    def __init__(self):
        self.learning_type = "supervised" #gumbeldore or supervised

        # Problem generation.
        # ACTIVE REGIME: the original "embed-all + minimize cost" problem (loose
        # resources so every request fits, enable_admission=False below). This is
        # the regime the prior training used, so scaling the dataset isolates
        # "more data" as the only change vs. the epoch-6 validation drop.
        #
        # CONTENDED REGIME (parked, for the admission-control follow-up): set
        #   num_virtual_requests_range = (8, 16)
        #   substrate_communication_bandwidth_range = (3, 6)
        #   substrate_compute_capacity_range = (3, 6)
        #   virtual_compute_demand_range = (2, 4)
        #   virtual_communication_demand_range = (2, 4)
        # plus enable_admission=True and validation_objective="lex". That makes
        # max-acceptance packing-limited (heavy/variable rejection), which needs
        # an acceptance-aware trajectory.py before validation/search are valid.
        self.num_substrate_comm_nodes_range = (60, 80)
        self.num_virtual_requests_range = (10, 20)
        self.num_virtual_nodes_range = (2, 5)
        self.substrate_topology = "line"

        self.substrate_edge_probability = 0.4
        self.substrate_compute_attach_probability = 1.0
        self.substrate_communication_bandwidth_range = (4, 12)
        self.substrate_compute_capacity_range = (3, 10)
        self.virtual_compute_demand_range = (1, 4)
        self.virtual_communication_demand_range = (1, 4)

        # Validation-set generation.
        self.validation_num_instances = 1000
        self.validation_generation_seed = 1
        self.validation_with_solutions = True
        # Solver time limit (seconds). The loose embed-all regime solves fast;
        # if a hard instance hits this limit, the best feasible embedding found
        # is kept (no discard/resample) so there is no timeout bias.
        self.validation_solver_time_limit = 60
        # MILP backend for label generation. "auto" tries fast/free HiGHS first,
        # then any licensed commercial solver (gurobi/cplex), then CBC. Force one
        # with "highs" | "gurobi" | "cplex" | "cbc". HiGHS needs no license and
        # runs on every cluster node; commercial backends need their own license/
        # install. All backends return the exact ILP optimum (or within mip_gap).
        self.validation_solver = "highs"
        self.validation_solver_threads = 0
        # Relative MIP gap. 0.0 = prove exact min cost (the loose regime is fast
        # enough). Raise (e.g. 0.03) if a harder regime makes proofs slow.
        self.validation_solver_mip_gap = 0.0
        self.validation_output_path = None

        # Label objective.
        #   "lex"    -> LEXICOGRAPHIC: maximize acceptance ratio FIRST (embed as
        #               many requests as physically fit), then minimize routing
        #               cost as a tiebreak. Implemented as a clean two-stage MILP
        #               (stage 1 max acceptance exact; stage 2 min cost within
        #               validation_solver_mip_gap), so the cost gap actually bites
        #               on cost instead of being swamped by a big-M term.
        #   "profit" -> single objective sum_r revenue_r*accept[r] - costs, with
        #               revenue_r = revenue_per_request + revenue_per_demand_unit
        #               * total_demand(r); the solver drops unprofitable requests.
        self.validation_objective = "lex"
        # Admission OFF for the active embed-all regime: every request is forced
        # (sum_c f == 1) and the objective reduces to pure min-cost. Set True
        # (with the contended ranges) for the admission-control follow-up.
        self.enable_admission = False
        # Used only by the "profit" objective.
        self.validation_revenue_per_request = 0.0
        self.validation_revenue_per_demand_unit = 1.0
        self.validation_cost_comm_per_unit = 1.0
        self.validation_cost_comp_per_unit = 0.0

        # Supervised training-set generation.
        self.supervised_training_num_instances = 10000
        self.supervised_training_generation_seed = 0

        # Network (scaled to ~2M params, matching TSP LEHD capacity).
        # Architecture family: "lehd" (encoder-decoder) or "bq" (unified transformer).
        self.architecture = "lehd"
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.dropout = 0.0
        self.latent_dimension = self.embedding_dim
        self.feedforward_dimension = 4 * self.embedding_dim  # 512
        self.num_attention_heads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.num_transformer_blocks = 9  # BQ unified-stack depth (paper: 9 blocks)
        self.use_rezero_transformer = True

        self.load_checkpoint_from_path = None
        self.load_optimizer_state = True
        self.reset_best_validation = False

        # Training.
        self.seed = 42
        self.num_dataloader_workers = 1
        self.CUDA_VISIBLE_DEVICES = "0"   # MPS exposes the shared GPU as device 0
        self.training_device = "cuda"
        # 15 epochs is enough to pass the prior epoch-6 validation drop. The
        # per-example (non-batched) forward makes epochs slow (~8 min each), so
        # we keep the count modest; raise once the forward is vectorized.
        self.num_epochs = 30
        self.validation_every_n_epochs = 1
        self.batch_size_training = 128

        self.optimizer = {
            "lr": 2e-4,
            "weight_decay": 0.0,
            "gradient_clipping": 1.0,
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
            # CPU workers avoid GPU memory conflicts with training. Use enough
            # of them (8-16) to keep generation throughput reasonable.
            "devices_for_workers": ["cpu"] * 8,
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
        # Validation beam search runs on the shared GPU (one Ray worker).
        self.devices_for_eval_workers = ["cuda:0"] * 4
        # Beam-1 (greedy) only during the per-epoch validation: it already shows
        # the overfitting/epoch-6 drop and is ~4x cheaper than also running
        # beam-4. (Add 4: 32 back for a final, more thorough eval if desired.)
        self.beams_with_batch_sizes = {
            1: 32,
        }
        self.validation_relevant_beam_width = 1


        self.training_set_path = "./data/vne/vne_supervised_training_dataset_10k.pickle"
        self.custom_num_instances = None
        # Bound the epoch length so epochs are comparable across pool sizes and
        # the per-example forward doesn't make a 50k epoch enormous: sample this
        # many replay decisions per epoch (matches the prior ~10k-decision epoch).
        self.custom_num_batches = ("absolute", 10000)
        self.validation_set_path = "./data/vne/vne_validation_dataset_1k.pickle"
        # Validate on a fixed subset each epoch to keep beam-search validation fast.
        self.validation_custom_num_instances = 128
        self.test_set_path = "./data/vne/vne_test_dataset_2k.pickle"

        self.results_path = os.path.join(
            "./model_checkpoints/vne/results",
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
        )
        self.log_to_file = True
        self.log_to_mlflow = False
        self.mlflow_server_uri = "<mlflow_server_uri>"
