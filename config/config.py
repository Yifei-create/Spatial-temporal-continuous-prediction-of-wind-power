import os.path as osp

from config.dataset_registry import DATASET_REGISTRY
from data.graph_generation import GRAPH_VARIANT_BASELINE, SUPPORTED_GRAPH_VARIANTS


SUPPORTED_DATASETS = tuple(DATASET_REGISTRY.keys())
STATIC_EMBEDDING_METHODS = {"ScaleShift", "VariationalScaleShift"}
FREQ_EMBEDDING_METHODS = set()
WARMUP_METHODS = {"ScaleShift", "VariationalScaleShift", "EAC"}
WARMUP_METHODS.update(
    {
        "STGNNInputBias",
        "STGNNStageResidual",
    }
)


def _preprocess_cache_dir_name(graph_variant, seed, x_len, y_len, num_expansions):
    exp_token = "auto" if num_expansions is None else str(int(num_expansions))
    return (
        f"preprocess"
        f"__graph-{graph_variant}"
        f"__seed-{int(seed)}"
        f"__x{int(x_len)}_y{int(y_len)}"
        f"__exp-{exp_token}"
    )


class Config:
    def __init__(self, method="EAC", logname="eac", seed=42, gpuid=0, dataset="sdwpf", graph_variant=GRAPH_VARIANT_BASELINE):
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset='{dataset}'. Expected one of {SUPPORTED_DATASETS}."
            )
        if graph_variant not in SUPPORTED_GRAPH_VARIANTS:
            raise ValueError(f"Unsupported graph_variant='{graph_variant}'. Expected one of {SUPPORTED_GRAPH_VARIANTS}.")
        self.dataset = dataset
        dataset_cfg = DATASET_REGISTRY[self.dataset]

        self.raw_data_path = dataset_cfg["raw_data_path"]
        self.location_path = dataset_cfg["location_path"]
        self.graph_variant = graph_variant
        self.save_data_path = osp.join("data", "processed", self.dataset, self.graph_variant)
        self.graph_path = osp.join("data", "graph", self.dataset, self.graph_variant)
        self.model_path = "results"

        self.data_process = False
        self.train = True

        self.x_len = 12
        self.y_len = 12
        self.batch_size = 64
        self.epoch = 100
        self.lr = 0.00005
        self.dropout = 0.0
        self.loss = "mse"
        self.num_workers = 0

        self.num_features = len(dataset_cfg["feature_cols"])
        self.static_dim = len(dataset_cfg["static_feature_names"])
        self.num_stages = len(dataset_cfg["default_expansion_groups"]) + 1

        self.method = method
        self.logname = logname
        self.seed = seed
        self.gpuid = gpuid
        self.num_expansions = None

        preprocess_dir = _preprocess_cache_dir_name(
            graph_variant=self.graph_variant,
            seed=self.seed,
            x_len=self.x_len,
            y_len=self.y_len,
            num_expansions=self.num_expansions,
        )
        self.save_data_path = osp.join("data", "processed", self.dataset, preprocess_dir)
        self.graph_path = osp.join("data", "graph", self.dataset, preprocess_dir)

        self.use_static_embedding = method in STATIC_EMBEDDING_METHODS
        self.use_freq_embedding = method in FREQ_EMBEDDING_METHODS and dataset_cfg["streaming_freq_mode"] == "dynamic"
        self.use_warmup = method in WARMUP_METHODS
        self.streaming_freq_mode = dataset_cfg["streaming_freq_mode"]
        self.supported_frequency_minutes = dataset_cfg.get("supported_frequency_minutes", []) if self.streaming_freq_mode == "dynamic" else []
        self.frequency_minutes = dataset_cfg.get("frequency_minutes")
        self.freq_num_embeddings = len(self.supported_frequency_minutes) if self.use_freq_embedding else 1

        in_dim = self.x_len * self.num_features
        self.gcn = {
            "in_channel": in_dim,
            "hidden_channel": 64,
            "out_channel": in_dim,
        }
        self.tcn = {
            "in_channel": 1,
            "out_channel": 1,
            "kernel_size": 3,
            "dilation": 1,
        }

        self.rank = 8
        self.base_node_size = len(dataset_cfg["default_initial_turbines"])
        self.kl_weight = 1e-4
        self.warmup_days = 4
        self.warmup_lr = 0.00005
        self.warmup_gradient_steps = 10
        self.baseline_weight_threshold = 0.95
        self.local_upstream_top_k = 16

        self.patchtst = {
            "patch_len": 4,
            "stride": 2,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
        }
