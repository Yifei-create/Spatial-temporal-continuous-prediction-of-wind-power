import os.path as osp
from config.dataset_registry import DATASET_REGISTRY


MULTIFREQ_METHODS = {'EAC', 'ScaleShift', 'VariationalScaleShift'}


class Config:
    """STCWPF Configuration"""

    def __init__(self, method="EAC", logname="eac", seed=42, gpuid=0, dataset="sdwpf"):
        self.dataset = dataset
        dataset_cfg = DATASET_REGISTRY[self.dataset]

        self.raw_data_path = dataset_cfg["raw_data_path"]
        self.location_path = dataset_cfg["location_path"]
        self.processing_tag = dataset_cfg["processing_tag"]
        self.save_data_path = osp.join("data", "processed", self.dataset, self.processing_tag, "distance")
        self.graph_path = osp.join("data", "graph", self.dataset, self.processing_tag, "distance")
        self.model_path = "log/"

        self.data_process = False

        self.x_len = 12
        self.y_len = 12

        self.num_features = len(dataset_cfg["feature_cols"])
        self.static_dim = len(dataset_cfg["static_feature_names"])
        self.use_static_embedding = True
        self.streaming_freq_mode = dataset_cfg["streaming_freq_mode"]
        self.use_freq_embedding = (self.streaming_freq_mode == "dynamic" and method in MULTIFREQ_METHODS)
        self.pretrain_freq_minutes = dataset_cfg.get("pretrain_freq_minutes", []) if method in MULTIFREQ_METHODS else []
        self.base_resolution_minutes = dataset_cfg.get("base_resolution_minutes")
        self.frequency_minutes = dataset_cfg.get("frequency_minutes")

        in_dim = self.x_len * self.num_features
        self.pos_x_bins = 32
        self.pos_y_bins = 32
        self.freq_num_embeddings = max(len(self.pretrain_freq_minutes), 1)

        self.gcn = {
            "in_channel": in_dim,
            "hidden_channel": 64,
            "out_channel": in_dim
        }

        self.tcn = {
            "in_channel": 1,
            "out_channel": 1,
            "kernel_size": 3,
            "dilation": 1
        }

        self.batch_size = 64
        self.epoch = 100
        self.lr = 0.001
        self.dropout = 0.0
        self.loss = "mse"

        self.rank = 8
        self.base_node_size = len(dataset_cfg["default_initial_turbines"])

        self.train = True
        self.kl_weight = 1e-4

        self.unified_data = True
        self.num_expansions = len(dataset_cfg["default_expansion_groups"])
        self.warmup_days = 4
        self.warmup_lr = 0.001

        self.adj_type = "distance"
        self.wind_top_k = 16

        self.patchtst = {
            "patch_len": 4,
            "stride": 2,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
        }

        self.method = method
        self.logname = logname
        self.seed = seed
        self.gpuid = gpuid
