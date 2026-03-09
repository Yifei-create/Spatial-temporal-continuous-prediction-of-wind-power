import os.path as osp

class Config:
    """STCWPF Configuration"""

    def __init__(self, method="EAC", logname="eac", seed=42, gpuid=0):
        self.raw_data_path = "data/processed/"
        self.save_data_path = "data/processed/"
        self.graph_path = "data/graph/"
        self.model_path = "log/"

        self.data_process = False

        # ====== window lengths ======
        self.x_len = 12
        self.y_len = 12

        # ====== feature dimension in raw data ======
        self.num_features = 16

        # ====== GCN channels: follow EAC's idea "time-window as node feature" ======
        # Each node feature vector = (x_len * num_features)
        in_dim = self.x_len * self.num_features

        self.gcn = {
            "in_channel": in_dim,
            "hidden_channel": 64,
            # To keep EAC residual x + data.x valid, out_channel must equal in_channel
            "out_channel": in_dim
        }

        # ====== TCN config ======
        self.tcn = {
            "in_channel": 1,
            "out_channel": 1,
            "kernel_size": 3,
            "dilation": 1
        }

        self.batch_size = 64
        self.epoch = 100
        self.lr = 0.0001
        self.dropout = 0.0
        self.loss = "mse"

        self.rank = 6
        self.base_node_size = 60   # initial turbines in pretrain phase

        self.train = True
        self.kl_weight = 1e-4

        # ====== unified streaming framework ======
        self.unified_data = True
        self.num_expansions = 3    # how many times new turbines join during streaming test
        self.warmup_days = 2       # days of warmup data when new turbines join
        self.online_lr = 1e-4      # learning rate for online adaptive updates

        self.method = method
        self.logname = logname
        self.seed = seed
        self.gpuid = gpuid
