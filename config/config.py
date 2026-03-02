import os.path as osp

class Config:
    """STCWPF Configuration"""
    
    def __init__(self, method="EAC", logname="eac", seed=42, gpuid=0):
        self.raw_data_path = "data/processed/"
        self.save_data_path = "data/processed/"
        self.graph_path = "data/graph/"
        self.model_path = "log/"
        
        self.begin_year = 0
        self.end_year = 3
        self.data_process = False
        
        self.gcn = {
            "in_channel": 16,
            "hidden_channel": 64,
            "out_channel": 16
        }
        
        self.tcn = {
            "in_channel": 1,
            "out_channel": 1,
            "kernel_size": 3,
            "dilation": 1
        }
        
        self.x_len = 12
        self.y_len = 12
        
        self.batch_size = 64
        self.epoch = 100
        self.lr = 0.03
        self.dropout = 0.0
        self.loss = "mse"
        
        self.rank = 6
        self.base_node_size = 44
        
        self.strategy = "retrain"
        self.init = True
        self.train = True
        self.auto_test = True
        
        self.increase = False
        self.detect = False
        self.replay = False
        self.ewc = False
        self.num_hops = 2
        
        self.load_first_year = False
        self.kl_weight = 1e-4
        
        self.method = method
        self.logname = logname
        self.seed = seed
        self.gpuid = gpuid
