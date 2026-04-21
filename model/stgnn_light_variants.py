import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchGCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super().__init__()
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter("weight_self", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x, adj):
        out = self.weight_neigh(torch.matmul(adj, x))
        if self.weight_self is not None:
            out = out + self.weight_self(x)
        return out


class AdaptiveSTGNNBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank
        self.num_nodes = args.base_node_size
        self.input_dim = args.gcn["in_channel"]
        self.hidden_dim = args.gcn["hidden_channel"]
        self.output_dim = args.gcn["out_channel"]
        self.y_len = args.y_len
        self.num_features = args.num_features
        self.x_len = args.x_len
        self.num_stages = getattr(args, "num_stages", 1)

        self.gcn1 = BatchGCNConv(self.input_dim, self.hidden_dim, bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(self.hidden_dim, self.output_dim, bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(
            args.tcn["in_channel"],
            args.tcn["out_channel"],
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"],
            padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
        )
        self.fc = nn.Linear(self.output_dim, self.y_len)
        self.activation = nn.GELU()

        self._node_param_init = {}
        self._global_param_names = set()
        self._init_adaptive_params()

    def _init_adaptive_params(self):
        raise NotImplementedError

    def node_adaptive_param_names(self):
        return set(self._node_param_init.keys())

    def global_adaptive_param_names(self):
        return set(self._global_param_names)

    def _register_node_param(self, name, shape, init="zeros"):
        param = nn.Parameter(self._make_param_tensor(shape, init))
        setattr(self, name, param)
        self._node_param_init[name] = init

    def _register_global_param(self, name, shape, init="zeros"):
        param = nn.Parameter(self._make_param_tensor(shape, init))
        setattr(self, name, param)
        self._global_param_names.add(name)

    def _make_param_tensor(self, shape, init):
        if init == "zeros":
            return torch.zeros(*shape)
        if init == "uniform_small":
            return torch.empty(*shape).uniform_(-0.1, 0.1)
        if init == "normal_small":
            return torch.empty(*shape).normal_(mean=0.0, std=0.02)
        raise ValueError(f"Unsupported init strategy: {init}")

    def _extend_node_param(self, param, n_add, init):
        ext_shape = (n_add,) + tuple(param.shape[1:])
        ext = self._make_param_tensor(ext_shape, init).to(device=param.device, dtype=param.dtype)
        return nn.Parameter(torch.cat([param, ext], dim=0))

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def freeze_backbone(self):
        adaptive_names = self.node_adaptive_param_names() | self.global_adaptive_param_names()
        for name, param in self.named_parameters():
            param.requires_grad = any(name == target or name.endswith("." + target) for target in adaptive_names)

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return
        n_add = new_num_nodes - self.num_nodes
        for name, init in self._node_param_init.items():
            param = getattr(self, name)
            setattr(self, name, self._extend_node_param(param, n_add, init))
        self.num_nodes = new_num_nodes

    def _stage_index_from_data(self, data, batch_size, device):
        if hasattr(data, "stage_idx") and data.stage_idx is not None:
            stage_idx = data.stage_idx.view(batch_size).to(device=device, dtype=torch.long)
        else:
            stage_idx = torch.zeros(batch_size, device=device, dtype=torch.long)
        return torch.clamp(stage_idx, min=0, max=self.num_stages - 1)

    def adapt_input(self, x, data, num_nodes):
        return x

    def adapt_hidden(self, h, data, num_nodes):
        return h

    def adapt_output(self, out, data, num_nodes):
        return out

    def forward(self, data, adj):
        num_nodes = adj.shape[0]
        x_in = data.x
        x = x_in.reshape((-1, num_nodes, self.input_dim))
        x = self.adapt_input(x, data, num_nodes)
        x = F.relu(self.gcn1(x, adj))
        x = self.adapt_hidden(x, data, num_nodes)
        x = x.reshape((-1, 1, self.hidden_dim))
        x = self.tcn1(x)
        x = x.reshape((-1, num_nodes, self.hidden_dim))
        x = self.gcn2(x, adj)
        x = x.reshape((-1, self.output_dim))
        x = x + x_in
        x = self.fc(self.activation(x))
        x = x.reshape((-1, num_nodes, self.y_len))
        x = self.adapt_output(x, data, num_nodes)
        x = x.reshape((-1, self.y_len))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class STGNNInputBias_Model(AdaptiveSTGNNBase):
    def _init_adaptive_params(self):
        self._register_node_param("input_bias", (self.num_nodes, self.input_dim), init="zeros")

    def adapt_input(self, x, data, num_nodes):
        return x + self.input_bias[:num_nodes].unsqueeze(0)


class STGNNStageResidual_Model(AdaptiveSTGNNBase):
    def _init_adaptive_params(self):
        self._register_node_param("stage_residual", (self.num_nodes, self.input_dim), init="zeros")
        self._register_global_param("stage_bias", (self.num_stages, self.input_dim), init="zeros")

    def adapt_input(self, x, data, num_nodes):
        batch_size = x.shape[0]
        stage_idx = self._stage_index_from_data(data, batch_size, x.device)
        stage_term = self.stage_bias[stage_idx].unsqueeze(1)
        return x + stage_term + self.stage_residual[:num_nodes].unsqueeze(0)
