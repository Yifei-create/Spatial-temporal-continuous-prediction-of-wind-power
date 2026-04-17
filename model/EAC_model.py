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


class EAC_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(
            args.tcn["in_channel"],
            args.tcn["out_channel"],
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"],
            padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
        )
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
        self.num_nodes = args.base_node_size

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            param.requires_grad = name in ("U", "V")

    def forward(self, data, adj):
        num_nodes = adj.shape[0]
        input_dim = self.args.gcn["in_channel"]
        x_in = data.x
        x = x_in.reshape((-1, num_nodes, input_dim))
        batch_size = x.shape[0]
        adaptive_params = torch.mm(self.U[:num_nodes, :], self.V)
        x = x + adaptive_params.unsqueeze(0).expand(batch_size, *adaptive_params.shape)
        x = F.relu(self.gcn1(x, adj))
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))
        x = self.tcn1(x)
        x = x.reshape((-1, num_nodes, self.args.gcn["hidden_channel"]))
        x = self.gcn2(x, adj)
        x = x.reshape((-1, self.args.gcn["out_channel"]))
        x = x + x_in
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            n_add = new_num_nodes - self.num_nodes
            new_params = nn.Parameter(torch.empty(n_add, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            self.num_nodes = new_num_nodes
