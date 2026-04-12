import torch
import torch.nn as nn
import torch.nn.functional as F
from model.static_embedding import StaticFreqMixin


class BatchGCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super().__init__()
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)
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


class VariationalScaleShift_Model(nn.Module, StaticFreqMixin):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn['in_channel'], args.gcn['hidden_channel'], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn['hidden_channel'], args.gcn['out_channel'], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(args.tcn['in_channel'], args.tcn['out_channel'], kernel_size=args.tcn['kernel_size'], dilation=args.tcn['dilation'], padding=int((args.tcn['kernel_size']-1)*args.tcn['dilation']/2))
        self.fc = nn.Linear(args.gcn['out_channel'], args.y_len)
        self.activation = nn.GELU()
        self.mu_scale = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.mu_shift = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.log_var_scale = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.log_var_shift = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.num_nodes = args.base_node_size
        self.init_static_freq(args)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def freeze_backbone(self):
        adaptive_names = {'mu_scale', 'mu_shift', 'log_var_scale', 'log_var_shift'}
        for name, param in self.named_parameters():
            param.requires_grad = name in adaptive_names

    def _kl_gaussian_vs_standard_normal(self, mu, log_var, N_used):
        mu = mu[:N_used, :]
        log_var = log_var[:N_used, :]
        var = torch.exp(log_var)
        kl = 0.5 * (var + mu.pow(2) - 1 - log_var)
        return kl.sum()

    def forward(self, data, adj):
        N = adj.shape[0]
        D_in = self.args.gcn['in_channel']
        x_in = data.x
        x = x_in.reshape((-1, N, D_in))
        x = self.apply_static_freq(x, data, N)
        if self.training:
            std_scale = torch.exp(0.5 * self.log_var_scale[:N, :])
            std_shift = torch.exp(0.5 * self.log_var_shift[:N, :])
            eps_scale = torch.randn_like(std_scale, device=x.device, dtype=x.dtype)
            eps_shift = torch.randn_like(std_shift, device=x.device, dtype=x.dtype)
            adaptive_scale = self.mu_scale[:N, :] + std_scale * eps_scale
            adaptive_shift = self.mu_shift[:N, :] + std_shift * eps_shift
            kl_term = self._kl_gaussian_vs_standard_normal(self.mu_scale, self.log_var_scale, N) + self._kl_gaussian_vs_standard_normal(self.mu_shift, self.log_var_shift, N)
        else:
            adaptive_scale = self.mu_scale[:N, :]
            adaptive_shift = self.mu_shift[:N, :]
            kl_term = None
        x = x * (1 + adaptive_scale.unsqueeze(0)) + adaptive_shift.unsqueeze(0)
        x = F.relu(self.gcn1(x, adj))
        x = x.reshape((-1, 1, self.args.gcn['hidden_channel']))
        x = self.tcn1(x)
        x = x.reshape((-1, N, self.args.gcn['hidden_channel']))
        x = self.gcn2(x, adj)
        x = x.reshape((-1, self.args.gcn['out_channel']))
        x = x + x_in
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.training and kl_term is not None:
            return x, kl_term
        return x

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            n_add = new_num_nodes - self.num_nodes
            self.mu_scale = nn.Parameter(torch.cat([self.mu_scale, nn.Parameter(torch.zeros(n_add, 1, dtype=self.mu_scale.dtype, device=self.mu_scale.device))], dim=0))
            self.mu_shift = nn.Parameter(torch.cat([self.mu_shift, nn.Parameter(torch.zeros(n_add, 1, dtype=self.mu_shift.dtype, device=self.mu_shift.device))], dim=0))
            self.log_var_scale = nn.Parameter(torch.cat([self.log_var_scale, nn.Parameter(torch.zeros(n_add, 1, dtype=self.log_var_scale.dtype, device=self.log_var_scale.device))], dim=0))
            self.log_var_shift = nn.Parameter(torch.cat([self.log_var_shift, nn.Parameter(torch.zeros(n_add, 1, dtype=self.log_var_shift.dtype, device=self.log_var_shift.device))], dim=0))
            self.num_nodes = new_num_nodes
