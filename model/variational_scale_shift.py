import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchGCNConv(nn.Module):
    """Batch Graph Convolutional Layer"""
    def __init__(self, in_channels, out_channels, bias=True, gcn=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn = gcn
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        out = torch.matmul(x, self.weight)
        out = torch.matmul(adj.unsqueeze(0), out)
        if self.bias is not None:
            out = out + self.bias
        return out


class VariationalScaleShift_Model(nn.Module):
    """
    Variational Scale-Shift Model
    scale ~ N(mu_scale, sigma_scale^2)
    shift ~ N(mu_shift, sigma_shift^2)
    x' = (1 + scale) * x + shift
    
    Use reparameterization trick for sampling during training, use mean during inference
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        
        # GCN layers
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        
        # TCN layer
        self.tcn1 = nn.Conv1d(
            in_channels=args.tcn["in_channel"], 
            out_channels=args.tcn["out_channel"], 
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"], 
            padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2)
        )
        
        # Output layer
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        # Variational parameters: mean and log variance (initialized to 0, corresponding to prior N(0,1))
        self.mu_scale = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.mu_shift = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.log_var_scale = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.log_var_shift = nn.Parameter(torch.zeros(args.base_node_size, 1))
        
        self.year = args.year
        self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def _kl_gaussian_vs_standard_normal(self, mu, log_var, N_used):
        """
        Compute KL divergence: KL(q(z|x) || N(0,1))
        q(z|x) = N(mu, exp(log_var))
        """
        mu = mu[:N_used, :]
        log_var = log_var[:N_used, :]
        var = torch.exp(log_var)
        kl = 0.5 * (var + mu.pow(2) - 1 - log_var)
        return kl.sum()
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  # (B, N, D)
        B, N, T = x.shape
        
        if self.training:
            # Training: sample using reparameterization trick
            std_scale = torch.exp(0.5 * self.log_var_scale[:N, :])
            std_shift = torch.exp(0.5 * self.log_var_shift[:N, :])
            eps_scale = torch.randn_like(std_scale, device=x.device, dtype=x.dtype)
            eps_shift = torch.randn_like(std_shift, device=x.device, dtype=x.dtype)
            adaptive_scale = self.mu_scale[:N, :] + std_scale * eps_scale
            adaptive_shift = self.mu_shift[:N, :] + std_shift * eps_shift
            
            # Compute KL divergence
            kl_scale = self._kl_gaussian_vs_standard_normal(self.mu_scale, self.log_var_scale, N)
            kl_shift = self._kl_gaussian_vs_standard_normal(self.mu_shift, self.log_var_shift, N)
            kl_term = kl_scale + kl_shift
        else:
            # Inference: use mean
            adaptive_scale = self.mu_scale[:N, :]
            adaptive_shift = self.mu_shift[:N, :]
            kl_term = None
        
        # Apply scale-shift transformation
        x = x * (1 + adaptive_scale.unsqueeze(0)) + adaptive_shift.unsqueeze(0)  # (B, N, D)
        
        # GCN1
        x = F.relu(self.gcn1(x, adj))  # (B, N, hidden)
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))  # (B*N, 1, hidden)
        
        # TCN
        x = self.tcn1(x)  # (B*N, 1, hidden)
        
        # GCN2
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))  # (B, N, hidden)
        x = self.gcn2(x, adj)  # (B, N, out)
        x = x.reshape((-1, self.args.gcn["out_channel"]))  # (B*N, out)
        
        # Residual connection
        x = x + data.x
        
        # Output
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Return prediction and KL divergence during training, only prediction during inference
        if self.training and kl_term is not None:
            return x, kl_term
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        """Expand parameters to accommodate new nodes"""
        if new_num_nodes > self.num_nodes:
            n_add = new_num_nodes - self.num_nodes
            new_mu_scale = nn.Parameter(torch.zeros(n_add, 1, dtype=self.mu_scale.dtype, device=self.mu_scale.device))
            new_mu_shift = nn.Parameter(torch.zeros(n_add, 1, dtype=self.mu_shift.dtype, device=self.mu_shift.device))
            new_log_var_scale = nn.Parameter(torch.zeros(n_add, 1, dtype=self.log_var_scale.dtype, device=self.log_var_scale.device))
            new_log_var_shift = nn.Parameter(torch.zeros(n_add, 1, dtype=self.log_var_shift.dtype, device=self.log_var_shift.device))
            
            self.mu_scale = nn.Parameter(torch.cat([self.mu_scale, new_mu_scale], dim=0))
            self.mu_shift = nn.Parameter(torch.cat([self.mu_shift, new_mu_shift], dim=0))
            self.log_var_scale = nn.Parameter(torch.cat([self.log_var_scale, new_log_var_scale], dim=0))
            self.log_var_shift = nn.Parameter(torch.cat([self.log_var_shift, new_log_var_shift], dim=0))
            
            self.num_nodes = new_num_nodes
