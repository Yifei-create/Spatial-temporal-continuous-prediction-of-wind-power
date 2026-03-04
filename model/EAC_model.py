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
        """
        x: (B, N, in_channels)
        adj: (N, N)
        Returns: (B, N, out_channels)
        """
        # x @ weight
        out = torch.matmul(x, self.weight)  # (B, N, out_channels)
        
        # adj @ out
        out = torch.matmul(adj.unsqueeze(0), out)  # (B, N, out_channels)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class EAC_Model(nn.Module):
    """
    EAC Model: Using low-rank decomposition for prompt parameters
    x' = x + U @ V
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank
        
        # GCN layers
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        
        # TCN layer (EAC style: Conv1d on (B*N, 1, hidden))
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
        
        # EAC low-rank parameters
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
        
        self.period = args.period
        self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        D_in = self.args.gcn["in_channel"]
        
        # data.x is expected to be (B*N, D_in) after PyG batching
        x_in = data.x  # keep for residual, shape: (B*N, D_in)
        
        # (B*N, D_in) -> (B, N, D_in)
        x = x_in.reshape((-1, N, D_in))
        B = x.shape[0]
        
        # Compute adaptive parameters: p = U @ V  -> (N, D_in)
        adaptive_params = torch.mm(self.U[:N, :], self.V)
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)
        
        # GCN1: (B, N, hidden)
        x = F.relu(self.gcn1(x, adj))
        
        # (B, N, hidden) -> (B*N, 1, hidden)  (EAC style)
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))
        
        # TCN
        x = self.tcn1(x)
        
        # Back to (B, N, hidden)
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))
        
        # GCN2: (B, N, out)
        x = self.gcn2(x, adj)
        
        # (B, N, out) -> (B*N, out)
        x = x.reshape((-1, self.args.gcn["out_channel"]))
        
        # Residual connection (requires out_channel == in_channel, ensured by config)
        x = x + x_in
        
        # Output: (B*N, y_len)
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        """Expand parameters to accommodate new nodes"""
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(
                torch.empty(new_num_nodes - self.num_nodes, self.rank, 
                           dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1)
            )
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            self.num_nodes = new_num_nodes