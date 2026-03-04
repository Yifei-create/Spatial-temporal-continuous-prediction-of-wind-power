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


class ScaleShift_Model(nn.Module):
    """
    Deterministic Scale-Shift Model
      x' = (1 + scale) * x + shift
    Two scalar parameters per node: scale and shift
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
        
        # Scale-Shift parameters: one scale and one shift per node
        self.scale = nn.Parameter(torch.zeros(args.base_node_size, 1))
        self.shift = nn.Parameter(torch.zeros(args.base_node_size, 1))
        
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

        # data.x is expected to be (B*N, D_in)
        x_in = data.x  # keep a stable residual tensor

        # (B*N, D_in) -> (B, N, D_in)
        x = x_in.reshape((-1, N, D_in))
        B = x.shape[0]

        # Apply scale-shift transformation: x' = (1 + scale) * x + shift
        # scale/shift are (N, 1) and broadcast over D_in
        adaptive_scale = self.scale[:N]  # (N, 1)
        adaptive_shift = self.shift[:N]  # (N, 1)
        x = x * (1 + adaptive_scale.unsqueeze(0)) + adaptive_shift.unsqueeze(0)  # (B, N, D_in)
        
        # GCN1
        x = F.relu(self.gcn1(x, adj))  # (B, N, hidden)
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))  # (B*N, 1, hidden)
        
        # TCN
        x = self.tcn1(x)  # (B*N, 1, hidden)
        
        # GCN2
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))  # (B, N, hidden)
        x = self.gcn2(x, adj)  # (B, N, out)
        x = x.reshape((-1, self.args.gcn["out_channel"]))  # (B*N, out)
        
        # Residual connection (requires out_channel == in_channel to match x_in)
        x = x + x_in
        
        # Output
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        """Expand parameters to accommodate new nodes"""
        if new_num_nodes > self.num_nodes:
            n_add = new_num_nodes - self.num_nodes
            new_scale = nn.Parameter(torch.zeros(n_add, 1, dtype=self.scale.dtype, device=self.scale.device))
            new_shift = nn.Parameter(torch.zeros(n_add, 1, dtype=self.shift.dtype, device=self.shift.device))
            self.scale = nn.Parameter(torch.cat([self.scale, new_scale], dim=0))
            self.shift = nn.Parameter(torch.cat([self.shift, new_shift], dim=0))
            self.num_nodes = new_num_nodes