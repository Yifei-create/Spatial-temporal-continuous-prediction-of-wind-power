import torch
import torch.nn as nn
import math


class PatchTST_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.x_len = args.x_len
        self.y_len = args.y_len
        self.num_features = args.num_features
        self.dropout_rate = args.dropout
        self.num_nodes = args.base_node_size
        self.use_static_embedding = getattr(args, 'use_static_embedding', False)
        self.use_freq_embedding = getattr(args, 'use_freq_embedding', False)
        self.static_proj = nn.Linear(args.static_dim, self.num_features) if self.use_static_embedding else None
        self.freq_embedding = nn.Embedding(3, args.freq_emb_dim) if self.use_freq_embedding else None
        self.freq_proj = nn.Linear(args.freq_emb_dim, self.num_features) if self.use_freq_embedding else None

        cfg = args.patchtst
        self.patch_len = cfg['patch_len']
        self.stride = cfg['stride']
        self.d_model = cfg['d_model']
        self.n_heads = cfg['n_heads']
        self.n_layers = cfg['n_layers']
        self.d_ff = cfg['d_ff']
        self.padding_len = self.stride
        self.num_patches = math.floor((self.x_len - self.patch_len) / self.stride) + 2
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches, self.d_model) * 0.02)
        self.pos_dropout = nn.Dropout(p=self.dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=self.dropout_rate, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers, enable_nested_tensor=False)
        self.head_dropout = nn.Dropout(p=self.dropout_rate)
        self.head = nn.Linear(self.num_patches * self.d_model, self.y_len)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters:     {total}")
        self.args.logger.info(f"Trainable Parameters: {trainable}")

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False

    def expand_adaptive_params(self, new_num_nodes: int):
        self.num_nodes = new_num_nodes

    def forward(self, data, adj=None):
        x_flat = data.x
        BN = x_flat.size(0)
        x = x_flat.reshape(BN, self.x_len, self.num_features)
        if self.static_proj is not None and hasattr(data, 'static_x') and data.static_x is not None:
            static_bias = self.static_proj(data.static_x).unsqueeze(1)
            x = x + static_bias
        if self.freq_embedding is not None and hasattr(data, 'freq_id') and data.freq_id is not None:
            freq_bias = self.freq_proj(self.freq_embedding(data.freq_id.view(-1))).unsqueeze(1)
            x = x + freq_bias
        x = x.permute(0, 2, 1).reshape(BN * self.num_features, self.x_len)
        x_mean = x.mean(dim=1, keepdim=True).detach()
        x_std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - x_mean) / x_std
        pad = x[:, :self.padding_len]
        x = torch.cat([pad, x], dim=1)
        x_unfold = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = self.patch_embedding(x_unfold)
        z = z + self.pos_embedding.squeeze(0)
        z = self.pos_dropout(z)
        z = self.transformer(z)
        z = z.reshape(BN * self.num_features, -1)
        z = self.head_dropout(z)
        out = self.head(z)
        out = out * x_std + x_mean
        out = out.reshape(BN, self.num_features, self.y_len)
        return out[:, -1, :]
