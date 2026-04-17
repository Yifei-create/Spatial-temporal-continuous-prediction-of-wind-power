import math

import torch
import torch.nn as nn


class PatchTST_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.x_len = args.x_len
        self.y_len = args.y_len
        self.num_features = args.num_features
        self.dropout_rate = args.dropout
        self.num_nodes = args.base_node_size

        cfg = args.patchtst
        self.patch_len = cfg["patch_len"]
        self.stride = cfg["stride"]
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.n_layers = cfg["n_layers"]
        self.d_ff = cfg["d_ff"]
        self.padding_len = self.stride
        self.num_patches = math.floor((self.x_len - self.patch_len) / self.stride) + 2

        self.patch_embedding = nn.Linear(self.patch_len * self.num_features, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.d_model) * 0.02)
        self.input_dropout = nn.Dropout(p=self.dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers, enable_nested_tensor=False)
        self.head_dropout = nn.Dropout(p=self.dropout_rate)
        self.head = nn.Linear(self.num_patches * self.d_model, self.y_len)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters:     {total}")
        self.args.logger.info(f"Trainable Parameters: {trainable}")

    def freeze_backbone(self):
        return

    def expand_adaptive_params(self, new_num_nodes):
        self.num_nodes = new_num_nodes

    def forward(self, data, adj=None):
        x_flat = data.x
        batch_size = x_flat.size(0)
        x = x_flat.reshape(batch_size, self.x_len, self.num_features)

        pad = x[:, :self.padding_len, :]
        x = torch.cat([pad, x], dim=1)
        x_unfold = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x_unfold = x_unfold.contiguous().reshape(batch_size, self.num_patches, self.patch_len * self.num_features)

        z = self.patch_embedding(x_unfold)
        z = z + self.pos_embedding
        z = self.input_dropout(z)
        z = self.transformer(z)
        z = z.reshape(batch_size, -1)
        z = self.head_dropout(z)
        return self.head(z)
