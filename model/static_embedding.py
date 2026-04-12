import torch
import torch.nn as nn


class StaticFreqMixin:
    def init_static_freq(self, args):
        self.use_static_embedding = getattr(args, 'use_static_embedding', False)
        self.use_freq_embedding = getattr(args, 'use_freq_embedding', False)
        in_channel = args.gcn['in_channel']
        self.x_embedding = nn.Embedding(args.pos_x_bins, in_channel) if self.use_static_embedding else None
        self.y_embedding = nn.Embedding(args.pos_y_bins, in_channel) if self.use_static_embedding else None
        self.freq_embedding = nn.Embedding(args.freq_num_embeddings, in_channel) if self.use_freq_embedding else None

    def apply_static_freq(self, x, data, N):
        B = x.shape[0]
        if self.use_static_embedding and self.x_embedding is not None:
            has_pos_bins = all(hasattr(data, name) and getattr(data, name) is not None for name in ('static_x_bin', 'static_y_bin'))
            if has_pos_bins:
                x_bin = data.static_x_bin.reshape(B, N)
                y_bin = data.static_y_bin.reshape(B, N)
                pos_emb = self.x_embedding(x_bin) + self.y_embedding(y_bin)
                x = x + pos_emb
        if self.freq_embedding is not None and hasattr(data, 'freq_id') and data.freq_id is not None:
            freq_id = data.freq_id.view(B)
            freq_bias = self.freq_embedding(freq_id).unsqueeze(1)
            x = x + freq_bias
        return x
