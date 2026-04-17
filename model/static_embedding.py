import torch
import torch.nn as nn


class StaticFreqMixin:
    def init_static_freq(self, args):
        self.use_static_embedding = getattr(args, "use_static_embedding", False)
        self.use_freq_embedding = getattr(args, "use_freq_embedding", False)
        in_channel = args.gcn["in_channel"]
        self.static_projection = None
        if self.use_static_embedding:
            static_dim = int(getattr(args, "static_dim", 0))
            if static_dim <= 0:
                raise ValueError(f"Invalid static_dim={static_dim} for static coordinate projection.")
            static_mean = torch.as_tensor(getattr(args, "static_mean", None), dtype=torch.float32)
            static_std = torch.as_tensor(getattr(args, "static_std", None), dtype=torch.float32)
            if static_mean.numel() != static_dim or static_std.numel() != static_dim:
                raise ValueError(
                    f"Static normalization buffers must have length {static_dim}. Received mean={tuple(static_mean.shape)}, std={tuple(static_std.shape)}."
                )
            if torch.any(static_std <= 0.0):
                raise ValueError(f"Invalid static_std values: {static_std.tolist()}")
            self.register_buffer("static_mean", static_mean.view(1, 1, static_dim))
            self.register_buffer("static_scale", static_std.view(1, 1, static_dim))
            self.static_projection = nn.Linear(static_dim, in_channel)
        self.freq_projection = None
        if self.use_freq_embedding:
            supported_frequency_minutes = [int(freq) for freq in getattr(args, "supported_frequency_minutes", [])]
            if len(supported_frequency_minutes) < 2:
                raise ValueError("Dynamic frequency projection requires at least two supported frequencies.")
            freq_values = torch.tensor(supported_frequency_minutes, dtype=torch.float32)
            self.register_buffer("freq_values", freq_values)
            self.register_buffer("freq_center", freq_values.mean())
            self.register_buffer("freq_scale", freq_values.std(unbiased=False))
            if float(self.freq_scale.item()) <= 0.0:
                raise ValueError(f"Invalid supported_frequency_minutes: {supported_frequency_minutes}")
            self.freq_projection = nn.Linear(1, in_channel)

    def apply_static_freq(self, x, data, N):
        B = x.shape[0]
        if self.static_projection is not None:
            if not hasattr(data, "static_data") or data.static_data is None:
                raise ValueError("static_data is required when use_static_embedding is enabled.")
            static_dim = self.static_mean.shape[-1]
            if data.static_data.dim() != 2 or data.static_data.shape[1] != static_dim:
                raise ValueError(
                    f"static_data must have shape (B*N, {static_dim}). Received {tuple(data.static_data.shape)}."
                )
            static_data = data.static_data.reshape(B, N, static_dim).to(dtype=x.dtype)
            normalized_static = (static_data - self.static_mean.to(dtype=x.dtype)) / self.static_scale.to(dtype=x.dtype)
            x = x + self.static_projection(normalized_static)
        if self.freq_projection is not None and hasattr(data, "freq_id") and data.freq_id is not None:
            freq_id = data.freq_id.view(B)
            if torch.any(freq_id < 0) or torch.any(freq_id >= self.freq_values.numel()):
                raise ValueError(f"freq_id is out of range: {freq_id}")
            freq_minutes = self.freq_values[freq_id].unsqueeze(-1)
            normalized_minutes = (freq_minutes - self.freq_center) / self.freq_scale
            freq_bias = self.freq_projection(normalized_minutes).unsqueeze(1)
            x = x + freq_bias
        return x
