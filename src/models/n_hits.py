# src/models/nhits_forecaster.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class NHitsBlock(nn.Module):
    """
    Simplified N-HiTS block:
    - temporal pooling (downsampling)
    - MLP on flattened pooled sequence
    - outputs only forecast (no backcast here, but you can add it later)
    """
    def __init__(
        self,
        input_length: int,
        in_features: int,
        horizon: int,
        hidden_dim: int = 256,
        pool_size: int = 1,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.input_length = input_length
        self.in_features = in_features
        self.horizon = horizon

        # after pooling, effective length
        pooled_len = input_length // pool_size

        self.fc = nn.Sequential(
            nn.Linear(pooled_len * in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head_forecast = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns forecast: (B, H)
        """
        B, T, F_ = x.shape

        if self.pool_size > 1:
            # downsample along time with average pooling
            # reshape to apply pooling
            # x -> (B*F, 1, T) then pool, then reshape back
            x_ = x.permute(0, 2, 1).contiguous()      # (B, F, T)
            x_ = F.avg_pool1d(x_, kernel_size=self.pool_size, stride=self.pool_size)
            x_ = x_.permute(0, 2, 1).contiguous()     # (B, T_pool, F)
        else:
            x_ = x

        x_ = x_.reshape(B, -1)
        h = self.fc(x_)
        forecast = self.head_forecast(h)
        return forecast


class NHitsForecaster(nn.Module):
    """
    N-HiTS-style multi-horizon forecaster with detector embeddings.
    Input:  x (B, T, F), det_ids (B,)
    Output: (B, horizon)
    """
    def __init__(
        self,
        input_length: int,
        num_features: int,
        horizon: int,
        num_detectors: int,
        emb_dim: int = 16,
        n_blocks: int = 3,
        hidden_dim: int = 256,
        pool_sizes=(1, 2, 4),
    ):
        super().__init__()

        self.horizon = horizon
        self.embedding = nn.Embedding(num_detectors, emb_dim)

        # if fewer pool_sizes than blocks, cycle them
        if len(pool_sizes) < n_blocks:
            repeats = (n_blocks + len(pool_sizes) - 1) // len(pool_sizes)
            pool_sizes = (pool_sizes * repeats)[:n_blocks]

        in_features = num_features + emb_dim

        self.blocks = nn.ModuleList([
            NHitsBlock(
                input_length=input_length,
                in_features=in_features,
                horizon=horizon,
                hidden_dim=hidden_dim,
                pool_size=pool_sizes[i],
            )
            for i in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        det_ids: (B,)
        """
        B, T, F = x.shape

        emb = self.embedding(det_ids)          # (B, emb_dim)
        emb_rep = emb.unsqueeze(1).expand(B, T, -1)  # (B, T, emb_dim)
        x_cat = torch.cat([x, emb_rep], dim=-1)      # (B, T, F+emb_dim)

        # Sum of block forecasts
        forecast = 0.0
        for block in self.blocks:
            forecast = forecast + block(x_cat)

        return forecast
