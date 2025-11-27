# src/models/tcn_forecaster.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    Removes the extra padding at the end to keep causality.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network forecaster with detector embeddings.

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
        num_channels=(64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.det_embedding = nn.Embedding(num_detectors, emb_dim)

        in_channels = num_features + emb_dim

        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)

        # final head: take last timestep representation (causal)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1]),
            nn.ReLU(),
            nn.Linear(num_channels[-1], horizon),
        )

    def forward(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        det_ids: (B,)
        """
        B, T, F = x.shape

        det_emb = self.det_embedding(det_ids)          # (B, emb_dim)
        det_rep = det_emb.unsqueeze(1).expand(B, T, -1)
        x_cat = torch.cat([x, det_rep], dim=-1)        # (B, T, F+emb_dim)

        # TCN expects (B, C, T)
        x_cat = x_cat.permute(0, 2, 1).contiguous()    # (B, C, T)

        h = self.network(x_cat)                        # (B, C_out, T)
        h_last = h[:, :, -1]                           # (B, C_out)

        out = self.head(h_last)                        # (B, horizon)
        return out
