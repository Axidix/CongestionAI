# src/models/transformer_forecaster.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)          # (T, D)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                        # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerForecaster(nn.Module):
    """
    Transformer-based forecaster with detector embeddings.

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
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.det_embedding = nn.Embedding(num_detectors, emb_dim)

        # project features + detector embedding to d_model
        self.input_proj = nn.Linear(num_features + emb_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, T, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=input_length)

        # readout from sequence representation
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        det_ids: (B,)
        """
        B, T, F = x.shape

        det_emb = self.det_embedding(det_ids)              # (B, emb_dim)
        det_rep = det_emb.unsqueeze(1).expand(B, T, -1)    # (B, T, emb_dim)

        x_cat = torch.cat([x, det_rep], dim=-1)            # (B, T, F+emb_dim)
        h = self.input_proj(x_cat)                         # (B, T, d_model)
        h = self.pos_encoding(h)                           # (B, T, d_model)

        h_enc = self.encoder(h)                            # (B, T, d_model)

        # simple choice: take representation at the last timestep
        h_last = h_enc[:, -1, :]                           # (B, d_model)

        out = self.head(h_last)                            # (B, horizon)
        return out
