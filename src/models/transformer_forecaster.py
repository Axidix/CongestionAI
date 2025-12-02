import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Head config utilities (same spirit as in your TCN file)
# ----------------------------------------------------------------------


def compute_head_config(
    horizon: int,
    head_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Dict]:
    """
    Build configuration for range-based heads.

    Args:
        horizon: Total forecast horizon H.
        head_ranges: Optional dict mapping name -> (start_idx, end_idx), 1-based.
            If None, defaults to:
                - immediate: 1-3
                - short:    4-8
                - medium:   9-16
                - long:     17-24
            and truncated to the requested horizon.

    Returns:
        Dict[name] -> {
            "start": int (0-based inclusive),
            "end":   int (0-based inclusive),
            "size":  int
        }
    """
    if head_ranges is None:
        head_ranges = {
            "immediate": (1, 3),
            "short": (4, 8),
            "medium": (9, 16),
            "long": (17, 24),
        }

    config = {}
    for name, (s, e) in head_ranges.items():
        if s > horizon:
            continue
        e = min(e, horizon)
        size = e - s + 1
        if size <= 0:
            continue
        # Convert to 0-based indices
        config[name] = {
            "start": s - 1,
            "end": e - 1,
            "size": size,
        }

    # If nothing created (e.g. very small horizon), fall back to a single head
    if not config:
        config["all"] = {"start": 0, "end": horizon - 1, "size": horizon}

    return config


class RangeHead(nn.Module):
    """
    Simple MLP head that predicts a contiguous block of the horizon.

    Input:  (B, D)
    Output: (B, size)
    """

    def __init__(self, in_dim: int, size: int, name: str, dropout: float = 0.1):
        super().__init__()
        self.name = name
        hidden = max(in_dim // 2, 64)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------
# Positional encoding (optional)
# ----------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic transformer positional encoding.
    We keep this lightweight; you already have time features,
    but this can still help the attention layers.

    Input / output shape: (B, T, D)
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ----------------------------------------------------------------------
# Transformer encoder layer
# ----------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    """
    Slightly customized Transformer encoder layer tailored for short sequences
    (24–48 steps) and regression.

    Uses:
      - Multi-head self-attention (batch_first=True)
      - Pre-LN
      - GELU activation
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        # Self-attention block (pre-norm)
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout1(attn_output)

        # FFN block (pre-norm)
        x_norm = self.norm2(x)
        ffn = self.linear2(self.dropout2(self.activation(self.linear1(x_norm))))
        x = x + ffn

        return x


# ----------------------------------------------------------------------
# Multi-head Transformer forecaster
# ----------------------------------------------------------------------


class MultiHeadTransformerForecaster(nn.Module):
    """
    Transformer-based forecaster with range-based heads, analogous to
    MultiHeadTCNForecaster.

    Pipeline:
      - Detector ID → embedding
      - Concatenate detector embedding to each timestep features
      - Linear projection → d_model
      - Optional sinusoidal positional encoding
      - L stacked TransformerEncoderLayer
      - Pool over time (last / mean / max / attention)
      - Range heads predict horizon segments
    """

    def __init__(
        self,
        num_features: int,
        horizon: int,
        num_detectors: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        det_emb_dim: int = 64,
        pooling: str = "attention",
        use_positional_encoding: bool = True,
        max_seq_len: int = 64,
        head_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__()

        assert 1 <= horizon <= 24, "Horizon must be between 1 and 24"
        self.horizon = horizon
        self.pooling = pooling
        self.d_model = d_model

        # Head configuration (similar concept as TCN forecaster)
        self.head_config = compute_head_config(horizon, head_ranges)
        self.head_names = list(self.head_config.keys())

        # Detector embedding
        self.det_embedding = nn.Embedding(num_detectors, det_emb_dim)

        # Project input (features + detector embedding) → d_model
        in_dim = num_features + det_emb_dim
        self.input_proj = nn.Linear(in_dim, d_model)

        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        # Transformer encoder stack
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Attention pooling over time if requested
        if pooling == "attention":
            self.attn_pool = nn.Linear(d_model, 1)

        # Range heads: from pooled feature (B, D) → horizon chunks
        self.heads = nn.ModuleDict(
            {
                name: RangeHead(d_model, cfg["size"], name, dropout=dropout)
                for name, cfg in self.head_config.items()
            }
        )

        self._init_parameters()

    def _init_parameters(self):
        # Slightly better initializer for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------ pooling ---------------------------------

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """
        Pool over time dimension.

        Args:
            h: (B, T, D)
        Returns:
            (B, D)
        """
        if self.pooling == "last":
            return h[:, -1, :]
        elif self.pooling == "avg":
            return h.mean(dim=1)
        elif self.pooling == "max":
            return h.max(dim=1)[0]
        elif self.pooling == "attention":
            # (B, T, D) → attention over T
            scores = self.attn_pool(h)  # (B, T, 1)
            weights = torch.softmax(scores, dim=1)  # (B, T, 1)
            return (h * weights).sum(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

    # ------------------------ encoding --------------------------------

    def encode(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        (B, T, F)   input features
            det_ids:  (B,)        detector indices

        Returns:
            pooled_features: (B, D)
        """
        B, T, F = x.shape

        # Detector embedding, repeated across time
        det_emb = self.det_embedding(det_ids)  # (B, det_emb_dim)
        det_emb = det_emb.unsqueeze(1).expand(B, T, -1)  # (B, T, det_emb_dim)

        x = torch.cat([x, det_emb], dim=-1)  # (B, T, F + det_emb_dim)

        # Project to d_model
        h = self.input_proj(x)  # (B, T, D)

        # Positional encoding
        if self.use_positional_encoding:
            h = self.pos_encoder(h)

        # Transformer encoder stack
        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)

        # Pool over time
        pooled = self._pool(h)  # (B, D)
        return pooled

    # ------------------------ forward ---------------------------------

    def forward(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) input history features
            det_ids: (B,) detector ids

        Returns:
            preds: (B, horizon) forecast for each step
        """
        features = self.encode(x, det_ids)  # (B, D)
        chunks = [self.heads[name](features) for name in self.head_names]
        preds = torch.cat(chunks, dim=-1)  # (B, H)
        return preds

    def forward_heads(self, x: torch.Tensor, det_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict of {head_name: (B, head_size)}.
        """
        features = self.encode(x, det_ids)
        return {name: self.heads[name](features) for name in self.head_names}

    def get_head_config(self) -> Dict[str, dict]:
        return self.head_config.copy()

