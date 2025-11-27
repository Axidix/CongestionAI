# src/models/tcn_forecaster.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


# ----------------------------------------------------------------------
# Utility blocks
# ----------------------------------------------------------------------

class Chomp1d(nn.Module):
    """Removes extra padding at end to maintain causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Learns to weight channels based on global context.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
    
    def forward(self, x):
        # x: (B, C, T)
        y = x.mean(dim=-1)  # Global avg pool: (B, C)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))  # Channel weights: (B, C)
        return x * y.unsqueeze(-1)


class TemporalBlock(nn.Module):
    """
    Standard TCN residual block.
    
    Architecture: Conv → Norm → Act → Dropout → Conv → Norm → Act → Dropout → Residual
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation: int, 
        dropout: float = 0.1,
        use_se: bool = False,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self.se = SEBlock(out_channels) if use_se else None

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        if self.se is not None:
            out = self.se(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


# ----------------------------------------------------------------------
# TCN Encoder
# ----------------------------------------------------------------------

class TCNEncoder(nn.Module):
    """Shared TCN encoder backbone."""
    def __init__(
        self,
        input_dim: int,
        num_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_se: bool = False,
    ):
        super().__init__()
        
        layers = []
        in_ch = input_dim
        
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout, use_se)
            )
            in_ch = out_ch
        
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]
    
    def forward(self, x):
        """x: (B, C_in, T) -> (B, C_out, T)"""
        return self.network(x)


# ----------------------------------------------------------------------
# Range-based heads configuration
# ----------------------------------------------------------------------

# Ranges are 1-indexed (h1 = 1 hour ahead, h24 = 24 hours ahead)
# Output tensor is 0-indexed: pred[:, 0] = h1, pred[:, 23] = h24
DEFAULT_HEAD_RANGES = {
    "immediate": (1, 3),    # h1-h3: 0-2 in tensor
    "short": (4, 8),        # h4-h8: 3-7 in tensor
    "medium": (9, 16),      # h9-h16: 8-15 in tensor
    "long": (17, 24),       # h17-h24: 16-23 in tensor
}


def compute_head_config(
    horizon: int,
    head_ranges: Dict[str, Tuple[int, int]] = None,
) -> Dict[str, dict]:
    """
    Compute which heads are needed and their output sizes.
    
    Args:
        horizon: Total forecast horizon (1-24)
        head_ranges: Dict of {name: (start_h, end_h)} where h is 1-indexed
    
    Returns:
        Dict of {name: {"start_h": int, "end_h": int, "start_idx": int, "end_idx": int, "size": int}}
    """
    if head_ranges is None:
        head_ranges = DEFAULT_HEAD_RANGES
    
    config = {}
    
    for name, (start_h, end_h) in head_ranges.items():
        # Skip if range starts beyond horizon
        if start_h > horizon:
            continue
        
        # Clip end to horizon
        actual_end_h = min(end_h, horizon)
        
        # Convert to 0-indexed tensor indices
        start_idx = start_h - 1  # h1 -> idx 0
        end_idx = actual_end_h   # h3 -> idx 3 (exclusive end for slicing)
        size = end_idx - start_idx
        
        if size > 0:
            config[name] = {
                "start_h": start_h,
                "end_h": actual_end_h,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "size": size,
            }
    
    return config


# ----------------------------------------------------------------------
# Range Head
# ----------------------------------------------------------------------

class RangeHead(nn.Module):
    """
    Prediction head for a specific horizon range.
    Deeper for longer-term predictions.
    """
    def __init__(
        self,
        input_dim: int,
        output_size: int,
        range_name: str,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Deeper heads for harder (longer-term) predictions
        depth = {"immediate": 1, "short": 1, "medium": 2, "long": 2}.get(range_name, 1)
        
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(input_dim, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# Multi-Head TCN Forecaster
# ----------------------------------------------------------------------

class MultiHeadTCNForecaster(nn.Module):
    """
    Multi-head TCN Forecaster with range-based heads.
    
    Default ranges (only creates heads needed for given horizon):
        - immediate: h1-h3
        - short: h4-h8
        - medium: h9-h16
        - long: h17-h24
    """
    def __init__(
        self,
        num_features: int,
        horizon: int,
        num_detectors: int,
        emb_dim: int = 32,
        num_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_se: bool = False,
        head_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        pooling: str = "last",
    ):
        super().__init__()
        
        assert 1 <= horizon <= 24, "Horizon must be between 1 and 24"
        
        self.horizon = horizon
        self.pooling = pooling
        
        # Compute head configuration
        self.head_config = compute_head_config(horizon, head_ranges)
        self.head_names = list(self.head_config.keys())
        
        # Detector embedding
        self.det_embedding = nn.Embedding(num_detectors, emb_dim)
        
        # TCN encoder
        self.encoder = TCNEncoder(
            input_dim=num_features + emb_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_se=use_se,
        )
        
        encoder_dim = self.encoder.output_dim
        
        # Attention pooling
        if pooling == "attention":
            self.attn_weights = nn.Linear(encoder_dim, 1)
        
        # Range heads
        self.heads = nn.ModuleDict({
            name: RangeHead(encoder_dim, cfg["size"], name, dropout)
            for name, cfg in self.head_config.items()
        })
    
    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, C, T) -> (B, C)"""
        if self.pooling == "last":
            return h[:, :, -1]
        elif self.pooling == "avg":
            return h.mean(dim=-1)
        elif self.pooling == "max":
            return h.max(dim=-1)[0]
        elif self.pooling == "attention":
            h_t = h.permute(0, 2, 1)  # (B, T, C)
            weights = torch.softmax(self.attn_weights(h_t), dim=1)  # (B, T, 1)
            return (h_t * weights).sum(dim=1)  # (B, C)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def encode(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        """Encode input to pooled features."""
        B, T, F = x.shape
        
        det_emb = self.det_embedding(det_ids).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x, det_emb], dim=-1)
        x = x.permute(0, 2, 1)  # (B, C, T)
        
        h = self.encoder(x)
        return self._pool(h)
    
    def forward(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) input features
            det_ids: (B,) detector indices
        Returns:
            (B, horizon) predictions
        """
        features = self.encode(x, det_ids)
        preds = [self.heads[name](features) for name in self.head_names]
        return torch.cat(preds, dim=-1)
    
    def forward_heads(self, x: torch.Tensor, det_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dict of {head_name: (B, head_size)}"""
        features = self.encode(x, det_ids)
        return {name: self.heads[name](features) for name in self.head_names}
    
    def get_head_config(self) -> Dict[str, dict]:
        """Get head configuration with index mappings."""
        return self.head_config.copy()


# ----------------------------------------------------------------------
# Standard TCN (backward compatible)
# ----------------------------------------------------------------------

class TCNForecaster(nn.Module):
    """Standard single-head TCN forecaster."""
    def __init__(
        self,
        num_features: int,
        horizon: int,
        num_detectors: int,
        emb_dim: int = 32,
        num_channels: Tuple[int, ...] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.det_embedding = nn.Embedding(num_detectors, emb_dim)
        
        self.encoder = TCNEncoder(
            input_dim=num_features + emb_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.output_dim, horizon),
        )

    def forward(self, x: torch.Tensor, det_ids: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        det_emb = self.det_embedding(det_ids).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x, det_emb], dim=-1)
        x = x.permute(0, 2, 1)
        
        h = self.encoder(x)
        return self.head(h[:, :, -1])

