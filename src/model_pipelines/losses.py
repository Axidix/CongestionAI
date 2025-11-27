from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional


def _compute_deltas(values: torch.Tensor, 
                    prev_values: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Efficiently compute deltas (changes) for a batch of sequences.
    
    Args:
        values: (batch, horizon) tensor
        prev_values: (batch,) tensor of values at t=0
    
    Returns:
        deltas: (batch, horizon) tensor
    """
    batch, horizon = values.shape
    
    if horizon == 1:
        if prev_values is not None:
            return values - prev_values.unsqueeze(1)
        return torch.zeros_like(values)
    
    # Efficient: single diff operation for horizons 1+
    later_deltas = values[:, 1:] - values[:, :-1]  # (batch, horizon-1)
    
    if prev_values is not None:
        first_delta = (values[:, 0] - prev_values).unsqueeze(1)  # (batch, 1)
    else:
        first_delta = torch.zeros(batch, 1, device=values.device, dtype=values.dtype)
    
    return torch.cat([first_delta, later_deltas], dim=1)


class SpikeWeightedMSELoss(nn.Module):
    """
    MSE loss with higher weight on spike time steps.
    Optimized: no boolean indexing, fully vectorized.
    """
    
    def __init__(self, spike_weight: float = 3.0, 
                 threshold: float = 0.15,
                 enabled: bool = True):
        super().__init__()
        self.spike_weight = spike_weight
        self.threshold = threshold
        self.enabled = enabled
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor, 
                prev_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        sq_errors = (pred - target).square()  # .square() slightly faster than ** 2
        
        if not self.enabled:
            return sq_errors.mean()
        
        deltas = _compute_deltas(target, prev_values)
        
        # Soft spike mask (avoids discrete jumps, still differentiable)
        # weights = 1 + (spike_weight - 1) * sigmoid((|delta| - threshold) * scale)
        # Or hard mask (faster):
        is_spike = (deltas.abs() > self.threshold).float()
        weights = 1.0 + (self.spike_weight - 1.0) * is_spike
        
        return (weights * sq_errors).mean()


class TwoTermSpikeLoss(nn.Module):
    """
    Loss = MSE_all + lambda * MSE_spikes_only
    
    Optimized: uses masked mean instead of boolean indexing.
    """
    
    def __init__(self, spike_lambda: float = 2.0, 
                 threshold: float = 0.15,
                 enabled: bool = True):
        super().__init__()
        self.spike_lambda = spike_lambda
        self.threshold = threshold
        self.enabled = enabled
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor, 
                prev_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        sq_errors = (pred - target).square()
        base_loss = sq_errors.mean()
        
        if not self.enabled:
            return base_loss
        
        deltas = _compute_deltas(target, prev_values)
        spike_mask = (deltas.abs() > self.threshold).float()
        
        # Masked mean without boolean indexing
        spike_count = spike_mask.sum()
        if spike_count > 0:
            spike_loss = (spike_mask * sq_errors).sum() / spike_count
        else:
            spike_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        
        return base_loss + self.spike_lambda * spike_loss


class DeltaLoss(nn.Module):
    """
    Loss = alpha * MSE(pred, target) + beta * MSE(pred_delta, target_delta)
    Already efficient, minor cleanup.
    """
    
    def __init__(self, alpha: float = 1.0, 
                 beta: float = 1.0,
                 enabled: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.enabled = enabled
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor, 
                prev_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        level_loss = (pred - target).square().mean()
        
        if not self.enabled:
            return level_loss
        
        target_deltas = _compute_deltas(target, prev_values)
        pred_deltas = _compute_deltas(pred, prev_values)
        
        delta_loss = (pred_deltas - target_deltas).square().mean()
        
        return self.alpha * level_loss + self.beta * delta_loss


# ============================================================
# LOSS FACTORY
# ============================================================

@dataclass
class LossConfig:
    """Configuration for loss function."""
    loss_type: str = "mse"
    spike_weight: float = 3.0
    spike_lambda: float = 2.0
    spike_threshold: float = 0.15
    delta_alpha: float = 1.0
    delta_beta: float = 1.0


def create_loss(config: LossConfig) -> nn.Module:
    """Factory function to create loss based on configuration."""
    if config.loss_type == "mse":
        return nn.MSELoss()
    
    elif config.loss_type == "spike_weighted":
        return SpikeWeightedMSELoss(
            spike_weight=config.spike_weight,
            threshold=config.spike_threshold
        )
    
    elif config.loss_type == "two_term":
        return TwoTermSpikeLoss(
            spike_lambda=config.spike_lambda,
            threshold=config.spike_threshold
        )
    
    elif config.loss_type == "delta":
        return DeltaLoss(
            alpha=config.delta_alpha,
            beta=config.delta_beta
        )
    
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")