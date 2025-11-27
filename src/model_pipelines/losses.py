from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


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

# ----------------------------------------------------------------------
# Multi-Head Loss Wrapper
# ----------------------------------------------------------------------

class MultiHeadLoss(nn.Module):
    """
    Wraps any existing loss function to work with multi-head models.
    
    Applies the base loss to each head separately, with optional per-head weighting.
    """
    
    def __init__(
        self,
        head_config: Dict[str, dict],
        base_loss: nn.Module,
        head_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            head_config: From model.get_head_config(), contains start_idx/end_idx
            base_loss: Any loss module (MSELoss, SpikeWeightedMSELoss, etc.)
            head_weights: Optional weight per head, e.g., {"immediate": 1.5, "short": 1.0}
        """
        super().__init__()
        
        self.head_config = head_config
        self.head_names = list(head_config.keys())
        self.base_loss = base_loss
        self.head_weights = head_weights or {n: 1.0 for n in self.head_names}
        
        # Check if base_loss accepts prev_values
        import inspect
        sig = inspect.signature(base_loss.forward)
        self.loss_accepts_prev = "prev_values" in sig.parameters
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred: (B, horizon) predictions
            target: (B, horizon) targets
            prev_values: (B,) values at t=0 (passed to base_loss if it accepts it)
        
        Returns:
            total_loss: Weighted sum of head losses
            head_losses: Dict of per-head losses for logging
        """
        total_loss = 0.0
        head_losses = {}
        
        for name in self.head_names:
            cfg = self.head_config[name]
            start, end = cfg["start_idx"], cfg["end_idx"]
            
            pred_h = pred[:, start:end]
            target_h = target[:, start:end]
            
            # Call base loss with or without prev_values
            if self.loss_accepts_prev and prev_values is not None:
                head_loss = self.base_loss(pred_h, target_h, prev_values)
            else:
                head_loss = self.base_loss(pred_h, target_h)
            
            head_losses[name] = head_loss
            total_loss = total_loss + self.head_weights.get(name, 1.0) * head_loss
        
        return total_loss, head_losses


def create_multihead_loss(
    head_config: Dict[str, dict],
    loss_config: LossConfig,
    head_weights: Optional[Dict[str, float]] = None,
) -> MultiHeadLoss:
    """
    Factory to create MultiHeadLoss from existing loss config.
    
    Args:
        head_config: From model.get_head_config()
        loss_config: LossConfig for the base loss
        head_weights: Optional per-head weights
    
    Returns:
        MultiHeadLoss wrapping the configured base loss
    """
    base_loss = create_loss(loss_config)
    return MultiHeadLoss(
        head_config=head_config,
        base_loss=base_loss,
        head_weights=head_weights,
    )