"""
Model Loader
============

Loads the trained TCN model checkpoint and prepares it for inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Singleton cache
_model_cache: Dict[str, Any] = {}


def load_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
    force_reload: bool = False,
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Uses singleton caching - model is loaded once and reused.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: "cuda" or "cpu"
        force_reload: If True, reload even if cached
    
    Returns:
        Model ready for inference (eval mode, on device)
    """
    global _model_cache
    
    checkpoint_path = Path(checkpoint_path)
    cache_key = str(checkpoint_path)
    
    # Return cached model if available
    if not force_reload and cache_key in _model_cache:
        logger.debug("Using cached model")
        return _model_cache[cache_key]
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config and reconstruct model
    model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    num_features = checkpoint.get("num_features", 47)
    num_detectors = checkpoint.get("num_detectors", 200)
    forecast_horizon = checkpoint.get("forecast_horizon", 24)
    
    # Model config (may be dict or dataclass)
    model_cfg = checkpoint.get("model_config", {})
    if hasattr(model_cfg, "__dict__"):
        model_cfg = model_cfg.__dict__
    
    # Import model class
    try:
        from src.models.tcn_forecaster import MultiHeadTCNForecaster
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.models.tcn_forecaster import MultiHeadTCNForecaster
    
    # Create model with architecture from checkpoint
    model = MultiHeadTCNForecaster(
        num_features=num_features,
        num_detectors=num_detectors,
        horizon=forecast_horizon,  # Model uses 'horizon' not 'forecast_horizon'
        emb_dim=model_cfg.get("emb_dim", 256),
        num_channels=model_cfg.get("num_channels", (128, 256, 256, 256, 256)),
        kernel_size=model_cfg.get("kernel_size", 5),
        dropout_encoder=model_cfg.get("dropout_encoder", 0.20),
        dropout_heads=model_cfg.get("dropout_heads", 0.20),
        use_se=model_cfg.get("use_se", False),
        pooling=model_cfg.get("pooling", "last"),
    )
    
    # Handle torch.compile'd models (have _orig_mod prefix)
    if any(k.startswith("_orig_mod.") for k in model_state.keys()):
        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # Cache and return
    _model_cache[cache_key] = model
    logger.info(f"Model loaded: {num_features} features, {num_detectors} detectors, horizon={forecast_horizon}")
    
    return model


def get_model_info(checkpoint_path: str | Path) -> Dict[str, Any]:
    """Extract metadata from checkpoint without loading full model."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    return {
        "num_features": checkpoint.get("num_features"),
        "num_detectors": checkpoint.get("num_detectors"),
        "forecast_horizon": checkpoint.get("forecast_horizon", 24),
        "model_config": checkpoint.get("model_config"),
        "data_config": checkpoint.get("data_config"),
        "checkpoint_keys": list(checkpoint.keys()),
    }


def clear_model_cache():
    """Clear the model cache (useful for testing)."""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")
