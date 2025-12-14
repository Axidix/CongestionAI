"""
Model Predictor
===============

Runs inference with the loaded model and postprocesses results.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def predict(
    model: torch.nn.Module,
    X: np.ndarray,
    detector_indices: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run model inference.
    
    Args:
        model: Loaded model (from loader.py)
        X: Features array of shape (N, history_length, num_features)
        detector_indices: Detector indices array of shape (N,)
        device: Device for inference
    
    Returns:
        Predictions array of shape (N, forecast_horizon)
    """
    model.eval()
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X).float().to(device)
    det_tensor = torch.from_numpy(detector_indices).long().to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(X_tensor, det_tensor)
    
    # Convert back to numpy
    return predictions.cpu().numpy()


def predict_batch(
    model: torch.nn.Module,
    X: np.ndarray,
    detector_indices: np.ndarray,
    device: str = "cuda",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run batched inference for large datasets.
    
    Args:
        model: Loaded model
        X: Features (N, history, features)
        detector_indices: Detector indices (N,)
        device: Device
        batch_size: Batch size for inference
    
    Returns:
        Predictions (N, horizon)
    """
    model.eval()
    n_samples = len(X)
    
    if n_samples <= batch_size:
        return predict(model, X, detector_indices, device)
    
    all_preds = []
    
    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        X_batch = X[i:end_i]
        det_batch = detector_indices[i:end_i]
        
        preds = predict(model, X_batch, det_batch, device)
        all_preds.append(preds)
    
    return np.concatenate(all_preds, axis=0)


def postprocess_predictions(
    predictions: np.ndarray,
    detector_ids: List[str],
    timestamp: Optional[datetime] = None,
    forecast_horizon: int = 24,
) -> Dict:
    """
    Format predictions into GUI-expected structure.
    
    Args:
        predictions: (N, horizon) array of congestion index predictions
        detector_ids: List of detector IDs matching predictions order
        timestamp: Timestamp of prediction (default: now UTC)
        forecast_horizon: Number of hours predicted
    
    Returns:
        dict suitable for JSON serialization:
        {
            "timestamp": "2024-12-14T15:00:00Z",
            "forecast_horizon": 24,
            "generated_at": "2024-12-14T15:01:23Z",
            "num_detectors": 150,
            "data": {
                "detector_abc": [0.12, 0.15, ..., 0.20],  # 24 values
                ...
            }
        }
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0.0, 1.0)
    
    # Build data dict
    data = {}
    for i, det_id in enumerate(detector_ids):
        # Round to 4 decimal places for JSON size
        data[det_id] = [round(float(v), 4) for v in predictions[i]]
    
    output = {
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_horizon": forecast_horizon,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_detectors": len(detector_ids),
        "data": data,
    }
    
    return output


def get_forecast_summary(predictions: np.ndarray, detector_ids: List[str]) -> Dict:
    """
    Generate summary statistics for the forecast.
    
    Returns:
        dict with: mean_congestion_by_hour, most_congested_detectors, etc.
    """
    # Mean congestion per hour across all detectors
    mean_by_hour = predictions.mean(axis=0).tolist()
    
    # Mean congestion per detector (across all hours)
    mean_by_detector = predictions.mean(axis=1)
    
    # Top 10 most congested detectors
    top_indices = np.argsort(mean_by_detector)[-10:][::-1]
    most_congested = [
        {"detector_id": detector_ids[i], "mean_congestion": round(float(mean_by_detector[i]), 3)}
        for i in top_indices
    ]
    
    # Peak congestion hour (0-23)
    peak_hour = int(np.argmax(mean_by_hour))
    
    return {
        "mean_congestion_by_hour": [round(v, 3) for v in mean_by_hour],
        "overall_mean": round(float(predictions.mean()), 3),
        "peak_hour": peak_hour,
        "peak_congestion": round(float(mean_by_hour[peak_hour]), 3),
        "most_congested_detectors": most_congested,
    }
