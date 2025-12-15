"""
Backend Service
===============

Main orchestration layer that coordinates data fetching, inference, and caching.

This is the primary interface for the GUI and scheduler.

Functions:
----------
- refresh_forecast() -> bool
    Main function called hourly:
    1. Fetch latest weather
    2. Fetch latest traffic
    3. Build features for all detectors
    4. Run model inference
    5. Save forecast.json for GUI
    Returns True if successful, False on error.

- get_current_forecast() -> dict
    Load and return the latest cached forecast.

- get_forecast_for_hour(hour_offset: int) -> dict
    Get forecast for specific hour ahead (0-23).

- get_forecast_age() -> timedelta
    How old is the current cached forecast?

- is_forecast_stale() -> bool
    Check if forecast needs refresh.

Workflow:
---------
    ┌─────────────────────────────────────────────────────────┐
    │  refresh_forecast()                                      │
    │                                                          │
    │  1. weather_df = fetch_weather_history(72)              │
    │  2. traffic_df = fetch_traffic_history(336)             │
    │  3. features = build_features_all_detectors(...)        │
    │  4. model = load_model() [cached singleton]             │
    │  5. predictions = predict_all_detectors(model, ...)     │
    │  6. output = postprocess_predictions(predictions)       │
    │  7. save_json(output, "gui/data/forecast.json")         │
    │  8. log success, return True                            │
    └─────────────────────────────────────────────────────────┘

Error Handling:
---------------
- If weather fetch fails: use cached weather, log warning
- If traffic fetch fails: use historical fallback, log warning  
- If model fails: keep old forecast, log error, return False
- Never crash the service

Logging:
--------
- Log each refresh: timestamp, duration, success/failure
- Log data quality: missing detectors, stale data warnings
- Save logs to backend/logs/

Usage:
------
    # Called by scheduler every hour
    from backend.service import refresh_forecast
    success = refresh_forecast()
    
    # Called by GUI
    from backend.service import get_current_forecast
    forecast = get_current_forecast()
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import config
from backend import config
from backend.data.traffic import fetch_traffic_history, get_traffic_at_lag, fetch_and_accumulate
from backend.data.weather import fetch_weather_history, format_weather_for_gui
from backend.data.features import prepare_inference_batch, EXPECTED_NUM_FEATURES
from backend.model.loader import load_model
from backend.model.predictor import predict_batch, postprocess_predictions, get_forecast_summary


# =============================================================================
# STARTUP VALIDATION
# =============================================================================

def validate_startup() -> None:
    """
    Validate that all required files exist before running.
    Raises FileNotFoundError if critical files are missing.
    """
    critical_files = [
        (config.MAPPING_PATH, "Detector mapping (run mapping script first)"),
        (config.MODEL_CHECKPOINT_PATH, "Model checkpoint (run train_final_model.py first)"),
    ]
    
    # Scalers are recommended but not strictly required (will use no scaling)
    recommended_files = [
        (config.STD_SCALER_PATH, "StandardScaler"),
        (config.MM_SCALER_PATH, "MinMaxScaler"),
        (config.DET2IDX_PATH, "Detector index mapping"),
    ]
    
    missing_critical = []
    missing_recommended = []
    
    for path, desc in critical_files:
        if not path.exists():
            missing_critical.append(f"  - {path} ({desc})")
    
    for path, desc in recommended_files:
        if not path.exists():
            missing_recommended.append(f"  - {path} ({desc})")
    
    if missing_critical:
        raise FileNotFoundError(
            f"Missing critical files:\n" + "\n".join(missing_critical) +
            "\n\nCannot start service without these files."
        )
    
    if missing_recommended:
        logger.warning(
            "Missing recommended files (service may work with reduced accuracy):\n" +
            "\n".join(missing_recommended)
        )
    
    logger.info("Startup validation passed")


# =============================================================================
# GLOBAL CACHE
# =============================================================================

_model = None
_startup_validated = False


def _get_model():
    """Get model singleton (lazy load)."""
    global _model
    if _model is None:
        _model = load_model(
            config.MODEL_CHECKPOINT_PATH,
            device=config.DEVICE,
        )
    return _model


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def refresh_forecast() -> bool:
    """
    Main refresh function. Called hourly by scheduler.
    
    Workflow:
    1. Accumulate latest traffic data to cache
    2. Fetch weather history
    3. Build features for all detectors
    4. Run model inference
    5. Save forecast.json for GUI
    
    Returns:
        True if successful, False on error
    """
    global _startup_validated
    
    # Validate once on first call
    if not _startup_validated:
        validate_startup()
        _startup_validated = True
    
    start_time = time.time()
    logger.info("="*50)
    logger.info(f"Starting forecast refresh at {datetime.utcnow()}")
    
    try:
        # Step 1: Accumulate traffic (fetch current + merge with cache)
        logger.info("[1/5] Accumulating traffic data...")
        traffic_df = fetch_and_accumulate()
        if traffic_df.empty:
            raise ValueError("No traffic data available")
        logger.info(f"  Traffic: {len(traffic_df)} rows, {traffic_df['detector_id'].nunique()} detectors")
        
        # Step 2: Fetch weather
        logger.info("[2/5] Fetching weather history...")
        weather_df = fetch_weather_history(hours=100)
        if weather_df.empty:
            raise ValueError("No weather data available")
        logger.info(f"  Weather: {len(weather_df)} rows")
        
        # Step 3: Get 168h lag for weekly feature
        logger.info("[3/5] Getting 168h lag snapshot...")
        lag_168h_df = get_traffic_at_lag(lag_hours=168)
        if lag_168h_df.empty:
            logger.warning("  No 168h lag data available, using fallback")
            lag_168h_df = None
        else:
            logger.info(f"  168h lag: {len(lag_168h_df)} detectors")
        
        # Step 4: Build features
        logger.info("[4/5] Building features...")
        X, det_indices, detector_ids = prepare_inference_batch(
            traffic_history=traffic_df,
            weather_history=weather_df,
            traffic_lag_168h=lag_168h_df,
        )
        
        if len(X) == 0:
            raise ValueError("No valid detector features built")
        logger.info(f"  Features: {X.shape} for {len(detector_ids)} detectors")
        

        # --- FIX 3: Log input X stats before inference ---
        logger.warning(
            "INPUT X stats: min=%.3f max=%.3f mean=%.3f",
            float(X.min()), float(X.max()), float(X.mean())
        )

        # Step 5: Run inference
        logger.info("[5/5] Running model inference...")
        model = _get_model()
        predictions = predict_batch(model, X, det_indices, device=config.DEVICE)
        logger.info(f"  Predictions: {predictions.shape}")

        # Save the FULL inference batch and predictions for deep debugging
        np.save("debug_X_full.npy", X)
        pd.Series(detector_ids).to_csv("debug_detector_ids_full.csv", index=False)
        np.save("debug_det_indices_full.npy", det_indices)

        
        # Clip predictions to valid range [0, 1] BEFORE summary and postprocessing
        predictions = np.clip(predictions, 0.0, 1.0)

        # Diagnostics: log min, max, mean, and check for all-zero or near-zero predictions
        min_pred = float(np.min(predictions))
        max_pred = float(np.max(predictions))
        mean_pred = float(np.mean(predictions))
        zero_count = int(np.sum(predictions == 0.0))
        total_count = int(np.prod(predictions.shape))
        logger.info(f"Prediction stats: min={min_pred:.4f}, max={max_pred:.4f}, mean={mean_pred:.4f}, zero_count={zero_count}/{total_count}")
        if zero_count == total_count:
            logger.warning("All predictions are zero! Check model, data, or feature pipeline.")
        elif zero_count > 0.9 * total_count:
            logger.warning(f"More than 90% of predictions are zero. Possible data/model issue.")

        # Extract current congestion from input features (assumes 'congestion_index' is in FEATURE_COLS)
        try:
            from backend.data.features import FEATURE_COLS
            congestion_idx = FEATURE_COLS.index('congestion_index')
            current_congestion = X[:, -1, congestion_idx]
        except Exception as e:
            logger.warning(f"Could not extract current congestion from features: {e}")
            current_congestion = None

        # Postprocess and save
        from datetime import timezone
        output = postprocess_predictions(
            predictions=predictions,
            detector_ids=detector_ids,
            current_congestion=current_congestion,
            timestamp=datetime.now(tz=timezone.utc),
        )

        # --- Zero origin diagnostics ---
        # If road-level expansion was used, compare zeros in detector_data vs road_data
        if 'data' in output and output.get('num_roads', 0) > 0:
            # Rebuild detector_data (should match what was passed to expand_detector_forecast_to_roads)
            detector_data = {}
            for i, det_id in enumerate(detector_ids):
                detector_data[str(det_id)] = [round(float(v), 4) for v in predictions[i]]
            # Count zeros in detector_data
            det_zeros = sum(v == 0.0 for values in detector_data.values() for v in values)
            det_total = sum(len(values) for values in detector_data.values())
            # Count zeros in road_data
            road_zeros = sum(v == 0.0 for values in output['data'].values() for v in values)
            road_total = sum(len(values) for values in output['data'].values())
            logger.info(f"Zero diagnostics: detectors {det_zeros}/{det_total} zeros, roads {road_zeros}/{road_total} zeros")
            if road_zeros > det_zeros:
                logger.warning("More zeros in road-level data than detector-level: likely from interpolation/mapping.")
            elif road_zeros == det_zeros:
                logger.info("Zero count matches: zeros originate from model inference.")
            else:
                logger.info("Fewer zeros in road-level data than detector-level (unexpected).")
        
        # Add summary stats (now on clipped predictions)
        output["summary"] = get_forecast_summary(predictions, detector_ids)
        
        # Add weather forecast for GUI
        output["weather"] = format_weather_for_gui(weather_df)
        
        # Save to JSON
        save_forecast(output)
        
        duration = time.time() - start_time
        logger.info(f"Forecast refresh completed in {duration:.1f}s")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Forecast refresh FAILED after {duration:.1f}s: {e}", exc_info=True)
        return False


def save_forecast(data: Dict, path: Optional[Path] = None) -> None:
    """Save forecast to JSON file."""
    path = path or config.FORECAST_OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved forecast to {path}")


def get_current_forecast() -> Dict:
    """
    Load and return the latest cached forecast.
    
    Returns empty dict if no forecast available.
    """
    path = config.FORECAST_OUTPUT_PATH
    
    if not path.exists():
        logger.warning(f"No forecast file at {path}")
        return {}
    
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load forecast: {e}")
        return {}


def get_forecast_age() -> Optional[timedelta]:
    """
    Get age of current forecast.
    
    Returns None if no forecast available.
    """
    forecast = get_current_forecast()
    if not forecast:
        return None
    
    try:
        ts = datetime.strptime(forecast["generated_at"], "%Y-%m-%dT%H:%M:%SZ")
        return datetime.utcnow() - ts
    except (KeyError, ValueError):
        return None


def is_forecast_stale(max_age_minutes: int = 90) -> bool:
    """
    Check if forecast needs refresh.
    
    Args:
        max_age_minutes: Maximum acceptable age (default 90 = 1.5 hours)
    
    Returns:
        True if forecast is stale or missing
    """
    age = get_forecast_age()
    
    if age is None:
        return True
    
    return age > timedelta(minutes=max_age_minutes)


def get_forecast_for_detector(detector_id: str) -> Optional[Dict]:
    """
    Get forecast for a specific detector.
    
    Returns:
        dict with 24-hour forecast, or None if not found
    """
    forecast = get_current_forecast()
    
    if not forecast or "data" not in forecast:
        return None
    
    if detector_id not in forecast["data"]:
        return None
    
    return {
        "detector_id": detector_id,
        "timestamp": forecast.get("timestamp"),
        "predictions": forecast["data"][detector_id],
    }


def get_forecast_for_hour(hour_offset: int = 0) -> Dict:
    """
    Get all detector predictions for a specific hour.
    
    Args:
        hour_offset: Hours ahead (0 = now, 1 = +1h, ..., 23 = +23h)
    
    Returns:
        dict mapping detector_id -> congestion_index for that hour
    """
    forecast = get_current_forecast()
    
    if not forecast or "data" not in forecast:
        return {}
    
    hour_offset = max(0, min(23, hour_offset))
    
    return {
        det_id: preds[hour_offset]
        for det_id, preds in forecast["data"].items()
        if len(preds) > hour_offset
    }
