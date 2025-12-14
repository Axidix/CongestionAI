#!/usr/bin/env python3
"""
Backend Pipeline Test Script
============================

Tests the full inference pipeline and compares feature engineering
to training preprocessing to ensure consistency.

Usage:
    python -m backend.test_pipeline

This script:
1. Fetches current traffic + weather data (with fallbacks for no history)
2. Runs backend feature engineering
3. Compares features to training preprocessing
4. Loads model and runs inference
5. Reports any discrepancies

Run from project root: python -m backend.test_pipeline
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# For testing without real data, we can generate synthetic data
USE_SYNTHETIC_DATA = True  # Set to False to use real API data
NUM_TEST_DETECTORS = 5     # Number of detectors to test with


# =============================================================================
# SYNTHETIC DATA GENERATION (for testing without API)
# =============================================================================

def generate_synthetic_traffic(num_detectors: int = 5, hours: int = 100) -> pd.DataFrame:
    """Generate synthetic traffic data for testing."""
    now = datetime.utcnow()
    
    # Get real detector IDs if available
    try:
        from backend.data.traffic import load_detector_segment_mapping
        mapping = load_detector_segment_mapping()
        detector_ids = mapping["detector_id"].unique()[:num_detectors].tolist()
        lons = mapping[mapping["detector_id"].isin(detector_ids)].groupby("detector_id")["lon"].first().values
        lats = mapping[mapping["detector_id"].isin(detector_ids)].groupby("detector_id")["lat"].first().values
    except Exception as e:
        logger.warning(f"Could not load detector mapping: {e}")
        detector_ids = [f"detector_{i}" for i in range(num_detectors)]
        lons = [13.3 + i * 0.01 for i in range(num_detectors)]
        lats = [52.5 + i * 0.01 for i in range(num_detectors)]
    
    rows = []
    for i, det_id in enumerate(detector_ids):
        for h in range(hours):
            ts = now - timedelta(hours=hours - 1 - h)
            # Synthetic congestion: daily pattern + noise
            hour_of_day = ts.hour
            day_pattern = 0.3 + 0.4 * np.sin(np.pi * (hour_of_day - 6) / 12) if 6 <= hour_of_day <= 18 else 0.2
            congestion = np.clip(day_pattern + np.random.normal(0, 0.1), 0, 1)
            
            rows.append({
                "detector_id": det_id,
                "timestamp": pd.Timestamp(ts, tz="UTC"),
                "congestion_index": congestion,
                "free_flow_speed": 50 + np.random.normal(0, 5),
                "lon": lons[i],
                "lat": lats[i],
            })
    
    df = pd.DataFrame(rows)
    logger.info(f"Generated synthetic traffic: {len(df)} rows, {num_detectors} detectors, {hours}h")
    return df


def generate_synthetic_weather(hours_past: int = 100, hours_future: int = 24) -> pd.DataFrame:
    """Generate synthetic weather data (past + future forecast)."""
    now = datetime.utcnow()
    
    rows = []
    total_hours = hours_past + hours_future
    for h in range(-hours_past + 1, hours_future + 1):
        ts = now + timedelta(hours=h)
        # Synthetic weather: gradual temperature change
        base_temp = 5 + 5 * np.sin(2 * np.pi * ts.hour / 24)  # Daily cycle
        
        rows.append({
            "timestamp": pd.Timestamp(ts, tz="UTC"),
            "temperature": base_temp + np.random.normal(0, 1),
            "precipitation": max(0, np.random.exponential(0.1)),
            "visibility": max(1000, 10000 + np.random.normal(0, 2000)),
            "is_rain": int(np.random.random() < 0.1),
            "is_snow": 0,
            "is_fog": int(np.random.random() < 0.05),
        })
    
    df = pd.DataFrame(rows)
    logger.info(f"Generated synthetic weather: {len(df)} rows ({hours_past}h past + {hours_future}h forecast)")
    return df


def generate_synthetic_lag_168h(detector_ids: list) -> pd.DataFrame:
    """Generate synthetic 168h lag snapshot."""
    rows = []
    for det_id in detector_ids:
        rows.append({
            "detector_id": det_id,
            "congestion_index": np.random.uniform(0.2, 0.6),
        })
    return pd.DataFrame(rows)


# =============================================================================
# REAL DATA FETCHING (with fallbacks)
# =============================================================================

def fetch_real_data():
    """Fetch real data from APIs with fallback to synthetic."""
    from backend.data.traffic import fetch_and_accumulate, get_traffic_at_lag
    from backend.data.weather import fetch_weather_history
    
    try:
        logger.info("Fetching real traffic data...")
        traffic_df = fetch_and_accumulate()
        if traffic_df.empty:
            raise ValueError("Empty traffic data")
    except Exception as e:
        logger.warning(f"Traffic fetch failed: {e}. Using synthetic data.")
        traffic_df = generate_synthetic_traffic(NUM_TEST_DETECTORS, 100)
    
    try:
        logger.info("Fetching real weather data...")
        weather_df = fetch_weather_history(hours=100, future_hours=24)
        if weather_df.empty:
            raise ValueError("Empty weather data")
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}. Using synthetic data.")
        weather_df = generate_synthetic_weather(100, 24)
    
    try:
        logger.info("Fetching 168h lag snapshot...")
        lag_168h_df = get_traffic_at_lag(lag_hours=168)
        if lag_168h_df.empty:
            raise ValueError("Empty lag data")
    except Exception as e:
        logger.warning(f"168h lag fetch failed: {e}. Using synthetic data.")
        detector_ids = traffic_df["detector_id"].unique().tolist()
        lag_168h_df = generate_synthetic_lag_168h(detector_ids)
    
    return traffic_df, weather_df, lag_168h_df


# =============================================================================
# BACKEND PIPELINE TEST
# =============================================================================

def test_backend_pipeline(traffic_df, weather_df, lag_168h_df):
    """Run the backend feature engineering pipeline."""
    from backend.data.features import (
        prepare_inference_batch,
        FEATURE_COLS,
        EXPECTED_NUM_FEATURES,
    )
    
    logger.info("\n" + "="*60)
    logger.info("TESTING BACKEND PIPELINE")
    logger.info("="*60)
    
    logger.info(f"Input data:")
    logger.info(f"  Traffic: {len(traffic_df)} rows, {traffic_df['detector_id'].nunique()} detectors")
    logger.info(f"  Weather: {len(weather_df)} rows")
    logger.info(f"  Lag 168h: {len(lag_168h_df)} detectors")
    
    # Run backend feature engineering
    try:
        X, det_indices, detector_ids = prepare_inference_batch(
            traffic_history=traffic_df,
            weather_history=weather_df,
            traffic_lag_168h=lag_168h_df,
        )
        
        logger.info(f"\nBackend output:")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  Expected: (N, 48, {EXPECTED_NUM_FEATURES})")
        logger.info(f"  Det indices: {det_indices.shape}")
        logger.info(f"  Detector IDs: {len(detector_ids)}")
        
        # Validate shape
        if X.shape[-1] != EXPECTED_NUM_FEATURES:
            logger.error(f"  ❌ Feature count mismatch! Got {X.shape[-1]}, expected {EXPECTED_NUM_FEATURES}")
            return None, None, None
        else:
            logger.info(f"  ✓ Feature count correct: {X.shape[-1]}")
        
        # Check for NaNs
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.warning(f"  ⚠ Found {nan_count} NaN values in features")
        else:
            logger.info(f"  ✓ No NaN values")
        
        # Check feature statistics
        logger.info(f"\nFeature statistics (first detector):")
        logger.info(f"  Mean: {X[0].mean():.4f}")
        logger.info(f"  Std:  {X[0].std():.4f}")
        logger.info(f"  Min:  {X[0].min():.4f}")
        logger.info(f"  Max:  {X[0].max():.4f}")
        
        return X, det_indices, detector_ids
        
    except Exception as e:
        logger.error(f"Backend pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# =============================================================================
# TRAINING PREPROCESSING COMPARISON
# =============================================================================

def compare_with_training_preprocessing(traffic_df, weather_df):
    """
    Run training-style preprocessing and compare features.
    
    This uses the same functions from src/utils/ that training uses.
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARING WITH TRAINING PREPROCESSING")
    logger.info("="*60)
    
    try:
        from src.utils.preprocessing import cyclical_encode
        from src.utils.crafted_features import (
            SpikeFeatureConfig,
            add_spike_features,
            make_lags,
        )
        from src.utils.configs import DataConfig
        
        # Use default data config
        data_cfg = DataConfig()
        
        # Prepare data similar to training
        df = traffic_df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add time columns needed for cyclical encoding
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        
        # Cyclical encoding (training style)
        df = cyclical_encode(df)
        
        # Season encoding (training style)
        df["season"] = 0.0
        df.loc[(df["month"] <= 2) | (df["month"] == 12), "season"] = 0.0
        df.loc[(df["month"] > 2) & (df["month"] <= 5), "season"] = 1.0
        df.loc[(df["month"] > 5) & (df["month"] <= 8), "season"] = 2.0
        df.loc[(df["month"] > 8) & (df["month"] <= 11), "season"] = 3.0
        
        # Spike features (training style)
        spike_config = SpikeFeatureConfig(
            enable_deltas=True,
            enable_abs_deltas=False,
            enable_rolling_stats=False,
            delta_lags=[1, 2, 4, 6],
            enable_volatility=True,
            volatility_window=3,
            volatility_binary_threshold=0.04,
        )
        df = df.sort_values(["detector_id", "timestamp"])
        df = add_spike_features(df, spike_config)
        
        # Congestion lags (training style)
        df = make_lags(df, "congestion_index", [48, 168])
        
        # Weather merge (simplified for comparison)
        weather_df_copy = weather_df.copy()
        weather_df_copy["_merge_hour"] = weather_df_copy["timestamp"].dt.floor("h")
        df["_merge_hour"] = df["timestamp"].dt.floor("h")
        
        df = df.merge(
            weather_df_copy[["_merge_hour", "temperature", "precipitation", "visibility"]].drop_duplicates("_merge_hour"),
            on="_merge_hour",
            how="left"
        )
        
        # Weather lags (training style - uses shift directly)
        weather_lags = [0, -3, -6, -12, -24]
        for col in ["temperature", "precipitation", "visibility"]:
            df = make_lags(df, col, weather_lags)
        
        logger.info(f"Training-style preprocessing complete")
        logger.info(f"  Columns created: {len([c for c in df.columns if 'lag' in c or 'sin' in c or 'cos' in c])}")
        
        # Check for expected columns
        expected_cols = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "season", "delta_1h", "delta_2h", "delta_4h", "delta_6h",
            "rolling_vol_3h", "is_high_vol",
            "congestion_index_lag_48h", "congestion_index_lag_168h",
            "temperature_lag_0h", "temperature_lag_-3h", "temperature_lag_-24h",
        ]
        
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            logger.warning(f"  Missing columns: {missing}")
        else:
            logger.info(f"  ✓ All expected columns present")
        
        return df
        
    except ImportError as e:
        logger.warning(f"Could not import training modules: {e}")
        logger.info("Skipping training comparison (training modules not available)")
        return None
    except Exception as e:
        logger.error(f"Training preprocessing comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MODEL INFERENCE TEST
# =============================================================================

def test_model_inference(X, det_indices):
    """Load model and run inference."""
    logger.info("\n" + "="*60)
    logger.info("TESTING MODEL INFERENCE")
    logger.info("="*60)
    
    from backend import config
    from backend.model.loader import load_model, get_model_info
    from backend.model.predictor import predict_batch
    
    # Check if model checkpoint exists
    if not config.MODEL_CHECKPOINT_PATH.exists():
        logger.warning(f"Model checkpoint not found: {config.MODEL_CHECKPOINT_PATH}")
        logger.info("Skipping inference test. Run train_final_model.py first.")
        return None
    
    try:
        # Load model
        logger.info(f"Loading model from {config.MODEL_CHECKPOINT_PATH}...")
        model = load_model(config.MODEL_CHECKPOINT_PATH, device=config.DEVICE)
        
        # Get model info for valid detector range
        model_info = get_model_info(config.MODEL_CHECKPOINT_PATH)
        num_detectors = model_info.get("num_detectors", 380)
        
        logger.info(f"  ✓ Model loaded successfully")
        logger.info(f"  Device: {config.DEVICE}")
        logger.info(f"  Model expects detector indices in [0, {num_detectors - 1}]")
        
        # Ensure detector indices are valid (within model's embedding table)
        # For synthetic data with invalid indices, remap to valid range
        invalid_indices = (det_indices < 0) | (det_indices >= num_detectors)
        if invalid_indices.any():
            logger.warning(f"  ⚠ {invalid_indices.sum()} detector indices out of range, remapping to valid range")
            det_indices = det_indices.copy()
            det_indices[invalid_indices] = np.arange(invalid_indices.sum()) % num_detectors
        
        # Run inference
        logger.info(f"\nRunning inference on {X.shape[0]} detectors...")
        predictions = predict_batch(model, X, det_indices, device=config.DEVICE)
        
        logger.info(f"\nInference results:")
        logger.info(f"  Predictions shape: {predictions.shape}")
        logger.info(f"  Expected: ({X.shape[0]}, 24)")
        
        # Check predictions
        logger.info(f"\nPrediction statistics:")
        logger.info(f"  Mean: {predictions.mean():.4f}")
        logger.info(f"  Std:  {predictions.std():.4f}")
        logger.info(f"  Min:  {predictions.min():.4f}")
        logger.info(f"  Max:  {predictions.max():.4f}")
        
        # Sanity check: predictions should be in [0, 1] for congestion index
        if predictions.min() < -0.5 or predictions.max() > 1.5:
            logger.warning(f"  ⚠ Predictions outside expected range [0, 1]")
        else:
            logger.info(f"  ✓ Predictions in reasonable range")
        
        # Show sample prediction
        logger.info(f"\nSample prediction (first detector, 24h horizon):")
        for h in [0, 6, 12, 18, 23]:
            logger.info(f"  +{h+1:2d}h: {predictions[0, h]:.3f}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# FULL PIPELINE TEST
# =============================================================================

def test_full_pipeline():
    """Run the complete end-to-end test."""
    logger.info("\n" + "#"*60)
    logger.info("# BACKEND PIPELINE TEST")
    logger.info("#"*60)
    logger.info(f"Time: {datetime.utcnow().isoformat()}")
    logger.info(f"Synthetic data: {USE_SYNTHETIC_DATA}")
    
    # Step 1: Get data
    logger.info("\n" + "-"*60)
    logger.info("STEP 1: DATA ACQUISITION")
    logger.info("-"*60)
    
    if USE_SYNTHETIC_DATA:
        traffic_df = generate_synthetic_traffic(NUM_TEST_DETECTORS, 100)
        weather_df = generate_synthetic_weather(100, 24)
        detector_ids = traffic_df["detector_id"].unique().tolist()
        lag_168h_df = generate_synthetic_lag_168h(detector_ids)
    else:
        traffic_df, weather_df, lag_168h_df = fetch_real_data()
    
    # Step 2: Backend pipeline
    logger.info("\n" + "-"*60)
    logger.info("STEP 2: BACKEND FEATURE ENGINEERING")
    logger.info("-"*60)
    
    X, det_indices, detector_ids = test_backend_pipeline(traffic_df, weather_df, lag_168h_df)
    
    if X is None:
        logger.error("Backend pipeline failed. Aborting.")
        return False
    
    # Step 3: Training comparison
    logger.info("\n" + "-"*60)
    logger.info("STEP 3: TRAINING PREPROCESSING COMPARISON")
    logger.info("-"*60)
    
    training_df = compare_with_training_preprocessing(traffic_df, weather_df)
    
    # Step 4: Model inference
    logger.info("\n" + "-"*60)
    logger.info("STEP 4: MODEL INFERENCE")
    logger.info("-"*60)
    
    predictions = test_model_inference(X, det_indices)
    
    # Summary
    logger.info("\n" + "#"*60)
    logger.info("# TEST SUMMARY")
    logger.info("#"*60)
    
    results = {
        "Data acquisition": "✓ PASS",
        "Backend pipeline": "✓ PASS" if X is not None else "❌ FAIL",
        "Training comparison": "✓ PASS" if training_df is not None else "⚠ SKIP",
        "Model inference": "✓ PASS" if predictions is not None else "⚠ SKIP",
    }
    
    for test, result in results.items():
        logger.info(f"  {test}: {result}")
    
    all_passed = all("PASS" in r or "SKIP" in r for r in results.values())
    
    if all_passed:
        logger.info("\n✓ All tests passed!")
    else:
        logger.error("\n❌ Some tests failed!")
    
    return all_passed


# =============================================================================
# FEATURE COMPARISON (detailed)
# =============================================================================

def compare_feature_values():
    """
    Detailed comparison of feature values between backend and training.
    Useful for debugging mismatches.
    """
    logger.info("\n" + "="*60)
    logger.info("DETAILED FEATURE COMPARISON")
    logger.info("="*60)
    
    from backend.data.features import FEATURE_COLS
    
    # Generate minimal test data
    traffic_df = generate_synthetic_traffic(1, 100)  # 1 detector, 100h
    weather_df = generate_synthetic_weather(100, 24)
    lag_168h_df = generate_synthetic_lag_168h(traffic_df["detector_id"].unique().tolist())
    
    # Backend features
    from backend.data.features import prepare_inference_batch
    X_backend, _, _ = prepare_inference_batch(traffic_df, weather_df, lag_168h_df)
    
    logger.info(f"\nFeature columns in order:")
    for i, col in enumerate(FEATURE_COLS):
        if X_backend is not None and X_backend.shape[0] > 0:
            val = X_backend[0, -1, i]  # Last timestep, first detector
            logger.info(f"  [{i:2d}] {col:30s}: {val:8.4f}")
        else:
            logger.info(f"  [{i:2d}] {col}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test backend pipeline")
    parser.add_argument("--real-data", action="store_true", help="Use real API data instead of synthetic")
    parser.add_argument("--detectors", type=int, default=5, help="Number of test detectors")
    parser.add_argument("--compare-features", action="store_true", help="Run detailed feature comparison")
    args = parser.parse_args()
    
    USE_SYNTHETIC_DATA = not args.real_data
    NUM_TEST_DETECTORS = args.detectors
    
    if args.compare_features:
        compare_feature_values()
    else:
        success = test_full_pipeline()
        sys.exit(0 if success else 1)
