from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Import paths from centralized config
from backend.config import (
    SCALER_DIR,
    STD_SCALER_PATH,
    MM_SCALER_PATH,
    DET2IDX_PATH,
)

logger = logging.getLogger(__name__)

# Feature configuration (must match train_final_model.py)
HISTORY_HOURS = 48
FORECAST_HORIZON = 24
CONGESTION_LAGS = (48, 168)
WEATHER_LAGS = (0, -3, -6, -12, -24)
DELTA_LAGS = (1, 2, 4, 6)
VOLATILITY_WINDOW = 3
VOLATILITY_THRESHOLD = 0.04

# Columns to normalize (must match training)
NORM_COLS = [
    "temperature", "precipitation", "visibility", 
    "congestion_index", "free_flow_speed",
    "delta_1h", "delta_2h", "delta_4h", "delta_6h",
    f"rolling_vol_{VOLATILITY_WINDOW}h",
] + [f"congestion_index_lag_{lag}h" for lag in CONGESTION_LAGS]

MINMAX_COLS = ["lon", "lat", "year", "season"]

# Final feature order (must match training exactly)
FEATURE_COLS_BASE = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lon", "lat", "year", "season",
    "temperature", "precipitation", "visibility",
    "congestion_index", "free_flow_speed",
    "is_weekend", "is_holiday", "is_school_holiday", "is_rush_hour", "is_snow", "is_fog",
]

SPIKE_COLS = [f"delta_{lag}h" for lag in DELTA_LAGS] + [
    f"rolling_vol_{VOLATILITY_WINDOW}h", "is_high_vol"
]

CONGESTION_LAG_COLS = [f"congestion_index_lag_{lag}h" for lag in CONGESTION_LAGS]

WEATHER_LAG_COLS = (
    [f"temperature_lag_{lag}h" for lag in WEATHER_LAGS] +
    [f"precipitation_lag_{lag}h" for lag in WEATHER_LAGS] +
    [f"visibility_lag_{lag}h" for lag in WEATHER_LAGS]
)

# Full feature list in order
FEATURE_COLS = FEATURE_COLS_BASE + SPIKE_COLS + CONGESTION_LAG_COLS + WEATHER_LAG_COLS

# Expected feature count (must match training)
# This validates that FEATURE_COLS matches what the model was trained on
EXPECTED_NUM_FEATURES = 44


# =============================================================================
# GERMAN HOLIDAYS (Berlin)
# =============================================================================

def get_german_holidays(year: int) -> set:
    """Return set of holiday dates for Berlin."""
    from datetime import date
    
    holidays = set()
    
    # Fixed holidays
    holidays.add(date(year, 1, 1))   # New Year
    holidays.add(date(year, 5, 1))   # Labour Day
    holidays.add(date(year, 10, 3))  # German Unity
    holidays.add(date(year, 12, 25)) # Christmas
    holidays.add(date(year, 12, 26)) # Boxing Day
    
    # Easter-based (approximate - for production, use a proper library)
    # Simple Gaussian algorithm for Easter Sunday
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = date(year, month, day)
    
    holidays.add(easter - timedelta(days=2))  # Good Friday
    holidays.add(easter + timedelta(days=1))  # Easter Monday
    holidays.add(easter + timedelta(days=39)) # Ascension
    holidays.add(easter + timedelta(days=50)) # Whit Monday
    
    return holidays


def get_school_holidays_berlin(timestamp: pd.Timestamp) -> bool:
    """Approximate Berlin school holidays."""
    month, day = timestamp.month, timestamp.day
    
    # Summer: mid-July to end of August
    if (month == 7 and day >= 15) or month == 8:
        return True
    # Christmas: ~Dec 23 to Jan 3
    if (month == 12 and day >= 23) or (month == 1 and day <= 3):
        return True
    # Easter: ~2 weeks around Easter (simplified: April 1-15)
    if month == 4 and day <= 15:
        return True
    # Fall: first week of October
    if month == 10 and day <= 7:
        return True
    
    return False


# =============================================================================
# CYCLICAL ENCODING
# =============================================================================

def cyclical_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings."""
    df = df.copy()
    
    # Extract time components if not present
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


# =============================================================================
# CALENDAR FEATURES
# =============================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based binary features."""
    df = df.copy()
    
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "year" not in df.columns:
        df["year"] = df["timestamp"].dt.year
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month
    
    # Weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    
    # Rush hour (7-9 and 16-19)
    df["is_rush_hour"] = ((df["hour"] >= 7) & (df["hour"] <= 9) | 
                          (df["hour"] >= 16) & (df["hour"] <= 19)).astype(float)
    
    # Holidays
    years = df["timestamp"].dt.year.unique()
    all_holidays = set()
    for y in years:
        all_holidays.update(get_german_holidays(y))
    df["is_holiday"] = df["timestamp"].dt.date.isin(all_holidays).astype(float)
    
    # School holidays
    df["is_school_holiday"] = df["timestamp"].apply(get_school_holidays_berlin).astype(float)
    
    # Season (0=winter, 1=spring, 2=summer, 3=fall)
    # MUST match training: month <=2 OR ==12 is winter
    df["season"] = 0.0  # default winter
    df.loc[(df["month"] <= 2) | (df["month"] == 12), "season"] = 0.0  # winter
    df.loc[(df["month"] > 2) & (df["month"] <= 5), "season"] = 1.0   # spring
    df.loc[(df["month"] > 5) & (df["month"] <= 8), "season"] = 2.0   # summer
    df.loc[(df["month"] > 8) & (df["month"] <= 11), "season"] = 3.0  # fall
    
    return df


# =============================================================================
# SPIKE / DELTA FEATURES
# =============================================================================

def add_delta_features(df: pd.DataFrame, lags: Tuple[int, ...] = DELTA_LAGS) -> pd.DataFrame:
    """Add congestion change (delta) features per detector."""
    df = df.copy()
    
    for lag in lags:
        df[f"delta_{lag}h"] = df.groupby("detector_id")["congestion_index"].diff(lag)
    
    return df


def add_volatility_features(
    df: pd.DataFrame, 
    window: int = VOLATILITY_WINDOW,
    threshold: float = VOLATILITY_THRESHOLD
) -> pd.DataFrame:
    """Add rolling volatility and binary high-volatility flag."""
    df = df.copy()
    
    col_name = f"rolling_vol_{window}h"
    df[col_name] = (
        df.groupby("detector_id")["congestion_index"]
          .transform(lambda x: x.rolling(window, min_periods=1).std())
          .fillna(0)
    )
    
    df["is_high_vol"] = (df[col_name] > threshold).astype(float)
    
    return df


# =============================================================================
# LAG FEATURES
# =============================================================================

def add_congestion_lags(
    df: pd.DataFrame, 
    lag_168h_values: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add congestion lag features.
    
    IMPORTANT: Training uses shift(48) and shift(168) per-row.
    - 48h lag: MUST be computed via shift(48), not broadcast from a single value.
      Each row gets its own lag (value from 48 rows earlier).
    - 168h lag: Since we likely don't have 168h of history, we accept a snapshot
      DataFrame and broadcast it. This is an approximation but necessary.
    
    NOTE: shift(k) gives value from k rows earlier. If data has gaps (missing hours),
    this becomes "k rows ago" not "k hours ago". Ensure data is hourly-aligned.
    
    COLD START: When insufficient history exists, we use fallbacks:
      - 48h lag NaN → use current congestion_index (less accurate but works)
      - 168h lag NaN → use current congestion_index or provided snapshot
    """
    df = df.copy()
    
    # 48h lag - MUST use shift(48) to match training (per-row computation)
    # Need 48+ rows of history before current time for valid lags
    df["congestion_index_lag_48h"] = df.groupby("detector_id")["congestion_index"].shift(48)
    
    # Count how many rows have valid 48h lag
    valid_48h = df["congestion_index_lag_48h"].notna().sum()
    total = len(df)
    if valid_48h < total:
        missing_pct = 100 * (1 - valid_48h / total)
        if valid_48h == 0:
            logger.warning(
                f"48h lag: NO valid values (cold start). "
                f"Using current congestion as fallback for all rows."
            )
        else:
            logger.info(
                f"48h lag: {valid_48h}/{total} rows have valid values "
                f"({100*valid_48h/total:.1f}%). Missing rows will use fallback."
            )
        # COLD START FALLBACK: Fill missing 48h lags with current congestion
        df["congestion_index_lag_48h"] = df["congestion_index_lag_48h"].fillna(
            df["congestion_index"]
        )
    
    # 168h lag - use snapshot if provided, otherwise fallback
    if lag_168h_values is not None:
        # Broadcast single snapshot value to all rows per detector
        df = df.merge(
            lag_168h_values[["detector_id", "congestion_index"]].rename(
                columns={"congestion_index": "congestion_index_lag_168h"}
            ),
            on="detector_id",
            how="left"
        )
        # Fill any detectors not in snapshot with current congestion
        df["congestion_index_lag_168h"] = df["congestion_index_lag_168h"].fillna(
            df["congestion_index"]
        )
    else:
        # Fallback: use current congestion as proxy (not ideal but prevents NaN)
        # Training would have used shift(168), but we can't replicate without 168h history
        df["congestion_index_lag_168h"] = df["congestion_index"]
        logger.warning(
            "168h lag snapshot not provided. Using current congestion as fallback. "
            "This may degrade prediction quality."
        )
    
    return df


def add_weather_lags(df: pd.DataFrame, lags: Tuple[int, ...] = WEATHER_LAGS) -> pd.DataFrame:
    """
    Add weather lag features using FORECAST data.
    
    WEATHER_LAGS = (0, -3, -6, -12, -24) with shift(lag):
      - shift(0)   = current weather
      - shift(-3)  = weather 3 rows LATER = 3h forecast
      - shift(-24) = weather 24 rows LATER = 24h forecast
    
    This is NOT data leakage - the negative lags represent weather FORECASTS
    which are available from the BrightSky API. The model uses predicted
    future weather conditions to improve traffic forecasting.
    
    IMPORTANT: For per-detector data, shift happens within each detector group.
    For weather-only data (single station), shift happens across all rows.
    """
    df = df.copy()
    
    # Check if this is per-detector data or single-station weather
    has_detector = "detector_id" in df.columns
    
    for col in ["temperature", "precipitation", "visibility"]:
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag_{lag}h"
            if lag == 0:
                df[lag_col] = df[col]
            else:
                # shift(lag) where negative lag = future forecast rows
                # shift(-3) = value from 3 rows later = 3h forecast
                if has_detector:
                    df[lag_col] = df.groupby("detector_id")[col].shift(lag)
                else:
                    df[lag_col] = df[col].shift(lag)
    
    return df


def precompute_weather_lags(weather_df: pd.DataFrame, lags: Tuple[int, ...] = WEATHER_LAGS) -> pd.DataFrame:
    """
    Pre-compute weather lag columns on weather DataFrame BEFORE merging with traffic.
    
    This is needed because weather_df includes future forecast rows, but after
    merging with traffic (which only has past/current timestamps), we would lose
    the ability to look ahead via shift.
    
    By pre-computing lag columns on weather_df (which has future rows), we can
    then merge the expanded weather with traffic.
    
    CRITICAL: Weather columns are scaled BEFORE computing lags, matching training.
    Training does: scale(temperature) -> shift -> temperature_lag_Xh
    So lag columns inherit the scaled values.
    
    Example:
        weather_df has timestamps: T-100h to T+24h
        For row at T, shift(-24) gives weather at T+24h (forecast)
        After merge with traffic (T-100h to T), row at T has correct T+24h weather
    """
    # Handle empty or missing weather data
    if weather_df is None or weather_df.empty:
        logger.warning("Empty weather DataFrame - returning empty DataFrame with lag columns")
        # Create empty DataFrame with expected lag columns
        lag_cols = []
        for col in ["temperature", "precipitation", "visibility"]:
            for lag in lags:
                lag_cols.append(f"{col}_lag_{lag}h")
        empty_df = pd.DataFrame(columns=["timestamp"] + lag_cols)
        return empty_df
    
    df = weather_df.copy()
    
    # Ensure timestamp is datetime
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df = df.sort_values("timestamp")
    
    # Scale base weather columns FIRST (matching training order)
    # In training: scale_features() is called before add_lags_and_drop()
    std_scaler, _, _ = load_scalers()
    
    if std_scaler is not None and hasattr(std_scaler, 'feature_names_in_'):
        # Get the weather columns that were scaled in training
        weather_cols_to_scale = [c for c in ["temperature", "precipitation", "visibility"] 
                                  if c in std_scaler.feature_names_in_ and c in df.columns]
        
        if weather_cols_to_scale:
            # Get indices of these columns in the scaler's fitted order
            scaler_cols = list(std_scaler.feature_names_in_)
            
            # Apply scaling to each weather column independently
            # (We can't use transform directly since df doesn't have all scaler columns)
            for col in weather_cols_to_scale:
                col_idx = scaler_cols.index(col)
                mean = std_scaler.mean_[col_idx]
                scale = std_scaler.scale_[col_idx]
                df[col] = (df[col] - mean) / scale
            
            logger.debug(f"Scaled weather columns before lag computation: {weather_cols_to_scale}")
    else:
        logger.warning("StandardScaler not available - weather lags will be unscaled")
    
    # Now compute lags on the SCALED values
    for col in ["temperature", "precipitation", "visibility"]:
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag_{lag}h"
            if lag == 0:
                df[lag_col] = df[col]
            else:
                # shift(-24) on weather with future rows = look 24h ahead
                df[lag_col] = df[col].shift(lag)
    
    # Fill NaN at edges with current value
    for col in ["temperature", "precipitation", "visibility"]:
        for lag in lags:
            if lag < 0:
                lag_col = f"{col}_lag_{lag}h"
                if lag_col in df.columns:
                    df[lag_col] = df[lag_col].fillna(df[col])
    
    return df


# =============================================================================
# SCALERS (load from training)
# =============================================================================

_std_scaler = None
_mm_scaler = None
_det2idx = None


def load_scalers():
    """Load scalers saved from training."""
    global _std_scaler, _mm_scaler, _det2idx
    
    if _std_scaler is None:
        if STD_SCALER_PATH.exists():
            _std_scaler = joblib.load(STD_SCALER_PATH)
            logger.info(f"Loaded StandardScaler from {STD_SCALER_PATH}")
        else:
            logger.warning(f"StandardScaler not found at {STD_SCALER_PATH}")
            _std_scaler = None
    
    if _mm_scaler is None:
        if MM_SCALER_PATH.exists():
            _mm_scaler = joblib.load(MM_SCALER_PATH)
            logger.info(f"Loaded MinMaxScaler from {MM_SCALER_PATH}")
        else:
            logger.warning(f"MinMaxScaler not found at {MM_SCALER_PATH}")
            _mm_scaler = None
    
    if _det2idx is None:
        if DET2IDX_PATH.exists():
            _det2idx = joblib.load(DET2IDX_PATH)
            logger.info(f"Loaded det2idx mapping with {len(_det2idx)} detectors")
        else:
            logger.warning(f"det2idx not found at {DET2IDX_PATH}")
            _det2idx = None
    
    return _std_scaler, _mm_scaler, _det2idx


def scale_features(df: pd.DataFrame, skip_weather_cols: bool = True) -> pd.DataFrame:
    """
    Apply scalers to features (must match training).
    
    CRITICAL: Scaler expects columns in exact order they were fit on.
    Uses feature_names_in_ if available (sklearn 1.0+), otherwise NORM_COLS.
    
    Args:
        df: DataFrame with features
        skip_weather_cols: If True, skip temperature/precipitation/visibility
                          (they're already scaled in precompute_weather_lags)
    """
    df = df.copy()
    std_scaler, mm_scaler, _ = load_scalers()
    
    # Weather columns that are pre-scaled in precompute_weather_lags
    pre_scaled_cols = {"temperature", "precipitation", "visibility"}
    
    if std_scaler is not None:
        # Prefer feature_names_in_ (sklearn 1.0+) for exact column order
        if hasattr(std_scaler, 'feature_names_in_'):
            cols_to_scale = list(std_scaler.feature_names_in_)
        else:
            # Fallback to NORM_COLS - must match training order exactly
            expected_n = getattr(std_scaler, 'n_features_in_', len(NORM_COLS))
            cols_to_scale = NORM_COLS[:expected_n]
            logger.warning(
                f"StandardScaler lacks feature_names_in_, using NORM_COLS[:{expected_n}]. "
                f"Ensure this matches training column order exactly."
            )
        
        # Skip weather columns if they're already scaled
        if skip_weather_cols:
            cols_to_scale = [c for c in cols_to_scale if c not in pre_scaled_cols]
        
        missing = [c for c in cols_to_scale if c not in df.columns]
        if missing:
            raise ValueError(
                f"StandardScaler missing required columns: {missing}. "
                f"Cannot proceed with inference - feature engineering is incomplete."
            )
        
        if cols_to_scale:
            # Need to get subset of scaler parameters for these columns
            if skip_weather_cols and hasattr(std_scaler, 'feature_names_in_'):
                # Apply scaling column by column to handle subset
                scaler_cols = list(std_scaler.feature_names_in_)
                for col in cols_to_scale:
                    col_idx = scaler_cols.index(col)
                    mean = std_scaler.mean_[col_idx]
                    scale = std_scaler.scale_[col_idx]
                    df[col] = (df[col] - mean) / scale
            else:
                # Apply in exact order (full scaler)
                df[cols_to_scale] = std_scaler.transform(df[cols_to_scale])
    
    if mm_scaler is not None:
        if hasattr(mm_scaler, 'feature_names_in_'):
            cols_to_scale = list(mm_scaler.feature_names_in_)
        else:
            expected_n = getattr(mm_scaler, 'n_features_in_', len(MINMAX_COLS))
            cols_to_scale = MINMAX_COLS[:expected_n]
            logger.warning(
                f"MinMaxScaler lacks feature_names_in_, using MINMAX_COLS[:{expected_n}]. "
                f"Ensure this matches training column order exactly."
            )
        
        missing = [c for c in cols_to_scale if c not in df.columns]
        if missing:
            raise ValueError(
                f"MinMaxScaler missing required columns: {missing}. "
                f"Cannot proceed with inference - feature engineering is incomplete."
            )
        
        df[cols_to_scale] = mm_scaler.transform(df[cols_to_scale])
    
    return df


def get_detector_index(detector_id) -> int:
    """
    Get model's detector index for a detector_id.
    
    Handles both string and int detector IDs for robustness.
    """
    _, _, det2idx = load_scalers()
    
    if det2idx is None:
        logger.warning("det2idx not loaded, using hash fallback")
        return hash(str(detector_id)) % 1000
    
    # Try direct lookup
    if detector_id in det2idx:
        return det2idx[detector_id]
    
    # Try converting string to int (det2idx keys might be int)
    try:
        det_id_int = int(detector_id)
        if det_id_int in det2idx:
            return det2idx[det_id_int]
    except (ValueError, TypeError):
        pass
    
    # Try converting int to string
    try:
        det_id_str = str(detector_id)
        if det_id_str in det2idx:
            return det2idx[det_id_str]
    except (ValueError, TypeError):
        pass
    
    return -1


# =============================================================================
# MAIN FEATURE BUILDING FUNCTIONS
# =============================================================================

def merge_traffic_weather(
    traffic_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge traffic and weather data on timestamp (hourly).
    
    Traffic: detector_id, timestamp, congestion_index, free_flow_speed, lon, lat
    Weather: timestamp, temperature, precipitation, visibility, ..., is_rain, is_snow, is_fog,
             PLUS pre-computed weather lag columns (temperature_lag_-3h, etc.)
    
    IMPORTANT: weather_df should have weather lag columns pre-computed via
    precompute_weather_lags() BEFORE calling this function. This is necessary
    because weather_df includes future forecast rows that traffic_df doesn't have.
    """
    traffic_df = traffic_df.copy()
    
    # Handle empty weather - use defaults for all traffic rows
    if weather_df is None or weather_df.empty:
        logger.warning("Empty weather DataFrame - using default weather values for all rows")
        merged = traffic_df.copy()
        merged["temperature"] = 10.0
        merged["precipitation"] = 0.0
        merged["visibility"] = 10000.0
        merged["is_rain"] = 0
        merged["is_snow"] = 0
        merged["is_fog"] = 0
        # Add empty lag columns (will be filled below with defaults)
        for col in ["temperature", "precipitation", "visibility"]:
            for lag in WEATHER_LAGS:
                merged[f"{col}_lag_{lag}h"] = merged[col]  # Use base value
        return merged
    
    weather_df = weather_df.copy()
    
    # Round timestamps to hour for merge
    traffic_df["_merge_hour"] = traffic_df["timestamp"].dt.floor("h")
    weather_df["_merge_hour"] = weather_df["timestamp"].dt.floor("h")
    
    # Weather columns to merge (base + lag columns)
    base_weather_cols = ["temperature", "precipitation", "visibility", "is_rain", "is_snow", "is_fog"]
    weather_lag_cols = [c for c in weather_df.columns if "_lag_" in c]
    
    weather_cols = ["_merge_hour"] + [c for c in base_weather_cols if c in weather_df.columns] + weather_lag_cols
    
    merged = traffic_df.merge(
        weather_df[weather_cols].drop_duplicates("_merge_hour"),
        on="_merge_hour",
        how="left"
    )
    merged = merged.drop(columns=["_merge_hour"])
    
    # Track and log missing weather (can indicate API failures)
    weather_missing = merged["temperature"].isna().sum()
    if weather_missing > 0:
        total_rows = len(merged)
        pct_missing = 100 * weather_missing / total_rows
        logger.warning(
            f"Weather data missing for {weather_missing}/{total_rows} rows ({pct_missing:.1f}%). "
            f"Filling with defaults. Check weather API if this is unexpected."
        )
    
    # Fill missing weather with reasonable defaults
    merged["temperature"] = merged["temperature"].fillna(10.0)
    merged["precipitation"] = merged["precipitation"].fillna(0.0)
    merged["visibility"] = merged["visibility"].fillna(10000.0)
    
    # Ensure weather condition columns exist as Series before fillna
    if "is_rain" not in merged.columns:
        merged["is_rain"] = 0
    merged["is_rain"] = merged["is_rain"].fillna(0)
    
    if "is_snow" not in merged.columns:
        merged["is_snow"] = 0
    merged["is_snow"] = merged["is_snow"].fillna(0)
    
    if "is_fog" not in merged.columns:
        merged["is_fog"] = 0
    merged["is_fog"] = merged["is_fog"].fillna(0)
    
    return merged


def build_features_for_detector(
    detector_id: str,
    merged_df: pd.DataFrame,
    lag_48h_value: Optional[float] = None,
    lag_168h_value: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Build (history_length, num_features) array for one detector.
    
    Args:
        detector_id: Detector to build features for
        merged_df: Traffic+weather merged DataFrame with all features
        lag_48h_value: Pre-fetched congestion value from 48h ago
        lag_168h_value: Pre-fetched congestion value from 168h ago
    
    Returns:
        np.ndarray of shape (48, num_features) or None if no data at all
        
    NOTE: If history < 48h, we PAD by repeating the earliest row.
    This allows inference during cold start (less accurate initially).
    """
    # Filter to detector
    det_df = merged_df[merged_df["detector_id"] == detector_id].copy()
    
    if len(det_df) == 0:
        logger.warning(f"Detector {detector_id}: no data at all")
        return None
    
    # Sort by time and take last available hours
    det_df = det_df.sort_values("timestamp")
    available_hours = len(det_df)
    
    # COLD START HANDLING: If history < 48h, pad by repeating earliest row
    if available_hours < HISTORY_HOURS:
        rows_needed = HISTORY_HOURS - available_hours
        logger.info(
            f"Detector {detector_id}: only {available_hours}h history, "
            f"padding with {rows_needed} copies of earliest row (cold start mode)"
        )
        # Get the earliest row and repeat it
        earliest_row = det_df.iloc[[0]]
        padding = pd.concat([earliest_row] * rows_needed, ignore_index=True)
        # Prepend padding to actual data
        det_df = pd.concat([padding, det_df], ignore_index=True)
    
    # Take last 48 hours (after potential padding)
    det_df = det_df.tail(HISTORY_HOURS)
    
    # Set lag values if provided
    if lag_48h_value is not None:
        det_df["congestion_index_lag_48h"] = lag_48h_value
    if lag_168h_value is not None:
        det_df["congestion_index_lag_168h"] = lag_168h_value
    
    # Select features in correct order
    feature_cols = [c for c in FEATURE_COLS if c in det_df.columns]
    
    if len(feature_cols) != len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(feature_cols)
        # Issue #5: Don't silently fill with 0 - this masks errors
        # Log which features are missing and their likely cause
        logger.error(
            f"Detector {detector_id} missing {len(missing)} features: {missing}. "
            f"This indicates incomplete feature engineering. Check upstream."
        )
        # Still add as 0 to prevent crash, but this is a WARNING not a fix
        for col in missing:
            det_df[col] = 0.0
        feature_cols = FEATURE_COLS
    
    # Validate no NaNs in final features (would cause model issues)
    X = det_df[feature_cols].values.astype(np.float32)
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning(f"Detector {detector_id}: {nan_count} NaN values in features, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    return X


def build_features_all_detectors(
    traffic_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    lag_168h_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build features for all detectors.
    
    Args:
        traffic_df: Traffic history (~100h per detector for valid 48h lags)
        weather_df: Weather data (past ~100h + future 24h forecast for weather lags)
        lag_168h_df: Traffic snapshot from 168h ago (per detector, broadcast)
    
    Returns:
        X: (N, 48, num_features) array
        det_indices: (N,) detector indices for model
        detector_ids: list of detector_id strings in order
    
    Weather Lag Strategy:
        weather_df should include ~24h of FORECAST data (future timestamps).
        We pre-compute weather lag columns on weather_df BEFORE merging.
        This allows shift(-24) to access 24h-ahead forecasts.
    
    NOTE: 48h congestion lag is computed via shift(48) per-row, matching training.
    You need ~100h of traffic history (48h sequence + 48h for lag).
    """
    # Pre-compute weather lags on weather DataFrame (which has future forecasts)
    # This must happen BEFORE merge, since traffic_df doesn't have future rows
    weather_with_lags = precompute_weather_lags(weather_df, WEATHER_LAGS)
    
    # Merge traffic with expanded weather (now includes lag columns)
    merged = merge_traffic_weather(traffic_df, weather_with_lags)
    
    # Add time features
    merged = cyclical_encode(merged)
    merged = add_calendar_features(merged)
    
    # Add spike features
    merged = merged.sort_values(["detector_id", "timestamp"])
    merged = add_delta_features(merged)
    merged = add_volatility_features(merged)
    
    # Weather lags already added via precompute_weather_lags + merge
    # (No need to call add_weather_lags here - lag columns came from weather_df)
    
    # Add congestion lags
    # 48h lag is always computed via shift(48) to match training
    # 168h lag uses provided snapshot (broadcast) since we lack 168h history
    merged = add_congestion_lags(merged, lag_168h_df)
    
    # Check cold start status
    hours_per_detector = merged.groupby("detector_id").size()
    min_hours = hours_per_detector.min() if len(hours_per_detector) > 0 else 0
    if min_hours < HISTORY_HOURS:
        logger.warning(
            f"COLD START MODE: Only {min_hours}h of history available (need {HISTORY_HOURS}h). "
            f"Predictions will be less accurate until cache fills. "
            f"Full accuracy after {HISTORY_HOURS - min_hours} more hours of data collection."
        )
    
    # Scale features
    merged = scale_features(merged)
    
    # Fill any remaining NaNs
    merged = merged.fillna(0.0)
    
    # Build per-detector
    detector_ids = merged["detector_id"].unique().tolist()
    X_list = []
    valid_detector_ids = []
    det_indices = []
    
    # Load det2idx to filter to known detectors
    _, _, det2idx = load_scalers()
    
    skipped_unknown = 0
    skipped_incomplete = 0
    
    for det_id in detector_ids:
        # Issue #7: Skip detectors not in training universe
        det_idx = get_detector_index(det_id)
        if det_idx == -1:
            skipped_unknown += 1
            continue
        
        X = build_features_for_detector(det_id, merged)
        if X is not None and X.shape[0] == HISTORY_HOURS:
            X_list.append(X)
            valid_detector_ids.append(det_id)
            det_indices.append(det_idx)
        else:
            skipped_incomplete += 1
    
    if skipped_unknown > 0:
        logger.warning(f"Skipped {skipped_unknown} detectors not in training det2idx")
    if skipped_incomplete > 0:
        logger.warning(f"Skipped {skipped_incomplete} detectors with incomplete history")
    
    if not X_list:
        logger.error("No valid detector data after filtering!")
        return np.array([]), np.array([]), []
    
    X = np.stack(X_list, axis=0)  # (N, 48, features)
    det_indices = np.array(det_indices, dtype=np.int64)
    
    # CRITICAL: Validate feature count matches model expectation
    actual_features = X.shape[-1]
    if actual_features != EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Feature count mismatch! Got {actual_features}, expected {EXPECTED_NUM_FEATURES}. "
            f"FEATURE_COLS has {len(FEATURE_COLS)} columns. "
            f"This will cause model dimension errors. Check feature engineering."
        )
    
    logger.info(f"Built features for {len(valid_detector_ids)} detectors, shape={X.shape}")
    
    return X, det_indices, valid_detector_ids


# =============================================================================
# CONVENIENCE FUNCTION FOR INFERENCE
# =============================================================================

def prepare_inference_batch(
    traffic_history: pd.DataFrame,
    weather_history: pd.DataFrame,
    traffic_lag_168h: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Main entry point: prepare a batch for model inference.
    
    Args:
        traffic_history: Last 48+ hours of traffic per detector (ideally ~100h for 48h lag)
        weather_history: Last 48+ hours of weather
        traffic_lag_168h: Traffic snapshot from 168h ago (optional)
    
    Returns:
        X: (N, 48, num_features) float32 tensor-ready array
        det_indices: (N,) int64 detector indices
        detector_ids: list of detector IDs in same order
    
    IMPORTANT: 48h lag is computed via shift(48) per-row, NOT as a single broadcast value.
    This means you need ~100h of history: 48h for the sequence + 48h for the lag.
    If you only have 48h of history, the first rows will have NaN for lag_48h.
    
    Usage:
        from backend.data.traffic import fetch_traffic_history, get_traffic_at_lag
        from backend.data.weather import fetch_weather_history
        from backend.data.features import prepare_inference_batch
        
        # Need ~100h to have valid 48h lags for full 48h sequence
        traffic_df = fetch_traffic_history(hours=100)
        weather_df = fetch_weather_history(hours=100)
        lag_168h = get_traffic_at_lag(lag_hours=168)
        
        X, det_idx, det_ids = prepare_inference_batch(traffic_df, weather_df, lag_168h)
    """
    # DON'T pre-compute 48h lag as a single value!
    # Training uses shift(48) which gives EACH row its own lag value.
    # Let build_features_all_detectors compute it via shift(48).
    #
    # For the 168h lag, we still need an external snapshot since we likely
    # don't have 168+ hours of continuous history.
    
    return build_features_all_detectors(
        traffic_df=traffic_history,
        weather_df=weather_history,
        lag_168h_df=traffic_lag_168h,
    )
