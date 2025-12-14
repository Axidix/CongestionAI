"""
Backend Configuration
=====================

Centralized configuration for paths, API keys, and model settings.

Contents:
---------
- MODEL_CHECKPOINT_PATH: Path to trained .pt file
- FORECAST_OUTPUT_PATH: Where to save forecast.json for GUI
- CACHE_DIR: Directory for cached weather/traffic data
- API keys for weather services
- Feature column definitions (must match training)
- Device selection (cuda/cpu)
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
GUI_DATA_DIR = PROJECT_ROOT / "gui" / "data"

# Model checkpoint (update after training final model)
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "plots_training_dl" / "final_model" / "checkpoints" / "best_model_FINAL_MODEL_lags_48_168.pt"

# Output for GUI
FORECAST_OUTPUT_PATH = GUI_DATA_DIR / "forecast.json"

# Cache directories
CACHE_DIR = BACKEND_DIR / "cache"
WEATHER_CACHE_PATH = CACHE_DIR / "weather_cache.parquet"  # Fixed: was .json
TRAFFIC_CACHE_PATH = CACHE_DIR / "traffic_cache.parquet"

# Scalers and mappings from training
SCALER_DIR = BACKEND_DIR / "scalers"
STD_SCALER_PATH = SCALER_DIR / "std_scaler.joblib"
MM_SCALER_PATH = SCALER_DIR / "mm_scaler.joblib"
DET2IDX_PATH = SCALER_DIR / "det2idx.joblib"

# Detector mapping (WFS segment ID -> detector ID from training)
MAPPING_PATH = BACKEND_DIR / "data" / "detector_to_wfs_segment.parquet"

# =============================================================================
# API CONFIGURATION
# =============================================================================

# DWD (German Weather Service) - BrightSky API uses station IDs
# 00403 = Berlin-Tegel (used by BrightSky for Berlin area)
# 00433 = Berlin-Tempelhof (alternative)
DWD_STATION_ID = "00403"  # Berlin-Tegel (matches weather.py)

# OpenWeatherMap (backup) - requires free API key
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

# =============================================================================
# MODEL CONFIGURATION (must match training)
# =============================================================================

FORECAST_HORIZON = 24  # hours ahead
HISTORY_HOURS = 48
CONGESTION_LAGS = (48, 168)  # 2-day and 1-week lags

# Feature columns (must match training exactly)
FEATURE_COLS_BASE = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lon", "lat", "year", "season",
    "temperature", "precipitation", "visibility",
    "congestion_index", "free_flow_speed",
    "is_weekend", "is_holiday", "is_school_holiday", "is_rush_hour", "is_snow", "is_fog"
)

WEATHER_LAGS = (0, -3, -6, -12, -24)
DELTA_LAGS = (1, 2, 4, 6)

# =============================================================================
# DEVICE
# =============================================================================

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# REFRESH SETTINGS
# =============================================================================

REFRESH_INTERVAL_MINUTES = 60  # How often to regenerate forecast
