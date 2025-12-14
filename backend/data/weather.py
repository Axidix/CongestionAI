from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Import from centralized config
from backend.config import (
    DWD_STATION_ID,
    WEATHER_CACHE_PATH,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)

# BrightSky API (free DWD wrapper)
BRIGHTSKY_BASE = "https://api.brightsky.dev"

# We need: ~100h continuous history (to match traffic for feature alignment)
# Plus buffer around 168h mark for weekly lag feature
HISTORY_HOURS_CONTINUOUS = 100   # Match traffic cache window
LAG_HOUR = 168                    # Weekly lag point  
CACHE_BUFFER_HOURS = 2            # Buffer around lag point

# Columns to keep from API response
WEATHER_COLS = [
    "timestamp",
    "temperature",
    "dew_point",
    "precipitation",
    "relative_humidity",
    "visibility",
    "cloud_cover",
    "condition",
    "icon",
]


def _session() -> requests.Session:
    """Create a requests session with appropriate headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "CongestionAI/1.0 (+weather-fetcher)",
        "Accept": "application/json",
    })
    return s


def fetch_current_weather(station_id: str = DWD_STATION_ID) -> dict:
    """
    Fetch current weather conditions from BrightSky.
    
    Returns dict with: temperature, precipitation, visibility, condition, icon, etc.
    """
    url = f"{BRIGHTSKY_BASE}/current_weather?dwd_station_id={station_id}"
    
    try:
        sess = _session()
        r = sess.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        weather = data.get("weather", {})
        if not weather:
            raise ValueError("No weather data in response")
        
        return weather
    
    except Exception as e:
        logger.error(f"Failed to fetch current weather: {e}")
        raise


def fetch_weather_for_date(date: str, station_id: str = DWD_STATION_ID) -> list[dict]:
    """
    Fetch hourly weather data for a specific date (YYYY-MM-DD).
    
    Returns list of hourly weather records.
    """
    url = f"{BRIGHTSKY_BASE}/weather?dwd_station_id={station_id}&date={date}"
    
    try:
        sess = _session()
        r = sess.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        weather = data.get("weather", [])
        return weather if isinstance(weather, list) else [weather]
    
    except Exception as e:
        logger.warning(f"Failed to fetch weather for {date}: {e}")
        return []


# Maximum future hours needed for weather lags (WEATHER_LAGS = 0, -3, -6, -12, -24)
# Negative lags mean FUTURE weather (shift(-24) = 24h ahead forecast)
FUTURE_HOURS_NEEDED = 24


def fetch_weather_history(
    hours: int = HISTORY_HOURS_CONTINUOUS, 
    future_hours: int = FUTURE_HOURS_NEEDED,
    station_id: str = DWD_STATION_ID
) -> pd.DataFrame:
    """
    Fetch weather data: past history + future forecasts.
    
    Args:
        hours: Number of hours of PAST history to fetch (default ~100h)
        future_hours: Number of hours of FUTURE forecast to fetch (default 24h)
        station_id: DWD station ID
    
    Returns:
        DataFrame with columns: timestamp, temperature, precipitation, visibility,
        dew_point, relative_humidity, cloud_cover, condition, icon,
        is_rain, is_snow, is_fog
    
    Weather Lag Strategy:
        Training used WEATHER_LAGS = (0, -3, -6, -12, -24) with shift(lag).
        - shift(-3) = weather from 3 rows LATER = 3h forecast
        - shift(-24) = weather from 24 rows LATER = 24h forecast
        
        This is NOT data leakage - it's using weather FORECASTS which are
        available in real-time from BrightSky API. The model learns to use
        predicted future weather conditions for traffic forecasting.
    """
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    end = now + timedelta(hours=future_hours)  # Include future forecasts!
    
    all_rows = []
    
    # Fetch day by day (past + future)
    current = start.date()
    end_date = end.date()
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        rows = fetch_weather_for_date(date_str, station_id)
        all_rows.extend(rows)
        current += timedelta(days=1)
    
    if not all_rows:
        logger.warning("No weather data fetched, trying cache...")
        return load_cached_weather()
    
    # Build DataFrame
    df = pd.DataFrame(all_rows)
    
    # Keep only needed columns (handle missing gracefully)
    available_cols = [c for c in WEATHER_COLS if c in df.columns]
    df = df[available_cols].copy()
    
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Filter to requested time window (past history + future forecast)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    
    logger.info(f"Fetched weather: {len(df)} rows from {start_ts} to {end_ts} (includes {future_hours}h forecast)")
    
    # Sort and dedupe
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    
    # Derive weather flags
    df = derive_weather_flags(df)
    
    # Cache the result
    cache_weather(df)
    
    return df.reset_index(drop=True)


def derive_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary weather flags based on condition/icon columns.
    
    Adds: is_rain, is_snow, is_fog
    """
    df = df.copy()
    
    # Default to False
    df["is_rain"] = False
    df["is_snow"] = False
    df["is_fog"] = False
    
    # Derive from 'condition' or 'icon' columns
    if "condition" in df.columns:
        cond_lower = df["condition"].fillna("").str.lower()
        df["is_rain"] = cond_lower.str.contains("rain|drizzle|shower", regex=True)
        df["is_snow"] = cond_lower.str.contains("snow|sleet|hail", regex=True)
        df["is_fog"] = cond_lower.str.contains("fog|mist|haze", regex=True)
    
    if "icon" in df.columns:
        icon_lower = df["icon"].fillna("").str.lower()
        # Supplement with icon if condition didn't catch it
        df["is_rain"] = df["is_rain"] | icon_lower.str.contains("rain", regex=False)
        df["is_snow"] = df["is_snow"] | icon_lower.str.contains("snow|sleet", regex=True)
        df["is_fog"] = df["is_fog"] | icon_lower.str.contains("fog", regex=False)
    
    # Also check visibility for fog (< 1000m is fog by definition)
    if "visibility" in df.columns:
        vis = pd.to_numeric(df["visibility"], errors="coerce")
        df["is_fog"] = df["is_fog"] | (vis < 1000)
    
    # Convert to int for consistency
    df["is_rain"] = df["is_rain"].astype(int)
    df["is_snow"] = df["is_snow"].astype(int)
    df["is_fog"] = df["is_fog"].astype(int)
    
    return df


def cache_weather(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    """
    Save weather data to cache, keeping only:
    - Last HISTORY_HOURS_CONTINUOUS (~100h) continuous
    - Data around LAG_HOUR (168h) mark for weekly lag feature
    """
    path = path or WEATHER_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        now = pd.Timestamp.utcnow()
        
        # Keep: recent ~100h + buffer around 168h lag
        recent_cutoff = now - timedelta(hours=HISTORY_HOURS_CONTINUOUS + 1)
        lag_start = now - timedelta(hours=LAG_HOUR + CACHE_BUFFER_HOURS)
        lag_end = now - timedelta(hours=LAG_HOUR - CACHE_BUFFER_HOURS)
        
        if "timestamp" in df.columns:
            mask_recent = df["timestamp"] >= recent_cutoff
            mask_lag = (df["timestamp"] >= lag_start) & (df["timestamp"] <= lag_end)
            df = df[mask_recent | mask_lag].copy()
        
        df.to_parquet(path, index=False)
        logger.debug(f"Cached {len(df)} weather rows to {path}")
    except Exception as e:
        logger.warning(f"Failed to cache weather: {e}")


def load_cached_weather(path: Optional[Path] = None) -> pd.DataFrame:
    """Load weather from cache (fallback if API fails)."""
    path = path or WEATHER_CACHE_PATH
    
    if not path.exists():
        logger.warning(f"No weather cache at {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from weather cache")
        return df
    except Exception as e:
        logger.error(f"Failed to load weather cache: {e}")
        return pd.DataFrame()


def get_weather_at_timestamp(
    target_ts: pd.Timestamp,
    df: Optional[pd.DataFrame] = None,
    hours_history: int = HISTORY_HOURS_CONTINUOUS,
) -> dict:
    """
    Get weather data closest to a target timestamp.
    
    Args:
        target_ts: Target timestamp (UTC)
        df: Pre-fetched weather DataFrame (optional, will fetch if None)
        hours_history: Hours of history to fetch if df is None
    
    Returns:
        dict with weather values for that hour
    """
    if df is None:
        df = fetch_weather_history(hours=hours_history)
    
    if df.empty:
        return {}
    
    # Ensure UTC
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize("UTC")
    
    # Find closest timestamp (use copy to avoid mutating input)
    df = df.copy()
    df["_diff"] = (df["timestamp"] - target_ts).abs()
    closest_idx = df["_diff"].idxmin()
    row = df.loc[closest_idx].drop("_diff")
    
    return row.to_dict()


def get_weather_at_lag(lag_hours: int = LAG_HOUR) -> dict:
    """
    Get weather from ~lag_hours ago (for weekly lag feature).
    
    Args:
        lag_hours: Hours ago to fetch (default 168 = 1 week)
    
    Returns:
        dict with weather values for that time
    """
    target = pd.Timestamp.utcnow() - timedelta(hours=lag_hours)
    
    # Try cache first
    df = load_cached_weather()
    if df.empty:
        # Fetch fresh (will get recent data, may not have lag point)
        df = fetch_weather_history(hours=lag_hours + CACHE_BUFFER_HOURS)
    
    return get_weather_at_timestamp(target, df=df)


def format_weather_for_gui(weather_df: pd.DataFrame) -> dict:
    """
    Format weather data for GUI display.
    
    Takes the weather DataFrame (with history + forecasts) and extracts:
    - Current weather conditions
    - 24-hour forecast (hourly)
    
    Returns:
        dict matching GUI's expected weather.json format:
        {
            "current": {"temp": 8.5, "description": "cloudy", "precip": 0.0, ...},
            "hourly": [{"hour": 0, "temp": 8.5, "precip": 0.0}, ...]
        }
    """
    if weather_df.empty:
        return _default_weather_for_gui()
    
    now = pd.Timestamp.utcnow()
    
    # Ensure timestamps are UTC
    df = weather_df.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    
    # Find current hour (closest to now)
    df["_diff"] = (df["timestamp"] - now).abs()
    current_idx = df["_diff"].idxmin()
    current_row = df.loc[current_idx]
    
    # Build current weather (handle NaN values)
    def safe_float(val, default=0):
        """Convert to float, returning default if NaN or invalid."""
        try:
            f = float(val)
            return default if pd.isna(f) else f
        except (TypeError, ValueError):
            return default
    
    current = {
        "temp": round(safe_float(current_row.get("temperature", 10), 10), 1),
        "description": str(current_row.get("condition", "unknown")),
        "wind_speed": round(safe_float(current_row.get("wind_speed", 0), 0), 1),
        "precip": round(safe_float(current_row.get("precipitation", 0), 0), 1),
        "visibility": round(safe_float(current_row.get("visibility", 10000), 10000), 0),
        "humidity": round(safe_float(current_row.get("relative_humidity", 50), 50), 0),
        "icon": str(current_row.get("icon", "cloudy")),
    }
    
    # Build hourly forecast (next 25 hours: 0 = now, 24 = 24h ahead)
    hourly = []
    for h in range(25):
        target_ts = now + timedelta(hours=h)
        
        # Find closest timestamp to this hour
        df["_diff"] = (df["timestamp"] - target_ts).abs()
        closest_idx = df["_diff"].idxmin()
        row = df.loc[closest_idx]
        
        # Only use if within 30 minutes of target
        if df.loc[closest_idx, "_diff"] <= timedelta(minutes=30):
            hourly.append({
                "hour": h,
                "temp": round(safe_float(row.get("temperature", 10), 10), 1),
                "precip": round(safe_float(row.get("precipitation", 0), 0), 2),
                "visibility": round(safe_float(row.get("visibility", 10000), 10000), 0),
                "condition": str(row.get("condition", "unknown")),
                "icon": str(row.get("icon", "cloudy")),
            })
        else:
            # No data for this hour, use interpolated/default
            hourly.append({
                "hour": h,
                "temp": round(safe_float(current.get("temp", 10), 10), 1),
                "precip": 0.0,
                "visibility": 10000,
                "condition": "unknown",
                "icon": "cloudy",
            })
    
    return {
        "current": current,
        "hourly": hourly,
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _default_weather_for_gui() -> dict:
    """Return default weather structure when no data available."""
    now = datetime.utcnow()
    return {
        "current": {
            "temp": 10.0,
            "description": "unavailable",
            "wind_speed": 0,
            "precip": 0.0,
            "visibility": 10000,
            "humidity": 50,
            "icon": "cloudy",
        },
        "hourly": [
            {"hour": h, "temp": 10.0, "precip": 0.0, "visibility": 10000, "condition": "unavailable", "icon": "cloudy"}
            for h in range(25)
        ],
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
