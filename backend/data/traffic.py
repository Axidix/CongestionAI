from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Import from centralized config
from backend.config import (
    TRAFFIC_CACHE_PATH,
    CACHE_DIR,
    MAPPING_PATH,
)

logger = logging.getLogger(__name__)

WFS_URL = (
    "https://api.viz.berlin.de/geoserver/mdh/ows?"
    "service=WFS&version=1.0.0&request=GetFeature&"
    "typeName=mdh:vmzlos-step&outputFormat=application/json"
)

# We need: ~100h continuous history (48h sequence + 48h for lag features)
# Plus 1 snapshot at 168h ago for weekly lag
HISTORY_HOURS_CONTINUOUS = 100   # 48h sequence + 48h for shift(48) + buffer
LAG_HOUR = 168                    # Weekly lag point
CACHE_BUFFER_HOURS = 2            # Buffer around lag point


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "CongestionAI/1.0 (+streamlit-backend)",
            "Accept": "application/json",
            "Referer": "https://viz.berlin.de/",
            "Origin": "https://viz.berlin.de",
        }
    )
    return s


def load_detector_segment_mapping(path: Path = MAPPING_PATH) -> pd.DataFrame:
    """
    Loads detector_id -> unique_id mapping built offline.
    Contains: detector_id, unique_id, lon, lat (and dist_m optionally).
    """
    m = pd.read_parquet(path)
    # Ensure unique_id is str to match WFS output (fixes dtype mismatch on merge)
    m["unique_id"] = m["unique_id"].astype(str)
    return m[["detector_id", "unique_id", "lon", "lat"]].copy()


def fetch_wfs_snapshot(url: str = WFS_URL, timeout: int = 20, retries: int = 2) -> dict:
    """
    Fetch GeoJSON snapshot from WFS. Retries lightly to stay robust but lightweight.
    """
    sess = _session()
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))  # small backoff
            else:
                raise RuntimeError(f"WFS fetch failed after {retries+1} tries: {e}") from e

    # Unreachable
    raise RuntimeError(str(last_err))


def extract_segments_to_frame(wfs_geojson: dict) -> pd.DataFrame:
    """
    Returns a DataFrame keyed by unique_id with:
      unique_id, free_flow_speed, congestion_index, timestamp
    """
    feats = wfs_geojson.get("features", [])
    if not feats:
        raise ValueError("WFS response has no features")

    # Extract properties only (fast, no GeoPandas needed)
    props = [f.get("properties", {}) for f in feats]
    df = pd.DataFrame(props)

    required = {"unique_id", "speedavg", "freeflowspeed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WFS properties missing columns: {missing}")

    # Timestamp: sometimes present per feature or global; handle safely
    # Prefer per-feature "timeStamp" if present; otherwise use now (UTC)
    # NOTE: If WFS lacks timestamp, we assume data is current. WFS is real-time,
    # but if their cache is stale, our timestamps will be slightly off.
    if "timeStamp" in df.columns and df["timeStamp"].notna().any():
        ts = pd.to_datetime(df["timeStamp"], errors="coerce", utc=True)
    else:
        ts = pd.Timestamp.utcnow()
        ts = pd.Series([ts] * len(df), dtype="datetime64[ns, UTC]")

    # Convert speeds to numeric
    speed = pd.to_numeric(df["speedavg"], errors="coerce")
    ff = pd.to_numeric(df["freeflowspeed"], errors="coerce")

    # Avoid divide-by-zero
    ff = ff.where(ff > 0)

    speed_ratio = speed / ff
    cong = 1.0 - speed_ratio

    # If road is closed, you may want to force congestion=1.0 (optional policy)
    if "closed" in df.columns:
        closed = pd.to_numeric(df["closed"], errors="coerce").fillna(0).astype(int)
        cong = np.where(closed == 1, 1.0, cong)

    cong = np.clip(cong, 0.0, 1.0)

    out = pd.DataFrame(
        {
            "unique_id": df["unique_id"].astype(str),
            "free_flow_speed": ff.astype(float),
            "congestion_index": cong.astype(float),
            "timestamp": ts,
        }
    )

    # Keep one row per unique_id (if duplicates appear, keep latest timestamp)
    out = out.sort_values("timestamp").drop_duplicates("unique_id", keep="last")
    return out


def build_detector_inputs(
    mapping: pd.DataFrame,
    seg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join mapping to segment snapshot -> detector-level inputs.
    Output columns match training format: detector_id, timestamp, congestion_index, free_flow_speed, lon, lat
    """
    merged = mapping.merge(seg_df, on="unique_id", how="left")

    # Ensure required output columns (same as training data):
    out = merged[["detector_id", "timestamp", "congestion_index", "free_flow_speed", "lon", "lat"]].copy()

    # If timestamp missing (unmatched), set to current time to avoid NaT
    now = pd.Timestamp.utcnow()
    out["timestamp"] = out["timestamp"].fillna(now)

    return out


def fetch_detector_snapshot() -> pd.DataFrame:
    """
    Fetch WFS + return detector-level DataFrame for current time.
    Output columns: detector_id, timestamp, congestion_index, free_flow_speed, lon, lat
    """
    mapping = load_detector_segment_mapping()
    wfs = fetch_wfs_snapshot()
    seg_df = extract_segments_to_frame(wfs)
    return build_detector_inputs(mapping, seg_df)


# =============================================================================
# HISTORY & CACHING (needed for lag features)
# =============================================================================

def cache_traffic(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    """
    Save traffic data to cache, keeping only:
    - Last HISTORY_HOURS_CONTINUOUS (~100h) continuous
    - Data around LAG_HOUR (168h) mark for weekly lag feature
    """
    path = path or TRAFFIC_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        now = pd.Timestamp.utcnow()
        
        # Keep: recent ~100h + buffer around 168h lag
        recent_cutoff = now - timedelta(hours=HISTORY_HOURS_CONTINUOUS + 1)
        lag_start = now - timedelta(hours=LAG_HOUR + CACHE_BUFFER_HOURS)
        lag_end = now - timedelta(hours=LAG_HOUR - CACHE_BUFFER_HOURS)
        
        mask_recent = df["timestamp"] >= recent_cutoff
        mask_lag = (df["timestamp"] >= lag_start) & (df["timestamp"] <= lag_end)
        
        df = df[mask_recent | mask_lag].copy()
        
        df.to_parquet(path, index=False)
        logger.debug(f"Cached {len(df)} traffic rows to {path}")
    except Exception as e:
        logger.warning(f"Failed to cache traffic: {e}")


def load_cached_traffic(path: Optional[Path] = None) -> pd.DataFrame:
    """Load traffic from cache."""
    path = path or TRAFFIC_CACHE_PATH
    
    if not path.exists():
        logger.warning(f"No traffic cache at {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(path)
        # Ensure timestamp is UTC
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        logger.info(f"Loaded {len(df)} rows from traffic cache")
        return df
    except Exception as e:
        logger.error(f"Failed to load traffic cache: {e}")
        return pd.DataFrame()


def append_snapshot_to_history(
    snapshot: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Append a new snapshot to history, deduplicating and trimming.
    
    Args:
        snapshot: New detector snapshot (from fetch_detector_snapshot)
        history: Existing history DataFrame (or None to load from cache)
    
    Returns:
        Updated history DataFrame
    """
    if history is None:
        history = load_cached_traffic()
    
    if history.empty:
        return snapshot.copy()
    
    # Round timestamps to hour for deduplication
    snapshot = snapshot.copy()
    history = history.copy()
    snapshot["_hour"] = snapshot["timestamp"].dt.floor("h")
    history["_hour"] = history["timestamp"].dt.floor("h")
    
    # Concat and dedupe (keep latest per detector+hour)
    combined = pd.concat([history, snapshot], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(
        subset=["detector_id", "_hour"], keep="last"
    )
    combined = combined.drop(columns=["_hour"])
    
    return combined.reset_index(drop=True)


def fetch_and_accumulate() -> pd.DataFrame:
    """
    Fetch current snapshot and add to history cache.
    
    Returns:
        Updated full history DataFrame
    """
    try:
        snapshot = fetch_detector_snapshot()
        history = append_snapshot_to_history(snapshot)
        cache_traffic(history)
        return history
    except Exception as e:
        logger.error(f"Failed to fetch/accumulate traffic: {e}")
        return load_cached_traffic()


def fetch_traffic_history(hours: int = HISTORY_HOURS_CONTINUOUS) -> pd.DataFrame:
    """
    Get traffic history for feature building.
    
    This is the main function for feature building - matches weather.fetch_weather_history() pattern.
    
    Args:
        hours: Number of hours of continuous history (default ~100h for 48h lags)
    
    Returns:
        DataFrame with columns: detector_id, timestamp, congestion_index, free_flow_speed, lon, lat
        (same format as training data)
    
    Note:
        For 168h lag, use get_traffic_at_lag() separately.
    """
    history = load_cached_traffic()
    
    if history.empty:
        logger.warning("No traffic history available, fetching current snapshot only")
        try:
            snapshot = fetch_detector_snapshot()
            cache_traffic(snapshot)
            return snapshot
        except Exception as e:
            logger.error(f"Failed to fetch traffic: {e}")
            return pd.DataFrame()
    
    # Filter to requested hours
    cutoff = pd.Timestamp.utcnow() - timedelta(hours=hours)
    history = history[history["timestamp"] >= cutoff].copy()
    
    return history.reset_index(drop=True)


def get_traffic_at_lag(lag_hours: int = LAG_HOUR) -> pd.DataFrame:
    """
    Get traffic snapshot from ~lag_hours ago (for weekly lag feature).
    
    Args:
        lag_hours: Hours ago to fetch (default 168 = 1 week)
    
    Returns:
        DataFrame with closest data to target time per detector
    """
    history = load_cached_traffic()
    
    if history.empty:
        logger.warning(f"No traffic history for {lag_hours}h lag")
        return pd.DataFrame()
    
    target = pd.Timestamp.utcnow() - timedelta(hours=lag_hours)
    
    # Find closest timestamp per detector
    history = history.copy()
    history["_diff"] = (history["timestamp"] - target).abs()
    
    # Keep row with smallest diff per detector
    idx = history.groupby("detector_id")["_diff"].idxmin()
    result = history.loc[idx].drop(columns=["_diff"])
    
    return result.reset_index(drop=True)


def get_traffic_for_detector(
    detector_id: str,
    df: Optional[pd.DataFrame] = None,
    hours: int = HISTORY_HOURS_CONTINUOUS,
) -> pd.DataFrame:
    """
    Get traffic history for a specific detector.
    
    Args:
        detector_id: Detector ID to filter
        df: Pre-loaded history DataFrame (optional)
        hours: Hours of history if df is None
    
    Returns:
        DataFrame sorted by timestamp for that detector
    """
    if df is None:
        df = fetch_traffic_history(hours=hours)
    
    if df.empty:
        return pd.DataFrame()
    
    det_df = df[df["detector_id"] == detector_id].copy()
    return det_df.sort_values("timestamp").reset_index(drop=True)
