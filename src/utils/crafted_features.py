"""
Modular spike detection features and losses.

Usage:
    # In data preparation:
    from src.utils.spike_features import SpikeFeatureConfig, add_spike_features
    
    config = SpikeFeatureConfig(
        enable_deltas=True,
        enable_rolling_stats=True,
        delta_lags=[1, 2, 4, 6],
        rolling_windows=[3, 6]
    )
    df = add_spike_features(df, config)
    
    # Get feature column names to add to model input:
    new_cols = config.get_feature_columns()
    
    # In training:
    from src.utils.spike_features import SpikeWeightedMSELoss
    criterion = SpikeWeightedMSELoss(spike_weight=3.0, threshold=0.15)
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpikeFeatureConfig:
    """
    Configuration for spike-related features.
    
    Attributes:
        enable_deltas: Add delta (change) features
        enable_abs_deltas: Add absolute delta features
        enable_rolling_stats: Add rolling statistics
        enable_spike_labels: Add binary spike labels
        
        delta_lags: Lag horizons for delta computation
        rolling_windows: Window sizes for rolling stats
        spike_quantile: Quantile for automatic spike threshold
        spike_threshold: Fixed threshold (overrides quantile if set)
        
        target_col: Column to compute features on
    """
    # Feature toggles
    enable_deltas: bool = True
    enable_abs_deltas: bool = False
    enable_rolling_stats: bool = False
    enable_spike_labels: bool = False  # Optional, for analysis
    
    # Feature parameters
    delta_lags: List[int] = field(default_factory=lambda: [1, 2, 4, 6])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    
    # NEW: Volatility feature
    enable_volatility: bool = False
    volatility_window: int = 3
    volatility_binary_threshold: Optional[float] = None  # If set, adds is_high_vol binary feature
    
    # Target column
    target_col: str = "congestion_index"
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns that will be created."""
        cols = []
        if self.enable_deltas:
            cols += [f"delta_{lag}h" for lag in self.delta_lags]
        if self.enable_abs_deltas:
            cols += [f"abs_delta_{lag}h" for lag in self.delta_lags]
        if self.enable_rolling_stats:
            for w in self.rolling_windows:
                cols += [f"rolling_mean_{w}h", f"rolling_std_{w}h"]
        if self.enable_volatility:
            cols.append(f"rolling_vol_{self.volatility_window}h")
            if self.volatility_binary_threshold is not None:
                cols.append("is_high_vol")
        return cols
    
    def get_normalization_columns(self) -> List[str]:
        """Return columns that should be normalized (StandardScaler)."""
        cols = []
        if self.enable_deltas:
            cols += [f"delta_{lag}h" for lag in self.delta_lags]
        if self.enable_abs_deltas:
            cols += [f"abs_delta_{lag}h" for lag in self.delta_lags]
        if self.enable_rolling_stats:
            for w in self.rolling_windows:
                cols += [f"rolling_mean_{w}h", f"rolling_std_{w}h"]
        if self.enable_volatility:
            cols.append(f"rolling_vol_{self.volatility_window}h")
            # Note: is_high_vol is binary, no normalization needed
        return cols


def add_delta_features(df: pd.DataFrame, 
                       col: str = "congestion_index",
                       lags: List[int] = [1, 2, 4, 6],
                       include_abs: bool = True) -> pd.DataFrame:
    """
    Add delta (change) features for recent history.
    
    Args:
        df: DataFrame sorted by detector_id, timestamp
        col: Column to compute deltas on
        lags: List of lag horizons
        include_abs: Whether to include absolute values
    
    Returns:
        df with delta columns added
    """
    df = df.copy()
    
    for lag in lags:
        df[f"delta_{lag}h"] = df.groupby("detector_id")[col].diff(lag)
        if include_abs:
            df[f"abs_delta_{lag}h"] = df[f"delta_{lag}h"].abs()
    
    return df


def add_rolling_stats(df: pd.DataFrame,
                      col: str = "congestion_index",
                      windows: List[int] = [3, 6]) -> pd.DataFrame:
    """
    Add rolling statistics (mean, std, range) for short windows.
    
    Args:
        df: DataFrame sorted by detector_id, timestamp
        col: Column to compute stats on
        windows: List of window sizes (in hours)
    
    Returns:
        df with rolling stat columns added
    """
    df = df.copy()
    
    for w in windows:
        grouped = df.groupby("detector_id")[col]
        
        df[f"rolling_mean_{w}h"] = grouped.transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}h"] = grouped.transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )
        df[f"rolling_range_{w}h"] = grouped.transform(
            lambda x: x.rolling(w, min_periods=1).max() - x.rolling(w, min_periods=1).min()
        )
        
        # Fill NaN std with 0 (first few points)
        df[f"rolling_std_{w}h"] = df[f"rolling_std_{w}h"].fillna(0)
        df[f"rolling_range_{w}h"] = df[f"rolling_range_{w}h"].fillna(0)
    
    return df


def add_spike_labels(df: pd.DataFrame,
                     col: str = "congestion_index",
                     threshold: Optional[float] = None,
                     quantile: float = 0.90) -> tuple:
    """
    Add binary spike labels based on absolute change.
    
    Args:
        df: DataFrame sorted by detector_id, timestamp
        col: Column to compute changes on
        threshold: Fixed threshold. If None, use quantile.
        quantile: Quantile for defining spikes if threshold not provided
    
    Returns:
        (df, thresholds_dict)
    """
    df = df.copy()
    
    # Compute delta if not already present
    if "delta_1h" not in df.columns:
        df["delta_1h"] = df.groupby("detector_id")[col].diff()
    
    if threshold is None:
        up_thresh = df["delta_1h"].quantile(quantile)
        down_thresh = df["delta_1h"].quantile(1 - quantile)
    else:
        up_thresh = threshold
        down_thresh = -threshold
    
    thresholds = {"up": up_thresh, "down": down_thresh}
    
    df["spike_up"] = (df["delta_1h"] > up_thresh).astype(int)
    df["spike_down"] = (df["delta_1h"] < down_thresh).astype(int)
    df["is_spike"] = ((df["spike_up"] == 1) | (df["spike_down"] == 1)).astype(int)
    
    print(f"Spike thresholds: up={up_thresh:.4f}, down={down_thresh:.4f}")
    print(f"Spike frequency: {df['is_spike'].mean()*100:.2f}%")
    
    return df, thresholds


def add_spike_features(df: pd.DataFrame, 
                       config: SpikeFeatureConfig) -> pd.DataFrame:
    """
    Add all spike-related features based on configuration.
    
    Args:
        df: DataFrame sorted by detector_id, timestamp
        config: SpikeFeatureConfig specifying which features to add
    
    Returns:
        df with spike features added
    """
    df = df.copy()
    col = config.target_col
    
    if config.enable_deltas or config.enable_abs_deltas:
        df = add_delta_features(
            df, col=col, 
            lags=config.delta_lags,
            include_abs=config.enable_abs_deltas
        )
    
    if config.enable_rolling_stats:
        df = add_rolling_stats(
            df, col=col,
            windows=config.rolling_windows
        )
    
    if config.enable_spike_labels:
        df, _ = add_spike_labels(
            df, col=col,
            threshold=config.spike_threshold,
            quantile=config.spike_quantile
        )
    
    # NEW: Volatility feature
    if config.enable_volatility:
        col_name = f"rolling_vol_{config.volatility_window}h"
        df[col_name] = (
            df.groupby("detector_id")["congestion_index"]
              .transform(lambda x: x.rolling(config.volatility_window).std())
              .fillna(0)
        )
        
        if config.volatility_binary_threshold is not None:
            df["is_high_vol"] = (df[col_name] > config.volatility_binary_threshold).astype(float)
    
    return df


def make_lags(df, col, lags):
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df.groupby("detector_id")[col].shift(lag)
    return df

def add_lags_and_drop(df, weather_lags):
    df = make_lags(df, "temperature", weather_lags)
    df = make_lags(df, "precipitation", weather_lags)
    df = make_lags(df, "visibility", weather_lags)
    df = df.dropna()
    
    return df
