import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def cyclical_encode(df):
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def encode_detectors(df):
    unique_detectors = sorted(df["detector_id"].unique())
    det2idx = {d: i for i, d in enumerate(unique_detectors)}
    df["det_index"] = df["detector_id"].map(det2idx)
    return df, det2idx


def scale_features(train, val, test, norm_cols, latlon_cols=["lon", "lat"]):
    std = StandardScaler()
    train[norm_cols] = std.fit_transform(train[norm_cols])
    val[norm_cols] = std.transform(val[norm_cols])
    if test is not None:
        test[norm_cols] = std.transform(test[norm_cols])

    mm = MinMaxScaler()
    train[latlon_cols] = mm.fit_transform(train[latlon_cols])
    val[latlon_cols] = mm.transform(val[latlon_cols])
    if test is not None:
        test[latlon_cols] = mm.transform(test[latlon_cols])
    return train, val, test, std, mm


from typing import Tuple, Any
import os
from src.utils.configs import DataConfig
from src.utils.crafted_features import SpikeFeatureConfig, add_spike_features, add_lags_and_drop, make_lags
from src.utils.memmap_sequences import MemmapSequenceBuilder


def prepare_data_memmap(
    data_cfg: DataConfig,
) -> Tuple[np.memmap, np.memmap, np.ndarray, np.ndarray,
           np.memmap, np.memmap, np.ndarray, np.ndarray,
           pd.DataFrame, pd.DataFrame, Any, Any, MemmapSequenceBuilder, int]:
    """
    Prepare data using memory-mapped arrays.
    
    Returns:
        X_train, Y_train, idx_train, det_train,
        X_val, Y_val, idx_val, det_val,
        train_df, val_df, std_scaler, mm_scaler, memmap_builder, num_detectors
    """
    print("\n" + "="*70)
    print("DATA PREPARATION (MEMMAP)")
    print("="*70)
    
    # Load base data
    print(f"\n[1/7] Loading data from {data_cfg.file_path}...")
    df_base = pd.read_csv(data_cfg.file_path)
    df_base["timestamp"] = pd.to_datetime(df_base["timestamp"])
    df_base["orig_idx"] = df_base.index
    df_base = cyclical_encode(df_base)
    print(f"  Loaded {len(df_base):,} rows, {df_base['detector_id'].nunique()} detectors")
    
    # Filter detectors if needed
    if data_cfg.nb_detectors is not None and data_cfg.nb_detectors < df_base["detector_id"].nunique():
        detector_ids = df_base["detector_id"].unique()[:data_cfg.nb_detectors]
        df_small = df_base[df_base["detector_id"].isin(detector_ids)].copy()
        print(f"  Using {data_cfg.nb_detectors} detectors")
    else:
        df_small = df_base.copy()
        print(f"  Using ALL {df_base['detector_id'].nunique()} detectors")
    
    df_small = df_small.sort_values(["detector_id", "timestamp"])
    
    # Season encoding
    print("\n[2/7] Adding season encoding...")
    df_small.loc[(df_small["month"] <= 2) | (df_small["month"] == 12), "season"] = 0
    df_small.loc[(df_small["month"] > 2) & (df_small["month"] <= 5), "season"] = 1
    df_small.loc[(df_small["month"] > 5) & (df_small["month"] <= 8), "season"] = 2
    df_small.loc[(df_small["month"] > 8) & (df_small["month"] <= 11), "season"] = 3
    
    # Spike features
    print("\n[3/7] Adding spike features...")
    spike_config = SpikeFeatureConfig(
        enable_deltas=True,
        enable_abs_deltas=False,
        enable_rolling_stats=False,
        delta_lags=list(data_cfg.delta_lags),
        enable_volatility=True,
        volatility_window=3,
        volatility_binary_threshold=data_cfg.volatility_threshold
    )
    df_small = add_spike_features(df_small, spike_config)
    spike_feature_cols = spike_config.get_feature_columns()
    spike_norm_cols = spike_config.get_normalization_columns()
    
    feature_cols = list(data_cfg.feature_cols_base) + spike_feature_cols
    feature_cols_norm = list(data_cfg.feature_cols_norm) + spike_norm_cols
    print(f"  Spike features: {spike_feature_cols}")
    
    # Congestion lags (NEW)
    print("\n[4/7] Adding congestion lags...")
    congestion_lag_cols = []
    if data_cfg.congestion_lags is not None and len(data_cfg.congestion_lags) > 0:
        df_small = make_lags(df_small, "congestion_index", list(data_cfg.congestion_lags))
        congestion_lag_cols = [f"congestion_index_lag_{lag}h" for lag in data_cfg.congestion_lags]
        feature_cols = feature_cols + congestion_lag_cols
        feature_cols_norm = feature_cols_norm + congestion_lag_cols
        print(f"  Congestion lag features: {congestion_lag_cols}")
    else:
        print(f"  No congestion lags configured")
    
    # Detector encoding
    print("\n[5/7] Encoding detectors...")
    df_small, det2idx = encode_detectors(df_small)
    num_detectors = len(det2idx)
    print(f"  Encoded {num_detectors} detectors")
    
    # Add weather lag column names
    weather_lags = list(data_cfg.weather_lags)
    if "temperature" in feature_cols:
        feature_cols = feature_cols + [f"temperature_lag_{lag}h" for lag in weather_lags] \
            + [f"precipitation_lag_{lag}h" for lag in weather_lags] \
            + [f"visibility_lag_{lag}h" for lag in weather_lags]
    
    # Split by years
    print("\n[6/7] Splitting data...")
    years_split = [list(data_cfg.years_train), list(data_cfg.years_val), list(data_cfg.years_test)]
    
    train = df_small[df_small["timestamp"].dt.year.isin(years_split[0])].copy()
    val = df_small[df_small["timestamp"].dt.year.isin(years_split[1])].copy()
    test = df_small[df_small["timestamp"].dt.year.isin(years_split[2])].copy() if years_split[2] else None
    
    print(f"  Train: {len(train):,} rows (years: {years_split[0]})")
    print(f"  Val: {len(val):,} rows (years: {years_split[1]})")
    
    train = train.set_index("orig_idx")
    val = val.set_index("orig_idx")
    if test is not None:
        test = test.set_index("orig_idx")
    
    # Normalization
    minmax_cols = ["lon", "lat", "year", "season"]
    train, val, test, std_scaler, mm_scaler = scale_features(
        train, val, test, feature_cols_norm, latlon_cols=minmax_cols
    )
    
    # Weather lags
    if "temperature" in list(data_cfg.feature_cols_base):
        train = add_lags_and_drop(train, weather_lags)
        val = add_lags_and_drop(val, weather_lags)
        if test is not None:
            test = add_lags_and_drop(test, weather_lags)
    
    # Drop NaNs from spike features and congestion lags
    drop_na_cols = [c for c in spike_feature_cols + congestion_lag_cols if c in train.columns]
    train = train.dropna(subset=drop_na_cols)
    val = val.dropna(subset=drop_na_cols)
    if test is not None:
        test = test.dropna(subset=drop_na_cols)
    
    # Keep only needed columns
    keep_cols = feature_cols + ["timestamp", "detector_id", "det_index"]
    keep_cols = [c for c in keep_cols if c in train.columns]
    
    train = train[keep_cols]
    val = val[keep_cols]
    if test is not None:
        test = test[keep_cols]
    
    # Create memmap sequences
    print("\n[7/7] Creating MEMMAP sequences...")
    history_offsets = list(range(data_cfg.history_hours))
    
    os.makedirs(data_cfg.cache_dir, exist_ok=True)
    memmap_builder = MemmapSequenceBuilder(cache_dir=data_cfg.cache_dir)
    
    X_train, Y_train, idx_train, det_train = memmap_builder.create_sequences_memmap(
        train, feature_cols, history_offsets, data_cfg.forecast_horizon, prefix="train"
    )
    
    X_val, Y_val, idx_val, det_val = memmap_builder.create_sequences_memmap(
        val, feature_cols, history_offsets, data_cfg.forecast_horizon, prefix="val"
    )
    
    print(f"\n✓ Data preparation complete!")
    print(f"  Train samples: {len(Y_train):,}")
    print(f"  Val samples: {len(Y_val):,}")
    print(f"  Features: {X_train.shape[-1]}")
    print(f"  History length: {X_train.shape[1]}")
    print(f"  Num detectors: {num_detectors}")
    
    return (X_train, Y_train, idx_train, det_train,
            X_val, Y_val, idx_val, det_val,
            train, val, std_scaler, mm_scaler, memmap_builder, num_detectors)
