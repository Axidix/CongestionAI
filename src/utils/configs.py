from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DataConfig:
    """Data preparation configuration."""
    file_path: str = "prepared_data/preprocessed_full_data.csv"
    nb_detectors: Optional[int] = None  # None = use all
    forecast_horizon: int = 24
    history_hours: int = 48
    weather_lags: Tuple[int, ...] = (0, -3, -6, -12, -24)
    delta_lags: Tuple[int, ...] = (1, 2, 4, 6)
    volatility_threshold: float = 0.04
    years_train: Tuple[int, ...] = (2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024)
    years_val: Tuple[int, ...] = (2019,)
    years_test: Tuple[int, ...] = ()
    cache_dir: str = "prepared_data/memmap_cache"
    
    feature_cols_norm: Tuple[str, ...] = (
        "temperature", "precipitation", "visibility", "congestion_index", "free_flow_speed"
    )
    feature_cols_base: Tuple[str, ...] = (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "lon", "lat", "year", "season",
        "temperature", "precipitation", "visibility",
        "congestion_index", "free_flow_speed",
        "is_weekend", "is_holiday", "is_school_holiday", "is_rush_hour", "is_snow", "is_fog"
    )


@dataclass
class TCNModelConfig:
    """TCN model configuration."""
    model_type: str = "tcn"
    emb_dim: int = 256
    num_channels: Tuple[int, ...] = (128, 256, 256, 256)
    kernel_size: int = 5
    dropout_encoder: float = 0.20
    dropout_heads: float = 0.20
    use_se: bool = False
    pooling: str = "last"


@dataclass
class TransformerModelConfig:
    """Transformer model configuration."""
    model_type: str = "transformer"
    d_model: int = 192
    n_heads: int = 6
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.20
    det_emb_dim: int = 96
    pooling: str = "avg"


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 3e-4
    weight_decay: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.99)
    grad_clip: float = 0.5
    use_amp: bool = True
    num_workers: int = 4
    patience: Optional[int] = 5
    
    # Loss config
    spike_weight: float = 3.0
    spike_threshold: float = 0.15
    eval_spike_threshold: float = 0.38
