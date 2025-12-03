import os
import sys
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_pipelines.dl_pipeline import train_model, evaluate, predict
from src.utils.model_evaluation import evaluate_and_plot_block
from src.utils.preprocessing import cyclical_encode, scale_features, encode_detectors
from src.utils.plots import plot_training_curves
from src.model_pipelines.losses import LossConfig, create_loss
from src.utils.crafted_features import SpikeFeatureConfig, add_spike_features, add_lags_and_drop
from src.utils.memmap_sequences import MemmapSequenceBuilder, MemmapDataset
from src.models.tcn_forecaster import MultiHeadTCNForecaster
from src.models.transformer_forecaster import MultiHeadTransformerForecaster

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

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


# ============================================================================
# PREDEFINED CONFIGURATIONS
# ============================================================================

CONFIG_TCN = {
    "data": DataConfig(
        history_hours=48,
        weather_lags=(0, -3, -6, -12, -24),
        delta_lags=(1, 2, 4, 6),
        volatility_threshold=0.04,
    ),
    "model": TCNModelConfig(
        emb_dim=256,
        num_channels=(128, 256, 256, 256),
        kernel_size=5,
        dropout_encoder=0.20,
        dropout_heads=0.20,
        use_se=False,
        pooling="last",
    ),
    "training": TrainingConfig(
        epochs=20,
        batch_size=2048,
        lr=3e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.99),
        grad_clip=0.5,
        use_amp=True,
    ),
}

CONFIG_TRANSFORMER = {
    "data": DataConfig(
        history_hours=72,
        weather_lags=(0, -3, -6, -12, -24),
        delta_lags=(1, 2, 4, 6),
        volatility_threshold=0.04,
    ),
    "model": TransformerModelConfig(
        d_model=192,
        n_heads=6,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.20,
        det_emb_dim=96,
        pooling="avg",
    ),
    "training": TrainingConfig(
        epochs=25,
        batch_size=2048,
        lr=8e-5,
        weight_decay=1e-2,
        betas=(0.9, 0.98),
        grad_clip=0.5,
        use_amp=True,
    ),
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

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
    print(f"\n[1/6] Loading data from {data_cfg.file_path}...")
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
    print("\n[2/6] Adding season encoding...")
    df_small.loc[(df_small["month"] <= 2) | (df_small["month"] == 12), "season"] = 0
    df_small.loc[(df_small["month"] > 2) & (df_small["month"] <= 5), "season"] = 1
    df_small.loc[(df_small["month"] > 5) & (df_small["month"] <= 8), "season"] = 2
    df_small.loc[(df_small["month"] > 8) & (df_small["month"] <= 11), "season"] = 3
    
    # Spike features
    print("\n[3/6] Adding spike features...")
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
    
    # Detector encoding
    print("\n[4/6] Encoding detectors...")
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
    print("\n[5/6] Splitting data...")
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
    
    # Drop NaNs from spike features
    spike_cols_in_df = [c for c in spike_feature_cols if c in train.columns]
    train = train.dropna(subset=spike_cols_in_df)
    val = val.dropna(subset=spike_cols_in_df)
    if test is not None:
        test = test.dropna(subset=spike_cols_in_df)
    
    # Keep only needed columns
    keep_cols = feature_cols + ["timestamp", "detector_id", "det_index"]
    keep_cols = [c for c in keep_cols if c in train.columns]
    
    train = train[keep_cols]
    val = val[keep_cols]
    if test is not None:
        test = test[keep_cols]
    
    # Create memmap sequences
    print("\n[6/6] Creating MEMMAP sequences...")
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


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(
    model_cfg,
    num_features: int,
    num_detectors: int,
    forecast_horizon: int,
    device: str = "cuda"
) -> nn.Module:
    """Create model based on configuration."""
    
    if isinstance(model_cfg, TCNModelConfig):
        print(f"\nCreating TCN model...")
        model = MultiHeadTCNForecaster(
            num_features=num_features,
            horizon=forecast_horizon,
            num_detectors=num_detectors,
            emb_dim=model_cfg.emb_dim,
            num_channels=model_cfg.num_channels,
            kernel_size=model_cfg.kernel_size,
            dropout_encoder=model_cfg.dropout_encoder,
            dropout_heads=model_cfg.dropout_heads,
            use_se=model_cfg.use_se,
            pooling=model_cfg.pooling,
        )
    elif isinstance(model_cfg, TransformerModelConfig):
        print(f"\nCreating MultiHead Transformer model...")
        model = MultiHeadTransformerForecaster(
            num_features=num_features,
            horizon=forecast_horizon,
            num_detectors=num_detectors,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
            det_emb_dim=model_cfg.det_emb_dim,
            pooling=model_cfg.pooling,
        )
    else:
        raise ValueError(f"Unknown model config type: {type(model_cfg)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model.to(device)


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_full_scale(
    model: nn.Module,
    X_train: np.memmap,
    Y_train: np.memmap,
    det_train: np.ndarray,
    X_val: np.memmap,
    Y_val: np.memmap,
    det_val: np.ndarray,
    train_cfg: TrainingConfig,
    model_cfg,
    data_cfg: DataConfig,
    exp_name: str,
    output_dir: str,
    device: str = "cuda",
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Full-scale training with memmap datasets.
    """
    print("\n" + "="*70)
    print(f"TRAINING: {exp_name}")
    print("="*70)
    
    # Create dataloaders with MemmapDataset
    train_loader = DataLoader(
        MemmapDataset(X_train, Y_train, det_train),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=True if train_cfg.num_workers > 0 else False,
        prefetch_factor=2 if train_cfg.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        MemmapDataset(X_val, Y_val, det_val),
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=True if train_cfg.num_workers > 0 else False,
    )
    
    print(f"\nDataloaders:")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Num workers: {train_cfg.num_workers}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )
    print(f"\nOptimizer: AdamW(lr={train_cfg.lr}, weight_decay={train_cfg.weight_decay}, betas={train_cfg.betas})")
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
        eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingLR(T_max={train_cfg.epochs}, eta_min=1e-6)")
    
    # Create loss
    loss_cfg = LossConfig(
        loss_type="spike_weighted",
        spike_weight=train_cfg.spike_weight,
        spike_threshold=train_cfg.spike_threshold
    )
    criterion = create_loss(loss_cfg)
    print(f"Loss: SpikeWeightedMSE(weight={train_cfg.spike_weight}, threshold={train_cfg.spike_threshold})")
    
    # AMP scaler
    scaler = GradScaler('cuda') if train_cfg.use_amp else None
    print(f"AMP: {'enabled' if train_cfg.use_amp else 'disabled'}")
    print(f"Grad clip: {train_cfg.grad_clip}")
    
    # Train
    print(f"\nStarting training for {train_cfg.epochs} epochs...")
    start_time = time.time()
    
    train_losses, val_losses, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=train_cfg.epochs,
        grad_clip=train_cfg.grad_clip,
        patience=train_cfg.patience,
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    # Save losses to file
    with open(f"{output_dir}/losses_{exp_name}.txt", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{t_loss:.6f},{v_loss:.6f}\n")
    
    # Save training curves plot
    plot_training_curves(
        train_losses, val_losses,
        filename=f"training_curve_{exp_name}.png",
        dir=output_dir
    )
    
    # Prepare config dicts for saving
    model_cfg_dict = asdict(model_cfg)
    data_cfg_dict = asdict(data_cfg)
    train_cfg_dict = asdict(train_cfg)
    
    # Save BEST model checkpoint
    best_model_path = f"{output_dir}/checkpoints/best_model_{exp_name}.pt"
    torch.save({
        'model_state_dict': best_state,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': min(val_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1,
        'total_epochs': len(train_losses),
        'training_time_s': elapsed,
        'exp_name': exp_name,
        'model_config': model_cfg_dict,
        'data_config': data_cfg_dict,
        'training_config': train_cfg_dict,
        'num_features': X_train.shape[-1],
        'history_length': X_train.shape[1],
        'num_detectors': len(np.unique(det_train)),
    }, best_model_path)
    print(f"\n  Best model saved to: {best_model_path}")
    
    # Save FINAL model checkpoint (last epoch state)
    final_model_path = f"{output_dir}/checkpoints/final_model_{exp_name}.pt"
    
    # Get the underlying model if compiled
    if hasattr(model, '_orig_mod'):
        final_state = model._orig_mod.state_dict()
    else:
        final_state = model.state_dict()
    
    torch.save({
        'model_state_dict': final_state,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_val_loss': val_losses[-1],
        'total_epochs': len(train_losses),
        'training_time_s': elapsed,
        'exp_name': exp_name,
        'model_config': model_cfg_dict,
        'data_config': data_cfg_dict,
        'training_config': train_cfg_dict,
        'num_features': X_train.shape[-1],
        'history_length': X_train.shape[1],
        'num_detectors': len(np.unique(det_train)),
    }, final_model_path)
    print(f"  Final model saved to: {final_model_path}")
    
    print(f"\n✓ Training complete!")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best val loss: {min(val_losses):.6f}")
    print(f"  Best epoch: {val_losses.index(min(val_losses)) + 1}")
    print(f"  Results saved to: {output_dir}/")
    
    # Load best state back into model
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)
    
    return model, train_losses, val_losses


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    X_val: np.memmap,
    Y_val: np.memmap,
    det_val: np.ndarray,
    idx_val: np.ndarray,
    val_df: pd.DataFrame,
    forecast_horizon: int,
    eval_spike_threshold: float,
    exp_name: str,
    output_dir: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate model and generate plots."""
    print("\n" + "="*70)
    print(f"EVALUATION: {exp_name}")
    print("="*70)
    
    # Predict
    print("\nGenerating predictions...")
    preds = predict(model, X_val, det_val, device=device, batch_size=4096)
    
    # Prepare eval dataframe
    df_subset = val_df.loc[idx_val].copy()
    
    eval_df = pd.DataFrame({
        "row_idx": idx_val,
        "timestamp": df_subset["timestamp"].values,
        "detector_id": df_subset["detector_id"].values,
    })
    
    # Add predictions
    for h in range(1, forecast_horizon + 1):
        eval_df[f"pred_{h}h"] = preds[:, h-1].numpy()
    
    # Add ground truth
    for h in range(1, forecast_horizon + 1):
        eval_df[f"future_{h}h"] = (
            val_df.groupby("detector_id")["congestion_index"]
                  .shift(-h)
                  .loc[idx_val]
                  .values
        )
    
    eval_df["congestion_index"] = val_df.loc[idx_val, "congestion_index"].values
    eval_df = eval_df.dropna()
    
    # Evaluate
    metrics = evaluate_and_plot_block(
        eval_df,
        horizon=forecast_horizon,
        years=[2019],
        plot_years=[2019],
        filename=exp_name,
        dir=output_dir,
        max_blocks=20,
        eval_spikes=True,
        spike_threshold=eval_spike_threshold
    )
    
    print(f"\n✓ Evaluation complete!")
    if metrics:
        print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  Spike Recall: {metrics.get('spike_recall', 'N/A'):.4f}")
        print(f"  Spike Precision: {metrics.get('spike_precision', 'N/A'):.4f}")
        print(f"  Spike F1: {metrics.get('spike_f1', 'N/A'):.4f}")
    
    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_experiment(config_name: str, device: str = "cuda"):
    """Run a complete experiment with the specified configuration."""
    
    if config_name.lower() == "tcn":
        config = CONFIG_TCN
        exp_name = "TCN_full_scale"
    elif config_name.lower() == "transformer":
        config = CONFIG_TRANSFORMER
        exp_name = "Transformer_full_scale"
    else:
        raise ValueError(f"Unknown config: {config_name}. Use 'tcn' or 'transformer'.")
    
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    
    output_dir = f"plots_training_dl/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {exp_name}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model type: {model_cfg.model_type}")
    print(f"  History hours: {data_cfg.history_hours}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.lr}")
    print(f"  Device: {device}")
    
    # Set precision for faster matmul
    torch.set_float32_matmul_precision('high')
    
    # Prepare data
    (X_train, Y_train, idx_train, det_train,
     X_val, Y_val, idx_val, det_val,
     train_df, val_df, std_scaler, mm_scaler,
     memmap_builder, num_detectors) = prepare_data_memmap(data_cfg)
    
    # Create model
    model = create_model(
        model_cfg,
        num_features=X_train.shape[-1],
        num_detectors=num_detectors,
        forecast_horizon=data_cfg.forecast_horizon,
        device=device,
    )
    
    # Optional: torch.compile for PyTorch 2.0+
    if hasattr(torch, 'compile'):
        print("\nApplying torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Train
    model, train_losses, val_losses = train_full_scale(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        det_train=det_train,
        X_val=X_val,
        Y_val=Y_val,
        det_val=det_val,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        exp_name=exp_name,
        output_dir=output_dir,
        device=device,
    )
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        X_val=X_val,
        Y_val=Y_val,
        det_val=det_val,
        idx_val=idx_val,
        val_df=val_df,
        forecast_horizon=data_cfg.forecast_horizon,
        eval_spike_threshold=train_cfg.eval_spike_threshold,
        exp_name=exp_name,
        output_dir=output_dir,
        device=device,
    )
    
    # Summary
    print("\n" + "="*70)
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Best val loss: {min(val_losses):.6f}")
    print(f"  Best epoch: {val_losses.index(min(val_losses)) + 1}")
    if metrics:
        print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
        print(f"  Spike F1: {metrics.get('spike_f1', 'N/A'):.4f}")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - Best model: {output_dir}/checkpoints/best_model_{exp_name}.pt")
    print(f"  - Final model: {output_dir}/checkpoints/final_model_{exp_name}.pt")
    
    # Cleanup memmap files (optional)
    # memmap_builder.cleanup()
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train full-scale congestion models")
    parser.add_argument(
        "--config",
        type=str,
        choices=["tcn", "transformer", "both"],
        default="both",
        help="Which configuration to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()
    
    if args.config == "both":
        print("\n" + "#"*70)
        print("# RUNNING BOTH CONFIGURATIONS")
        print("#"*70)
        
        # Run TCN first
        print("\n>>> Starting TCN experiment...")
        run_experiment("tcn", device=args.device)
        
        # Clear cache between experiments
        torch.cuda.empty_cache()
        
        # Run Transformer
        print("\n>>> Starting Transformer experiment...")
        run_experiment("transformer", device=args.device)
        
    else:
        run_experiment(args.config, device=args.device)


if __name__ == "__main__":
    main()