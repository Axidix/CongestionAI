import os
import sys

# Add project root to PYTHONPATH automatically
PROJECT_ROOT = r"C:\Users\adib4\OneDrive\Documents\Projets perso\CongestionAI\find_issues.ipynb"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.model_pipelines.dl_pipeline import train_model, evaluate, predict
from src.utils.model_evaluation import evaluate_and_plot_block
from src.utils.hist_baseline import historical_baseline_multi
from src.utils.preprocessing import cyclical_encode, scale_features, encode_detectors
from src.utils.sequences import create_nhits_sequences, NHitsDataset
from src.utils.plots import plot_training_curves
from src.model_pipelines.losses import (
    SpikeWeightedMSELoss,
    TwoTermSpikeLoss,
    DeltaLoss,
    LossConfig,
    create_loss
)

from src.utils.crafted_features import (
    SpikeFeatureConfig,
    add_spike_features,
    add_lags_and_drop
)



from src.models.n_hits import NHitsForecaster
from src.models.tcn_forecaster import MultiHeadTCNForecaster

FILE_PATH = "prepared_data/preprocessed_full_data.csv"


def run_dl_experiment_optimized(
    model,
    optimizer,
    criterion,
    X_train_hist,
    Y_train,
    train_det_idx,
    X_val_hist,
    Y_val,
    val_det_idx,
    X_test_hist,
    Y_test,
    test_det_idx,
    device="cuda",
    batch_size=512,
    epochs=10,
    grad_clip=1.0,
    scheduler=None,
    scaler=None,
    exp_name="",
    patience=None,
    num_workers=4,
):
    """Optimized training pipeline with multi-worker dataloaders."""

    # -------------------------
    # OPTIMIZED DATALOADERS
    # -------------------------
    train_loader = DataLoader(
        NHitsDataset(X_train_hist, Y_train, train_det_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        NHitsDataset(X_val_hist, Y_val, val_det_idx),
        batch_size=batch_size * 2,  # Larger batch for eval (no gradients)
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if X_test_hist is None or Y_test is None or test_det_idx is None:
        test_loader = None
    else:
        test_loader = DataLoader(
            NHitsDataset(X_test_hist, Y_test, test_det_idx),
            batch_size=batch_size * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

    model.to(device)

    # -------------------------
    # TRAINING
    # -------------------------
    train_losses, val_losses, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=epochs,
        grad_clip=grad_clip,
        patience=patience,
    )

    # Save losses to file
    os.makedirs(f"plots_training_dl/{broad_exp_name}/", exist_ok=True)
    with open(f"plots_training_dl/{broad_exp_name}/losses_{exp_name}.txt", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{t_loss:.6f},{v_loss:.6f}\n")

    plot_training_curves(train_losses, val_losses, filename=f"training_curve{exp_name}.png", dir=f"plots_training_dl/{broad_exp_name}/")
    
    if test_loader is not None:
        _, test_loss = evaluate(model, test_loader, criterion, device, scaler=scaler)
        print(f"Test Loss ({exp_name}): {test_loss:.4f}")

    return model, train_losses, val_losses


def prepare_eval_df(df, idx_seq, preds, horizon):
    """
    df: the dataset (train, val or test)
    idx_seq: array of starting indices returned by create_nhits_sequences
    preds: model predictions (N, horizon)
    """

    df_subset = df.loc[idx_seq].copy()
    print(df.info())
    print(df_subset.info())
    print(len(df_subset), len(idx_seq), preds.shape)

    eval_df = pd.DataFrame({
        "row_idx": idx_seq,
        "timestamp": df_subset["timestamp"].values,
        "detector_id": df_subset["detector_id"].values,
    })
    

    # Add predictions
    for h in range(1, horizon + 1):
        eval_df[f"pred_{h}h"] = preds[:, h-1].numpy()

    # Add ground truth targets
    for h in range(1, horizon + 1):
        eval_df[f"future_{h}h"] = (
            df.groupby("detector_id")["congestion_index"]
              .shift(-h)
              .loc[idx_seq]
              .values
        )

    return eval_df.dropna()



def prepare_dl_data_with_spikes(history_offsets, forecast_horizon, nb_detectors, df_base,
                                years_split, feature_cols_norm, feature_cols_base,
                                weather_lags, spike_config=None):
    """Extended data prep with optional spike features."""
    
    print("Loading data...")
    df_small = df_base[df_base["detector_id"].isin(df_base["detector_id"].unique()[:nb_detectors])].copy()
    df_small = df_small.sort_values(["detector_id", "timestamp"])
    
    # Season encoding
    df_small.loc[(df_small["month"] <= 2) | (df_small["month"] == 12), "season"] = 0
    df_small.loc[(df_small["month"] > 2) & (df_small["month"] <= 5), "season"] = 1
    df_small.loc[(df_small["month"] > 5) & (df_small["month"] <= 8), "season"] = 2
    df_small.loc[(df_small["month"] > 8) & (df_small["month"] <= 11), "season"] = 3
    
    # Add spike features if configured
    feature_cols = feature_cols_base.copy()
    feature_cols_norm_full = feature_cols_norm.copy()
    
    if spike_config is not None:
        print(f"Adding spike features: deltas={spike_config.enable_deltas}, rolling={spike_config.enable_rolling_stats}")
        df_small = add_spike_features(df_small, spike_config)
        spike_feature_cols = spike_config.get_feature_columns()
        spike_norm_cols = spike_config.get_normalization_columns()
        feature_cols = feature_cols + spike_feature_cols
        feature_cols_norm_full = feature_cols_norm_full + spike_norm_cols
        print(f"  Added columns: {spike_feature_cols}")
    
    # Detector encoding
    df_small, det2idx = encode_detectors(df_small)
    
    # Add weather lag column names
    if "temperature" in feature_cols:
        feature_cols = feature_cols + [f"temperature_lag_{lag}h" for lag in weather_lags] \
            + [f"precipitation_lag_{lag}h" for lag in weather_lags] \
            + [f"visibility_lag_{lag}h" for lag in weather_lags]
    
    # Split
    train = df_small[df_small["timestamp"].dt.year.isin(years_split[0])].copy()
    val = df_small[df_small["timestamp"].dt.year.isin(years_split[1])].copy()
    test = df_small[df_small["timestamp"].dt.year.isin(years_split[2])].copy() if years_split[2] else None
    
    train = train.set_index("orig_idx")
    val = val.set_index("orig_idx")
    if test is not None:
        test = test.set_index("orig_idx")
    
    # Normalization
    minmax_cols = ["lon", "lat", "year", "season"]
    train, val, test, std_scaler, mm_scaler = scale_features(
        train, val, test, feature_cols_norm_full, latlon_cols=minmax_cols
    )
    
    # Weather lags
    if "temperature" in feature_cols_base:
        train = add_lags_and_drop(train, weather_lags)
        val = add_lags_and_drop(val, weather_lags)
        if test is not None:
            test = add_lags_and_drop(test, weather_lags)
    
    # Drop NaNs from spike features
    if spike_config is not None:
        spike_cols_in_df = [c for c in spike_feature_cols if c in train.columns]
        train = train.dropna(subset=spike_cols_in_df)
        val = val.dropna(subset=spike_cols_in_df)
        if test is not None:
            test = test.dropna(subset=spike_cols_in_df)
    
    # Keep only needed columns (congestion_index is already in feature_cols)
    keep_cols = feature_cols + ["timestamp", "detector_id", "det_index"]
    keep_cols = [c for c in keep_cols if c in train.columns]
    
    train = train[keep_cols]
    val = val[keep_cols]
    if test is not None:
        test = test[keep_cols]
    
    # Build sequences
    X_train_hist, Y_train, idx_train, det_train = create_nhits_sequences(
        train, feature_cols, history_offsets, forecast_horizon)
    X_val_hist, Y_val, idx_val, det_val = create_nhits_sequences(
        val, feature_cols, history_offsets, forecast_horizon)
    
    if test is not None:
        X_test_hist, Y_test, idx_test, det_test = create_nhits_sequences(
            test, feature_cols, history_offsets, forecast_horizon)
    else:
        X_test_hist, Y_test, idx_test, det_test = None, None, None, None
    
    print(f"Sequences created. Features: {len(feature_cols)}, Train samples: {len(Y_train)}")
    
    return (X_train_hist, Y_train, idx_train, det_train,
            X_val_hist, Y_val, idx_val, det_val,
            X_test_hist, Y_test, idx_test, det_test,
            train, val, test, std_scaler, mm_scaler)



df_base = pd.read_csv(FILE_PATH)
df_base["timestamp"] = pd.to_datetime(df_base["timestamp"])
df_base["orig_idx"] = df_base.index
df_base = cyclical_encode(df_base)



# Experiment: Model capacity scaling at 100 detectors

import time
import torch
from torch.amp import GradScaler

torch.set_float32_matmul_precision('high')

broad_exp_name = "TCN-100det_24horizon-capacity_scaling"

# ========================================================
# DATA CONFIGURATION
# ========================================================
feature_cols_norm_base = [
    "temperature", "precipitation", "visibility", "congestion_index", "free_flow_speed"
]
feature_cols_base = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lon", "lat", "year", "season",
    "temperature", "precipitation", "visibility",
    "congestion_index", "free_flow_speed",
    "is_weekend", "is_holiday", "is_school_holiday", "is_rush_hour", "is_snow", "is_fog"
]

nb_detectors = 100
years_split = [[2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024], [2019], []]
forecast_horizon = 24
epochs = 15

# ========================================================
# SPIKE FEATURE CONFIGURATION
# ========================================================
spike_config = SpikeFeatureConfig(
    enable_deltas=True,
    enable_abs_deltas=False,
    enable_rolling_stats=False,
    delta_lags=[1, 2, 4, 6],
    enable_volatility=True,
    volatility_window=3,
    volatility_binary_threshold=0.04
)

# ========================================================
# HISTORY & WEATHER LAGS
# ========================================================
h_offsets = list(range(24))
w_lags = [0, -3, -6, -12, -24]

# ========================================================
# LOSS CONFIGURATION
# ========================================================
spike_trigger_threshold = 0.15
eval_spike_threshold = 0.38
cfg_loss = LossConfig(
    loss_type="spike_weighted",
    spike_weight=3.0,
    spike_threshold=spike_trigger_threshold
)

# ========================================================
# FIXED TRAINING CONFIG
# ========================================================
optim_config = {"type": "adamW", "lr": 3e-4, "weight_decay": 1e-3, "betas": (0.9, 0.99)}
schedule_config = {"type": "CosineAnnealingLR", "T_max": epochs, "eta_min": 1e-6}
grad_clip = 0.5
batch_size = 2048
num_workers = 4
use_amp = True
use_compile = True

# ========================================================
# EXPERIMENT DEFINITIONS
# ========================================================
experiments = [
    {
        "name": "1_CONTROL_baseline",
        "emb_dim": 256,
        "num_channels": (128, 256, 256),
        "dropout_encoder": 0.25,
        "dropout_heads": 0.25,
    },
    {
        "name": "2_WIDER_large_tcn",
        "emb_dim": 384,
        "num_channels": (256, 384, 384),
        "dropout_encoder": 0.25,
        "dropout_heads": 0.25,
    },
    {
        "name": "3_DEEPER_more_layers",
        "emb_dim": 256,
        "num_channels": (128, 256, 256, 256),
        "dropout_encoder": 0.25,
        "dropout_heads": 0.25,
    },
    {
        "name": "4_WIDE_DEEP_max_capacity",
        "emb_dim": 384,
        "num_channels": (256, 384, 384, 384),
        "dropout_encoder": 0.25,
        "dropout_heads": 0.25,
    },
]

# ========================================================
# PREPARE DATA ONCE (shared across all experiments)
# ========================================================
print("=" * 70)
print("PREPARING DATA (shared across all experiments)")
print("=" * 70)

data_start = time.time()
X_train_hist, Y_train, idx_train, det_train, \
X_val_hist, Y_val, idx_val, det_val, \
X_test_hist, Y_test, idx_test, det_test, \
train, val, test, \
std_scaler, mm_scaler = prepare_dl_data_with_spikes(
    h_offsets, forecast_horizon, nb_detectors, df_base,
    feature_cols_norm=feature_cols_norm_base,
    feature_cols_base=feature_cols_base,
    weather_lags=w_lags,
    years_split=years_split,
    spike_config=spike_config
)
print(f"Data prep time: {time.time() - data_start:.1f}s")
print(f"Train samples: {len(Y_train)}, Val samples: {len(Y_val)}")

# ========================================================
# RUN EXPERIMENTS
# ========================================================
results = []

for exp in experiments:
    exp_name = exp["name"]
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}")
    print(f"  emb_dim: {exp['emb_dim']}")
    print(f"  num_channels: {exp['num_channels']}")
    print(f"  dropout: {exp['dropout_encoder']}")
    
    start_time = time.time()
    criterion = create_loss(cfg_loss)
    
    # Build model config
    model_config = {
        "horizon": forecast_horizon,
        "num_detectors": nb_detectors,
        "emb_dim": exp["emb_dim"],
        "num_channels": exp["num_channels"],
        "kernel_size": 5,
        "dropout_encoder": exp["dropout_encoder"],
        "dropout_heads": exp["dropout_heads"],
        "use_se": False,
        "pooling": "last"
    }
    
    try:
        # Create model
        model = MultiHeadTCNForecaster(**model_config, num_features=X_train_hist.shape[-1])
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {num_params:,}")
        
        # Apply torch.compile
        if use_compile and hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")
        
        # Create optimizer
        optim = torch.optim.AdamW(
            model.parameters(), 
            lr=optim_config["lr"], 
            weight_decay=optim_config["weight_decay"], 
            betas=optim_config["betas"]
        )
        
        # Create scheduler
        sched_class = getattr(torch.optim.lr_scheduler, schedule_config["type"])
        sched_params = {k: v for k, v in schedule_config.items() if k != "type"}
        scheduler = sched_class(optim, **sched_params)
        
        # Create AMP scaler
        scaler = GradScaler('cuda') if use_amp else None
        
        # Run experiment
        model, train_losses, val_losses = run_dl_experiment_optimized(
            model=model,
            optimizer=optim,
            criterion=criterion,
            X_train_hist=X_train_hist,
            Y_train=Y_train,
            train_det_idx=det_train,
            X_val_hist=X_val_hist,
            Y_val=Y_val,
            val_det_idx=det_val,
            X_test_hist=X_test_hist,
            Y_test=Y_test,
            test_det_idx=det_test,
            device="cuda",
            batch_size=batch_size,
            epochs=epochs,
            grad_clip=grad_clip,
            scheduler=scheduler,
            scaler=scaler,
            exp_name=exp_name,
            num_workers=num_workers,
        )
        
        # Evaluate
        eval_df = prepare_eval_df(val, idx_val, predict(model, X_val_hist, det_val), forecast_horizon)
        eval_df["congestion_index"] = val.loc[idx_val, "congestion_index"].values
        metrics = evaluate_and_plot_block(
            eval_df, horizon=forecast_horizon, years=[2019], plot_years=[2019],
            filename=exp_name, dir=f"plots_training_dl/{broad_exp_name}/",
            max_blocks=15, eval_spikes=True, spike_threshold=eval_spike_threshold
        )
        
        elapsed = time.time() - start_time
        
        # Build result
        result = {
            "exp_name": exp_name,
            "emb_dim": exp["emb_dim"],
            "num_channels": str(exp["num_channels"]),
            "num_params": num_params,
            "train_samples": len(Y_train),
            "val_samples": len(Y_val),
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": min(val_losses),
            "best_epoch": val_losses.index(min(val_losses)) + 1,
            "training_time_s": round(elapsed, 1),
        }
        
        # Add metrics
        if metrics is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    result[k] = v
        
        results.append(result)
        
        print(f"\n✓ {exp_name} completed in {elapsed:.1f}s")
        print(f"  Parameters: {num_params:,}")
        print(f"  best_val_loss: {result['best_val_loss']:.4f}")
        print(f"  best_epoch: {result['best_epoch']}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ FAILED: {exp_name} - {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "exp_name": exp_name, 
            "error": str(e),
            "training_time_s": round(elapsed, 1)
        })
    
    torch.cuda.empty_cache()

# ========================================================
# SAVE RESULTS
# ========================================================
os.makedirs(f"plots_training_dl/{broad_exp_name}/", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(f"plots_training_dl/{broad_exp_name}/capacity_scaling_results.csv", index=False)

print("\n\n" + "=" * 80)
print("CAPACITY SCALING EXPERIMENTS COMPLETE")
print("=" * 80)

display_cols = ["exp_name", "emb_dim", "num_channels", "num_params", "best_val_loss", "best_epoch", "training_time_s"]
display_cols = [c for c in display_cols if c in results_df.columns]
print(results_df[display_cols].to_string())

# Show spike metrics
spike_cols = ["exp_name", "spike_recall", "spike_precision", "spike_f1"]
spike_cols = [c for c in spike_cols if c in results_df.columns]
if len(spike_cols) > 1:
    print("\n--- Spike Metrics ---")
    print(results_df[spike_cols].to_string())

print(f"\nResults saved to: plots_training_dl/{broad_exp_name}/capacity_scaling_results.csv")

