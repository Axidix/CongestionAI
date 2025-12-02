import os
import sys

# Add project root to PYTHONPATH automatically
PROJECT_ROOT = r"C:\Users\adib4\OneDrive\Documents\Projets perso\CongestionAI"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from torch.amp import GradScaler
import gc

from src.model_pipelines.dl_pipeline import train_model, evaluate, predict
from src.utils.model_evaluation import evaluate_and_plot_block
from src.utils.preprocessing import cyclical_encode, scale_features, encode_detectors
from src.utils.sequences import create_nhits_sequences, NHitsDataset
from src.utils.plots import plot_training_curves
from src.model_pipelines.losses import LossConfig, create_loss

from src.utils.crafted_features import (
    SpikeFeatureConfig,
    add_spike_features,
    add_lags_and_drop
)

from src.models.tcn_forecaster import MultiHeadTCNForecaster
from src.models.transformer_forecaster import MultiHeadTransformerForecaster

FILE_PATH = "prepared_data/preprocessed_full_data.csv"

# ========================================================
# PYTORCH OPTIMIZATIONS
# ========================================================
torch.set_float32_matmul_precision('high')
torch.set_num_threads(16)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    broad_exp_name="",
):
    """Optimized training pipeline with multi-worker dataloaders."""

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
        batch_size=batch_size * 2,
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

    os.makedirs(f"plots_training_dl/{broad_exp_name}/", exist_ok=True)
    with open(f"plots_training_dl/{broad_exp_name}/losses_{exp_name}.txt", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{t_loss:.6f},{v_loss:.6f}\n")

    plot_training_curves(train_losses, val_losses, filename=f"training_curve_{exp_name}.png", dir=f"plots_training_dl/{broad_exp_name}/")
    
    if test_loader is not None:
        _, test_loss = evaluate(model, test_loader, criterion, device, scaler=scaler)
        print(f"Test Loss ({exp_name}): {test_loss:.4f}")

    return model, train_losses, val_losses


def prepare_eval_df(df, idx_seq, preds, horizon):
    df_subset = df.loc[idx_seq].copy()

    eval_df = pd.DataFrame({
        "row_idx": idx_seq,
        "timestamp": df_subset["timestamp"].values,
        "detector_id": df_subset["detector_id"].values,
    })

    for h in range(1, horizon + 1):
        eval_df[f"pred_{h}h"] = preds[:, h-1].numpy()

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
    
    df_small.loc[(df_small["month"] <= 2) | (df_small["month"] == 12), "season"] = 0
    df_small.loc[(df_small["month"] > 2) & (df_small["month"] <= 5), "season"] = 1
    df_small.loc[(df_small["month"] > 5) & (df_small["month"] <= 8), "season"] = 2
    df_small.loc[(df_small["month"] > 8) & (df_small["month"] <= 11), "season"] = 3
    
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
    
    df_small, det2idx = encode_detectors(df_small)
    
    if "temperature" in feature_cols:
        feature_cols = feature_cols + [f"temperature_lag_{lag}h" for lag in weather_lags] \
            + [f"precipitation_lag_{lag}h" for lag in weather_lags] \
            + [f"visibility_lag_{lag}h" for lag in weather_lags]
    
    train = df_small[df_small["timestamp"].dt.year.isin(years_split[0])].copy()
    val = df_small[df_small["timestamp"].dt.year.isin(years_split[1])].copy()
    test = df_small[df_small["timestamp"].dt.year.isin(years_split[2])].copy() if years_split[2] else None
    
    train = train.set_index("orig_idx")
    val = val.set_index("orig_idx")
    if test is not None:
        test = test.set_index("orig_idx")
    
    minmax_cols = ["lon", "lat", "year", "season"]
    train, val, test, std_scaler, mm_scaler = scale_features(
        train, val, test, feature_cols_norm_full, latlon_cols=minmax_cols
    )
    
    if "temperature" in feature_cols_base:
        train = add_lags_and_drop(train, weather_lags)
        val = add_lags_and_drop(val, weather_lags)
        if test is not None:
            test = add_lags_and_drop(test, weather_lags)
    
    if spike_config is not None:
        spike_cols_in_df = [c for c in spike_feature_cols if c in train.columns]
        train = train.dropna(subset=spike_cols_in_df)
        val = val.dropna(subset=spike_cols_in_df)
        if test is not None:
            test = test.dropna(subset=spike_cols_in_df)
    
    keep_cols = feature_cols + ["timestamp", "detector_id", "det_index", "congestion_index"]
    keep_cols = list(set([c for c in keep_cols if c in train.columns]))
    
    train = train[keep_cols]
    val = val[keep_cols]
    if test is not None:
        test = test[keep_cols]
    
    X_train_hist, Y_train, idx_train, det_train = create_nhits_sequences(
        train, feature_cols, history_offsets, forecast_horizon)
    X_val_hist, Y_val, idx_val, det_val = create_nhits_sequences(
        val, feature_cols, history_offsets, forecast_horizon)
    
    if test is not None:
        X_test_hist, Y_test, idx_test, det_test = create_nhits_sequences(
            test, feature_cols, history_offsets, forecast_horizon)
    else:
        X_test_hist, Y_test, idx_test, det_test = None, None, None, None
    
    # Keep only idx for eval_df reconstruction
    train_info = {"index": train.index.values, "timestamp": train["timestamp"].values, 
                  "detector_id": train["detector_id"].values, "congestion_index": train["congestion_index"].values}
    val_info = {"index": val.index.values, "timestamp": val["timestamp"].values,
                "detector_id": val["detector_id"].values, "congestion_index": val["congestion_index"].values}
    
    del train, val, test, df_small
    gc.collect()
    
    print(f"Sequences created. Features: {len(feature_cols)}, Train samples: {len(Y_train)}")
    
    return (X_train_hist, Y_train, idx_train, det_train,
            X_val_hist, Y_val, idx_val, det_val,
            X_test_hist, Y_test, idx_test, det_test,
            train_info, val_info, std_scaler, mm_scaler, feature_cols)


# ========================================================
# LOAD BASE DATA
# ========================================================
print("=" * 80)
print("LOADING BASE DATA")
print("=" * 80)
df_base = pd.read_csv(FILE_PATH)
df_base["timestamp"] = pd.to_datetime(df_base["timestamp"])
df_base["orig_idx"] = df_base.index
df_base = cyclical_encode(df_base)
print("Base data loaded.")

# ========================================================
# EXPERIMENT CONFIGURATION
# ========================================================
broad_exp_name = "Transformer_vs_TCN_20det_history_scaling"

# DATA CONFIG (frozen)
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

nb_detectors = 20
years_split = [[2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024], [2019], []]
forecast_horizon = 24

# SPIKE CONFIG (frozen)
spike_config = SpikeFeatureConfig(
    enable_deltas=True,
    enable_abs_deltas=False,
    enable_rolling_stats=False,
    delta_lags=[1, 2, 4, 6],
    enable_volatility=True,
    volatility_window=3,
    volatility_binary_threshold=0.04
)

# WEATHER LAGS (frozen)
w_lags = [0, -3, -6, -12, -24]

# LOSS CONFIG (frozen)
spike_trigger_threshold = 0.15
eval_spike_threshold = 0.38
cfg_loss = LossConfig(
    loss_type="spike_weighted",
    spike_weight=3.0,
    spike_threshold=spike_trigger_threshold
)

# TRAINING CONFIG
batch_size = 1024
num_workers = 0
use_amp = True
use_compile = True
grad_clip = 1.0

# ========================================================
# EXPERIMENT DEFINITIONS
# ========================================================
# Model complexity scaled to history length:
# - 24h: BASELINE config (fast, sanity check)
# - 48h: BASELINE config (still manageable)
# - 72h: RECOMMENDED config (sweet spot)
# - 96h: HIGH-CAPACITY config (max performance)

experiments = [
    # ============================================
    # 0. GPU SATURATION TEST - Heaviest config first
    # ============================================
    #{
    #    "name": "00_GPU_TEST_Transformer_H96_heavy",
    #    "model_type": "transformer",
    #    "history_hours": 96,
    #    "epochs": 1,  # Short test to check memory/speed
    #   "config": {
    #        "d_model": 256,
    #        "n_heads": 8,
    #        "num_layers": 8,
    #        "dim_feedforward": 1024,
    #        "dropout": 0.2,
    #        "det_emb_dim": 128,
    #        "pooling": "avg",
    #    },
    #    "optim_config": {"lr": 5e-5, "weight_decay": 1e-2, "betas": (0.9, 0.98)},
    #},
    
    # ============================================
    # 1. TCN BASELINES (best previous config)
    # ============================================
    {
        "name": "01_TCN_H24",
        "model_type": "tcn",
        "history_hours": 24,
        "epochs": 15,
        "config": {
            "emb_dim": 256,
            "num_channels": (128, 256, 256),
            "kernel_size": 5,
            "dropout_encoder": 0.25,
            "dropout_heads": 0.25,
            "use_se": False,
            "pooling": "last",
        },
        "optim_config": {"lr": 3e-4, "weight_decay": 1e-3, "betas": (0.9, 0.99)},
    },
    
    # ============================================
    # 2. TRANSFORMER BASELINE (48h) - Fast config
    # ============================================
    {
        "name": "06_Transformer_BASELINE_H48",
        "model_type": "transformer",
        "history_hours": 48,
        "epochs": 25,
        "config": {
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "det_emb_dim": 64,
            "pooling": "last",
        },
        "optim_config": {"lr": 1e-4, "weight_decay": 1e-2, "betas": (0.9, 0.98)},
    },
    
    # ============================================
    # 3. TRANSFORMER RECOMMENDED (72h) - Sweet spot
    # ============================================
    {
        "name": "07_Transformer_RECOMMENDED_H72",
        "model_type": "transformer",
        "history_hours": 72,
        "epochs": 25,
        "config": {
            "d_model": 192,
            "n_heads": 6,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.2,
            "det_emb_dim": 96,
            "pooling": "avg",
        },
        "optim_config": {"lr": 8e-5, "weight_decay": 1e-2, "betas": (0.9, 0.98)},
    },
    {
        "name": "08_Transformer_RECOMMENDED_H96",
        "model_type": "transformer",
        "history_hours": 96,
        "epochs": 25,
        "config": {
            "d_model": 192,
            "n_heads": 6,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.2,
            "det_emb_dim": 96,
            "pooling": "avg",
        },
        "optim_config": {"lr": 8e-5, "weight_decay": 1e-2, "betas": (0.9, 0.98)},
    },
    
    # ============================================
    # 4. TRANSFORMER HIGH-CAPACITY (96h) - Max performance
    # ============================================
    {
        "name": "09_Transformer_HIGHCAP_H96",
        "model_type": "transformer",
        "history_hours": 96,
        "epochs": 30,
        "config": {
            "d_model": 256,
            "n_heads": 8,
            "num_layers": 8,
            "dim_feedforward": 1024,
            "dropout": 0.2,
            "det_emb_dim": 128,
            "pooling": "avg",
        },
        "optim_config": {"lr": 5e-5, "weight_decay": 1e-2, "betas": (0.9, 0.98)},
    },
    
    # ============================================
    # 5. ABLATION: Pooling comparison at 72h
    # ============================================
    {
        "name": "10_Transformer_RECOMMENDED_H72_poolLast",
        "model_type": "transformer",
        "history_hours": 72,
        "epochs": 25,
        "config": {
            "d_model": 192,
            "n_heads": 6,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.2,
            "det_emb_dim": 96,
            "pooling": "last",
        },
        "optim_config": {"lr": 8e-5, "weight_decay": 1e-2, "betas": (0.9, 0.98)},
    },
    {
        "name": "11_Transformer_RECOMMENDED_H72_poolAttn",
        "model_type": "transformer",
        "history_hours": 72,
        "epochs": 25,
        "config": {
            "d_model": 192,
            "n_heads": 6,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.2,
            "det_emb_dim": 96,
            "pooling": "attention",
        },
        "optim_config": {"lr": 8e-5, "weight_decay": 1e-2, "betas": (0.9, 0.98)},
    },
]

# ========================================================
# RUN EXPERIMENTS
# ========================================================
results = []

os.makedirs(f"plots_training_dl/{broad_exp_name}/", exist_ok=True)

for exp in experiments:
    exp_name = exp["name"]
    history_hours = exp["history_hours"]
    h_offsets = list(range(history_hours))
    epochs = exp["epochs"]
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    print(f"  Model type: {exp['model_type']}")
    print(f"  History: {history_hours}h")
    print(f"  Epochs: {epochs}")
    print(f"  Config: {exp['config']}")
    print(f"  Optim: {exp['optim_config']}")
    
    start_time = time.time()
    
    # -------------------------
    # PREPARE DATA (fresh each time - reload df_base too)
    # -------------------------
    print(f"\n  [Data] Preparing data for {history_hours}h history...")
    data_start = time.time()
    
    # Reload df_base fresh for each experiment to ensure clean memory
    df_base = pd.read_csv(FILE_PATH)
    df_base["timestamp"] = pd.to_datetime(df_base["timestamp"])
    df_base["orig_idx"] = df_base.index
    df_base = cyclical_encode(df_base)
    
    (X_train_hist, Y_train, idx_train, det_train,
     X_val_hist, Y_val, idx_val, det_val,
     X_test_hist, Y_test, idx_test, det_test,
     train_info, val_info, std_scaler, mm_scaler, feature_cols) = prepare_dl_data_with_spikes(
        h_offsets, forecast_horizon, nb_detectors, df_base,
        feature_cols_norm=feature_cols_norm_base,
        feature_cols_base=feature_cols_base,
        weather_lags=w_lags,
        years_split=years_split,
        spike_config=spike_config
    )
    
    # Free df_base immediately after data prep
    del df_base
    gc.collect()
    
    print(f"  [Data] Prepared in {time.time() - data_start:.1f}s")
    
    num_features = X_train_hist.shape[-1]
    print(f"  [Data] Train: {len(Y_train)}, Val: {len(Y_val)}, Features: {num_features}")
    
    # -------------------------
    # CREATE MODEL
    # -------------------------
    try:
        if exp["model_type"] == "transformer":
            exp["config"]["max_seq_len"] = max(history_hours + 16, 128)
            model = MultiHeadTransformerForecaster(
                num_features=num_features,
                horizon=forecast_horizon,
                num_detectors=nb_detectors,
                **exp["config"]
            )
        elif exp["model_type"] == "tcn":
            model = MultiHeadTCNForecaster(
                num_features=num_features,
                horizon=forecast_horizon,
                num_detectors=nb_detectors,
                **exp["config"]
            )
        else:
            raise ValueError(f"Unknown model type: {exp['model_type']}")
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [Model] Parameters: {num_params:,}")
        
        # GPU memory check before training
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"  [GPU] Initial memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Apply torch.compile (skip for GPU test)
        if use_compile and hasattr(torch, 'compile') and "GPU_TEST" not in exp_name:
            print("  [Model] Applying torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
        
        # -------------------------
        # CREATE OPTIMIZER & SCHEDULER
        # -------------------------
        optim_cfg = exp["optim_config"]
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=optim_cfg["lr"],
            weight_decay=optim_cfg["weight_decay"],
            betas=optim_cfg["betas"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=1e-6
        )
        
        scaler = GradScaler('cuda') if use_amp else None
        criterion = create_loss(cfg_loss)
        
        # -------------------------
        # TRAIN
        # -------------------------
        print(f"\n  [Training] Starting {epochs} epochs...")
        train_start = time.time()
        
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
            broad_exp_name=broad_exp_name,
        )
        
        train_time = time.time() - train_start
        time_per_epoch = train_time / epochs
        print(f"  [Training] Completed in {train_time:.1f}s ({time_per_epoch:.1f}s/epoch)")
        
        # GPU peak memory
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [GPU] Peak memory: {peak_mem:.2f} GB")
        else:
            peak_mem = 0
        
        # -------------------------
        # EVALUATE
        # -------------------------
        print(f"  [Eval] Running evaluation...")
        preds = predict(model, X_val_hist, det_val)
        
        # Convert preds to numpy if it's a tensor
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        
        # Build eval_df directly (Y_val contains future targets)
        eval_df = pd.DataFrame({
            "row_idx": idx_val,
            "timestamp": val_info["timestamp"][np.searchsorted(val_info["index"], idx_val)],
            "detector_id": val_info["detector_id"][np.searchsorted(val_info["index"], idx_val)],
            "congestion_index": val_info["congestion_index"][np.searchsorted(val_info["index"], idx_val)],
        })
        
        # Add predictions (preds is now numpy)
        for h in range(1, forecast_horizon + 1):
            eval_df[f"pred_{h}h"] = preds[:, h-1]
        
        # Add ground truth targets (Y_val is already numpy)
        for h in range(1, forecast_horizon + 1):
            eval_df[f"future_{h}h"] = Y_val[:, h-1]
        
        metrics = evaluate_and_plot_block(
            eval_df, horizon=forecast_horizon, years=[2019], plot_years=[2019],
            filename=exp_name, dir=f"plots_training_dl/{broad_exp_name}/",
            max_blocks=15, eval_spikes=True, spike_threshold=eval_spike_threshold
        )
        
        elapsed = time.time() - start_time
        
        # -------------------------
        # BUILD RESULT
        # -------------------------
        result = {
            "exp_name": exp_name,
            "model_type": exp["model_type"],
            "history_hours": history_hours,
            "num_params": num_params,
            "train_samples": len(Y_train),
            "val_samples": len(Y_val),
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": min(val_losses),
            "best_epoch": val_losses.index(min(val_losses)) + 1,
            "training_time_s": round(train_time, 1),
            "time_per_epoch_s": round(time_per_epoch, 1),
            "peak_gpu_gb": round(peak_mem, 2),
            "total_time_s": round(elapsed, 1),
        }
        
        # Add config details
        for k, v in exp["config"].items():
            result[f"cfg_{k}"] = str(v) if isinstance(v, tuple) else v
        
        # Add metrics
        if metrics is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    result[k] = v
        
        results.append(result)
        
        # Print summary
        print(f"\n  ✓ {exp_name} COMPLETE")
        print(f"    Parameters: {num_params:,}")
        print(f"    Best val loss: {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")
        print(f"    Time: {train_time:.1f}s total, {time_per_epoch:.1f}s/epoch")
        print(f"    Peak GPU: {peak_mem:.2f} GB")
        if metrics:
            print(f"    Spike Recall: {metrics.get('spike_recall', 'N/A')}")
            print(f"    Spike F1: {metrics.get('spike_f1', 'N/A')}")
            print(f"    Delta Corr: {metrics.get('delta_corr', 'N/A')}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ✗ FAILED: {exp_name}")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "exp_name": exp_name,
            "model_type": exp["model_type"],
            "history_hours": history_hours,
            "error": str(e),
            "total_time_s": round(elapsed, 1)
        })
    
    # Save intermediate results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"plots_training_dl/{broad_exp_name}/results_intermediate.csv", index=False)
    
    # Aggressive cleanup after each experiment
    torch.cuda.empty_cache()
    try:
        del model, optim, scheduler, criterion
        del X_train_hist, Y_train, idx_train, det_train
        del X_val_hist, Y_val, idx_val, det_val
        del X_test_hist, Y_test, idx_test, det_test
        del train_info, val_info
        del preds, eval_df
        del std_scaler, mm_scaler, feature_cols
    except NameError:
        pass
    gc.collect()
    
    # Log memory status
    import psutil
    mem = psutil.Process().memory_info().rss / 1e9
    print(f"  [RAM] After cleanup: {mem:.2f} GB")

# ========================================================
# FINAL RESULTS SUMMARY
# ========================================================
print("\n\n" + "=" * 100)
print("TRANSFORMER vs TCN EXPERIMENTS COMPLETE")
print("=" * 100)

results_df = pd.DataFrame(results)
results_df.to_csv(f"plots_training_dl/{broad_exp_name}/results_final.csv", index=False)

# Display summary table
display_cols = [
    "exp_name", "model_type", "history_hours", "num_params", 
    "best_val_loss", "best_epoch", "time_per_epoch_s", "peak_gpu_gb"
]
display_cols = [c for c in display_cols if c in results_df.columns]
print("\n--- Performance Summary ---")
print(results_df[display_cols].to_string())

# Spike metrics
spike_cols = ["exp_name", "model_type", "history_hours", "spike_recall", "spike_precision", "spike_f1", "delta_corr"]
spike_cols = [c for c in spike_cols if c in results_df.columns]
if len(spike_cols) > 3:
    print("\n--- Spike Metrics ---")
    print(results_df[spike_cols].to_string())

# Best per model type
print("\n--- Best Config per Model Type ---")
for model_type in ["tcn", "transformer"]:
    subset = results_df[results_df["model_type"] == model_type]
    if len(subset) > 0 and "best_val_loss" in subset.columns:
        best_idx = subset["best_val_loss"].idxmin()
        if pd.notna(best_idx):
            best = subset.loc[best_idx]
            print(f"\n{model_type.upper()}:")
            print(f"  Best: {best['exp_name']}")
            print(f"  Val Loss: {best['best_val_loss']:.4f}")
            print(f"  History: {best['history_hours']}h")
            if "spike_f1" in best:
                print(f"  Spike F1: {best['spike_f1']}")

print(f"\nResults saved to: plots_training_dl/{broad_exp_name}/results_final.csv")