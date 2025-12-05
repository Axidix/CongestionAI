"""
Comprehensive parameter search experiment runner.
Runs 20 experiments with different configurations and compiles results.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.configs import DataConfig, TCNModelConfig, TrainingConfig
from src.utils.preprocessing import prepare_data_memmap
from src.model_pipelines.dl_pipeline import create_model, train_full_scale
from src.utils.model_evaluation import evaluate_model


# ============================================================================
# EXTENDED CONFIGS FOR PARAMETER SEARCH
# ============================================================================

@dataclass
class ExtendedDataConfig(DataConfig):
    """Extended data config with congestion lags."""
    congestion_lags: Tuple[int, ...] = ()  # e.g., (168,) for weekly lag


@dataclass 
class ExtendedTrainingConfig(TrainingConfig):
    """Extended training config with optimizer/scheduler options."""
    epochs: int = 8
    optimizer_type: str = "adamw"  # "adamw", "radam", "lion"
    scheduler_type: str = "cosine"  # "cosine", "onecycle", "cosine_warm"
    
    # OneCycleLR params
    onecycle_max_lr: float = 5e-4
    onecycle_div_factor: float = 10.0
    onecycle_final_div_factor: float = 100.0
    
    # CosineAnnealingWarmRestarts params
    cosine_warm_t0: int = 4
    cosine_warm_tmult: int = 2


# ============================================================================
# BASE CONFIGURATION (shared by all experiments)
# ============================================================================

def get_base_data_config(**overrides) -> ExtendedDataConfig:
    """Get base data config with optional overrides."""
    config = ExtendedDataConfig(
        history_hours=48,
        weather_lags=(0, -3, -6, -12, -24),
        delta_lags=(1, 2, 4, 6),
        volatility_threshold=0.04,
        congestion_lags=(),
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_base_model_config(**overrides) -> TCNModelConfig:
    """Get base model config with optional overrides."""
    config = TCNModelConfig(
        emb_dim=256,
        num_channels=(128, 256, 256, 256, 256),
        kernel_size=5,
        dropout_encoder=0.20,
        dropout_heads=0.20,
        use_se=False,
        pooling="last",
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_base_training_config(**overrides) -> ExtendedTrainingConfig:
    """Get base training config with optional overrides."""
    config = ExtendedTrainingConfig(
        epochs=8,
        batch_size=2048,
        lr=3e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.99),
        grad_clip=0.5,
        use_amp=True,
        patience=None,  # No early stopping for fair comparison
        spike_weight=3.0,
        spike_threshold=0.15,
        optimizer_type="adamw",
        scheduler_type="cosine",
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


# ============================================================================
# ALL 20 EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENTS = {
    # -------------------------------------------------------------------------
    # 1️⃣ Spike-loss variants (3 experiments)
    # -------------------------------------------------------------------------
    "EXP_spike_sw2_thr015": {
        "data": get_base_data_config(),
        "model": get_base_model_config(),
        "training": get_base_training_config(spike_weight=2.0, spike_threshold=0.15),
    },
    "EXP_spike_sw3_thr010": {
        "data": get_base_data_config(),
        "model": get_base_model_config(),
        "training": get_base_training_config(spike_weight=3.0, spike_threshold=0.10),
    },
    "EXP_spike_sw4_thr015": {
        "data": get_base_data_config(),
        "model": get_base_model_config(),
        "training": get_base_training_config(spike_weight=4.0, spike_threshold=0.15),
    },
    
    # -------------------------------------------------------------------------
    # 2️⃣ Lag-based feature variants (4 experiments)
    # -------------------------------------------------------------------------
    "EXP_lags_week_168h": {
        "data": get_base_data_config(congestion_lags=(168,)),
        "model": get_base_model_config(),
        "training": get_base_training_config(),
    },
    "EXP_lags_week_168h_336h": {
        "data": get_base_data_config(congestion_lags=(168, 336)),
        "model": get_base_model_config(),
        "training": get_base_training_config(),
    },
    "EXP_lags_halfweek_84h": {
        "data": get_base_data_config(congestion_lags=(84,)),
        "model": get_base_model_config(),
        "training": get_base_training_config(),
    },
    "EXP_lags_multiscale_48_168": {
        "data": get_base_data_config(congestion_lags=(48, 168)),
        "model": get_base_model_config(),
        "training": get_base_training_config(),
    },
    
    # -------------------------------------------------------------------------
    # 3️⃣ Architecture variations (6 experiments)
    # -------------------------------------------------------------------------
    "EXP_arch_deeper6": {
        "data": get_base_data_config(),
        "model": get_base_model_config(num_channels=(128, 256, 256, 256, 256, 256)),
        "training": get_base_training_config(),
    },
    "EXP_arch_narrow": {
        "data": get_base_data_config(),
        "model": get_base_model_config(num_channels=(128, 192, 192, 192, 192)),
        "training": get_base_training_config(),
    },
    "EXP_arch_wide": {
        "data": get_base_data_config(),
        "model": get_base_model_config(num_channels=(128, 256, 256, 512, 512)),
        "training": get_base_training_config(),
    },
    "EXP_arch_kernel7": {
        "data": get_base_data_config(),
        "model": get_base_model_config(kernel_size=7),
        "training": get_base_training_config(),
    },
    "EXP_arch_kernel9": {
        "data": get_base_data_config(),
        "model": get_base_model_config(kernel_size=9),
        "training": get_base_training_config(),
    },
    "EXP_arch_high_emb384": {
        "data": get_base_data_config(),
        "model": get_base_model_config(emb_dim=384),
        "training": get_base_training_config(),
    },
    
    # -------------------------------------------------------------------------
    # 4️⃣ Dropout variants (3 experiments)
    # -------------------------------------------------------------------------
    "EXP_drop_015": {
        "data": get_base_data_config(),
        "model": get_base_model_config(dropout_encoder=0.15, dropout_heads=0.15),
        "training": get_base_training_config(),
    },
    "EXP_drop_030": {
        "data": get_base_data_config(),
        "model": get_base_model_config(dropout_encoder=0.30, dropout_heads=0.30),
        "training": get_base_training_config(),
    },
    "EXP_drop_asymmetric": {
        "data": get_base_data_config(),
        "model": get_base_model_config(dropout_encoder=0.25, dropout_heads=0.35),
        "training": get_base_training_config(),
    },
    
    # -------------------------------------------------------------------------
    # 5️⃣ Optimizer / LR schedule variants (4 experiments)
    # -------------------------------------------------------------------------
    "EXP_opt_radam": {
        "data": get_base_data_config(),
        "model": get_base_model_config(),
        "training": get_base_training_config(optimizer_type="radam"),
    },
    "EXP_opt_lion_lr1e4": {
        "data": get_base_data_config(),
        "model": get_base_model_config(),
        "training": get_base_training_config(
            optimizer_type="lion",
            lr=1e-4,
            weight_decay=1e-2
        ),
    },
    "EXP_sched_onecycle": {
        "data": get_base_data_config(),
        "model": get_base_model_config(),
        "training": get_base_training_config(
            scheduler_type="onecycle",
            onecycle_max_lr=5e-4,
            onecycle_div_factor=10.0,
            onecycle_final_div_factor=100.0,
        ),
    },
    "EXP_sched_cosineWarm": {
        "data": get_base_data_config(),
        "model": get_base_model_config(),
        "training": get_base_training_config(
            scheduler_type="cosine_warm",
            cosine_warm_t0=4,
            cosine_warm_tmult=2,
        ),
    },
    
    # -------------------------------------------------------------------------
    # 6️⃣ Embedding dimension scaling (1 experiments)
    # -------------------------------------------------------------------------
    "EXP_emb512": {
        "data": get_base_data_config(),
        "model": get_base_model_config(emb_dim=512),
        "training": get_base_training_config(),
    },
}


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    exp_name: str,
    config: Dict,
    output_base_dir: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    
    output_dir = f"{output_base_dir}/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {exp_name}")
    print("="*70)
    
    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Model: TCN")
    print(f"  History hours: {data_cfg.history_hours}")
    print(f"  Congestion lags: {data_cfg.congestion_lags}")
    print(f"  Num channels: {model_cfg.num_channels}")
    print(f"  Kernel size: {model_cfg.kernel_size}")
    print(f"  Emb dim: {model_cfg.emb_dim}")
    print(f"  Dropout enc/heads: {model_cfg.dropout_encoder}/{model_cfg.dropout_heads}")
    print(f"  Optimizer: {train_cfg.optimizer_type}")
    print(f"  Scheduler: {train_cfg.scheduler_type}")
    print(f"  LR: {train_cfg.lr}")
    print(f"  Spike weight/threshold: {train_cfg.spike_weight}/{train_cfg.spike_threshold}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Device: {device}")
    
    # Set precision for faster matmul
    torch.set_float32_matmul_precision('high')
    
    try:
        # Prepare data (with congestion lags support)
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
        
        # Train (with extended optimizer/scheduler support)
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
        
        # Compile results
        results = {
            "exp_name": exp_name,
            "status": "success",
            "best_val_loss": min(val_losses),
            "final_val_loss": val_losses[-1],
            "best_epoch": val_losses.index(min(val_losses)) + 1,
            "total_epochs": len(val_losses),
            "train_loss_final": train_losses[-1],
            # Metrics
            "mae": metrics.get("mae") if metrics else None,
            "rmse": metrics.get("rmse") if metrics else None,
            "corr": metrics.get("corr") if metrics else None,
            "spike_f1": metrics.get("spike_f1") if metrics else None,
            "spike_precision": metrics.get("spike_precision") if metrics else None,
            "spike_recall": metrics.get("spike_recall") if metrics else None,
            # Config summary
            "history_hours": data_cfg.history_hours,
            "congestion_lags": str(data_cfg.congestion_lags),
            "num_channels": str(model_cfg.num_channels),
            "kernel_size": model_cfg.kernel_size,
            "emb_dim": model_cfg.emb_dim,
            "dropout_encoder": model_cfg.dropout_encoder,
            "dropout_heads": model_cfg.dropout_heads,
            "optimizer": train_cfg.optimizer_type,
            "scheduler": train_cfg.scheduler_type,
            "lr": train_cfg.lr,
            "weight_decay": train_cfg.weight_decay,
            "spike_weight": train_cfg.spike_weight,
            "spike_threshold": train_cfg.spike_threshold,
            "num_features": X_train.shape[-1],
        }
        
        print(f"\n✓ Experiment {exp_name} complete!")
        print(f"  Best val loss: {results['best_val_loss']:.6f}")
        print(f"  MAE: {results['mae']}")
        print(f"  Spike F1: {results['spike_f1']}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        results = {
            "exp_name": exp_name,
            "status": f"failed: {str(e)}",
            "best_val_loss": None,
            "final_val_loss": None,
            "best_epoch": None,
            "total_epochs": None,
            "train_loss_final": None,
            "mae": None,
            "rmse": None,
            "corr": None,
            "spike_f1": None,
            "spike_precision": None,
            "spike_recall": None,
        }
        print(f"\n✗ Experiment {exp_name} FAILED: {e}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    return results


def compile_results(results_list: List[Dict], output_dir: str) -> pd.DataFrame:
    """Compile all experiment results into a CSV."""
    df = pd.DataFrame(results_list)
    
    # Sort by best_val_loss (ascending, best first)
    df_sorted = df.sort_values("best_val_loss", ascending=True, na_position="last")
    
    # Save to CSV
    csv_path = f"{output_dir}/all_results.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n✓ Results compiled to: {csv_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY (sorted by best_val_loss)")
    print("="*70)
    
    success_df = df_sorted[df_sorted["status"] == "success"]
    if len(success_df) > 0:
        print(f"\nTop 5 experiments:")
        print(success_df[["exp_name", "best_val_loss", "mae", "spike_f1"]].head(10).to_string(index=False))
        
        print(f"\nOverall statistics:")
        print(f"  Total experiments: {len(df)}")
        print(f"  Successful: {len(success_df)}")
        print(f"  Failed: {len(df) - len(success_df)}")
        print(f"  Best val loss: {success_df['best_val_loss'].min():.6f} ({success_df.iloc[0]['exp_name']})")
        if success_df['mae'].notna().any():
            best_mae_idx = success_df['mae'].idxmin()
            print(f"  Best MAE: {success_df.loc[best_mae_idx, 'mae']:.4f} ({success_df.loc[best_mae_idx, 'exp_name']})")
        if success_df['spike_f1'].notna().any():
            best_f1_idx = success_df['spike_f1'].idxmax()
            print(f"  Best Spike F1: {success_df.loc[best_f1_idx, 'spike_f1']:.4f} ({success_df.loc[best_f1_idx, 'exp_name']})")
    
    return df_sorted


def main():
    parser = argparse.ArgumentParser(description="Run parameter search experiments")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["all"],
        help="Which experiments to run (experiment names or 'all')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots_training_dl/param_search",
        help="Base output directory"
    )
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("\n" + "#"*70)
    print("# PARAMETER SEARCH EXPERIMENT SUITE")
    print(f"# Output: {output_base_dir}")
    print("#"*70)
    
    # Determine which experiments to run
    if "all" in args.experiments:
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [e for e in args.experiments if e in EXPERIMENTS]
        if len(exp_names) != len(args.experiments):
            missing = set(args.experiments) - set(exp_names) - {"all"}
            print(f"Warning: Unknown experiments ignored: {missing}")
    
    print(f"\nRunning {len(exp_names)} experiments:")
    for i, name in enumerate(exp_names, 1):
        print(f"  {i:2d}. {name}")
    
    # Run experiments
    results_list = []
    for i, exp_name in enumerate(exp_names, 1):
        print(f"\n>>> [{i}/{len(exp_names)}] Starting {exp_name}...")
        config = EXPERIMENTS[exp_name]
        result = run_single_experiment(exp_name, config, output_base_dir, args.device)
        results_list.append(result)
        
        # Save intermediate results after each experiment
        compile_results(results_list, output_base_dir)
    
    # Final summary
    print("\n" + "#"*70)
    print("# ALL EXPERIMENTS COMPLETE")
    print("#"*70)
    compile_results(results_list, output_base_dir)


if __name__ == "__main__":
    main()