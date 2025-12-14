"""
Train Final Model for Deployment
================================
Best config from parameter search: EXP_lags_multiscale_48_168
- Congestion lags: (48, 168) hours
- All detectors, all years (2016-2024)
- 8 epochs (best validation was at epoch 6/8)

Run with:
    python train_final_model.py
"""

import os
import sys
import torch
import joblib
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.configs import DataConfig, TCNModelConfig, TrainingConfig
from src.utils.preprocessing import prepare_data_memmap
from src.model_pipelines.dl_pipeline import create_model, train_full_scale
from src.utils.model_evaluation import evaluate_model


# ============================================================================
# FINAL MODEL CONFIGURATION
# ============================================================================

# Data config - ALL years, ALL detectors
data_cfg = DataConfig(
    file_path="prepared_data/preprocessed_full_data.csv",
    nb_detectors=None,  # Use ALL detectors
    forecast_horizon=24,
    history_hours=48,
    weather_lags=(0, -3, -6, -12, -24),
    delta_lags=(1, 2, 4, 6),
    volatility_threshold=0.04,
    # Include ALL years for training (2019 was validation before)
    years_train=(2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024),
    years_val=(2024,),  # Use 2024 as validation (most recent)
    years_test=(),
    cache_dir="prepared_data/memmap_cache_final",
    # Best feature from param search
    congestion_lags=(48, 168),  # 2-day and 1-week lags
)

# Model config - base TCN (same as param search)
model_cfg = TCNModelConfig(
    emb_dim=256,
    num_channels=(128, 256, 256, 256, 256),
    kernel_size=5,
    dropout_encoder=0.20,
    dropout_heads=0.20,
    use_se=False,
    pooling="last",
)

# Training config
train_cfg = TrainingConfig(
    epochs=8,
    batch_size=2048,
    lr=3e-4,
    weight_decay=1e-3,
    betas=(0.9, 0.99),
    grad_clip=0.5,
    use_amp=True,
    num_workers=4,
    patience=None, 
    spike_weight=3.0,
    spike_threshold=0.15,
    eval_spike_threshold=0.38,
    optimizer="adamw",
    scheduler="cosine",
)


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = "FINAL_MODEL_lags_48_168"
    output_dir = "plots_training_dl/final_model"
    
    print("\n" + "#"*70)
    print("# FINAL MODEL TRAINING FOR DEPLOYMENT")
    print("#"*70)
    print(f"\nConfiguration:")
    print(f"  Congestion lags: {data_cfg.congestion_lags}")
    print(f"  Training years: {data_cfg.years_train}")
    print(f"  Validation year: {data_cfg.years_val}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set precision for faster matmul
    torch.set_float32_matmul_precision('high')
    
    # Prepare data
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    
    (X_train, Y_train, idx_train, det_train,
     X_val, Y_val, idx_val, det_val,
     train_df, val_df, std_scaler, mm_scaler,
     memmap_builder, num_detectors) = prepare_data_memmap(data_cfg)
    
    # Save scalers for inference (backend/scalers/)
    scaler_dir = os.path.join(PROJECT_ROOT, "backend", "scalers")
    os.makedirs(scaler_dir, exist_ok=True)
    
    joblib.dump(std_scaler, os.path.join(scaler_dir, "std_scaler.joblib"))
    joblib.dump(mm_scaler, os.path.join(scaler_dir, "mm_scaler.joblib"))
    print(f"  Saved scalers to {scaler_dir}")
    
    # Save detector mapping (det2idx) - need to extract from train_df
    if "det_index" in train_df.columns:
        det2idx = dict(zip(
            train_df["detector_id"].unique(),
            train_df.groupby("detector_id")["det_index"].first().values
        ))
        # Sort by index to ensure correct mapping
        det2idx = {k: int(v) for k, v in sorted(det2idx.items(), key=lambda x: x[1])}
        joblib.dump(det2idx, os.path.join(scaler_dir, "det2idx.joblib"))
        print(f"  Saved det2idx mapping ({len(det2idx)} detectors)")
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  Num detectors: {num_detectors}")
    print(f"  Num features: {X_train.shape[-1]}")
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
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
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
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
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
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
    print("\n" + "#"*70)
    print("# FINAL MODEL TRAINING COMPLETE")
    print("#"*70)
    print(f"\nResults:")
    print(f"  Best val loss: {min(val_losses):.6f}")
    print(f"  Best epoch: {val_losses.index(min(val_losses)) + 1}")
    if metrics:
        print(f"  MAE: {metrics.get('mae', 'N/A')}")
        print(f"  RMSE: {metrics.get('rmse', 'N/A')}")
        print(f"  Spike F1: {metrics.get('spike_f1', 'N/A')}")
    
    print(f"\nModel saved to:")
    print(f"  {output_dir}/checkpoints/best_model_{exp_name}.pt")
    print(f"\nReady for deployment!")


if __name__ == "__main__":
    main()
