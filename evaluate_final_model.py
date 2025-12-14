"""
Evaluate Final Model (without retraining)
=========================================
Run this after training to evaluate the saved model.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.configs import DataConfig, TCNModelConfig, TrainingConfig
from src.utils.preprocessing import prepare_data_memmap
from src.model_pipelines.dl_pipeline import create_model
from src.utils.model_evaluation import evaluate_model


# Same configs as training
data_cfg = DataConfig(
    file_path="prepared_data/preprocessed_full_data.csv",
    nb_detectors=None,
    forecast_horizon=24,
    history_hours=48,
    weather_lags=(0, -3, -6, -12, -24),
    delta_lags=(1, 2, 4, 6),
    volatility_threshold=0.04,
    years_train=(2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024),
    years_val=(2024,),
    years_test=(),
    cache_dir="prepared_data/memmap_cache_final",
    congestion_lags=(48, 168),
)

model_cfg = TCNModelConfig(
    emb_dim=256,
    num_channels=(128, 256, 256, 256, 256),
    kernel_size=5,
    dropout_encoder=0.20,
    dropout_heads=0.20,
    use_se=False,
    pooling="last",
)

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = "FINAL_MODEL_lags_48_168"
    output_dir = "plots_training_dl/final_model"
    checkpoint_path = f"{output_dir}/checkpoints/best_model_{exp_name}.pt"
    
    print("\n" + "#"*70)
    print("# EVALUATING FINAL MODEL")
    print("#"*70)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Prepare data (uses cache, should be fast)
    print("\nPreparing data from cache...")
    (X_train, Y_train, idx_train, det_train,
     X_val, Y_val, idx_val, det_val,
     train_df, val_df, std_scaler, mm_scaler,
     memmap_builder, num_detectors) = prepare_data_memmap(data_cfg)
    
    print(f"\nValidation data:")
    print(f"  X_val: {X_val.shape}")
    print(f"  Num detectors: {num_detectors}")
    print(f"  Num features: {X_val.shape[-1]}")
    
    # Create model architecture
    model = create_model(
        model_cfg,
        num_features=X_val.shape[-1],
        num_detectors=num_detectors,
        forecast_horizon=data_cfg.forecast_horizon,
        device=device,
    )
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle compiled model state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Remove _orig_mod prefix if present (from torch.compile)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[new_key] = v
    
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    print(f"✓ Loaded model weights from epoch {checkpoint.get('epoch', '?')}")
    
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
    
    print("\n" + "#"*70)
    print("# EVALUATION COMPLETE")
    print("#"*70)


if __name__ == "__main__":
    main()
