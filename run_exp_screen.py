import os
import sys
import argparse
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.configs import DataConfig, TCNModelConfig, TransformerModelConfig, TrainingConfig
from src.utils.preprocessing import prepare_data_memmap
from src.model_pipelines.dl_pipeline import create_model, train_full_scale
from src.utils.model_evaluation import evaluate_model


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
        epochs=5,
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
        epochs=5,
        batch_size=2048,
        lr=8e-5,
        weight_decay=1e-2,
        betas=(0.9, 0.98),
        grad_clip=0.5,
        use_amp=True,
    ),
}



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
        mae = metrics.get('mae')
        spike_f1 = metrics.get('spike_f1')
        if mae is not None and not isinstance(mae, str):
            print(f"  MAE: {mae:.4f}")
        else:
            print(f"  MAE: {mae}")
        if spike_f1 is not None and not isinstance(spike_f1, str):
            print(f"  Spike F1: {spike_f1:.4f}")
        else:
            print(f"  Spike F1: {spike_f1}")
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