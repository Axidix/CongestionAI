import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from torch import nn

from src.utils.plots import plot_block_predictions
from src.model_pipelines.dl_pipeline import predict


def evaluate_block_predictions(Y_true, Y_pred):
    """
    Ultra-fast evaluation of block forecasts using vectorized operations.
    Shape: Y_true, Y_pred = (N, H)
    """

    # ---- Basic metrics (vectorized) ----
    diff = Y_true - Y_pred
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff * diff))
    print("MAE:", mae, "RMSE:", rmse)

    # ---- Vectorized correlation ----
    # Center sequences
    Yt = Y_true - Y_true.mean(axis=1, keepdims=True)
    Yp = Y_pred - Y_pred.mean(axis=1, keepdims=True)
    print("Mean centered Y_true and Y_pred for correlation calculation.")

    # Compute numerator: cov
    num = np.sum(Yt * Yp, axis=1)

    # Denominator: std(True)*std(Pred)
    denom = np.sqrt(np.sum(Yt * Yt, axis=1) * np.sum(Yp * Yp, axis=1))
    print("Computed denominator for correlation calculation.")

    # Avoid division by zero
    corr_per_block = np.where(denom == 0, np.nan, num / denom)

    # Average correlation (ignoring NaN)
    corr = np.nanmean(corr_per_block)
    print("Correlation:", corr)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Corr": corr
    }


def evaluate_per_timestep(Y_true, Y_pred):
    """
    Evaluate multi-step forecasts per time step.
    Shape: Y_true, Y_pred = (N, H)
    """

    horizon = Y_true.shape[1]
    metrics = {}

    for h in range(horizon):
        diff_h = Y_true[:, h] - Y_pred[:, h]
        mae_h  = np.mean(np.abs(diff_h))
        rmse_h = np.sqrt(np.mean(diff_h * diff_h))

        metrics[f"MAE_h{h+1}"] = mae_h
        metrics[f"RMSE_h{h+1}"] = rmse_h

    return metrics


# ============================================================
# SPIKE EVALUATION METRICS
# ============================================================

def compute_deltas(Y, prev_values=None):
    """
    Compute hour-to-hour deltas for a target array.
    
    Args:
        Y: (N, H) array of values
        prev_values: (N,) array of values at t=0 (before forecast starts)
                     If None, first delta is set to 0
    
    Returns:
        deltas: (N, H) array of changes
    """
    N, H = Y.shape
    deltas = np.zeros_like(Y)
    
    # First horizon: compare to previous value
    if prev_values is not None:
        deltas[:, 0] = Y[:, 0] - prev_values
    
    # Subsequent horizons: compare to previous horizon
    if H > 1:
        deltas[:, 1:] = Y[:, 1:] - Y[:, :-1]
    
    return deltas


def evaluate_spike_performance(Y_true, Y_pred, prev_values=None, threshold=0.15):
    """
    Compute spike-specific metrics.
    
    Args:
        Y_true: (N, H) ground truth
        Y_pred: (N, H) predictions
        prev_values: (N,) values at t=0 for computing first delta
        threshold: Absolute change threshold to define a spike
    
    Returns:
        dict with spike metrics
    """
    # Compute deltas
    true_deltas = compute_deltas(Y_true, prev_values)
    pred_deltas = compute_deltas(Y_pred, prev_values)
    
    # Identify true spikes (up or down)
    true_spike_up = true_deltas > threshold
    true_spike_down = true_deltas < -threshold
    true_spike = true_spike_up | true_spike_down
    
    # Identify predicted spikes
    pred_spike_up = pred_deltas > threshold
    pred_spike_down = pred_deltas < -threshold
    pred_spike = pred_spike_up | pred_spike_down
    
    # Counts
    n_true_spikes = true_spike.sum()
    n_pred_spikes = pred_spike.sum()
    n_total = Y_true.size
    
    # Spike frequency
    spike_frequency = n_true_spikes / n_total if n_total > 0 else 0.0
    
    if n_true_spikes > 0:
        # Recall: how many true spikes did we catch?
        spike_recall = (true_spike & pred_spike).sum() / n_true_spikes
        
        # MAE on spike positions only
        spike_mae = np.abs(Y_pred[true_spike] - Y_true[true_spike]).mean()
        
        # MAE on non-spike positions
        non_spike_mae = np.abs(Y_pred[~true_spike] - Y_true[~true_spike]).mean() if (~true_spike).sum() > 0 else 0.0
        
        # Delta MAE on spikes (how well do we predict the change magnitude?)
        spike_delta_mae = np.abs(pred_deltas[true_spike] - true_deltas[true_spike]).mean()
        
        # Delta correlation on spike positions
        if true_spike.sum() > 1:
            spike_delta_corr = np.corrcoef(
                true_deltas[true_spike].flatten(),
                pred_deltas[true_spike].flatten()
            )[0, 1]
            if np.isnan(spike_delta_corr):
                spike_delta_corr = 0.0
        else:
            spike_delta_corr = 0.0
    else:
        spike_recall = 0.0
        spike_mae = 0.0
        non_spike_mae = np.abs(Y_pred - Y_true).mean()
        spike_delta_mae = 0.0
        spike_delta_corr = 0.0
    
    if n_pred_spikes > 0:
        # Precision: how many predicted spikes were real?
        spike_precision = (true_spike & pred_spike).sum() / n_pred_spikes
    else:
        spike_precision = 0.0
    
    # F1 score
    if spike_precision + spike_recall > 0:
        spike_f1 = 2 * spike_precision * spike_recall / (spike_precision + spike_recall)
    else:
        spike_f1 = 0.0
    
    # Overall delta correlation
    if true_deltas.size > 1:
        overall_delta_corr = np.corrcoef(
            true_deltas.flatten(),
            pred_deltas.flatten()
        )[0, 1]
        if np.isnan(overall_delta_corr):
            overall_delta_corr = 0.0
    else:
        overall_delta_corr = 0.0
    
    # Direction accuracy: did we predict the right direction of change?
    true_direction = np.sign(true_deltas)
    pred_direction = np.sign(pred_deltas)
    direction_accuracy = (true_direction == pred_direction).mean()
    
    # Direction accuracy on spikes only
    if n_true_spikes > 0:
        spike_direction_accuracy = (true_direction[true_spike] == pred_direction[true_spike]).mean()
    else:
        spike_direction_accuracy = 0.0
    
    return {
        "n_true_spikes": int(n_true_spikes),
        "n_pred_spikes": int(n_pred_spikes),
        "spike_frequency": float(spike_frequency),
        "spike_recall": float(spike_recall),
        "spike_precision": float(spike_precision),
        "spike_f1": float(spike_f1),
        "spike_mae": float(spike_mae),
        "non_spike_mae": float(non_spike_mae),
        "spike_delta_mae": float(spike_delta_mae),
        "spike_delta_corr": float(spike_delta_corr),
        "overall_delta_corr": float(overall_delta_corr),
        "direction_accuracy": float(direction_accuracy),
        "spike_direction_accuracy": float(spike_direction_accuracy),
    }


def print_spike_report(metrics):
    """Pretty-print spike metrics."""
    print("\n" + "="*55)
    print("SPIKE PERFORMANCE REPORT")
    print("="*55)
    print(f"{'Metric':<30} {'Value':>20}")
    print("-"*55)
    print(f"{'True spikes':<30} {metrics['n_true_spikes']:>20,}")
    print(f"{'Predicted spikes':<30} {metrics['n_pred_spikes']:>20,}")
    print(f"{'Spike frequency':<30} {metrics['spike_frequency']:>19.2%}")
    print("-"*55)
    print(f"{'Spike Recall':<30} {metrics['spike_recall']:>19.2%}")
    print(f"{'Spike Precision':<30} {metrics['spike_precision']:>19.2%}")
    print(f"{'Spike F1':<30} {metrics['spike_f1']:>19.2%}")
    print("-"*55)
    print(f"{'Spike MAE':<30} {metrics['spike_mae']:>20.4f}")
    print(f"{'Non-Spike MAE':<30} {metrics['non_spike_mae']:>20.4f}")
    print(f"{'Spike Delta MAE':<30} {metrics['spike_delta_mae']:>20.4f}")
    print("-"*55)
    print(f"{'Spike Delta Corr':<30} {metrics['spike_delta_corr']:>20.4f}")
    print(f"{'Overall Delta Corr':<30} {metrics['overall_delta_corr']:>20.4f}")
    print(f"{'Direction Accuracy':<30} {metrics['direction_accuracy']:>19.2%}")
    print(f"{'Spike Direction Accuracy':<30} {metrics['spike_direction_accuracy']:>19.2%}")
    print("="*55)


def evaluate_and_plot_block(df, horizon=24,
                            detector_id=None,
                            years=None, plot_years=None,
                            months=None,
                            true_prefix="future_",
                            pred_prefix="pred_",
                            filename="",
                            dir="plots_training_dl",
                            max_blocks=10,
                            eval_spikes=True,
                            spike_threshold=0.15,
                            prev_col="congestion_index"):
    """
    Full multi-step forecast evaluation and plotting.
    
    Args:
        df: DataFrame with predictions and ground truth
        horizon: Forecast horizon
        detector_id: Optional detector filter
        years: Years to filter for evaluation
        plot_years: Years to plot
        months: Optional month filter
        true_prefix: Prefix for ground truth columns
        pred_prefix: Prefix for prediction columns
        filename: Output filename suffix
        dir: Output directory
        max_blocks: Max blocks to plot
        eval_spikes: Whether to compute spike metrics
        spike_threshold: Threshold for spike detection
        prev_col: Column name for previous value (t=0)
    """

    # ---- Extract arrays ----
    true_cols = [f"{true_prefix}{h}h" for h in range(1, horizon+1)]
    pred_cols = [f"{pred_prefix}{h}h" for h in range(1, horizon+1)]

    if years is not None:
        df = df[df["timestamp"].dt.year.isin(years)]
    Y_true = df[true_cols].values
    Y_pred = df[pred_cols].values

    # ---- Compute basic metrics ----
    metrics = evaluate_block_predictions(Y_true, Y_pred)

    print("=== Block Forecast Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    metrics_per_timestep = evaluate_per_timestep(Y_true, Y_pred)

    # ---- Compute spike metrics ----
    spike_metrics = {}
    if eval_spikes:
        # Get previous values if available
        if prev_col in df.columns:
            prev_values = df[prev_col].values
        else:
            prev_values = None
        
        spike_metrics = evaluate_spike_performance(
            Y_true, Y_pred, 
            prev_values=prev_values, 
            threshold=spike_threshold
        )
        print_spike_report(spike_metrics)

    # Save metrics to txt file
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/block_forecast_metrics_{filename}.txt", "w") as f:
        f.write("=== Overall Block Forecast Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        
        f.write("\n--- Per Timestep Metrics ---\n")
        for k, v in metrics_per_timestep.items():
            f.write(f"{k}: {v:.4f}\n")
        
        if eval_spikes:
            f.write("\n--- Spike Metrics ---\n")
            for k, v in spike_metrics.items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.4f}\n")
                else:
                    f.write(f"{k}: {v}\n")

    # ---- Plot ----
    plot_block_predictions(
        df,
        horizon=horizon,
        detector_id=detector_id,
        years=plot_years,
        months=months,
        true_prefix=true_prefix,
        pred_prefix=pred_prefix,
        max_blocks=max_blocks,
        filename=filename,
        dir=dir
    )

    # Merge all metrics
    all_metrics = {**metrics, **spike_metrics}
    return all_metrics




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
    
    # idx_val contains orig_idx label values, use .loc
    df_subset = val_df.loc[idx_val].copy()
    
    eval_df = pd.DataFrame({
        "row_idx": idx_val,
        "timestamp": df_subset["timestamp"].values,
        "detector_id": df_subset["detector_id"].values,
    })
    
    # Add predictions
    for h in range(1, forecast_horizon + 1):
        eval_df[f"pred_{h}h"] = preds[:, h-1].numpy()
    
    # Add ground truth - use .loc since idx_val contains labels
    for h in range(1, forecast_horizon + 1):
        shifted = val_df.groupby("detector_id")["congestion_index"].shift(-h)
        eval_df[f"future_{h}h"] = shifted.loc[idx_val].values
    
    eval_df["congestion_index"] = df_subset["congestion_index"].values
    eval_df = eval_df.dropna()
    
    print(f"  Eval samples: {len(eval_df):,}")
    
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
        mae = metrics.get('mae', 'N/A')
        rmse = metrics.get('rmse', 'N/A')
        spike_recall = metrics.get('spike_recall', 'N/A')
        spike_precision = metrics.get('spike_precision', 'N/A')
        spike_f1 = metrics.get('spike_f1', 'N/A')
        
        print(f"  MAE: {mae:.4f}" if isinstance(mae, (int, float)) else f"  MAE: {mae}")
        print(f"  RMSE: {rmse:.4f}" if isinstance(rmse, (int, float)) else f"  RMSE: {rmse}")
        print(f"  Spike Recall: {spike_recall:.4f}" if isinstance(spike_recall, (int, float)) else f"  Spike Recall: {spike_recall}")
        print(f"  Spike Precision: {spike_precision:.4f}" if isinstance(spike_precision, (int, float)) else f"  Spike Precision: {spike_precision}")
        print(f"  Spike F1: {spike_f1:.4f}" if isinstance(spike_f1, (int, float)) else f"  Spike F1: {spike_f1}")
    
    return metrics