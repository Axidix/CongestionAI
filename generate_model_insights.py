import os
import json
import numpy as np
import pandas as pd
import torch
from src.utils.configs import DataConfig, TCNModelConfig, TrainingConfig
from src.utils.preprocessing import prepare_data_memmap
from src.model_pipelines.dl_pipeline import create_model


# --- Load configs (reuse from evaluate_final_model.py) ---
data_cfg = DataConfig(
    file_path="prepared_data/preprocessed_full_data.csv",
    nb_detectors=None,  # Use ALL detectors
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

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "plots_training_dl/final_model"
checkpoint_path = f"{output_dir}/checkpoints/best_model_FINAL_MODEL_lags_48_168.pt"

# --- Load validation data ---
(X_train, Y_train, idx_train, det_train,
 X_val, Y_val, idx_val, det_val,
 train_df, val_df, std_scaler, mm_scaler,
 memmap_builder, num_detectors) = prepare_data_memmap(data_cfg)

# --- Load model ---
model = create_model(
    model_cfg,
    num_features=X_val.shape[-1],
    num_detectors=num_detectors,
    forecast_horizon=data_cfg.forecast_horizon,
    device=device,
)
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
model.eval()


# --- Run predictions in batches to avoid OOM ---
def batched_predict(model, X, det, batch_size=2048, device="cpu"):
    preds = []
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            X_batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            det_batch = torch.from_numpy(det[i:i+batch_size]).long().to(device)
            out = model(X_batch, det_batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)

preds = batched_predict(model, X_val, det_val, batch_size=2048, device=device)
# ✅ Fix 1: Apply sigmoid before evaluation
preds = 1.0 / (1.0 + np.exp(-preds))

Y_true = Y_val
Y_pred = preds
abs_err = np.abs(Y_pred - Y_true)

# ✅ Fix 3: Correct R² (flatten arrays)
y_true_flat = Y_true.ravel()
y_pred_flat = Y_pred.ravel()
ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
r2 = float(1 - ss_res / ss_tot)

mae = float(np.mean(abs_err))
rmse = float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)))
within_010 = float(np.mean(abs_err < 0.10))
within_015 = float(np.mean(abs_err < 0.15))
n_samples = int(np.prod(Y_true.shape))

with open("gui/data/metrics_overall.json", "w") as f:
    json.dump({
        "mae": mae, "rmse": rmse, "r2": r2,
        "within_010": within_010, "within_015": within_015,
        "n_samples": n_samples
    }, f, indent=2)

# MAE by horizon
mae_by_horizon = np.mean(abs_err, axis=0).tolist()
with open("gui/data/metrics_by_horizon.json", "w") as f:
    json.dump({"horizon": list(range(1, Y_true.shape[1]+1)), "mae": mae_by_horizon}, f, indent=2)




# ✅ Fix 2: Use only rows corresponding to idx_val for time slicing (positional alignment)
val_df_reset = val_df.reset_index(drop=True)
row_idx = np.arange(len(idx_val))
hours = val_df_reset.iloc[row_idx]["timestamp"].dt.hour.values
mae_by_hour = [float(np.mean(abs_err[hours == h])) for h in range(24)]
with open("gui/data/metrics_by_hour.json", "w") as f:
    json.dump({"hour": list(range(24)), "mae": mae_by_hour}, f, indent=2)

dows = val_df_reset.iloc[row_idx]["timestamp"].dt.dayofweek.values
mae_by_dow = [float(np.mean(abs_err[dows == d])) for d in range(7)]
with open("gui/data/metrics_by_dow.json", "w") as f:
    json.dump({"dow": list(range(7)), "mae": mae_by_dow}, f, indent=2)

# Error distribution
hist, bin_edges = np.histogram(abs_err, bins=30, range=(0, 1))
with open("gui/data/error_distribution.json", "w") as f:
    json.dump({"bins": bin_edges.tolist(), "counts": hist.tolist()}, f, indent=2)

# ✅ Fix 4: Aggregate MAE by detector_id for spatial metrics (align with validation rows)
if "lat" in val_df.columns and "lon" in val_df.columns:
    df_err = val_df_reset.iloc[row_idx].copy()
    df_err["mae"] = np.mean(abs_err, axis=1)
    by_detector = (
        df_err
        .groupby("detector_id")
        .agg(
            mae=("mae", "mean"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .reset_index()
    )
    with open("gui/data/metrics_by_detector.json", "w") as f:
        json.dump(by_detector.to_dict(orient="records"), f, indent=2)