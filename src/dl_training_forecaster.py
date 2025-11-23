import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.model_pipelines.dl_pipeline import train_model, evaluate
from src.models.mlp_forecaster import MLPForecaster
from src.utils.model_evaluation import evaluate_and_plot_block, historical_baseline_multi
from src.utils.preprocessing import make_lags, cyclical_encode, scale_features, encode_detectors
from src.utils.sequences import create_nhits_sequences, NHitsDataset
from src.utils.plots import plot_training_curves

FILE_PATH = "prepared_data/preprocessed_full_data.csv"


def run_dl_experiment(
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
    batch_size=128,
    epochs=10,
    grad_clip=1.0,
    scheduler=None,
    scaler=None,
    exp_name="",
):
    """
    Runs the full deep-learning training pipeline.

    - Builds model + dataloaders
    - Trains the model
    - Returns model, predictions, losses
    """

    # -------------------------
    # DATALOADERS
    # -------------------------
    train_loader = DataLoader(
        NHitsDataset(X_train_hist, Y_train, train_det_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        NHitsDataset(X_val_hist, Y_val, val_det_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        NHitsDataset(X_test_hist, Y_test, test_det_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
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
        early_stopping={"patience": 5},
    )

    plot_training_curves(train_losses, val_losses, filename=f"training_curve{exp_name}.png")

    # -------------------------
    # TEST
    # -------------------------
    model.load_state_dict(best_state)
    preds, test_loss = evaluate(model, test_loader, criterion, device)

    return model, preds, test_loss, train_losses, val_losses



def prepare_dl_data(history_offsets, forecast_horizon, nb_detectors,
                    data_splits=(0.6, 0.2, 0.2)):
    
    # LOAD + BASIC FIXES
    keep_cols = [
        "timestamp", "detector_id", "congestion_index",
        "hour", "day_of_week", "month", "is_weekend", "lon", "lat",
        "is_holiday", "is_school_holiday", "is_rush_hour", "free_flow_speed",
        "temperature", "precipitation", "visibility", "is_snow", "is_fog"
    ]
    df = pd.read_csv(FILE_PATH, usecols=keep_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = cyclical_encode(df)

    
    # DETECTOR SAMPLING
    df_small = df[df["detector_id"].isin(df["detector_id"].unique()[:nb_detectors])]
    df_small = df_small.sort_values(["detector_id", "timestamp"]).reset_index(drop=True)

    
    # WEATHER LAGS
    weather_lags = [-i for i in range(1, 25, 8)]
    df_small = make_lags(df_small, "temperature", weather_lags)
    df_small = make_lags(df_small, "precipitation", weather_lags)
    df_small = make_lags(df_small, "visibility", weather_lags)

    df_small = df_small.dropna().reset_index(drop=True)

    
    # DETECTOR ENCODING
    df_small, det2idx = encode_detectors(df_small)
    
    # CONFIG
    feature_cols_norm = [
        "temperature", "precipitation", "visibility", "congestion_index", "free_flow_speed"
    ] + [f"temperature_lag_{lag}h" for lag in weather_lags] \
      + [f"precipitation_lag_{lag}h" for lag in weather_lags] \
      + [f"visibility_lag_{lag}h" for lag in weather_lags]

    feature_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday", "is_school_holiday", "is_rush_hour",
        "lon", "lat",
        "temperature", "precipitation", "visibility", "is_snow", "is_fog",
        "congestion_index", "free_flow_speed"
    ] + [f"temperature_lag_{lag}h" for lag in weather_lags] \
      + [f"precipitation_lag_{lag}h" for lag in weather_lags] \
      + [f"visibility_lag_{lag}h" for lag in weather_lags]

    
    # TIME SPLIT (Keeping order)
    timestamps = np.sort(df_small["timestamp"].unique())
    n_ts = len(timestamps)
    # ensure indices valid
    idx1 = max(1, int(n_ts * data_splits[0]))
    idx2 = max(idx1 + 1, int(n_ts * (data_splits[0] + data_splits[1])))

    t_cut1 = timestamps[idx1 - 1]
    t_cut2 = timestamps[idx2 - 1]

    train = df_small[df_small["timestamp"] <= t_cut1].copy()
    val = df_small[(df_small["timestamp"] > t_cut1) & (df_small["timestamp"] <= t_cut2)].copy()
    test = df_small[df_small["timestamp"] > t_cut2].copy()

    
    # NORMALIZATION  
    train, val, test, std_scaler, mm_scaler = scale_features(
        train, val, test, feature_cols_norm, latlon_cols=["lon", "lat"]
    )

    # BUILD ALL SEQUENCES
    X_train_hist, Y_train, idx_train, det_train = create_nhits_sequences(
        train, feature_cols, history_offsets, forecast_horizon)

    X_val_hist, Y_val, idx_val, det_val = create_nhits_sequences(
        val, feature_cols, history_offsets, forecast_horizon)

    X_test_hist, Y_test, idx_test, det_test = create_nhits_sequences(
        test, feature_cols, history_offsets, forecast_horizon)
    
    return (X_train_hist, Y_train, idx_train, det_train,
            X_val_hist, Y_val, idx_val, det_val,
            X_test_hist, Y_test, idx_test, det_test,
            train, val, test, std_scaler, mm_scaler)


def prepare_eval_df(test, idx_test, preds, forecast_horizon):
    # BUILD DF FOR EVAL
    eval_df = pd.DataFrame({
        "row_idx": idx_test,
        "timestamp": test.loc[idx_test, "timestamp"].values,
        "detector_id": test.loc[idx_test, "detector_id"].values
    })

    for h in range(1, forecast_horizon+1):
        eval_df[f"pred_{h}h"] = preds[:, h-1].numpy()

    for h in range(1, forecast_horizon+1):
        eval_df[f"future_{h}h"] = (
            test.groupby("detector_id")["congestion_index"]
                .shift(-h)
                .loc[idx_test]
        ).values

    eval_df = eval_df.dropna()

    return eval_df


def main():
    history_offsets = [0, 1, 2, 3, 6, 12, 18, 24, 36, 48]
    forecast_horizon = 24
    nb_detectors = 50
    evaluation_years = [2022]

    X_train_hist, Y_train, idx_train, det_train, \
    X_val_hist, Y_val, idx_val, det_val, \
    X_test_hist, Y_test, idx_test, det_test, \
    train, val, test, std_scaler, mm_scaler = prepare_dl_data(history_offsets, forecast_horizon, nb_detectors)
    
    # BASELINE
    hist_df = historical_baseline_multi(test, horizon=forecast_horizon)
    evaluate_and_plot_block(hist_df, horizon=forecast_horizon, years=evaluation_years)

    model_1 = MLPForecaster(
        input_length=len(history_offsets),
        num_features=X_train_hist.shape[-1],
        horizon=forecast_horizon,
        num_detectors=nb_detectors,
    )
    optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=3e-4)
    params_experiment_1 = {
        "model": model_1,
        "optimizer": optimizer_1,
        "criterion": nn.MSELoss(),
        "X_train_hist": X_train_hist,
        "Y_train": Y_train,
        "train_det_idx": det_train,
        "X_val_hist": X_val_hist,
        "Y_val": Y_val,
        "val_det_idx": det_val,
        "X_test_hist": X_test_hist,
        "Y_test": Y_test,
        "test_det_idx": det_test,
        "feature_dim": X_train_hist.shape[-1],
        "history_length": len(history_offsets),
        "forecast_horizon": forecast_horizon,
        "num_detectors": nb_detectors,
        "device": "cuda",
        "batch_size": 256,
        "epochs": 30,
        "grad_clip": None,
        "scheduler": None
    }

    model_2 = MLPForecaster(
        input_length=len(history_offsets),
        num_features=X_train_hist.shape[-1],
        horizon=forecast_horizon,
        num_detectors=nb_detectors,
    )
    optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=3e-4)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=30)
    params_experiment_2 = params_experiment_1.copy()
    params_experiment_2["optimizer"] = optimizer_2
    params_experiment_2["scheduler"] = scheduler_2
    params_experiment_2["model"] = model_2
    
    model_3 = MLPForecaster(
        input_length=len(history_offsets),
        num_features=X_train_hist.shape[-1],
        horizon=forecast_horizon,
        num_detectors=nb_detectors,
    )
    optimizer_3 = torch.optim.AdamW(model_3.parameters(), lr=3e-4)
    params_experiment_3 = params_experiment_1.copy()
    params_experiment_3["criterion"] = nn.SmoothL1Loss()
    params_experiment_3["optimizer"] = optimizer_3
    params_experiment_3["model"] = model_3
    
    # RUN EXPERIMENTS
    model, preds, test_loss, train_losses, val_losses = run_dl_experiment(**params_experiment_1, exp_name="_experiment_1")
    print("Final test loss:", test_loss)
    eval_df = prepare_eval_df(test, idx_test, preds, forecast_horizon)
    evaluate_and_plot_block(eval_df, horizon=forecast_horizon, years=evaluation_years)

    model, preds, test_loss, train_losses, val_losses = run_dl_experiment(**params_experiment_2, exp_name="_experiment_2")
    print("Final test loss:", test_loss)
    eval_df = prepare_eval_df(test, idx_test, preds, forecast_horizon)
    evaluate_and_plot_block(eval_df, horizon=forecast_horizon, years=evaluation_years)

    model, preds, test_loss, train_losses, val_losses = run_dl_experiment(**params_experiment_3, exp_name="_experiment_3")
    print("Final test loss:", test_loss)
    eval_df = prepare_eval_df(test, idx_test, preds, forecast_horizon)
    evaluate_and_plot_block(eval_df, horizon=forecast_horizon, years=evaluation_years)


if __name__ == "__main__":
    main()