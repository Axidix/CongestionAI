import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
import torch.nn as nn


class MLPForecaster(nn.Module):
    def __init__(self, input_length, num_features, horizon, num_detectors, emb_dim=16):
        super().__init__()

        self.emb = nn.Embedding(num_detectors, emb_dim)

        # after concatenation embedding is repeated for each timestep → shape (input_length, emb_dim)
        self.input_dim = input_length * (num_features + emb_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, horizon)
        )

    def forward(self, x, det_ids):
        """
        x: (B, T, F)
        det_ids: (B,)
        """
        emb = self.emb(det_ids)          # (B, emb_dim)
        emb_rep = emb.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, T, emb_dim)

        x = torch.cat([x, emb_rep], dim=-1)  # (B, T, F+emb_dim)
        x = x.reshape(x.size(0), -1)         # flatten
        return self.net(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, scaler, device, num_epochs):
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(num_epochs)):
            model.train()
            loss_epoch = 0
            for X_batch, Y_batch, det_ids in tqdm(train_loader):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                det_ids = det_ids.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    preds = model(X_batch, det_ids)
                    loss = criterion(preds, Y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_epoch += loss.item()

            train_losses.append(loss_epoch / len(train_loader))

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for X_batch, Y_batch, det_ids in tqdm(val_loader):
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    det_ids = det_ids.to(device)
                    preds = model(X_batch, det_ids)
                    loss = criterion(preds, Y_batch)
                    val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))
            
            tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")

        return train_losses, val_losses

        
def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        preds_list = []

        with torch.no_grad():
            for X_batch, Y_batch, det_ids in tqdm(loader):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                det_ids = det_ids.to(device)
                preds = model(X_batch, det_ids)
                loss = criterion(preds, Y_batch)
                total_loss += loss.item()
                preds_list.append(preds.cpu())

        preds_list = torch.cat(preds_list, dim=0)
        return preds_list, total_loss / len(loader)


def plot_training_curves(train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.savefig("plots_training_dl/training_curves.png")
        plt.close()


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



def plot_block_predictions(df, horizon=24, detector_id=None,
                            years=None, months=None,
                            true_prefix="future_", pred_prefix="pred_",
                            max_blocks=10):
        """
        Plot 24h forecast trajectories:
        - t+1 ... t+horizon for each chosen block
        - sample every 'horizon' timestamps
        """

        df_plot = df.copy()

        # ---- FILTERING ----
        if detector_id is None:
            detector_id = df_plot["detector_id"].iloc[0]
        df_plot = df_plot[df_plot["detector_id"] == detector_id]

        if years is not None:
            df_plot = df_plot[df_plot["timestamp"].dt.year.isin(years)]

        if months is not None:
            df_plot = df_plot[df_plot["timestamp"].dt.month.isin(months)]

        # ---- Take blocks every 'horizon' timesteps ----
        df_blocks = df_plot.iloc[::horizon].copy()
        df_blocks = df_blocks.head(max_blocks)

        # ---- Column lists ----
        true_cols = [f"{true_prefix}{h}h" for h in range(1, horizon+1)]
        pred_cols = [f"{pred_prefix}{h}h" for h in range(1, horizon+1)]

        plt.figure(figsize=(14, 7))
        print("Plotting...")
        for _, row in df_blocks.iterrows():
            base_time = row["timestamp"]
            horizon_times = base_time + pd.to_timedelta(np.arange(1, horizon+1), "h")

            plt.plot(horizon_times, row[true_cols].values,
                    label=f"True (start {base_time})", alpha=0.6)

            plt.plot(horizon_times, row[pred_cols].values,
                    label=f"Pred (start {base_time})", alpha=0.6)

        plt.title(f"{horizon}-hour Forecast Trajectories")
        plt.xlabel("Time")
        plt.ylabel("Congestion Index")
        plt.legend()
        plt.savefig("plots_training_dl/block_forecast_trajectories.png")
        plt.close()



def evaluate_and_plot_block(df, horizon=24,
                                detector_id=None,
                                years=None,
                                months=None,
                                true_prefix="future_",
                                pred_prefix="pred_"):
        """
        Full multi-step forecast evaluation and plotting.
        """

        # ---- Extract arrays ----
        true_cols = [f"{true_prefix}{h}h" for h in range(1, horizon+1)]
        pred_cols = [f"{pred_prefix}{h}h" for h in range(1, horizon+1)]

        Y_true = df[true_cols].values
        Y_pred = df[pred_cols].values

        # ---- Compute metrics ----
        metrics = evaluate_block_predictions(Y_true, Y_pred)

        print("=== Block Forecast Evaluation ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # ---- Plot ----
        plot_block_predictions(
            df,
            horizon=horizon,
            detector_id=detector_id,
            years=years,
            months=months,
            true_prefix=true_prefix,
            pred_prefix=pred_prefix
        )

        return metrics


    
def historical_baseline_multi(df, window_size=5, horizon=24):
    df_h = df[["detector_id", "timestamp", "congestion_index"]].copy()

    # For each horizon h = 1..24 create a future target
    for h in range(1, horizon+1):
        df_h[f"future_{h}h"] = (
            df_h.groupby("detector_id")["congestion_index"]
                .shift(-h)
        )

    # Baseline uses rolling mean of past
    df_h["hist_baseline"] = (
        df_h.groupby("detector_id")["congestion_index"]
             .rolling(window_size, min_periods=1)
             .mean()
             .reset_index(level=0, drop=True)
    )

    # Expand baseline into 24 identical horizons
    for h in range(1, horizon+1):
        df_h[f"pred_{h}h"] = df_h["hist_baseline"]

    # Drop rows where ANY future target is missing
    future_cols = [f"future_{h}h" for h in range(1, horizon+1)]
    df_h = df_h.dropna(subset=future_cols)

    return df_h


class NHitsDataset(Dataset):
    def __init__(self, X_hist, Y, det_ids):
        self.X_hist = X_hist
        self.Y = Y
        self.det_ids = det_ids

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_hist[idx]).float(), 
            torch.from_numpy(self.Y[idx]).float(),
            torch.tensor(self.det_ids[idx], dtype=torch.long),
        )



if __name__ == "__main__":

    # Path to your cleaned giant file
    FILE_PATH = "prepared_data/preprocessed_full_data.csv"

    # Columns we actually need for deep learning
    keep_cols = [
        "timestamp", "detector_id", "congestion_index",
        "hour", "day_of_week", "month", "is_weekend", "lon", "lat",
        "is_holiday", "is_school_holiday", "is_rush_hour", "free_flow_speed",
        "temperature", "precipitation", "visibility", "is_snow", "is_fog"
    ]

    print("Loading only required columns...")
    df = pd.read_csv(FILE_PATH, usecols=keep_cols)

    # Memory reduction
    for col in df.select_dtypes("float64"):
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes("int64"):
        df[col] = df[col].astype("int32")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(df.info())

    
    # Cyclical encoding of time features
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    required_years = df["timestamp"].dt.year.unique().tolist()
    nb_detectors = 50

    # Pick a few detectors for fast experimenting
    sample_detectors = df["detector_id"].unique()[:nb_detectors]  # first n detectors
    df_small = df[df["detector_id"].isin(sample_detectors)]

    # Restrict to some years for prototyping
    df_small = df_small[df_small["timestamp"].dt.year.isin(required_years)]
    df_small = df_small.sort_values(["detector_id", "timestamp"]).reset_index(drop=True)
    print("Small dataframe size:", df_small.shape)

    
    # Add lags
    def make_lags(df, col, lags):
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df.groupby("detector_id", observed=True)[col].shift(lag)
        return df


    weather_lags = [-i for i in range(1, 25, 8)] # Next 24 hours
    df_small = make_lags(df_small, "temperature", weather_lags)
    df_small = make_lags(df_small, "precipitation", weather_lags)
    df_small = make_lags(df_small, "visibility", weather_lags)

    # Remove rows with NaNs due to lagging
    df_small = df_small.dropna().reset_index(drop=True)

    # Encode detector_id as integer index for embedding
    unique_detectors = sorted(df_small["detector_id"].unique())
    det2idx = {d: i for i, d in enumerate(unique_detectors)}
    df_small["det_index"] = df_small["detector_id"].map(det2idx)

    # -----------------------------
    # 1) Define parameters
    # -----------------------------
    HISTORY_OFFSETS = [0, 1, 2, 3, 6, 12, 18, 24, 36, 48]   # hours before t
    forecast_horizon = 24      # next 24 hours → target

    feature_cols_norm = [
        "temperature", "precipitation", "visibility", "congestion_index", "free_flow_speed", #"lon", "lat"
    ]  + [f"temperature_lag_{lag}h" 
        for lag in weather_lags] + [f"precipitation_lag_{lag}h" 
        for lag in weather_lags] + [f"visibility_lag_{lag}h" for lag in weather_lags]

    feature_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday", "is_school_holiday", "is_rush_hour", "lon", "lat",
        "temperature", "precipitation", "visibility", "is_snow", "is_fog",
        "congestion_index", "free_flow_speed"
    ] + [f"temperature_lag_{lag}h" 
        for lag in weather_lags] + [f"precipitation_lag_{lag}h" 
        for lag in weather_lags] + [f"visibility_lag_{lag}h" for lag in weather_lags]


    # -----------------------------
    # 2) Train / Val / Test split
    # -----------------------------
    train = df_small[df_small["timestamp"] < "2021-05-01"].copy()
    val   = df_small[(df_small["timestamp"] >= "2021-05-01") &
                    (df_small["timestamp"] <  "2022-01-01")].copy()
    test  = df_small[df_small["timestamp"] >= "2022-01-01"].copy()

    print(train.shape, val.shape, test.shape)

    # -----------------------------
    # 3) Normalize continuous features
    # -----------------------------
    scaler = StandardScaler()
    train[feature_cols_norm] = scaler.fit_transform(train[feature_cols_norm])
    val[feature_cols_norm]   = scaler.transform(val[feature_cols_norm])
    test[feature_cols_norm]  = scaler.transform(test[feature_cols_norm])
    scaler_minmax = MinMaxScaler()
    train[["lon", "lat"]] = scaler_minmax.fit_transform(train[["lon", "lat"]])
    val[["lon", "lat"]]   = scaler_minmax.transform(val[["lon", "lat"]])
    test[["lon", "lat"]]  = scaler_minmax.transform(test[["lon", "lat"]])
    
    # -----------------------------
    # 4) Function to create sequences
    # -----------------------------
    def create_nhits_sequences(df, feature_cols, hist_offsets, horizon):
        X_list, Y_list, idx_list, det_list = [], [], [], []

        for det_id, df_det in df.groupby("detector_id", observed=True):
            df_det = df_det.sort_values("timestamp").reset_index(drop=False)
            #   keep original row index      ↑↑↑

            values = df_det[feature_cols].values.astype(np.float32)
            target = df_det["congestion_index"].values.astype(np.float32)
            idx    = df_det["index"].values    # original global index
            det_idx = df_det["det_index"].values

            n = len(df_det)
            for t in range(max(hist_offsets), n - horizon):
                X_list.append(values[[t-h for h in hist_offsets]])
                Y_list.append(target[t : t+horizon])
                idx_list.append(idx[t])   # <--- store index where prediction starts
                det_list.append(det_idx[t])  # detector index

        return (
            np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            np.array(idx_list, dtype=np.int64),
            np.array(det_list, dtype=np.int64)
        )


    # -----------------------------
    # 5) Choose the model input features
    # -----------------------------
    model_features = feature_cols

    # -----------------------------
    # 6) Build sequences for each split
    # -----------------------------
    X_train_hist, Y_train, train_idx, train_det_idx = create_nhits_sequences(
        train, model_features, HISTORY_OFFSETS, forecast_horizon)

    X_val_hist, Y_val, val_idx, val_det_idx = create_nhits_sequences(
        val, model_features, HISTORY_OFFSETS, forecast_horizon)

    X_test_hist, Y_test, test_idx, test_det_idx = create_nhits_sequences(
        test, model_features, HISTORY_OFFSETS, forecast_horizon)

    print("Shapes:")
    print("X_train_hist:", X_train_hist.shape)
    print("Y_train:", Y_train.shape)
    print("X_val_hist:",   X_val_hist.shape)
    print("Y_val:",   Y_val.shape)
    print("X_test_hist:",  X_test_hist.shape)
    print("Y_test:",  Y_test.shape)


    batch_size = 128     

    train_loader = DataLoader(
        NHitsDataset(X_train_hist, Y_train, train_det_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        NHitsDataset(X_val_hist, Y_val, val_det_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        NHitsDataset(X_test_hist, Y_test, test_det_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPForecaster(
        input_length=len(HISTORY_OFFSETS),
        num_features=X_train_hist.shape[-1],
        horizon=forecast_horizon,
        num_detectors=nb_detectors,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")  # AMP = automatic mixed precision

    epochs = 10

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scaler,
        device,
        epochs
    )

    model.eval()
    preds, test_loss = evaluate(model, test_loader, criterion, device)
    print("Test loss:", test_loss)


    eval_df = pd.DataFrame({
        "row_idx": test_idx,
        "timestamp": test.loc[test_idx, "timestamp"].values,
        "detector_id": test.loc[test_idx, "detector_id"].values
    })

    for h in range(1, forecast_horizon+1):
        eval_df[f"pred_{h}h"] = preds[:, h-1].numpy()

    for h in range(1, forecast_horizon+1):
        eval_df[f"future_{h}h"] = (
            test.groupby("detector_id", observed=True)["congestion_index"]
                .shift(-h)
                .loc[test_idx]
                .values
        )
    eval_df = eval_df.dropna()

    print("Baseline: historical average congestion.")
    df_historical = historical_baseline_multi(test, horizon=24)
    evaluate_and_plot_block(df_historical, horizon=24, years=[2022])

    print("DL Model Evaluation on Test Set")
    evaluate_and_plot_block(eval_df, horizon=24, years=[2022])



