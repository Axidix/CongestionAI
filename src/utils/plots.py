import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.lines import Line2D

DEBUG_PLOTS = True

def plot_training_curves(train_losses, val_losses, filename, dir="plots_training_dl"):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        os.makedirs(dir, exist_ok=True)
        plt.savefig(f"{dir}/{filename}")
        plt.close()


def plot_block_predictions(df, horizon=24, detector_id=None,
                            years=None, months=None,
                            true_prefix="future_", pred_prefix="pred_",
                            max_blocks=10, filename="", dir="plots_training_dl"):
        """
        Plot 24h forecast trajectories:
        - t+1 ... t+horizon for each chosen block
        - sample every 'horizon' timestamps
        """


        df_plot = df.copy()
        if DEBUG_PLOTS:
            print(len(df_plot), "total blocks in dataframe")
            print("Years in data:", df_plot["timestamp"].dt.year.unique())

        # ---- FILTERING ----
        if detector_id is None:
            detector_id = df_plot["detector_id"].unique()[0]
        df_plot = df_plot[df_plot["detector_id"] == detector_id]
        if DEBUG_PLOTS:
            print(f"Using detector_id = {detector_id}")
            print(len(df_plot), "blocks after filtering detector_id")

        if years is not None:
            df_plot = df_plot[df_plot["timestamp"].dt.year.isin(years)]
            if DEBUG_PLOTS:
                print(f"Filtering years = {years}")
                print(len(df_plot), "blocks after filtering years")

        if months is not None:
            df_plot = df_plot[df_plot["timestamp"].dt.month.isin(months)]
            if DEBUG_PLOTS:
                print(f"Filtering months = {months}")
                print(len(df_plot), "blocks after filtering months")

        if DEBUG_PLOTS:
            print(len(df_plot), "blocks after filtering")

        # ---- Take blocks every 'horizon' timesteps ----
        df_blocks = df_plot.iloc[::horizon].copy()
        start = np.random.randint(len(df_blocks)-max_blocks)
        df_blocks = df_blocks.iloc[start:start + max_blocks]
        if DEBUG_PLOTS:
            print(len(df_blocks), "blocks after sampling")

        # ---- Column lists ----
        true_cols = [f"{true_prefix}{h}h" for h in range(1, horizon+1)]
        pred_cols = [f"{pred_prefix}{h}h" for h in range(1, horizon+1)]
        if DEBUG_PLOTS:
            print("True columns:", true_cols)
            print("Predicted columns:", pred_cols)

        plt.figure(figsize=(14, 7))
        print("Plotting...")

        # fixed colors for all truths and preds
        true_color = "tab:blue"
        pred_color = "tab:orange"
        i=0

        for _, row in df_blocks.iterrows():
            base_time = row["timestamp"]
            horizon_times = pd.date_range(
                start=base_time + pd.Timedelta(hours=1),
                periods=horizon,
                freq="h"
            )

            y_true = row[true_cols].astype(float).to_numpy().reshape(-1)
            y_pred = row[pred_cols].astype(float).to_numpy().reshape(-1)
            if DEBUG_PLOTS and i < 5:
                print("Base time:", base_time)
                print("Horizon times:", horizon_times)
                print("y_true length:", len(y_true))
                print("y_pred length:", len(y_pred))
            if horizon == 1:
                # For 1-hour horizon, plot a single point
                plt.scatter(horizon_times, y_true, color=true_color, alpha=0.6)
                plt.scatter(horizon_times, y_pred, color=pred_color, alpha=0.6)
            else:
                plt.plot(horizon_times, y_true, color=true_color, alpha=0.6)
                plt.plot(horizon_times, y_pred, color=pred_color, alpha=0.6)
            i+=1

        plt.title(f"{horizon}-hour Forecast Trajectories")
        plt.xlabel("Time")
        plt.ylabel("Congestion Index")

        # create a legend with only two entries (True, Pred)
        handles = [
            Line2D([0], [0], color=true_color, lw=2, alpha=0.6),
            Line2D([0], [0], color=pred_color, lw=2, alpha=0.6),
        ]
        labels = ["True", "Predicted"]
        plt.legend(handles=handles, labels=labels)

        os.makedirs(dir, exist_ok=True)
        plt.savefig(f"{dir}/block_forecast_trajectories_{filename}.png")
        plt.close()
