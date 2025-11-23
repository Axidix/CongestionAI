import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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