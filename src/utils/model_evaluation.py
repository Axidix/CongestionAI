import numpy as np
from src.utils.plots import plot_block_predictions

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