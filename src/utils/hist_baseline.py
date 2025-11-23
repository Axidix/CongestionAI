import pandas as pd

def historical_baseline_multi(df, window_size=5, horizon=24):
    df_h = df[["detector_id", "timestamp", "congestion_index"]].copy()

    for h in range(1, horizon+1):
        df_h[f"future_{h}h"] = df_h.groupby("detector_id")["congestion_index"].shift(-h)

    df_h["hist_baseline"] = (
        df_h.groupby("detector_id")["congestion_index"]
             .rolling(window_size, min_periods=1)
             .mean()
             .reset_index(level=0, drop=True)
    )

    for h in range(1, horizon+1):
        df_h[f"pred_{h}h"] = df_h["hist_baseline"]

    future_cols = [f"future_{h}h" for h in range(1, horizon+1)]
    df_h = df_h.dropna(subset=future_cols)
    return df_h
