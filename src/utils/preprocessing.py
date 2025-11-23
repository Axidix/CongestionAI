import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def cyclical_encode(df):
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def make_lags(df, col, lags):
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df.groupby("detector_id")[col].shift(lag)
    return df


def encode_detectors(df):
    unique_detectors = sorted(df["detector_id"].unique())
    det2idx = {d: i for i, d in enumerate(unique_detectors)}
    df["det_index"] = df["detector_id"].map(det2idx)
    return df, det2idx


def scale_features(train, val, test, norm_cols, latlon_cols=["lon", "lat"]):
    std = StandardScaler()
    train[norm_cols] = std.fit_transform(train[norm_cols])
    val[norm_cols]   = std.transform(val[norm_cols])
    test[norm_cols]  = std.transform(test[norm_cols])

    mm = MinMaxScaler()
    train[latlon_cols] = mm.fit_transform(train[latlon_cols])
    val[latlon_cols]   = mm.transform(val[latlon_cols])
    test[latlon_cols]  = mm.transform(test[latlon_cols])

    return train, val, test, std, mm
