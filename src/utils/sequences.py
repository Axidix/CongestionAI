import numpy as np
import torch
from torch.utils.data import Dataset

class NHitsDataset(Dataset):
    def __init__(self, X_hist, Y, det_ids):
        # Pre-convert to tensors once
        if isinstance(X_hist, np.ndarray):
            self.X = torch.from_numpy(np.ascontiguousarray(X_hist)).float()
            self.Y = torch.from_numpy(np.ascontiguousarray(Y)).float()
            self.det_ids = torch.from_numpy(np.ascontiguousarray(det_ids)).long()
        else:
            self.X = X_hist.contiguous().float()
            self.Y = Y.contiguous().float()
            self.det_ids = det_ids.contiguous().long()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.det_ids[idx]


def create_nhits_sequences(df, feature_cols, hist_offsets, horizon):
    X_list, Y_list, idx_list, det_list = [], [], [], []

    for det_id, df_det in df.groupby("detector_id"):
        df_det = df_det.sort_values("timestamp")

        values = df_det[feature_cols].values.astype(np.float32)
        target = df_det["congestion_index"].values.astype(np.float32)
        idx = df_det.index.values
        det_idx = df_det["det_index"].values

        n = len(df_det)
        for t in range(max(hist_offsets), n - horizon):
            X_list.append(values[[t - h for h in hist_offsets]])
            Y_list.append(target[t:t + horizon])
            idx_list.append(idx[t])
            det_list.append(det_idx[t])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(Y_list, dtype=np.float32),
        np.array(idx_list, dtype=np.int64),
        np.array(det_list, dtype=np.int64),
    )
