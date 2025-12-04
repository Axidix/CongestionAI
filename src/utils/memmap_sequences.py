"""
Memory-mapped sequence creation for large-scale training.
Stores sequences on disk and loads them on-demand during training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
import tempfile


class MemmapSequenceBuilder:
    """
    Creates and manages memory-mapped numpy arrays for sequence data.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Args:
            cache_dir: Directory to store memmap files. If None, uses temp directory.
        """
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "congestion_memmap")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._files_created = []
    
    def create_sequences_memmap(
        self,
        df,
        feature_cols: List[str],
        history_offsets: List[int],
        forecast_horizon: int,
        target_col: str = "congestion_index",
        prefix: str = "train"
    ) -> Tuple[np.memmap, np.memmap, np.ndarray, np.ndarray]:
        """
        Create memory-mapped sequences from dataframe.
        
        Returns:
            X_hist: memmap of shape (N, history_len, n_features)
            Y: memmap of shape (N, forecast_horizon)
            idx: numpy array of starting indices
            det_idx: numpy array of detector indices
        """
        print(f"  Creating memmap sequences for {prefix}...")
        
        # First pass: count valid samples
        detectors = df["detector_id"].unique()
        history_len = len(history_offsets)
        max_offset = max(history_offsets)
        n_features = len(feature_cols)
        
        # Count samples per detector
        sample_counts = []
        for det_id in detectors:
            det_df = df[df["detector_id"] == det_id]
            n_rows = len(det_df)
            n_samples = n_rows - max_offset - forecast_horizon
            if n_samples > 0:
                sample_counts.append((det_id, n_samples))
        
        total_samples = sum(c[1] for c in sample_counts)
        print(f"    Total samples: {total_samples:,}")
        
        # Create memmap files
        X_path = os.path.join(self.cache_dir, f"{prefix}_X.dat")
        Y_path = os.path.join(self.cache_dir, f"{prefix}_Y.dat")
        
        X_shape = (total_samples, history_len, n_features)
        Y_shape = (total_samples, forecast_horizon)
        
        X_size_gb = np.prod(X_shape) * 4 / 1e9
        Y_size_gb = np.prod(Y_shape) * 4 / 1e9
        print(f"    X shape: {X_shape} ({X_size_gb:.2f} GB)")
        print(f"    Y shape: {Y_shape} ({Y_size_gb:.2f} GB)")
        
        # Create memmap arrays
        X_memmap = np.memmap(X_path, dtype='float32', mode='w+', shape=X_shape)
        Y_memmap = np.memmap(Y_path, dtype='float32', mode='w+', shape=Y_shape)
        self._files_created.extend([X_path, Y_path])
        
        # Collect indices
        idx_list = []
        det_idx_list = []
        
        # Second pass: fill memmap arrays
        sample_idx = 0
        for det_id, n_samples in sample_counts:
            det_df = df[df["detector_id"] == det_id].copy()
            det_df = det_df.sort_values("timestamp").reset_index()
            
            det_index = det_df["det_index"].iloc[0]
            feature_matrix = det_df[feature_cols].values.astype(np.float32)
            target_series = det_df[target_col].values.astype(np.float32)
            original_indices = det_df["orig_idx"].values
            
            for i in range(max_offset, len(det_df) - forecast_horizon):
                # History features
                hist_indices = [i - offset for offset in history_offsets]
                X_memmap[sample_idx] = feature_matrix[hist_indices]
                
                # Future targets
                Y_memmap[sample_idx] = target_series[i+1:i+1+forecast_horizon]
                
                # Indices
                idx_list.append(original_indices[i])
                det_idx_list.append(det_index)
                
                sample_idx += 1
        
        # Flush to disk
        X_memmap.flush()
        Y_memmap.flush()
        
        idx = np.array(idx_list, dtype=np.int64)
        det_idx = np.array(det_idx_list, dtype=np.int64)
        
        print(f"    Memmap files created: {X_path}")
        
        # Return read-only memmaps for safety
        X_memmap_ro = np.memmap(X_path, dtype='float32', mode='r', shape=X_shape)
        Y_memmap_ro = np.memmap(Y_path, dtype='float32', mode='r', shape=Y_shape)
        
        return X_memmap_ro, Y_memmap_ro, idx, det_idx
    
    def cleanup(self):
        """Remove all created memmap files."""
        for f in self._files_created:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed: {f}")
        self._files_created = []


class MemmapDataset(Dataset):
    """
    PyTorch Dataset that reads from memory-mapped arrays.
    Only loads data into RAM when __getitem__ is called.
    """
    
    def __init__(self, X_memmap: np.memmap, Y_memmap: np.memmap, det_idx: np.ndarray):
        self.X = X_memmap
        self.Y = Y_memmap
        self.det_idx = det_idx
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        # .copy() is crucial - it reads from disk into a new array
        x = torch.from_numpy(self.X[idx].copy())
        y = torch.from_numpy(self.Y[idx].copy())
        det = torch.tensor(self.det_idx[idx], dtype=torch.long)
        return x, y, det