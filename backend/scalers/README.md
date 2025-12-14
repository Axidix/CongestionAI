# Scalers Directory

This directory contains the fitted scalers from training, required for inference.

## Required Files

After running `train_final_model.py`, this directory should contain:

1. **std_scaler.joblib** - StandardScaler for normalizing:
   - temperature, precipitation, visibility
   - congestion_index, free_flow_speed
   - delta_1h, delta_2h, delta_4h, delta_6h
   - rolling_vol_3h
   - congestion_index_lag_48h, congestion_index_lag_168h

2. **mm_scaler.joblib** - MinMaxScaler for normalizing:
   - lon, lat, year, season

3. **det2idx.joblib** - Dictionary mapping detector_id → model index:
   - Used to get the correct embedding for each detector
   - Keys: detector_id strings
   - Values: integer indices (0 to num_detectors-1)

## Usage

```python
import joblib
from pathlib import Path

scaler_dir = Path("backend/scalers")

std_scaler = joblib.load(scaler_dir / "std_scaler.joblib")
mm_scaler = joblib.load(scaler_dir / "mm_scaler.joblib")
det2idx = joblib.load(scaler_dir / "det2idx.joblib")
```

## Generation

Run the training script to generate these files:

```bash
python train_final_model.py
```

The script will automatically save scalers to this directory.
