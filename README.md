# CongestionAI
Real-time traffic congestion forecasting and visualization for Berlin

---

## 1. Product Overview

CongestionAI is a traffic intelligence web application that helps users anticipate congestion and plan trips across Berlin.

### What it does for a user

- Forecasts congestion up to 24 hours ahead.
- Visualizes predicted congestion on a Berlin road map (road-segment level).
- Lets the user explore congestion evolution hour-by-hour.
- Supports itinerary decisions by highlighting when/where congestion is lower.

The system predicts a continuous congestion index in the range [0, 1] (not a binary label), enabling nuanced interpretation of traffic intensity.

---

## 2. Technical Summary

This repository is a full-stack ML system:
- data ingestion from external APIs (traffic + weather),
- feature engineering and temporal modeling,
- deep learning forecasting (PyTorch TCN),
- backend inference and caching,
- authenticated API exposure,
- and a Streamlit frontend for visualization.

The implementation emphasizes production constraints:
- clear separation of training vs inference,
- fault tolerance and caching,
- and observability (logs + runtime diagnostics).

---

## 3. Architecture

End-to-end flow:

1) Traffic API + Weather API ingestion
2) Feature engineering for all detectors
3) Temporal model inference (24h horizon)
4) Postprocessing + detector-to-road interpolation
5) Cached forecast artifact written to gui/data/forecast.json
6) API serves the cached forecast to the frontend
7) Streamlit UI renders the map and time controls

Conceptual diagram:

Traffic API      Weather API
    |               |
    +------v--------+
           |
   Feature engineering (backend)
           |
           v
   Temporal CNN model (TCN)
           |
           v
 Detector-level predictions (24h)
           |
           v
 Road-level interpolation (IDW)
           |
           v
        FastAPI backend
           |
           v
     Streamlit frontend

---

## 4. Repository Structure

CongestionAI/
  backend/                Inference, ingestion, API, scheduler
    api.py                FastAPI app (HTTP interface)
    service.py            Main orchestration logic for refresh + caching
    scheduler.py          Periodic refresh loop (hourly)
    config.py             Centralized configuration (paths, keys, constants)
    data/
      traffic.py          Traffic ingestion, history accumulation, cleaning
      weather.py          Weather ingestion and formatting
      features.py         Feature construction for inference (X tensor)
      mapping.py          Detector-to-road interpolation (IDW)
    model/
      loader.py           Model loading and caching (checkpoint -> torch model)
      predictor.py        Batched inference + postprocess (sigmoid, safety)
    scalers/              Training artifacts reused at inference
    logs/                 Runtime logs

  gui/                    Streamlit UI
    app.py                Frontend entry point
    data/
      forecast.json        Cached backend output consumed by the UI
      berlin_roads.geojson.gz

  src/                    Training + evaluation code (PyTorch)
    models/               Model definitions (TCN)
    model_pipelines/       Training loops, losses, evaluation
    utils/                Preprocessing, scaling, feature utilities

  deploy/
    congestion-api.service  systemd service file(s)

  train_final_model.py     Final training entry point (portfolio-grade run)
  training_journal.txt     Experiment notes and decisions
  README.md

---

## 5. Backend (Inference + API)

### Responsibilities

The backend:
- fetches and accumulates traffic data,
- fetches weather history,
- builds inference features for all detectors,
- loads the trained model checkpoint,
- runs batched inference,
- constrains outputs to [0, 1] via sigmoid,
- interpolates detector predictions to road segments,
- writes a single cached artifact (gui/data/forecast.json),
- serves it through a FastAPI endpoint.

### Key files

backend/service.py
- Primary orchestration layer.
- The function refresh_forecast() is the main pipeline:
  1) traffic_df = fetch_and_accumulate()
  2) weather_df = fetch_weather_history(...)
  3) X, det_indices, detector_ids = prepare_inference_batch(...)
  4) pred_raw = predict_batch(model, X, det_indices)
  5) pred_sigmoid = sigmoid(pred_raw) and clip to [0, 1]
  6) postprocess_predictions(...) including detector->road expansion
  7) save_forecast(...) -> gui/data/forecast.json

backend/api.py
- FastAPI app exposing endpoints (example):
  - GET /forecast  (requires API key)
  - GET /health

backend/scheduler.py
- APScheduler-based periodic refresh.
- Runs refresh_forecast() every hour.
- Designed to keep the forecast fresh without manual intervention.

backend/model/predictor.py
- predict_batch(): runs model inference on the full detector batch.
- postprocess_predictions(): prepares JSON output for the GUI and mapping layer.
- Output constraints:
  - raw model outputs are unbounded,
  - sigmoid is applied in inference to enforce [0, 1],
  - clipping is only a final safety net.

backend/data/mapping.py
- Expands detector predictions to road-segment predictions using IDW mapping.
- Handles roads without nearby detectors via a configurable default value.

---

## 6. Model and Output Semantics

### Model
- Temporal Convolutional Network (TCN), PyTorch implementation.

### Why TCN
- Efficient for multivariate time series.
- Stable and parallelizable training/inference.
- Suitable for medium-length sequences (48h history) with engineered lags.

### Output range
- The model’s direct output is a real-valued score.
- A sigmoid transformation is applied during backend inference:
  sigmoid(x) = 1 / (1 + exp(-x))
- The application-level congestion index is defined as the sigmoid output in [0, 1].

---

## 7. Training Pipeline

Training code lives in src/.

Entry point:
- train_final_model.py

Responsibilities:
- build training tensors from prepared historical data,
- fit scalers and persist them (StandardScaler + MinMaxScaler),
- train the TCN model,
- evaluate on validation slices,
- persist final model checkpoint and inference artifacts used by the backend.

Artifacts typically saved:
- model checkpoint (.pt)
- std_scaler.joblib
- mm_scaler.joblib
- det2idx.joblib (mapping from detector_id to embedding index)

Key principle:
- backend inference reuses the exact artifacts produced by training to avoid train/serve skew.

---

## 8. Frontend (Streamlit GUI)

gui/app.py
- Fetches the forecast from the backend API (or a proxy endpoint).
- Renders:
  - map overlay for road-level congestion values,
  - time navigation / hour selector,
  - optional supporting UI components (legend, metadata, freshness).

The GUI is intentionally stateless:
- It does not compute forecasts.
- It consumes backend outputs and focuses on visualization.

---

## 9. Deployment

### Components

Oracle Cloud VM
- Runs backend services (FastAPI + scheduler).
- Uses systemd to keep services running and auto-restart on failure.

DuckDNS
- Provides a stable domain name pointing to the VM public IP.

Cloudflare Worker
- Acts as an HTTPS-accessible proxy between Streamlit Cloud and the backend.
- Used to mitigate outbound connectivity constraints observed from Streamlit Cloud to the VM.

Streamlit Cloud
- Hosts the GUI.

### Deployment flow

Oracle VM (FastAPI + scheduler)
  -> DuckDNS domain
      -> Cloudflare Worker (HTTPS)
          -> Streamlit Cloud frontend

---

## 10. Observability and Operations

The system is designed for operational clarity:
- systemd supervises the backend processes.
- backend writes structured logs and statistics:
  - feature tensor stats,
  - raw and post-sigmoid prediction stats,
  - forecast age and refresh timing,
  - mapping expansion diagnostics (detector vs road-level coverage).

Recommended operational checks:
- GET /health for freshness
- monitor backend logs (journalctl) for anomalies

---

## 11. Collaboration Notes

This repository is structured to support collaboration:
- separation of concerns (training vs inference vs UI),
- reproducible artifacts for inference,
- centralized configuration,
- and modular data/model components.

Typical contribution areas:
- improved feature engineering,
- improved spatial interpolation / coverage,
- model iteration and evaluation,
- UI improvements (itinerary support, overlays, UX).

---

## 12. Known Limitations and Planned Work

- Further validation of road-level interpolation behavior.
- Additional itinerary tools (routing-aware aggregation over road segments).
- Performance improvements for large mapping expansion.
- Extended monitoring and alerting for forecast freshness and drift.
