# CongestionAI

CongestionAI is a city-scale traffic congestion forecasting system for Berlin.
It predicts road congestion up to 24 hours ahead and provides a web interface
to explore future traffic conditions over time.

The project is designed as an end-to-end machine learning system, covering
data ingestion, feature engineering, model training, production inference,
deployment, and visualization.

The objective of the project is twofold:
- deliver a usable congestion forecasting tool
- demonstrate the design and deployment of a production-grade ML system

---

## 1. Product Overview

CongestionAI provides:

- hourly congestion forecasts for the next 24 hours
- city-wide road-level congestion visualization
- time navigation to explore congestion evolution
- a foundation for itinerary-aware decision support

Rather than reacting to current traffic conditions, the system enables users
to anticipate congestion and plan accordingly.

The application is currently focused on Berlin but was designed so that
additional cities could be integrated with minimal changes.

## 2. Technical Overview

CongestionAI is implemented as a production-oriented ML pipeline:

- ingestion of traffic and weather data from external APIs
- time-series feature engineering
- deep learning inference using a Temporal Convolutional Network (TCN)
- spatial interpolation from detectors to road segments
- backend caching and API exposure
- frontend visualization via Streamlit

Design principles:
- strict separation between training and inference code
- reproducible inference using saved training artifacts
- fault-tolerant backend behavior
- observability through logs and diagnostics

---
## 3. Repository Structure

Top-level directories:

- `backend/` — Inference + ingestion + scheduler + FastAPI API
  - `api.py` — FastAPI HTTP interface (`/forecast`, `/health`, auth)
  - `service.py` — Main orchestration: refresh pipeline + caching
  - `scheduler.py` — Hourly refresh loop (APScheduler)
  - `config.py` — Central configuration (paths, keys, constants)
  - `data/` — Data ingestion + feature engineering
    - `traffic.py` — Traffic ingestion, history accumulation, cleaning
    - `weather.py` — Weather ingestion and formatting for GUI/features
    - `features.py` — Feature construction for inference (build X tensor)
    - `mapping.py` — Detector-to-road expansion (IDW interpolation)
  - `model/` — Model loading and inference helpers
    - `loader.py` — Load checkpoint into torch model (cached/singleton)
    - `predictor.py` — Batched inference + postprocess
  - `scalers/` — Training artifacts reused at inference (joblib)
  - `logs/` — Runtime logs

- `gui/` — Streamlit frontend
  - `app.py` — Streamlit entry point
  - `data/forecast.json` — Cached backend output consumed by the UI
  - `data/berlin_roads.geojson.gz` — Road geometry for rendering

- `src/` — Training + evaluation code (PyTorch)
  - `models/` — Model definitions (TCN and modules)
  - `model_pipelines/` — Training loops, losses, evaluation utilities
  - `utils/` — Preprocessing, scalers, feature utilities

- `deploy/` — Deployment assets
  - `congestion-api.service` — systemd unit(s)

Top-level entry points / docs:

- `train_final_model.py` — Final consolidated training run for deployment
- `training_journal.txt` — Experiment notes and decisions
- `README.md` — Project documentation

## 4. Backend Architecture

### Responsibilities

The backend produces and serves a single authoritative artifact:
a cached congestion forecast.

Main responsibilities:
- ingest and accumulate traffic data
- fetch recent weather data
- build inference-ready features
- run model inference
- interpolate predictions to road level
- cache and serve the forecast

### Core orchestration

The main pipeline is implemented in `backend/service.py` via
`refresh_forecast()`:

1. Accumulate recent traffic history
2. Fetch weather history
3. Build features for all detectors
4. Run model inference
5. Postprocess and interpolate predictions
6. Save `forecast.json` for the GUI

Failures are logged and the previous forecast is preserved whenever possible.

### API layer

`backend/api.py` exposes a FastAPI application with endpoints such as:
- `GET /forecast`
- `GET /health`

Access is protected using an API key.

### Scheduling

`backend/scheduler.py` runs the refresh pipeline hourly using APScheduler.

## 5. Machine Learning Model

### Model type

The forecasting model is a Temporal Convolutional Network (TCN) implemented
in PyTorch.

TCNs were selected for:
- stable training on long sequences
- parallel computation
- strong performance on multivariate time series

### Training and experimentation

Training code is located in `src/` and reflects an iterative experimentation
process rather than a single training script.

Experimentation included:
- feature engineering strategies
- temporal context lengths
- spike-aware loss weighting
- validation splits respecting temporal order
- train / serve consistency checks

Key experiment decisions and observations are documented in
`training_journal.txt`.

`train_final_model.py` consolidates the selected configuration into a
deployment-ready training run and outputs the artifacts reused by the backend.

---

## 6. Frontend (GUI)

The frontend is implemented using Streamlit and focuses exclusively on
visualization.

Responsibilities:
- fetch forecast data from the backend
- render road-level congestion maps
- allow time navigation across forecast horizons
- display forecast metadata

The frontend is intentionally stateless.

---

## 7. Deployment

Infrastructure components:
- Oracle Cloud VM: backend hosting
- systemd: backend process supervision
- DuckDNS: public domain name
- Cloudflare Worker: HTTPS proxy
- Streamlit Cloud: frontend hosting

This setup ensures reliability, security, and compatibility with Streamlit
Cloud outbound networking constraints.

---

## 8. Observability and Future Work

The system includes diagnostics for:
- refresh duration and scheduling
- input feature statistics
- prediction distribution checks
- detector vs road-level coverage

Planned improvements:
- a model insights page in the GUI
- latency optimization in feature building and interpolation
- itinerary-aware congestion aggregation

---

## 9. Project Status

- end-to-end pipeline operational
- hourly forecasting stable
- backend and frontend fully integrated
- ongoing refinement for performance and interpretability
