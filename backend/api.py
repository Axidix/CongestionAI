"""
Backend REST API
================

Simple FastAPI server to expose forecast data to remote clients (e.g., Streamlit Cloud).

Endpoints:
----------
- GET /forecast       → Full forecast JSON (roads + weather)
- GET /health         → Health check
- GET /forecast/age   → How old is the current forecast

Usage:
------
    # Run the API server
    uvicorn backend.api:app --host 0.0.0.0 --port 8000
    
    # Or with auto-reload for development
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

The scheduler should still run separately (via systemd or cron) to refresh forecasts.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from backend.service import get_current_forecast, get_forecast_age

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CongestionAI Backend API",
    description="Traffic congestion forecast API for Berlin",
    version="1.0.0",
)

# Allow CORS from Streamlit Cloud and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.streamlit.app",  # Streamlit Cloud
        "http://localhost:8501",     # Local Streamlit
        "http://127.0.0.1:8501",
        "*",  # Allow all for now (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "CongestionAI Backend",
        "version": "1.0.0",
        "endpoints": {
            "/forecast": "Get full forecast data",
            "/health": "Health check",
            "/forecast/age": "Get forecast age",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    forecast = get_current_forecast()
    age = get_forecast_age()
    
    return {
        "status": "healthy" if forecast else "degraded",
        "has_forecast": bool(forecast),
        "forecast_age_minutes": age.total_seconds() / 60 if age else None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/forecast")
async def forecast():
    """
    Get the current forecast data.
    
    Returns the full forecast.json including:
    - data: Road-level congestion predictions (73,687 roads × 25 hours)
    - weather: Current + 24h hourly weather forecast
    - summary: Aggregate statistics
    - metadata: Timestamp, model info, etc.
    """
    data = get_current_forecast()
    
    if not data:
        raise HTTPException(
            status_code=503,
            detail="No forecast available. The backend may be starting up."
        )
    
    return JSONResponse(
        content=data,
        headers={
            "Cache-Control": "public, max-age=300",  # Cache for 5 min
        }
    )


@app.get("/forecast/age")
async def forecast_age():
    """Get how old the current forecast is."""
    age = get_forecast_age()
    
    if age is None:
        raise HTTPException(status_code=503, detail="No forecast available")
    
    return {
        "age_minutes": age.total_seconds() / 60,
        "age_hours": age.total_seconds() / 3600,
        "is_stale": age.total_seconds() > 3600,  # Stale if > 1 hour
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
