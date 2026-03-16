"""FastAPI prediction service for air quality PM2.5 predictions."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from api.model_loader import load_latest_model, ModelArtifact
from api.prediction_logger import PredictionLogger
from api.schemas import (
    PredictionResponse,
    GridPredictionResponse,
    GridPoint,
    ForecastResponse,
    ForecastPoint,
    HealthResponse,
    get_aqi_category,
)
from config import settings
from ingestion.weather_client import OpenMeteoClient

model_artifact: ModelArtifact | None = None
prediction_logger = PredictionLogger()
weather_client = OpenMeteoClient()
# Cache of recent predictions per location for PM2.5 lag features
_recent_predictions: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifact
    logger.info("Loading model from S3...")
    model_artifact = load_latest_model()
    logger.info(f"Model ready: version={model_artifact.version}")
    prediction_logger.start()
    yield
    await prediction_logger.stop()


app = FastAPI(
    title="Air Quality Prediction API",
    description="PM2.5 predictions for Beijing using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_features(lat: float, lon: float, weather: dict) -> dict:
    """Compute model features for a single prediction point."""
    now = datetime.now(timezone.utc)

    # Get cached PM2.5 history for lag features
    loc_key = f"{lat:.4f},{lon:.4f}"
    history = _recent_predictions.get(loc_key, [])

    # Use last prediction as lag, or 0 if no history yet
    pm25_lag_1h = history[-1] if len(history) >= 1 else 0.0
    pm25_lag_3h = history[-3] if len(history) >= 3 else pm25_lag_1h
    pm25_rolling = sum(history[-3:]) / max(len(history[-3:]), 1) if history else 0.0

    features = {
        "latitude": lat,
        "longitude": lon,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "wind_speed": weather["wind_speed"],
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "is_rush_hour": 1 if now.hour in (7, 8, 9, 17, 18, 19) else 0,
        "pm25_lag_1h": pm25_lag_1h,
        "pm25_lag_3h": pm25_lag_3h,
        "pm25_rolling_mean_3h": pm25_rolling,
    }

    return features


def _cache_prediction(lat: float, lon: float, pm25: float):
    """Cache a prediction for future lag features."""
    loc_key = f"{lat:.4f},{lon:.4f}"
    if loc_key not in _recent_predictions:
        _recent_predictions[loc_key] = []
    _recent_predictions[loc_key].append(pm25)
    # Keep only last 24 entries
    if len(_recent_predictions[loc_key]) > 24:
        _recent_predictions[loc_key] = _recent_predictions[loc_key][-24:]


def _predict(features: dict) -> float:
    """Run model inference on a single feature dict."""
    feature_values = [features[f] for f in model_artifact.feature_names]
    X = pd.DataFrame([feature_values], columns=model_artifact.feature_names)
    prediction = model_artifact.predict(X)[0]
    return float(max(0, prediction))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_version=model_artifact.version if model_artifact else "not loaded",
        loaded_at=model_artifact.loaded_at if model_artifact else "",
    )


@app.get("/predict", response_model=PredictionResponse)
async def predict(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
):
    """Predict PM2.5 for a single location."""
    weather = weather_client.get_current_weather(lat, lon)
    features = _build_features(lat, lon, weather)
    pm25 = _predict(features)
    _cache_prediction(lat, lon, pm25)

    # Log prediction
    prediction_logger.log({
        "latitude": lat,
        "longitude": lon,
        "pm25_predicted": pm25,
        "model_version": model_artifact.version,
        **features,
    })

    return PredictionResponse(
        latitude=lat,
        longitude=lon,
        pm25_predicted=round(pm25, 1),
        aqi_category=get_aqi_category(pm25),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/predict-grid", response_model=GridPredictionResponse)
async def predict_grid(
    lat_min: float = Query(39.80, description="Minimum latitude"),
    lat_max: float = Query(40.05, description="Maximum latitude"),
    lon_min: float = Query(116.20, description="Minimum longitude"),
    lon_max: float = Query(116.60, description="Maximum longitude"),
    grid_size: int = Query(10, description="Number of grid points per axis"),
):
    """Generate PM2.5 predictions across a grid for heatmap visualization."""
    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)

    # Get weather at center point (approximation for the grid)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    weather = weather_client.get_current_weather(center_lat, center_lon)

    predictions = []
    for lat in lats:
        for lon in lons:
            features = _build_features(float(lat), float(lon), weather)
            pm25 = _predict(features)
            _cache_prediction(float(lat), float(lon), pm25)
            predictions.append(
                GridPoint(
                    lat=round(float(lat), 4),
                    lon=round(float(lon), 4),
                    pm25=round(pm25, 1),
                    aqi_category=get_aqi_category(pm25),
                )
            )

    return GridPredictionResponse(predictions=predictions, count=len(predictions))


@app.get("/forecast", response_model=ForecastResponse)
async def forecast(
    lat: float = Query(39.90, description="Latitude"),
    lon: float = Query(116.40, description="Longitude"),
    days: int = Query(3, description="Forecast days (1-7)", ge=1, le=7),
):
    """Multi-day PM2.5 forecast using Open-Meteo weather forecasts.

    Fetches future hourly weather, then runs the model for each hour
    using recursive PM2.5 predictions as lag features.
    """
    # Get hourly weather forecast
    weather_forecast = weather_client.get_forecast_weather(lat, lon, forecast_days=days)

    # Start with current prediction as seed for PM2.5 lags
    current_weather = weather_client.get_current_weather(lat, lon)
    seed_features = _build_features(lat, lon, current_weather)
    seed_pm25 = _predict(seed_features)

    # Rolling PM2.5 history for recursive forecasting
    pm25_history = [seed_pm25] * 3  # initialize with current estimate

    forecast_points = []
    for wx in weather_forecast:
        ts = datetime.fromisoformat(wx["timestamp"])

        # Build features using forecast weather + recursive PM2.5 lags
        features = {
            "latitude": lat,
            "longitude": lon,
            "temperature": wx["temperature"],
            "humidity": wx["humidity"],
            "wind_speed": wx["wind_speed"],
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "is_rush_hour": 1 if ts.hour in (7, 8, 9, 17, 18, 19) else 0,
            "pm25_lag_1h": pm25_history[-1],
            "pm25_lag_3h": pm25_history[-3] if len(pm25_history) >= 3 else pm25_history[0],
            "pm25_rolling_mean_3h": sum(pm25_history[-3:]) / len(pm25_history[-3:]),
        }

        pm25 = _predict(features)
        pm25_history.append(pm25)

        forecast_points.append(
            ForecastPoint(
                timestamp=wx["timestamp"],
                hour=ts.hour,
                pm25=round(pm25, 1),
                aqi_category=get_aqi_category(pm25),
                temperature=wx["temperature"],
                humidity=wx["humidity"],
                wind_speed=wx["wind_speed"],
            )
        )

    return ForecastResponse(
        latitude=lat,
        longitude=lon,
        forecast_hours=len(forecast_points),
        predictions=forecast_points,
    )


# Serve frontend (for local development — in production nginx handles this)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
