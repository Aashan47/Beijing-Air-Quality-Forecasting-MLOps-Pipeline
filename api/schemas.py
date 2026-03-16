"""Pydantic models and AQI category mapping for the prediction API."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    latitude: float
    longitude: float
    pm25_predicted: float
    aqi_category: str
    timestamp: str


class GridPoint(BaseModel):
    lat: float
    lon: float
    pm25: float
    aqi_category: str


class GridPredictionResponse(BaseModel):
    predictions: list[GridPoint]
    count: int


class ForecastPoint(BaseModel):
    timestamp: str
    hour: int
    pm25: float
    aqi_category: str
    temperature: float
    humidity: float
    wind_speed: float


class ForecastResponse(BaseModel):
    latitude: float
    longitude: float
    forecast_hours: int
    predictions: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    loaded_at: str


def get_aqi_category(pm25: float) -> str:
    """Map PM2.5 concentration to EPA AQI category.

    Based on EPA breakpoints for PM2.5 (24-hour average).
    """
    if pm25 <= 12.0:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"
