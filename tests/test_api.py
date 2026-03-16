"""Unit tests for the FastAPI prediction API."""

import pytest
from api.schemas import get_aqi_category, PredictionResponse, GridPoint


def test_aqi_good():
    assert get_aqi_category(5.0) == "Good"
    assert get_aqi_category(12.0) == "Good"


def test_aqi_moderate():
    assert get_aqi_category(12.1) == "Moderate"
    assert get_aqi_category(35.4) == "Moderate"


def test_aqi_usg():
    assert get_aqi_category(35.5) == "Unhealthy for Sensitive Groups"
    assert get_aqi_category(55.4) == "Unhealthy for Sensitive Groups"


def test_aqi_unhealthy():
    assert get_aqi_category(55.5) == "Unhealthy"
    assert get_aqi_category(150.4) == "Unhealthy"


def test_aqi_very_unhealthy():
    assert get_aqi_category(150.5) == "Very Unhealthy"
    assert get_aqi_category(250.4) == "Very Unhealthy"


def test_aqi_hazardous():
    assert get_aqi_category(250.5) == "Hazardous"
    assert get_aqi_category(500.0) == "Hazardous"


def test_prediction_response_model():
    resp = PredictionResponse(
        latitude=39.9,
        longitude=116.4,
        pm25_predicted=75.3,
        aqi_category="Unhealthy",
        timestamp="2024-01-01T00:00:00Z",
    )
    assert resp.pm25_predicted == 75.3


def test_grid_point_model():
    point = GridPoint(lat=39.9, lon=116.4, pm25=50.0, aqi_category="Unhealthy for Sensitive Groups")
    assert point.pm25 == 50.0
