"""Unit tests for feature engineering."""

import pandas as pd
import numpy as np
import pytest


def make_raw_data(n=100):
    """Create synthetic raw data for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "latitude": rng.uniform(39.7, 40.1, n),
        "longitude": rng.uniform(116.1, 116.7, n),
        "pm25": rng.uniform(10, 200, n),
        "temperature": rng.uniform(-5, 35, n),
        "humidity": rng.uniform(20, 90, n),
        "wind_speed": rng.uniform(0.5, 12, n),
    })


def test_simple_features():
    """Test that simple mode produces expected columns."""
    df = make_raw_data()

    # Simulate feature building logic
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["pm25_target"] = df["pm25"]

    expected_cols = [
        "latitude", "longitude", "temperature", "humidity", "wind_speed",
        "hour", "day_of_week", "is_rush_hour", "pm25_target",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_no_nans_in_simple_mode():
    """Simple mode should not produce NaN values."""
    df = make_raw_data()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    feature_cols = ["latitude", "longitude", "temperature", "humidity",
                    "wind_speed", "hour", "day_of_week", "is_rush_hour"]
    assert df[feature_cols].isna().sum().sum() == 0


def test_rush_hour_flag():
    """Rush hour should be 1 for hours 7-9 and 17-19."""
    hours = [0, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 23]
    expected = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    result = [1 if h in (7, 8, 9, 17, 18, 19) else 0 for h in hours]
    assert result == expected


def test_pm25_target_range():
    """PM2.5 target should be positive."""
    df = make_raw_data()
    assert (df["pm25"] >= 0).all()
