"""Generate synthetic air quality training data for Beijing (optional fallback)."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from loguru import logger

from pipelines.utils import upload_parquet_to_s3


def generate_synthetic_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic PM2.5 data for Beijing.

    PM2.5 patterns:
    - Higher in winter (Nov-Feb), lower in summer
    - Peaks during rush hours (7-9, 17-19)
    - Higher with low wind, high humidity
    - Higher at night due to temperature inversions
    """
    rng = np.random.default_rng(seed)

    # Beijing bbox: 116.10,39.70,116.70,40.15
    latitudes = rng.uniform(39.70, 40.15, n_rows)
    longitudes = rng.uniform(116.10, 116.70, n_rows)

    # Timestamps spanning 30 days, hourly
    base_time = pd.Timestamp("2024-11-01", tz="UTC")
    timestamps = [base_time + pd.Timedelta(hours=i) for i in range(n_rows)]

    hours = np.array([t.hour for t in timestamps])
    days_of_week = np.array([t.dayofweek for t in timestamps])
    months = np.array([t.month for t in timestamps])

    # Temperature: Beijing ranges -5C to 35C depending on season
    # Nov-Feb: cold (-5 to 10), Jun-Aug: hot (20-35)
    base_temp = 15 + 15 * np.sin(2 * np.pi * (months - 7) / 12)
    temperature = base_temp + rng.normal(0, 3, n_rows)

    # Humidity: 30-90%
    humidity = 50 + 20 * np.sin(2 * np.pi * (months - 8) / 12) + rng.normal(0, 10, n_rows)
    humidity = np.clip(humidity, 20, 95)

    # Wind speed: 0.5-12 m/s
    wind_speed = 3 + rng.exponential(2, n_rows)
    wind_speed = np.clip(wind_speed, 0.5, 15)

    # PM2.5 model: base + seasonal + hourly + weather effects + noise
    pm25_base = 80  # Beijing baseline is high

    # Seasonal: worse in winter (heating season)
    seasonal_effect = 60 * np.cos(2 * np.pi * (months - 1) / 12)

    # Rush hour peaks
    rush_hour = np.where(np.isin(hours, [7, 8, 9, 17, 18, 19]), 25, 0)

    # Night inversion (worse air at night)
    night_effect = np.where((hours >= 22) | (hours <= 5), 15, 0)

    # Weather effects
    wind_effect = -8 * wind_speed  # More wind = less pollution
    humidity_effect = 0.3 * humidity  # High humidity traps pollution

    # Weekend slightly lower
    weekend_effect = np.where(days_of_week >= 5, -15, 0)

    pm25 = (
        pm25_base
        + seasonal_effect
        + rush_hour
        + night_effect
        + wind_effect
        + humidity_effect
        + weekend_effect
        + rng.normal(0, 15, n_rows)
    )
    pm25 = np.clip(pm25, 5, 500)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "latitude": latitudes,
        "longitude": longitudes,
        "pm25": np.round(pm25, 1),
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 1),
    })

    logger.info(f"Generated {len(df)} synthetic rows. PM2.5 range: {df['pm25'].min()}-{df['pm25'].max()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows")
    parser.add_argument(
        "--output",
        default="raw/synthetic/training_bootstrap.parquet",
        help="S3 key for output",
    )
    args = parser.parse_args()

    df = generate_synthetic_data(args.rows)
    upload_parquet_to_s3(df, args.output)
    logger.info(f"Uploaded synthetic data to s3://{args.output}")


if __name__ == "__main__":
    main()
