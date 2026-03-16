"""Ingestion pipeline: fetch PM2.5 + weather from Open-Meteo (free, no API key)."""

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from config import settings
from pipelines.utils import upload_parquet_to_s3, generate_s3_key

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Grid of points across Beijing for spatial coverage
BEIJING_GRID = [
    (39.75, 116.15), (39.75, 116.40), (39.75, 116.65),
    (39.90, 116.15), (39.90, 116.40), (39.90, 116.65),
    (40.05, 116.15), (40.05, 116.40), (40.05, 116.65),
]


def _fetch_location_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch PM2.5 + weather for a single location from Open-Meteo."""
    import httpx

    # Fetch air quality
    aq_response = httpx.get(
        AIR_QUALITY_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5",
            "start_date": start_date,
            "end_date": end_date,
        },
        timeout=30.0,
    )
    aq_response.raise_for_status()
    aq_data = aq_response.json()["hourly"]

    # Fetch weather
    wx_response = httpx.get(
        WEATHER_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        },
        timeout=30.0,
    )
    wx_response.raise_for_status()
    wx_data = wx_response.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": aq_data["time"],
        "latitude": lat,
        "longitude": lon,
        "pm25": aq_data["pm2_5"],
        "temperature": wx_data["temperature_2m"],
        "humidity": wx_data["relative_humidity_2m"],
        "wind_speed": [round(w / 3.6, 1) if w is not None else None
                       for w in wx_data["wind_speed_10m"]],  # km/h -> m/s
    })
    return df.dropna()


def run_ingestion(days_back: int = 90) -> pd.DataFrame:
    """Fetch PM2.5 + weather for Beijing grid from Open-Meteo.

    Args:
        days_back: Number of days of historical data to fetch (default: 90).

    Returns:
        Combined DataFrame uploaded to S3.
    """
    now = datetime.now(timezone.utc)
    # Open-Meteo archive has a ~5 day lag, so end_date is 5 days ago
    end_date = (now - timedelta(days=5)).strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    logger.info(f"Fetching data from {start_date} to {end_date} for {len(BEIJING_GRID)} grid points")

    all_dfs = []
    for lat, lon in BEIJING_GRID:
        try:
            df = _fetch_location_data(lat, lon, start_date, end_date)
            all_dfs.append(df)
            logger.info(f"  ({lat}, {lon}): {len(df)} rows")
        except Exception as e:
            logger.warning(f"  ({lat}, {lon}): failed - {e}")

    if not all_dfs:
        logger.warning("No data fetched from any location")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    logger.info(f"Total: {len(combined)} rows from {len(all_dfs)} locations")

    # Upload to S3
    if not combined.empty:
        s3_key = generate_s3_key("raw")
        upload_parquet_to_s3(combined, s3_key)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Fetch air quality + weather data from Open-Meteo")
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Days of historical data to fetch (default: 90)",
    )
    args = parser.parse_args()

    df = run_ingestion(args.days_back)
    if df.empty:
        logger.warning("No data fetched")
    else:
        logger.info(f"Successfully ingested {len(df)} records")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f}")


if __name__ == "__main__":
    main()
