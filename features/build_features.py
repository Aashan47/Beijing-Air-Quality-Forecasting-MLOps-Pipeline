"""Feature engineering pipeline: transform raw data into model-ready features."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from loguru import logger

from pipelines.utils import download_all_parquets, upload_parquet_to_s3


def build_training_features(source_prefix: str = "raw/", mode: str = "simple") -> pd.DataFrame:
    """Build features from raw data.

    Args:
        source_prefix: S3 prefix to load raw parquet files from
        mode: "simple" (v1, weather + time + location) or "full" (v2, adds rolling features)

    Returns:
        Feature DataFrame ready for training
    """
    # Load raw data
    df = download_all_parquets(source_prefix)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} raw rows for feature engineering (mode={mode})")

    # Sort by location then time for correct lag computation
    df = df.sort_values(["latitude", "longitude", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby(["latitude", "longitude"])

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # PM2.5 lag features (available in both modes — strongest predictors)
    df["pm25_lag_1h"] = grouped["pm25"].transform(lambda x: x.shift(1))
    df["pm25_lag_3h"] = grouped["pm25"].transform(lambda x: x.shift(3))
    df["pm25_rolling_mean_3h"] = grouped["pm25"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    if mode == "full":
        # Additional lag/trend features
        df["pm25_lag_24h"] = grouped["pm25"].transform(lambda x: x.shift(24))

        # Temperature lag (1 hour)
        df["temperature_lag_1h"] = grouped["temperature"].transform(lambda x: x.shift(1))

        # Humidity change over 1 hour
        df["humidity_change_1h"] = grouped["humidity"].transform(lambda x: x.diff())

        # Wind speed trend (slope over 3-hour window)
        df["wind_speed_trend"] = grouped["wind_speed"].transform(
            lambda x: x.rolling(3, min_periods=2).apply(
                lambda w: np.polyfit(range(len(w)), w, 1)[0] if len(w) >= 2 else 0,
                raw=False,
            )
        )

    # Target: next hour's PM2.5 (features at time T predict PM2.5 at T+1)
    df["pm25_target"] = grouped["pm25"].transform(lambda x: x.shift(-1))

    # Select feature columns based on mode
    feature_cols = ["latitude", "longitude", "temperature", "humidity", "wind_speed",
                    "hour", "day_of_week", "is_rush_hour",
                    "pm25_lag_1h", "pm25_lag_3h", "pm25_rolling_mean_3h"]

    if mode == "full":
        feature_cols += ["pm25_lag_24h", "temperature_lag_1h",
                         "humidity_change_1h", "wind_speed_trend"]

    output_cols = ["timestamp"] + feature_cols + ["pm25_target"]
    df = df[output_cols].dropna().reset_index(drop=True)

    logger.info(f"Built {len(df)} feature rows with {len(feature_cols)} features")

    # Upload to S3
    upload_parquet_to_s3(df, "features/training_dataset.parquet")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build training features")
    parser.add_argument(
        "--source", default="raw/", help="S3 prefix for raw data (default: raw/)"
    )
    parser.add_argument(
        "--mode", choices=["simple", "full"], default="simple",
        help="Feature mode: simple (v1) or full (v2)",
    )
    args = parser.parse_args()

    df = build_training_features(args.source, args.mode)
    logger.info(f"Feature dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"PM2.5 target stats:\n{df['pm25_target'].describe()}")


if __name__ == "__main__":
    main()
