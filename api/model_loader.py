"""Download and load the latest XGBoost model from S3."""

import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from config import settings
from pipelines.utils import get_s3_client


class ModelArtifact:
    def __init__(self, model, feature_names: list[str], version: str, loaded_at: str, use_booster: bool = False):
        self.model = model
        self.feature_names = feature_names
        self.version = version
        self.loaded_at = loaded_at
        self.use_booster = use_booster

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using either XGBRegressor or raw Booster."""
        if self.use_booster:
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            return self.model.predict(dmatrix)
        return self.model.predict(X)


def load_latest_model() -> ModelArtifact:
    """Download model.tar.gz from S3, extract, and load XGBoost model."""
    s3 = get_s3_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "model.tar.gz")

        # Download model artifact
        s3_key = "models/model.tar.gz"
        logger.info(f"Downloading model from s3://{settings.s3_bucket}/{s3_key}")
        s3.download_file(settings.s3_bucket, s3_key, tar_path)

        # Get version from S3 object metadata
        response = s3.head_object(Bucket=settings.s3_bucket, Key=s3_key)
        version = response["LastModified"].strftime("%Y%m%d_%H%M%S")

        # Extract
        extract_dir = os.path.join(tmpdir, "model")
        os.makedirs(extract_dir)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        # List extracted files for debugging
        extracted_files = os.listdir(extract_dir)
        logger.info(f"Extracted files: {extracted_files}")

        # Try loading as XGBRegressor first (local training format: xgboost-model.json)
        json_model_path = os.path.join(extract_dir, "xgboost-model.json")
        binary_model_path = os.path.join(extract_dir, "xgboost-model")
        use_booster = False

        if os.path.exists(json_model_path):
            model = xgb.XGBRegressor()
            model.load_model(json_model_path)
            logger.info("Loaded model from xgboost-model.json (local/JSON format)")
        elif os.path.exists(binary_model_path):
            # SageMaker built-in XGBoost 1.7 saves in old binary format
            # Use raw Booster which can handle legacy formats
            model = xgb.Booster()
            model.load_model(binary_model_path)
            use_booster = True
            logger.info("Loaded model from xgboost-model (SageMaker binary format via Booster)")
        else:
            raise FileNotFoundError(f"No model file found in {extract_dir}. Files: {extracted_files}")

        # Load feature names (SageMaker built-in doesn't include this file)
        features_path = os.path.join(extract_dir, "feature_names.json")
        if os.path.exists(features_path):
            with open(features_path) as f:
                feature_names = json.load(f)
        else:
            # Default feature names matching build_features.py
            feature_names = [
                "latitude", "longitude", "temperature", "humidity", "wind_speed",
                "hour", "day_of_week", "is_rush_hour",
                "pm25_lag_1h", "pm25_lag_3h", "pm25_rolling_mean_3h",
            ]
            logger.warning(f"feature_names.json not found, using defaults: {feature_names}")

        loaded_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Model loaded: version={version}, features={feature_names}, booster={use_booster}")

    return ModelArtifact(
        model=model,
        feature_names=feature_names,
        version=version,
        loaded_at=loaded_at,
        use_booster=use_booster,
    )
