"""Local training: same logic as train_model.py but runs locally with S3 data."""

import json
import os
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger

from config import settings
from pipelines.utils import download_parquet_from_s3, get_s3_client
from training.train_model import train


def train_local():
    """Download data from S3, train locally, upload model back to S3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        model_dir = os.path.join(tmpdir, "model")
        os.makedirs(data_dir)
        os.makedirs(model_dir)

        # Download training dataset
        logger.info("Downloading training dataset from S3...")
        df = download_parquet_from_s3("features/training_dataset.parquet")
        local_path = os.path.join(data_dir, "training_dataset.parquet")
        df.to_parquet(local_path, index=False)
        logger.info(f"Dataset: {len(df)} rows, {len(df.columns)} columns")

        # Train
        hyperparams = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        metrics = train(data_dir, model_dir, hyperparams)

        logger.info(f"Training complete. RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")

        # Check RMSE threshold
        if metrics["rmse"] > settings.rmse_threshold:
            logger.warning(
                f"RMSE {metrics['rmse']:.2f} exceeds threshold {settings.rmse_threshold}. "
                "Model may not be suitable for deployment."
            )

        # Package as model.tar.gz and upload to S3
        tar_path = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for fname in os.listdir(model_dir):
                tar.add(os.path.join(model_dir, fname), arcname=fname)

        s3 = get_s3_client()
        s3_key = "models/model.tar.gz"
        s3.upload_file(tar_path, settings.s3_bucket, s3_key)
        logger.info(f"Model uploaded to s3://{settings.s3_bucket}/{s3_key}")

        # Also upload metrics separately for easy access
        metrics_key = "models/latest_metrics.json"
        s3.put_object(
            Bucket=settings.s3_bucket,
            Key=metrics_key,
            Body=json.dumps(metrics, indent=2),
        )
        logger.info(f"Metrics uploaded to s3://{settings.s3_bucket}/{metrics_key}")

    return metrics


if __name__ == "__main__":
    train_local()
