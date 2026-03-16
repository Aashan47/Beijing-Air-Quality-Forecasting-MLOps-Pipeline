"""Drift detection using Evidently: compare training data vs recent predictions."""

import json
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from loguru import logger

from config import settings
from pipelines.utils import (
    download_parquet_from_s3,
    download_all_parquets,
    upload_parquet_to_s3,
    get_s3_client,
)

# Feature columns to check for drift (must match training features)
DRIFT_FEATURES = [
    "latitude", "longitude", "temperature", "humidity",
    "wind_speed", "hour", "day_of_week", "is_rush_hour",
]

PSI_THRESHOLD = 0.2


def check_drift() -> dict:
    """Compare training feature distributions vs recent prediction feature distributions.

    Returns:
        Dict with drift_detected (bool), drifted_features (list), report_path (str)
    """
    # Load reference data (training dataset)
    logger.info("Loading training dataset as reference...")
    reference = download_parquet_from_s3("features/training_dataset.parquet")
    reference = reference[DRIFT_FEATURES]

    # Load current data (recent prediction logs)
    logger.info("Loading recent prediction logs...")
    try:
        current = download_all_parquets("predictions/")
    except FileNotFoundError:
        logger.warning("No prediction logs found. Skipping drift check.")
        return {"drift_detected": False, "drifted_features": [], "report_path": ""}

    # Align columns — prediction logs include extra fields
    available_cols = [c for c in DRIFT_FEATURES if c in current.columns]
    if len(available_cols) < len(DRIFT_FEATURES):
        missing = set(DRIFT_FEATURES) - set(available_cols)
        logger.warning(f"Missing columns in predictions: {missing}")
    current = current[available_cols]

    if len(current) < 50:
        logger.warning(f"Only {len(current)} prediction rows — too few for reliable drift detection")
        return {"drift_detected": False, "drifted_features": [], "report_path": ""}

    # Run Evidently drift report
    logger.info(f"Running drift detection: {len(reference)} reference vs {len(current)} current rows")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current[available_cols])

    # Extract results from new Evidently v0.7 API
    report_dict = snapshot.dict()
    metrics = report_dict["metrics"]

    # Find DriftedColumnsCount metric for overall drift summary
    drift_share = 0.0
    drifted_features = []
    for m in metrics:
        name = m.get("metric_name", "")
        if "DriftedColumnsCount" in name:
            drift_share = m["value"].get("share", 0) if isinstance(m["value"], dict) else 0
        elif "ValueDrift" in name and "column=" in name:
            # ValueDrift returns p-value; drift detected if p-value < 0.05
            col = name.split("column=")[1].split(",")[0]
            p_value = m["value"] if isinstance(m["value"], (int, float)) else 1.0
            if p_value < 0.05:
                drifted_features.append(col)

    drift_detected = drift_share > 0.5

    logger.info(
        f"Drift result: detected={drift_detected}, "
        f"share={drift_share:.2f}, drifted={drifted_features}"
    )

    # Save reports to S3
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save HTML report
    import tempfile
    html_path = os.path.join(tempfile.gettempdir(), f"drift_report_{ts}.html")
    snapshot.save_html(html_path)
    s3 = get_s3_client()
    s3.upload_file(html_path, settings.s3_bucket, f"monitoring/drift_report_{ts}.html")

    # Save JSON summary
    summary = {
        "timestamp": ts,
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "drifted_features": drifted_features,
        "reference_rows": len(reference),
        "current_rows": len(current),
    }
    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=f"monitoring/drift_summary_{ts}.json",
        Body=json.dumps(summary, indent=2),
    )

    report_path = f"s3://{settings.s3_bucket}/monitoring/drift_report_{ts}.html"
    logger.info(f"Reports saved to {report_path}")

    return {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "report_path": report_path,
    }


def trigger_retraining_if_needed() -> bool:
    """Check drift and trigger retraining if detected.

    Returns:
        True if retraining was triggered.
    """
    result = check_drift()

    if result["drift_detected"]:
        logger.warning(f"Drift detected in features: {result['drifted_features']}")
        logger.info("Triggering retraining pipeline...")

        from training.launch_training import launch_sagemaker_training
        launch_sagemaker_training()

        logger.info("Retraining complete")
        return True

    logger.info("No drift detected. No retraining needed.")
    return False


if __name__ == "__main__":
    trigger_retraining_if_needed()
