"""Airflow DAG configuration for the air quality ML pipeline."""

# Beijing bounding box
BBOX = "116.10,39.70,116.70,40.15"

# Hours of data to fetch per ingestion run
HOURS_BACK = 6

# Model quality threshold
RMSE_THRESHOLD = 15.0

# Drift detection threshold (fraction of drifted columns)
DRIFT_THRESHOLD = 0.5
