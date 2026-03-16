"""Unit tests for drift detection logic."""

import numpy as np
import pandas as pd


def test_drift_features_list():
    """Ensure drift feature list matches training features."""
    from monitoring.drift_detection import DRIFT_FEATURES

    expected = [
        "latitude", "longitude", "temperature", "humidity",
        "wind_speed", "hour", "day_of_week", "is_rush_hour",
    ]
    assert DRIFT_FEATURES == expected


def test_psi_threshold():
    """PSI threshold should be 0.2 (standard)."""
    from monitoring.drift_detection import PSI_THRESHOLD
    assert PSI_THRESHOLD == 0.2


def test_identical_distributions_no_drift():
    """If reference and current are identical, no drift should be detected."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "latitude": rng.uniform(39.7, 40.1, 500),
        "longitude": rng.uniform(116.1, 116.7, 500),
        "temperature": rng.normal(20, 5, 500),
        "humidity": rng.normal(60, 10, 500),
        "wind_speed": rng.exponential(3, 500),
        "hour": rng.integers(0, 24, 500),
        "day_of_week": rng.integers(0, 7, 500),
        "is_rush_hour": rng.integers(0, 2, 500),
    })

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=data, current_data=data)
    d = snapshot.dict()

    # Check DriftedColumnsCount shows 0 drift share
    for m in d["metrics"]:
        if "DriftedColumnsCount" in m.get("metric_name", ""):
            drift_share = m["value"]["share"]
            assert drift_share == 0.0, f"Expected 0 drift share, got {drift_share}"
            break
