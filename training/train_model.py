"""SageMaker training entry point: loads data, trains XGBoost, saves model + metrics."""

import argparse
import json
import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def train(data_dir: str, model_dir: str, hyperparams: dict):
    """Train XGBoost model on feature dataset.

    Args:
        data_dir: Directory containing training_dataset.parquet
        model_dir: Directory to save model artifact and metrics
        hyperparams: XGBoost hyperparameters
    """
    # Load data
    df = pd.read_parquet(os.path.join(data_dir, "training_dataset.parquet"))

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ("timestamp", "pm25_target")]
    X = df[feature_cols]
    y = df["pm25_target"]

    # Time-based split (last 20% for test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"Features: {feature_cols}")

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=int(hyperparams.get("n_estimators", 200)),
        max_depth=int(hyperparams.get("max_depth", 6)),
        learning_rate=float(hyperparams.get("learning_rate", 0.1)),
        subsample=float(hyperparams.get("subsample", 0.8)),
        colsample_bytree=float(hyperparams.get("colsample_bytree", 0.8)),
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print(f"Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost-model.json")
    model.save_model(model_path)

    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature names for inference
    feature_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f)

    print(f"Model saved to {model_path}")
    return metrics


if __name__ == "__main__":
    # SageMaker convention
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    args = parser.parse_args()

    hyperparams = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }

    data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train(data_dir, model_dir, hyperparams)
