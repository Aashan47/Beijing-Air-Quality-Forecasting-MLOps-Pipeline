"""Airflow DAG: air quality ML pipeline — ingest, features, drift check, retrain."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from config import BBOX, HOURS_BACK

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def _fetch_data(**kwargs):
    from ingestion.fetch_air_quality import run_ingestion
    df = run_ingestion(bbox=BBOX, hours_back=HOURS_BACK)
    return len(df)


def _build_features(**kwargs):
    from features.build_features import build_training_features
    df = build_training_features(source_prefix="raw/", mode="simple")
    return len(df)


def _check_drift(**kwargs):
    from monitoring.drift_detection import check_drift
    result = check_drift()
    kwargs["ti"].xcom_push(key="drift_result", value=result)
    return result["drift_detected"]


def _branch_on_drift(**kwargs):
    drift_detected = kwargs["ti"].xcom_pull(task_ids="check_drift")
    # Also retrain on Sundays (weekly refresh)
    is_sunday = datetime.now().weekday() == 6
    if drift_detected or is_sunday:
        return "trigger_training"
    return "end"


def _trigger_training(**kwargs):
    from training.launch_training import launch_sagemaker_training
    result = launch_sagemaker_training()
    kwargs["ti"].xcom_push(key="training_result", value=result)
    return result


def _evaluate_model(**kwargs):
    import json
    from config import settings
    from pipelines.utils import get_s3_client

    # Read metrics saved by training job
    s3 = get_s3_client()
    response = s3.get_object(Bucket=settings.s3_bucket, Key="models/latest_metrics.json")
    metrics = json.loads(response["Body"].read())

    rmse = metrics.get("rmse", float("inf"))
    approved = rmse < settings.rmse_threshold
    kwargs["ti"].xcom_push(key="approved", value=approved)
    return {"approved": approved, "rmse": rmse}


def _update_model_version(**kwargs):
    approved = kwargs["ti"].xcom_pull(task_ids="evaluate_model", key="approved")
    if not approved:
        print("Model not approved — skipping deployment")
        return
    # SageMaker training copies model to models/model.tar.gz
    # and registers it in Model Registry via launch_training.py
    # The API will pick up the new model on next restart
    print("Model updated in S3 and registered in SageMaker Model Registry")


with DAG(
    "air_quality_ml_pipeline",
    default_args=default_args,
    description="Beijing air quality ML pipeline: ingest → features → drift → retrain",
    schedule="0 */6 * * *",
    catchup=False,
    tags=["ml", "air-quality"],
) as dag:

    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=_fetch_data,
    )

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=_build_features,
    )

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=_check_drift,
    )

    branch_on_drift = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=_branch_on_drift,
    )

    trigger_training = PythonOperator(
        task_id="trigger_training",
        python_callable=_trigger_training,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    update_model_version = PythonOperator(
        task_id="update_model_version",
        python_callable=_update_model_version,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # DAG dependencies
    fetch_data >> build_features >> check_drift >> branch_on_drift
    branch_on_drift >> [trigger_training, end]
    trigger_training >> evaluate_model >> update_model_version >> end
