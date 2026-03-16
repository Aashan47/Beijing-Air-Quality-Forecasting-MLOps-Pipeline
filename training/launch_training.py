"""Launch SageMaker training job using Script Mode with our own train_model.py."""

import json
import os
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
from loguru import logger

from config import settings
from pipelines.utils import get_s3_client

# SKLearn framework containers (support Script Mode, have XGBoost pre-installed)
SKLEARN_CONTAINERS = {
    "us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "us-west-2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "eu-west-1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "eu-north-1": "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "ap-southeast-1": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
}


def _get_image_uri(region: str) -> str:
    if region in SKLEARN_CONTAINERS:
        return SKLEARN_CONTAINERS[region]
    raise ValueError(f"SKLearn container not configured for region {region}.")


def _upload_training_script(s3) -> str:
    """Package train_model.py as a source tarball and upload to S3."""
    script_path = os.path.join(os.path.dirname(__file__), "train_model.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "sourcedir.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(script_path, arcname="train_model.py")

        s3_key = "training-scripts/sourcedir.tar.gz"
        s3.upload_file(tar_path, settings.s3_bucket, s3_key)

    s3_uri = f"s3://{settings.s3_bucket}/{s3_key}"
    logger.info(f"Uploaded training script to {s3_uri}")
    return s3_uri


def launch_sagemaker_training() -> dict:
    """Launch a SageMaker training job with Script Mode (our own train_model.py).

    Returns:
        Dict with job_name, model_artifact_path, and model_package_arn.
    """
    region = settings.aws_default_region
    sm_client = boto3.client("sagemaker", region_name=region)
    s3 = get_s3_client()

    image_uri = _get_image_uri(region)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"air-quality-pm25-{timestamp}"

    logger.info(f"Launching SageMaker training job: {job_name}")
    logger.info(f"Region: {region}, Instance: {settings.training_instance_type}")

    # Upload our training script
    source_dir_uri = _upload_training_script(s3)

    sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        RoleArn=settings.sagemaker_role_arn,
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{settings.s3_bucket}/features/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{settings.s3_bucket}/models/sagemaker/",
        },
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": settings.training_instance_type,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={
            # Script Mode: tell the container to run our script
            "sagemaker_program": "train_model.py",
            "sagemaker_submit_directory": source_dir_uri,
            # XGBoost hyperparameters (passed as CLI args to train_model.py)
            "n-estimators": "200",
            "max-depth": "6",
            "learning-rate": "0.1",
            "subsample": "0.8",
            "colsample-bytree": "0.8",
        },
    )

    # Wait for completion
    logger.info("Waiting for training job to complete...")
    waiter = sm_client.get_waiter("training_job_completed_or_stopped")
    waiter.wait(TrainingJobName=job_name, WaiterConfig={"Delay": 30, "MaxAttempts": 120})

    # Check status
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]

    if status != "Completed":
        failure = response.get("FailureReason", "Unknown")
        raise RuntimeError(f"Training job {job_name} failed: {status} - {failure}")

    model_artifact = response["ModelArtifacts"]["S3ModelArtifacts"]
    logger.info(f"Training complete. Model artifact: {model_artifact}")

    # Copy model to standard location
    source_key = model_artifact.replace(f"s3://{settings.s3_bucket}/", "")
    s3.copy_object(
        Bucket=settings.s3_bucket,
        CopySource={"Bucket": settings.s3_bucket, "Key": source_key},
        Key="models/model.tar.gz",
    )
    logger.info("Copied model to models/model.tar.gz")

    # Extract metrics from model artifact (our script saves metrics.json)
    tar_path = os.path.join(tempfile.gettempdir(), "model.tar.gz")
    s3.download_file(settings.s3_bucket, "models/model.tar.gz", tar_path)
    metrics = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        try:
            f = tar.extractfile("metrics.json")
            if f:
                metrics = json.loads(f.read())
                s3.put_object(
                    Bucket=settings.s3_bucket,
                    Key="models/latest_metrics.json",
                    Body=json.dumps(metrics, indent=2),
                )
                logger.info(f"Metrics: {metrics}")
        except KeyError:
            logger.warning("No metrics.json found in model artifact")

    # Register in Model Registry
    model_package_arn = _register_model(sm_client, s3, model_artifact, image_uri)

    return {
        "job_name": job_name,
        "model_artifact_path": model_artifact,
        "model_package_arn": model_package_arn,
    }


def _register_model(sm_client, s3, model_data: str, image_uri: str) -> str:
    """Register model in SageMaker Model Registry."""
    group_name = settings.model_package_group

    # Ensure model package group exists
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Air Quality PM2.5 prediction models for Beijing",
        )
        logger.info(f"Created model package group: {group_name}")
    except sm_client.exceptions.ClientError as e:
        if "already exists" in str(e).lower():
            pass
        else:
            raise

    # Read metrics
    try:
        response = s3.get_object(
            Bucket=settings.s3_bucket, Key="models/latest_metrics.json",
        )
        metrics = json.loads(response["Body"].read())
    except Exception:
        metrics = {}

    rmse = metrics.get("rmse", float("inf"))
    approval_status = "Approved" if rmse < settings.rmse_threshold else "PendingManualApproval"

    response = sm_client.create_model_package(
        ModelPackageGroupName=group_name,
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": model_data,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        ModelApprovalStatus=approval_status,
        ModelPackageDescription=f"RMSE: {rmse:.2f}" if metrics else "No metrics available",
    )

    arn = response["ModelPackageArn"]
    logger.info(f"Registered model: {arn} (status={approval_status}, RMSE={rmse:.2f})")
    return arn


if __name__ == "__main__":
    result = launch_sagemaker_training()
    logger.info(f"Result: {result}")
