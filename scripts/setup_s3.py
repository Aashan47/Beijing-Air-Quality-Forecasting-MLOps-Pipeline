"""Create S3 bucket and folder structure for the air quality MLOps data lake."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from config import settings

PREFIXES = ["raw", "features", "models", "predictions", "monitoring"]


def setup_s3():
    s3 = boto3.client(
        "s3",
        region_name=settings.aws_default_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )

    bucket = settings.s3_bucket

    # Create bucket (idempotent)
    try:
        if settings.aws_default_region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": settings.aws_default_region},
            )
        logger.info(f"Created bucket: {bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info(f"Bucket already exists: {bucket}")
        else:
            raise

    # Create folder markers
    for prefix in PREFIXES:
        key = f"{prefix}/.keep"
        s3.put_object(Bucket=bucket, Key=key, Body=b"")
        logger.info(f"Created prefix: s3://{bucket}/{prefix}/")

    # Set lifecycle policy to delete old prediction logs
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "delete-old-predictions",
                    "Prefix": "predictions/",
                    "Status": "Enabled",
                    "Expiration": {"Days": 90},
                }
            ]
        },
    )
    logger.info("Set lifecycle policy: predictions/ expires after 90 days")

    logger.info(f"S3 data lake setup complete: s3://{bucket}/")


if __name__ == "__main__":
    setup_s3()
