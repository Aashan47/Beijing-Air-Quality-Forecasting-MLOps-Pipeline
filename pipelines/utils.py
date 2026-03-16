import io
from datetime import datetime

import boto3
import pandas as pd
from loguru import logger

from config import settings


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=settings.aws_default_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def upload_parquet_to_s3(df: pd.DataFrame, s3_key: str) -> str:
    s3 = get_s3_client()
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    s3.put_object(Bucket=settings.s3_bucket, Key=s3_key, Body=buffer.getvalue())
    full_path = f"s3://{settings.s3_bucket}/{s3_key}"
    logger.info(f"Uploaded {len(df)} rows to {full_path}")
    return full_path


def download_parquet_from_s3(s3_key: str) -> pd.DataFrame:
    s3 = get_s3_client()
    response = s3.get_object(Bucket=settings.s3_bucket, Key=s3_key)
    buffer = io.BytesIO(response["Body"].read())
    df = pd.read_parquet(buffer, engine="pyarrow")
    logger.info(f"Downloaded {len(df)} rows from s3://{settings.s3_bucket}/{s3_key}")
    return df


def list_s3_objects(prefix: str) -> list[str]:
    s3 = get_s3_client()
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_all_parquets(prefix: str) -> pd.DataFrame:
    keys = [k for k in list_s3_objects(prefix) if k.endswith(".parquet")]
    if not keys:
        raise FileNotFoundError(f"No parquet files found under s3://{settings.s3_bucket}/{prefix}")
    dfs = [download_parquet_from_s3(k) for k in keys]
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(keys)} files into {len(combined)} rows from {prefix}")
    return combined


def generate_s3_key(prefix: str, suffix: str = ".parquet") -> str:
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}/{date_str}/{ts_str}{suffix}"
