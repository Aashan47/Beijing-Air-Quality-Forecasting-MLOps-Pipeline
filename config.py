from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys — Open-Meteo is free, no key needed
    # OpenAQ key kept as optional if you want to use it for additional data
    openaq_api_key: str = ""

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"

    # S3
    s3_bucket: str = "air-quality-mlops-data"

    # SageMaker
    sagemaker_role_arn: str = ""
    model_package_group: str = "AirQualityPM25Model"
    training_instance_type: str = "ml.m5.xlarge"

    # Beijing defaults
    default_bbox: str = "116.10,39.70,116.70,40.15"
    default_lat: float = 39.90
    default_lon: float = 116.40

    # Model
    rmse_threshold: float = 15.0

    # Prediction logger
    log_buffer_size: int = 100
    log_flush_interval_seconds: int = 60

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
