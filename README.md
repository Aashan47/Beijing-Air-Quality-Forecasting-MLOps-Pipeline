<p align="center">
  <h1 align="center">Beijing Air Quality Forecasting & MLOps Platform</h1>
  <p align="center">
    End-to-end machine learning operations pipeline for real-time PM2.5 air quality prediction and multi-day forecasting over Beijing, China.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.1+-orange?logo=xgboost" alt="XGBoost">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/AWS-SageMaker%20%7C%20S3%20%7C%20EC2-yellow?logo=amazonaws" alt="AWS">
  <img src="https://img.shields.io/badge/Docker-Compose-blue?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Airflow-2.10-red?logo=apacheairflow" alt="Airflow">
  <img src="https://img.shields.io/badge/Evidently-0.7+-purple" alt="Evidently">
</p>

---

## Overview

A production-grade MLOps platform that ingests real-time air quality and weather data from Open-Meteo, engineers temporal features, trains an XGBoost model on AWS SageMaker, serves predictions via a FastAPI backend, and visualizes results on an interactive Leaflet.js dashboard with multi-day forecast charts. Automated drift detection triggers retraining when data distributions shift.

### Key Capabilities

- **Real-time PM2.5 predictions** across a 15x15 spatial grid over Beijing
- **7-day air quality forecasting** using Open-Meteo weather forecasts with recursive prediction
- **Automated MLOps pipeline** orchestrated by Apache Airflow (every 6 hours)
- **Data drift detection** via Evidently AI (PSI method) with automatic retraining
- **Model versioning** through SageMaker Model Registry with RMSE-based approval gates
- **Interactive dashboard** with heatmap, AQI categories, and forecast charts

---

## System Architecture

<p align="center">
  <img src="architecture-diagrams/High Level Platform Architecture.png" alt="High-Level Platform Architecture" width="600">
</p>

The platform follows a layered architecture — data flows from Open-Meteo APIs through ingestion, feature engineering, and SageMaker training, into a FastAPI prediction service backed by an S3 data lake, and surfaces on an interactive map dashboard.

### S3 Data Lake Structure

```
s3://air-quality-mlops-data/
├── raw/YYYY-MM-DD/*.parquet            Ingested PM2.5 + weather data
├── features/training_dataset.parquet   Engineered features with PM2.5 lags
├── models/
│   ├── model.tar.gz                    Current production model
│   └── latest_metrics.json             RMSE, MAE, R² scores
├── predictions/YYYY-MM-DD/*.parquet    Logged API predictions (90-day retention)
└── monitoring/drift_report_*.html      Evidently drift analysis reports
```

### Automated MLOps Pipeline

<p align="center">
  <img src="architecture-diagrams/Automated MLOps Pipeline.png" alt="Automated MLOps Pipeline" width="500">
</p>

The Airflow DAG runs **every 6 hours**, orchestrating the full cycle: data ingestion, feature engineering, drift detection, conditional retraining on SageMaker, model evaluation against the RMSE threshold, and model registry updates.

### AWS Infrastructure Architecture

<p align="center">
  <img src="architecture-diagrams/AWS Infrastructure Architecture.png" alt="AWS Infrastructure Architecture" width="700">
</p>

All application services (FastAPI, Nginx frontend, Airflow, PostgreSQL) run as Docker containers on a single EC2 instance. SageMaker handles training jobs on-demand, the Model Registry tracks model versions, and S3 serves as the central data lake connecting all components.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Source** | Open-Meteo API | Air quality (PM2.5) + weather data (free, no API key) |
| **Storage** | AWS S3 | Data lake for raw data, features, models, predictions |
| **Training** | AWS SageMaker | Managed XGBoost training with Script Mode |
| **Model Registry** | SageMaker Model Registry | Model versioning with approval gates |
| **API** | FastAPI + Uvicorn | REST prediction endpoints |
| **Frontend** | Leaflet.js + Chart.js | Interactive heatmap + forecast charts |
| **Orchestration** | Apache Airflow 2.10 | Scheduled pipeline execution |
| **Monitoring** | Evidently AI 0.7 | Data drift detection (PSI) |
| **Containers** | Docker Compose | Multi-service deployment |
| **Infrastructure** | AWS EC2 | Production hosting |
| **ML Framework** | XGBoost | Gradient-boosted PM2.5 regression |

---

## Project Structure

```
air-quality-mlops/
├── api/
│   ├── main.py                 # FastAPI app: /predict, /predict-grid, /forecast, /health
│   ├── schemas.py              # Response models + EPA AQI category mapping
│   ├── model_loader.py         # S3 model download + XGBoost loading
│   └── prediction_logger.py    # Async buffered prediction logging to S3
│
├── airflow/dags/
│   ├── ml_pipeline.py          # Main DAG: ingest → features → drift → train
│   └── config.py               # DAG configuration (bbox, schedule, thresholds)
│
├── training/
│   ├── train_model.py          # XGBoost training script (SageMaker entry point)
│   ├── launch_training.py      # SageMaker job launcher + model registry
│   └── train_local.py          # Local training for development
│
├── ingestion/
│   ├── fetch_air_quality.py    # Open-Meteo PM2.5 + weather ingestion (9-point grid)
│   ├── weather_client.py       # OpenMeteo client (current, forecast, historical)
│   └── openaq_client.py        # Alternative data source (optional)
│
├── features/
│   └── build_features.py       # Feature engineering with temporal lag features
│
├── monitoring/
│   └── drift_detection.py      # Evidently drift detection + auto-retrain trigger
│
├── pipelines/
│   └── utils.py                # S3 utilities (upload, download, list parquet)
│
├── scripts/
│   └── setup_s3.py             # S3 bucket initialization + lifecycle policies
│
├── frontend/
│   └── index.html              # Interactive dashboard (Leaflet + Chart.js)
│
├── docker/
│   ├── api.Dockerfile          # FastAPI service container
│   ├── airflow.Dockerfile      # Airflow worker container
│   ├── frontend.Dockerfile     # Nginx static file server
│   └── nginx.conf              # Nginx reverse proxy configuration
│
├── notebooks/
│   └── sagemaker_training.ipynb # Interactive training notebook for SageMaker Studio
│
├── tests/
│   ├── test_api.py             # AQI mapping + response validation (5 tests)
│   ├── test_drift.py           # Drift detection logic (3 tests)
│   └── test_features.py        # Feature engineering validation (5 tests)
│
├── config.py                   # Central configuration (Pydantic Settings)
├── docker-compose.yml          # 6-service orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md
```

---

## Features

### Model & Predictions

| Feature | Description |
|---------|-------------|
| **Target** | PM2.5 concentration (next-hour forecast) |
| **Algorithm** | XGBoost Regressor (200 rounds, max_depth=6) |
| **Input Features** | `latitude`, `longitude`, `temperature`, `humidity`, `wind_speed`, `hour`, `day_of_week`, `is_rush_hour`, `pm25_lag_1h`, `pm25_lag_3h`, `pm25_rolling_mean_3h` |
| **Training Split** | Time-based 80/20 (preserves temporal ordering) |
| **Spatial Coverage** | 9-point Beijing grid (39.75-40.05°N, 116.15-116.65°E) |
| **Forecast Range** | 1 hour (real-time) to 7 days (recursive forecast) |

### AQI Categories (EPA PM2.5 Breakpoints)

| Category | PM2.5 Range (µg/m³) | Color |
|----------|---------------------|-------|
| Good | 0 - 12 | Green |
| Moderate | 12 - 35.4 | Yellow |
| Unhealthy for Sensitive Groups | 35.4 - 55.4 | Orange |
| Unhealthy | 55.4 - 150.4 | Red |
| Very Unhealthy | 150.4 - 250.4 | Purple |
| Hazardous | 250.4+ | Maroon |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with model version info |
| `GET` | `/predict?lat=39.90&lon=116.40` | Single-point PM2.5 prediction |
| `GET` | `/predict-grid?lat_min=39.7&lat_max=40.15&lon_min=116.1&lon_max=116.7&grid_size=15` | Grid predictions for heatmap |
| `GET` | `/forecast?lat=39.90&lon=116.40&days=3` | Multi-day hourly forecast |
| `GET` | `/docs` | Interactive Swagger documentation |

### Example Response — `/predict`

```json
{
  "latitude": 39.90,
  "longitude": 116.40,
  "pm25_predicted": 28.4,
  "aqi_category": "Moderate",
  "timestamp": "2026-03-15T14:00:00+08:00"
}
```

### Example Response — `/forecast`

```json
{
  "latitude": 39.90,
  "longitude": 116.40,
  "forecast_hours": 72,
  "predictions": [
    {
      "timestamp": "2026-03-15T15:00:00",
      "hour": 15,
      "pm25": 31.2,
      "aqi_category": "Moderate",
      "temperature": 12.5,
      "humidity": 45.0,
      "wind_speed": 3.2
    }
  ]
}
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS Account with S3, SageMaker, and EC2 access
- AWS CLI configured (`aws configure`)

### 1. Clone the Repository

```bash
git clone git@github.com:Aashan47/Beijing-Air-Quality-Forecasting.git
cd Beijing-Air-Quality-Forecasting
```

### 2. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=eu-north-1
S3_BUCKET=air-quality-mlops-data
SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/your-sagemaker-role
```

### 4. Initialize S3 Bucket

```bash
python -m scripts.setup_s3
```

### 5. Ingest Data

```bash
python -m ingestion.fetch_air_quality --days-back 90
```

This fetches ~18,500 rows of PM2.5 + weather data across 9 Beijing grid points.

### 6. Build Features

```bash
python -m features.build_features --source raw/ --mode simple
```

### 7. Train Model

**Option A — Local training (development):**
```bash
python -m training.train_local
```

**Option B — SageMaker training (production):**
```bash
python -m training.launch_training
```

**Option C — Interactive notebook:**
Open `notebooks/sagemaker_training.ipynb` in SageMaker Studio or Jupyter.

### 8. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the API documentation.

---

## Production Deployment (EC2)

### 1. Launch an EC2 Instance

- **AMI:** Amazon Linux 2023
- **Instance type:** t3.medium or larger
- **Security group ports:** 22 (SSH), 80 (Frontend), 8000 (API), 8080 (Airflow)

### 2. Install Docker on EC2

```bash
sudo yum update -y
sudo yum install -y docker git
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. Deploy

```bash
git clone git@github.com:Aashan47/Beijing-Air-Quality-Forecasting.git
cd Beijing-Air-Quality-Forecasting

# Create .env with AWS credentials
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=eu-north-1
S3_BUCKET=air-quality-mlops-data
SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/your-sagemaker-role
AIRFLOW_UID=1000
EOF

# Launch all services
docker-compose up -d
```

### 4. Access Services

| Service | URL |
|---------|-----|
| **Dashboard** | `http://<ec2-ip>/` |
| **API Docs** | `http://<ec2-ip>:8000/docs` |
| **Airflow UI** | `http://<ec2-ip>:8080` (admin/admin) |

---

## MLOps Pipeline

### Automated Workflow (Every 6 Hours)

1. **Data Ingestion** — Fetches latest PM2.5 and weather data from Open-Meteo for 9 Beijing grid points
2. **Feature Engineering** — Computes temporal lag features (`pm25_lag_1h`, `pm25_lag_3h`, `pm25_rolling_mean_3h`) and time-based features
3. **Drift Detection** — Compares current prediction distribution against training data using Evidently AI's Population Stability Index (PSI)
4. **Conditional Retraining** — Triggers SageMaker training job if drift is detected or on weekly schedule (Sunday)
5. **Model Evaluation** — Validates RMSE against threshold (15.0 µg/m³)
6. **Model Registration** — Registers approved models in SageMaker Model Registry; rejected models require manual approval

### Model Registry Approval Gate

```
RMSE < 15.0 µg/m³  →  Approved (auto-deployed)
RMSE >= 15.0 µg/m³ →  PendingManualApproval
```

### Drift Detection

- **Method:** Population Stability Index (PSI) via Evidently AI
- **Monitored features:** latitude, longitude, temperature, humidity, wind_speed, hour, day_of_week, is_rush_hour
- **Trigger:** Drift flagged when >50% of monitored features show significant distribution shift
- **Action:** Automatic SageMaker retraining job

---

## Dashboard

The interactive frontend provides:

- **Heatmap** — 15x15 grid of PM2.5 predictions with dynamic color scaling
- **Click-to-predict** — Click any location on the map for a point prediction
- **Forecast chart** — Multi-day PM2.5 and temperature forecast (24h / 3d / 5d / 7d tabs)
- **AQI indicator** — Current average air quality with category and color coding
- **Auto-refresh** — Updates every 5 minutes and on map pan/zoom
- **API health status** — Live connection indicator

### Real-Time Prediction Flow

<p align="center">
  <img src="architecture-diagrams/Real-Time Prediction Flow.png" alt="Real-Time Prediction Flow" width="750">
</p>

When a user opens the dashboard, the frontend requests grid predictions from FastAPI, which loads the latest model from S3, generates spatial PM2.5 predictions using current weather data, logs each prediction asynchronously, and returns results to render the heatmap.

---

## AWS IAM Requirements

### SageMaker Execution Role

Trust policy must include:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": ["sts:AssumeRole", "sts:SetSourceIdentity"]
    }
  ]
}
```

### IAM User Permissions

Attach these managed policies to the IAM user whose access keys are in `.env`:

- `AmazonS3FullAccess` (or scoped to your bucket)
- `AmazonSageMakerFullAccess`

---

## Running Tests

```bash
pytest tests/ -v
```

```
tests/test_api.py       — AQI category mapping + response validation (5 tests)
tests/test_drift.py     — Drift detection logic + Evidently integration (3 tests)
tests/test_features.py  — Feature engineering + lag computation (5 tests)
```

---

## Configuration Reference

All settings are managed via environment variables or `.env` file, loaded by Pydantic Settings in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | — | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET` | `air-quality-mlops-data` | S3 bucket name |
| `SAGEMAKER_ROLE_ARN` | — | SageMaker execution role ARN |
| `MODEL_PACKAGE_GROUP` | `AirQualityPM25Model` | Model registry group name |
| `TRAINING_INSTANCE_TYPE` | `ml.m5.xlarge` | SageMaker training instance |
| `RMSE_THRESHOLD` | `15.0` | Model approval RMSE gate |
| `DEFAULT_BBOX` | `116.10,39.70,116.70,40.15` | Beijing bounding box |
| `LOG_BUFFER_SIZE` | `100` | Prediction log buffer size |
| `LOG_FLUSH_INTERVAL_SECONDS` | `60` | Log flush interval |


---

<p align="center">
  Built with FastAPI, XGBoost, AWS SageMaker, Apache Airflow, Evidently AI, Leaflet.js, and Chart.js
</p>
