# EC2 Deployment Guide

## Instance Setup

1. Launch EC2 instance:
   - Type: `t3.medium` (2 vCPU, 4GB RAM)
   - AMI: Amazon Linux 2023
   - Storage: 30GB EBS

2. Security group inbound rules:
   - Port 80: Frontend (0.0.0.0/0)
   - Port 8000: API (0.0.0.0/0 or restrict)
   - Port 8080: Airflow UI (your IP only)
   - Port 22: SSH (your IP only)

3. Attach IAM role with permissions:
   - S3: read/write to `air-quality-mlops-data`
   - SageMaker: CreateTrainingJob, CreateModelPackage, DescribeTrainingJob

## Install Dependencies

```bash
# Install Docker
sudo yum update -y
sudo yum install -y docker git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

## Deploy

```bash
# Clone repository
git clone <your-repo-url> ~/air-quality-mlops
cd ~/air-quality-mlops

# Configure environment
cp .env.example .env
# Edit .env with your API keys and AWS credentials
# (If using IAM role, AWS credentials can be omitted)

# Start all services
docker compose up -d

# Verify
docker compose ps
curl http://localhost:8000/health
```

## Access

- Frontend map: `http://<ec2-public-ip>/`
- API docs: `http://<ec2-public-ip>:8000/docs`
- Airflow UI: `http://<ec2-public-ip>:8080` (admin/admin)

## Estimated Costs

- EC2 t3.medium: ~$30/month
- SageMaker training (ml.m5.xlarge, 4x daily, ~5 min each): ~$2.50/month
- S3 storage: ~$1/month
- **Total: ~$35/month**
