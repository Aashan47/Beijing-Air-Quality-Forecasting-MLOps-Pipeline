FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY config.py /opt/airflow/
COPY ingestion/ /opt/airflow/ingestion/
COPY features/ /opt/airflow/features/
COPY training/ /opt/airflow/training/
COPY monitoring/ /opt/airflow/monitoring/
COPY pipelines/ /opt/airflow/pipelines/
