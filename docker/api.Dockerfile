FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY config.py .
COPY api/ api/
COPY ingestion/ ingestion/
COPY features/ features/
COPY training/ training/
COPY pipelines/ pipelines/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
