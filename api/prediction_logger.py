"""Buffered async prediction logger — flushes to S3 in batches."""

import asyncio
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from config import settings
from pipelines.utils import upload_parquet_to_s3, generate_s3_key


class PredictionLogger:
    def __init__(self):
        self.buffer: list[dict] = []
        self._flush_task: asyncio.Task | None = None

    def start(self):
        """Start the periodic flush background task."""
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self):
        """Flush remaining buffer and cancel background task."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_buffer()

    def log(self, prediction: dict):
        """Add a prediction to the buffer. Flushes if buffer is full."""
        self.buffer.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **prediction,
        })
        if len(self.buffer) >= settings.log_buffer_size:
            asyncio.create_task(self._flush_buffer())

    async def _periodic_flush(self):
        """Flush buffer every N seconds."""
        while True:
            await asyncio.sleep(settings.log_flush_interval_seconds)
            await self._flush_buffer()

    async def _flush_buffer(self):
        """Write buffered predictions to S3 as Parquet."""
        if not self.buffer:
            return

        records = self.buffer.copy()
        self.buffer.clear()

        try:
            df = pd.DataFrame(records)
            s3_key = generate_s3_key("predictions")
            await asyncio.to_thread(upload_parquet_to_s3, df, s3_key)
            logger.info(f"Flushed {len(records)} predictions to s3://{settings.s3_bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to flush predictions: {e}")
            # Put records back in buffer
            self.buffer = records + self.buffer
