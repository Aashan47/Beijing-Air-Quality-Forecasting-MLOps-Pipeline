"""OpenAQ v3 API client for fetching PM2.5 air quality measurements."""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from config import settings

BASE_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2


class OpenAQClient:
    def __init__(self):
        self.headers = {"X-API-Key": settings.openaq_api_key}

    def _get_client(self) -> httpx.Client:
        return httpx.Client(base_url=BASE_URL, headers=self.headers, timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_locations(self, bbox: str) -> list[dict]:
        """Get PM2.5 monitoring locations within a bounding box.

        Args:
            bbox: "lon_min,lat_min,lon_max,lat_max"

        Returns:
            List of location dicts with id, name, latitude, longitude, and sensor info.
        """
        locations = []
        page = 1

        with self._get_client() as client:
            while True:
                response = client.get(
                    "/locations",
                    params={
                        "bbox": bbox,
                        "parameters_id": PM25_PARAMETER_ID,
                        "limit": 100,
                        "page": page,
                    },
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for loc in results:
                    sensors = [
                        s for s in loc.get("sensors", [])
                        if s.get("parameter", {}).get("id") == PM25_PARAMETER_ID
                    ]
                    if sensors:
                        locations.append({
                            "location_id": loc["id"],
                            "name": loc.get("name", ""),
                            "latitude": loc["coordinates"]["latitude"],
                            "longitude": loc["coordinates"]["longitude"],
                            "sensors": [{"sensor_id": s["id"]} for s in sensors],
                        })

                if len(results) < 100:
                    break
                page += 1

        logger.info(f"Found {len(locations)} PM2.5 locations in bbox {bbox}")
        return locations

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_measurements(
        self, sensor_id: int, datetime_from: str, datetime_to: str
    ) -> list[dict]:
        """Get measurements for a specific sensor.

        Args:
            sensor_id: OpenAQ sensor ID
            datetime_from: ISO format start time
            datetime_to: ISO format end time

        Returns:
            List of measurement dicts with timestamp and pm25 value.
        """
        measurements = []
        page = 1

        with self._get_client() as client:
            while True:
                response = client.get(
                    f"/sensors/{sensor_id}/measurements",
                    params={
                        "datetime_from": datetime_from,
                        "datetime_to": datetime_to,
                        "limit": 100,
                        "page": page,
                    },
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for m in results:
                    period = m.get("period", {})
                    measurements.append({
                        "timestamp": period.get("datetimeTo", {}).get("utc", ""),
                        "pm25": m.get("value"),
                    })

                if len(results) < 100:
                    break
                page += 1

        logger.info(f"Fetched {len(measurements)} measurements for sensor {sensor_id}")
        return measurements
