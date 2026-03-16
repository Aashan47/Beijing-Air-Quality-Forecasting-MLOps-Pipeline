"""Weather client using Open-Meteo — free, no API key, historical + current."""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoClient:
    """Open-Meteo API — free, no API key, covers both historical and current weather."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_historical_weather(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> list[dict]:
        """Get hourly historical weather for a location and date range.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"

        Returns:
            List of dicts with timestamp, temperature, humidity, wind_speed.
        """
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                OPEN_METEO_HISTORICAL_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                    "timezone": "UTC",
                },
            )
            response.raise_for_status()
            data = response.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("wind_speed_10m", [])

        records = []
        for i in range(len(times)):
            if temps[i] is not None and humidities[i] is not None and winds[i] is not None:
                records.append({
                    "timestamp": times[i],
                    "temperature": temps[i],
                    "humidity": humidities[i],
                    "wind_speed": round(winds[i] / 3.6, 1),  # km/h -> m/s
                })

        logger.info(
            f"Open-Meteo: {len(records)} hourly records for ({lat}, {lon}) "
            f"from {start_date} to {end_date}"
        )
        return records

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_forecast_weather(
        self, lat: float, lon: float, forecast_days: int = 7
    ) -> list[dict]:
        """Get hourly weather forecast for a location (up to 16 days ahead).

        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: Number of days to forecast (1-16, default 7)

        Returns:
            List of dicts with timestamp, temperature, humidity, wind_speed.
        """
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                    "forecast_days": min(forecast_days, 16),
                    "timezone": "UTC",
                },
            )
            response.raise_for_status()
            data = response.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("wind_speed_10m", [])

        records = []
        for i in range(len(times)):
            if temps[i] is not None and humidities[i] is not None and winds[i] is not None:
                records.append({
                    "timestamp": times[i],
                    "temperature": temps[i],
                    "humidity": humidities[i],
                    "wind_speed": round(winds[i] / 3.6, 1),  # km/h -> m/s
                })

        logger.info(f"Open-Meteo forecast: {len(records)} hourly records for ({lat}, {lon}), {forecast_days} days")
        return records

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def get_current_weather(self, lat: float, lon: float) -> dict:
        """Get current weather for a location using Open-Meteo forecast API.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with temperature (C), humidity (%), wind_speed (m/s).
        """
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                    "timezone": "UTC",
                },
            )
            response.raise_for_status()
            data = response.json()

        current = data.get("current", {})
        weather = {
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": round(current["wind_speed_10m"] / 3.6, 1),  # km/h -> m/s
        }
        logger.debug(f"Open-Meteo current: ({lat}, {lon}) -> {weather}")
        return weather
