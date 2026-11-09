"""Poll Open-Meteo and forward each returned observation to this Flask app.

Run Flask in one terminal, then run this script in another terminal.
This polls the provider; it does not make the provider update its forecast
on every poll. For physical sensor data, use mqtt_consumer.py instead.
"""
import os
import time
import requests

FLASK_INGEST_URL = os.getenv(
    "FLASK_INGEST_URL", "http://127.0.0.1:5000/api/ingest"
)
LATITUDE = float(os.getenv("WEATHER_LATITUDE", "29.6857"))
LONGITUDE = float(os.getenv("WEATHER_LONGITUDE", "76.9905"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "300"))

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "UTC",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=20)
    response.raise_for_status()
    current = response.json()["current"]

    return {
        "timestamp": current.get("time"),
        "station_id": "open-meteo-karnal",
        "source": "open-meteo",
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "pressure": current["surface_pressure"],
        "rainfall": current["precipitation"],
        "wind_speed": current["wind_speed_10m"],
        "wind_direction": current["wind_direction_10m"],
    }


def main():
    print(f"Polling Open-Meteo every {POLL_SECONDS} seconds.")
    while True:
        try:
            observation = fetch_weather()
            response = requests.post(
                FLASK_INGEST_URL, json=observation, timeout=20
            )
            response.raise_for_status()
            print("Stored:", response.json()["observation"])
        except requests.RequestException as exc:
            print("Weather/API connection error:", exc)
        except (KeyError, ValueError) as exc:
            print("Invalid weather response:", exc)

        time.sleep(max(60, POLL_SECONDS))


if __name__ == "__main__":
    main()
