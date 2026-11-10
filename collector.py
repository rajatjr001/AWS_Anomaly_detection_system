"""Poll Open-Meteo and send new readings to the Flask app.

Open-Meteo updates its "current" values about every 15 minutes, so a reading
is forwarded only when its timestamp changes. The dashboard chart moves each
time a new Open-Meteo reading arrives.
"""
import os
import time
import requests

FLASK_INGEST_URL = os.getenv("FLASK_INGEST_URL", "http://127.0.0.1:5001/api/ingest")
LATITUDE = float(os.getenv("WEATHER_LATITUDE", "29.6857"))
LONGITUDE = float(os.getenv("WEATHER_LONGITUDE", "76.9905"))
POLL_SECONDS = max(30, int(os.getenv("POLL_SECONDS", "60")))
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                "precipitation", "wind_speed_10m", "wind_direction_10m"]


def fetch_weather():
    params = {"latitude": LATITUDE, "longitude": LONGITUDE,
              "current": ",".join(CURRENT_VARS), "timezone": "UTC"}
    response = requests.get(WEATHER_URL, params=params, timeout=20)
    response.raise_for_status()
    current = response.json()["current"]
    return {
        "timestamp": current["time"],
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
    print(f"Polling Open-Meteo every {POLL_SECONDS} seconds. Sending to {FLASK_INGEST_URL}. Ctrl+C to stop.")
    last_timestamp = None
    while True:
        try:
            observation = fetch_weather()
            if observation["timestamp"] == last_timestamp:
                print("No new Open-Meteo reading yet:", observation["timestamp"])
            else:
                response = requests.post(FLASK_INGEST_URL, json=observation, timeout=20)
                response.raise_for_status()
                saved = response.json()["observation"]
                last_timestamp = observation["timestamp"]
                print("Stored:", saved["timestamp"], saved["status"], saved["temperature"], "C")
        except requests.ConnectionError:
            print(f"Cannot reach the app at {FLASK_INGEST_URL}. Is app.py running?")
        except requests.HTTPError as exc:
            print("HTTP error (the app may still be training):", exc)
        except requests.RequestException as exc:
            print("Request error:", exc)
        except (KeyError, ValueError) as exc:
            print("Invalid response:", exc)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
