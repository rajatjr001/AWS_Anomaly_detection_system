"""Optional physical-sensor MQTT subscriber.

Expected topic: weather/station/<station_id>
Expected JSON keys: temperature, humidity, pressure, rainfall,
wind_speed, wind_direction.

Set MQTT_HOST, MQTT_PORT and FLASK_INGEST_URL as environment variables.
"""
import json
import os
import requests
import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", "weather/station/+")
FLASK_INGEST_URL = os.getenv(
    "FLASK_INGEST_URL", "http://127.0.0.1:5001/api/ingest"
)
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(TOPIC, qos=1)
    else:
        print("MQTT connection failed:", rc)


def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode("utf-8"))
        topic_parts = message.topic.split("/")
        station_id = topic_parts[-1] if topic_parts else "unknown"

        payload["station_id"] = payload.get("station_id", station_id)
        payload["source"] = "mqtt"

        response = requests.post(
            FLASK_INGEST_URL, json=payload, timeout=15
        )
        response.raise_for_status()
        print("Sensor observation accepted:", response.status_code)
    except Exception as exc:
        print("Message processing failed:", exc)


def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION1,
        client_id="aws-anomaly-consumer"
    )
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or "")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
