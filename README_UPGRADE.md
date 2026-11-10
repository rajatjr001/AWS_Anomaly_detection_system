# AWS Anomaly Detection System — Live Ingestion Upgrade

This version replaces the simulated stream queue with:
- HTTP ingestion endpoint (`POST /api/ingest`)
- Open-Meteo polling collector (`collector.py`)
- Optional MQTT subscriber for physical sensors (`mqtt_consumer.py`)
- SQLite persistence by default (`aws_monitor.db`)
- Cached Isolation Forest and novelty-enabled LOF models
- Database-backed dashboard refresh every five seconds

## 1. Setup (macOS)

```bash
cd AWS_Anomaly_detection_system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the files from this upgrade package into your existing repository:
- Replace `app.py`
- Replace `requirements.txt`
- Replace `static/index.html`
- Add `database.py`
- Add `collector.py`
- Add `mqtt_consumer.py`
- Add `.env.example` (optional)
- Keep your existing `pipeline.py`

Do not delete `pipeline.py`; the upgraded app imports its data generation,
preprocessing, feature engineering and model-building functions.

## 2. Start the Flask application

```bash
python app.py
```

Open http://127.0.0.1:5000

On first start, the app creates a synthetic historical baseline in
`data/training_history.csv`, fits both anomaly detectors once, and creates
`aws_monitor.db`. This synthetic data is the training baseline, not live
sensor data.

## 3. Start external weather ingestion

Open a second terminal, activate the same virtual environment, then run:

```bash
python collector.py
```

The collector polls Open-Meteo and posts readings to Flask. Set
`WEATHER_LATITUDE`, `WEATHER_LONGITUDE` and `POLL_SECONDS` before running
if you want a different location or polling interval. The default location
is Karnal, India. The provider's data update cadence may be slower than
your polling interval; repeated polls may return the same current observation.

## 4. Test the API manually

```bash
curl -X POST http://127.0.0.1:5000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "station_id":"test-station-01",
    "source":"curl",
    "temperature":28.5,
    "humidity":65,
    "pressure":1012,
    "rainfall":0,
    "wind_speed":12,
    "wind_direction":180
  }'
```

Read stored observations:

```bash
curl http://127.0.0.1:5000/api/readings/latest?limit=20
```

Read summary:

```bash
curl http://127.0.0.1:5000/api/summary
```

## 5. Physical sensors via MQTT (optional)

Install and start Mosquitto. For example, on macOS with Homebrew:

```bash
brew install mosquitto
brew services start mosquitto
```

In another terminal:

```bash
python mqtt_consumer.py
```

Publish JSON to `weather/station/station-01`:

```bash
mosquitto_pub -h localhost -t weather/station/station-01 -m \
'{"temperature":28.5,"humidity":65,"pressure":1012,"rainfall":0,"wind_speed":12,"wind_direction":180}'
```

For actual ESP32 deployment, configure MQTT authentication and TLS,
validate device identity, and do not expose an unauthenticated broker
to the public internet.

## 6. PostgreSQL (optional)

Install a PostgreSQL driver:

```bash
pip install psycopg2-binary
```

Create a database named `aws_monitor`, then set:

```bash
export DATABASE_URL="postgresql+psycopg2://USERNAME:PASSWORD@localhost:5432/aws_monitor"
```

Start the app again. SQLAlchemy creates the table automatically for a new
database. For schema changes in an existing deployment, use Alembic migrations.

## Important implementation notes

- The ML model is trained once at application startup, not on every request.
- The historical baseline is persisted in `data/training_history.csv`.
- Incoming readings and their model scores are stored in SQLite/PostgreSQL.
- The dashboard reads persisted data from the database.
- Physical range checks are separate from statistical anomaly detection.
- The Open-Meteo collector is external weather-service ingestion, not a
  physical sensor feed. MQTT is the physical-sensor integration path.
- This is an educational production-style foundation. Before public
  deployment add authentication, rate limiting, HTTPS, secret management,
  logging/monitoring, database migrations, backups, and a robust job/queue
  architecture if ingestion volume grows.
