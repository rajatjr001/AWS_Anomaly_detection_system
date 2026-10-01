import os
import json
import logging
import queue
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

import pipeline as pl
from database import init_db, save_reading, get_latest_readings, get_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), static_url_path="")
logging.basicConfig(level=logging.INFO)

SENSOR_FIELDS = pl.SENSOR_COLS
PHYSICAL_LIMITS = pl.PHYSICAL_LIMITS
LATITUDE = float(os.getenv("WEATHER_LATITUDE", "29.6857"))
LONGITUDE = float(os.getenv("WEATHER_LONGITUDE", "76.9905"))
TRAIN_DAYS = 365
LOCK = threading.RLock()  # re-entrant: train_models() can run inside a locked ingest call

SUBSCRIBERS = set()  # one queue per open dashboard stream
SUB_LOCK = threading.Lock()


class ModelNotReady(RuntimeError):
    """Raised when a reading arrives before the background training finishes."""


def publish(event):
    """Push a new-reading event to every connected dashboard."""
    message = json.dumps(event, default=str)
    with SUB_LOCK:
        for q in list(SUBSCRIBERS):
            try:
                q.put_nowait(message)
            except queue.Full:
                pass  # a stalled browser catches up on its next 5-second poll


STATE = {"history": None, "scaler": None, "isolation_forest": None, "lof": None, "trained_at": None}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_training_history(n_days=TRAIN_DAYS):
    """Real Open-Meteo history for this location, cached to disk. Synthetic data only if the download fails."""
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "training_history_real.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    try:
        raw = pl.fetch_openmeteo_history(LATITUDE, LONGITUDE, n_days=n_days)
        raw.to_csv(path, index=False)
        app.logger.info("Downloaded %s rows of Open-Meteo history", len(raw))
        return raw
    except Exception:
        app.logger.exception("Open-Meteo history download failed; using synthetic baseline.")
        return pl.generate_synthetic_aws_data(n_days=n_days, anomaly_fraction=0.02)


def train_models(n_days=TRAIN_DAYS):
    """Load the baseline and fit the detectors once, not per request."""
    raw = load_training_history(n_days)
    raw = pl.preprocess_data(raw)
    history = pl.engineer_features(raw)
    if len(history) < 30:
        raise RuntimeError("Training history must contain at least 30 rows.")

    scaler, iso, lof = pl.build_manual_detectors(history)
    with LOCK:
        STATE.update(history=raw.reset_index(drop=True), scaler=scaler,
                     isolation_forest=iso, lof=lof, trained_at=utc_now_iso())
    app.logger.info("Models trained on %s historical rows", len(history))
    return len(history)


def score_observation(payload):
    if STATE["history"] is None or STATE["scaler"] is None:
        raise ModelNotReady("The model is still training. Try again in about a minute.")

    values = {}
    for field in SENSOR_FIELDS:
        raw_value = payload.get(field, 0 if field == "rainfall" else None)
        if raw_value is None or raw_value == "":
            if field == "wind_direction":
                raw_value = float(STATE["history"]["wind_direction"].iloc[-1])
            else:
                raise ValueError(f"Missing required field: {field}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")
        values[field] = value

    feature_row, quality_info = pl.make_manual_feature_row(STATE["history"], **values)
    X = STATE["scaler"].transform(feature_row[pl.FEATURE_COLUMNS].values)

    iso_anomaly = int(STATE["isolation_forest"].predict(X)[0]) == -1
    iso_score = float(STATE["isolation_forest"].decision_function(X)[0])
    lof_anomaly = int(STATE["lof"].predict(X)[0]) == -1
    lof_score = float(STATE["lof"].decision_function(X)[0])
    hard_issue = bool(quality_info["hard_issue"])

    if hard_issue or (iso_anomaly and lof_anomaly):
        status = "Anomalous/Requires Review"
    elif iso_anomaly or lof_anomaly:
        status = "Suspicious"
    else:
        status = "Normal/Good"

    return {
        **values,
        "timestamp": payload.get("timestamp") or utc_now_iso(),
        "station_id": str(payload.get("station_id", "station-001")),
        "source": str(payload.get("source", "http")),
        "is_anomaly": bool(hard_issue or iso_anomaly or lof_anomaly),
        "anomaly_score": round(iso_score, 6),
        "status": status,
        "iso_forest_anomaly": iso_anomaly,
        "iso_forest_score": round(iso_score, 6),
        "lof_anomaly": lof_anomaly,
        "lof_score": round(lof_score, 6),
        "hard_quality_issue": hard_issue,
        "out_of_range": quality_info["out_of_range"],
    }


def _ingest(body):
    try:
        with LOCK:
            result = score_observation(body)
            invalid_fields = [f for f, (lo, hi) in PHYSICAL_LIMITS.items()
                              if not lo <= result[f] <= hi]
            if invalid_fields:
                result.update(hard_quality_issue=True, status="Anomalous/Requires Review",
                              is_anomaly=True, invalid_fields=invalid_fields)
            save_reading(result)
            STATE["history"] = pl.update_history(STATE["history"], result)
        publish({"timestamp": result["timestamp"], "source": result["source"],
                 "temperature": result["temperature"], "status": result["status"]})
        return jsonify({"ok": True, "observation": result}), 201
    except ModelNotReady as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        app.logger.exception("Ingestion failed")
        return jsonify({"ok": False, "error": "Processing failed. Check server logs."}), 500


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "trained": STATE["history"] is not None,
        "trained_at": STATE["trained_at"],
        "database": "connected",
    })


@app.post("/api/train")
def train():
    body = request.get_json(silent=True) or {}
    n_days = max(3, min(int(body.get("n_days", TRAIN_DAYS)), 730))
    count = train_models(n_days=n_days)
    return jsonify({"ok": True, "training_rows": count, "trained_at": STATE["trained_at"]})


@app.post("/api/ingest")
def ingest():
    return _ingest(request.get_json(silent=True) or {})


@app.post("/api/manual")
def manual():
    body = request.get_json(silent=True) or {}
    body["source"] = "manual"
    return _ingest(body)


@app.get("/api/readings/latest")
def readings_latest():
    limit = request.args.get("limit", default=100, type=int)
    limit = max(1, min(limit or 100, 1000))
    return jsonify({"ok": True, "readings": get_latest_readings(limit)})


@app.get("/api/summary")
def summary():
    return jsonify({"ok": True, **get_summary()})


@app.get("/api/timeseries")
def timeseries():
    variable = request.args.get("variable", "temperature")
    if variable not in set(SENSOR_FIELDS):
        return jsonify({"ok": False, "error": "Unsupported variable"}), 400
    rows = get_latest_readings(500)
    return jsonify({
        "ok": True,
        "variable": variable,
        "points": [{"timestamp": r["timestamp"], variable: r[variable],
                    "is_anomaly": r["is_anomaly"], "status": r["status"]} for r in rows],
    })


@app.get("/api/stream")
def stream():
    """Server-Sent Events: the browser gets a 'reading' event as soon as a reading is saved."""
    q = queue.Queue(maxsize=100)

    def events():
        with SUB_LOCK:
            SUBSCRIBERS.add(q)
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    message = q.get(timeout=20)
                    yield f"event: reading\ndata: {message}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            with SUB_LOCK:
                SUBSCRIBERS.discard(q)

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


init_db()


def background_train():
    try:
        train_models()
    except Exception:
        app.logger.exception("Model training failed; fix the data source and call POST /api/train.")


# Train in the background so the web server accepts requests immediately.
threading.Thread(target=background_train, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=False)
