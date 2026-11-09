import os
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import pipeline as pl
from database import init_db, save_reading, get_latest_readings, get_summary

app = Flask(__name__, static_folder="static", static_url_path="")
logging.basicConfig(level=logging.INFO)

SENSOR_FIELDS = [
    "temperature", "humidity", "pressure",
    "rainfall", "wind_speed", "wind_direction"
]

PHYSICAL_LIMITS = {
    "temperature": (-40, 55),
    "humidity": (0, 100),
    "pressure": (870, 1085),
    "rainfall": (0, 500),
    "wind_speed": (0, 100),
    "wind_direction": (0, 360),
}

STATE = {
    "history": None,
    "scaler": None,
    "isolation_forest": None,
    "lof": None,
    "trained_at": None,
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def train_models(n_days=30):
    """Create/load a baseline and fit inference models once, not per request."""
    os.makedirs("data", exist_ok=True)
    history_path = os.path.join("data", "training_history.csv")

    if os.path.exists(history_path):
        raw = pd.read_csv(history_path, parse_dates=["timestamp"])
    else:
        raw = pl.generate_synthetic_aws_data(
            n_days=n_days,
            freq_minutes=15,
            anomaly_fraction=0.02
        )
        raw.to_csv(history_path, index=False)

    raw = pl.preprocess_data(raw)
    history = pl.engineer_features(raw)

    if len(history) < 30:
        raise RuntimeError("Training history must contain at least 30 rows.")

    scaler, iso, lof = pl.build_manual_detectors(history)

    STATE["history"] = raw.reset_index(drop=True)
    STATE["scaler"] = scaler
    STATE["isolation_forest"] = iso
    STATE["lof"] = lof
    STATE["trained_at"] = utc_now_iso()

    app.logger.info("Models trained on %s historical rows", len(history))
    return len(history)


def score_observation(payload):
    """Validate and score one new observation using the cached fitted models."""
    if STATE["history"] is None or STATE["scaler"] is None:
        train_models()

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

    # Feature engineering is kept consistent with pipeline.py.
    feature_row, quality_info = pl.make_manual_feature_row(
        history_df=STATE["history"],
        temperature=values["temperature"],
        humidity=values["humidity"],
        pressure=values["pressure"],
        wind_speed=values["wind_speed"],
        rainfall=values["rainfall"],
        wind_direction=values["wind_direction"],
    )

    X = STATE["scaler"].transform(
        [feature_row[pl.FEATURE_COLUMNS].values]
    )

    iso_pred = int(STATE["isolation_forest"].predict(X)[0])
    iso_score = float(STATE["isolation_forest"].decision_function(X)[0])
    lof_pred = int(STATE["lof"].predict(X)[0])
    lof_score = float(STATE["lof"].decision_function(X)[0])

    iso_anomaly = iso_pred == -1
    lof_anomaly = lof_pred == -1
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
        "iso_forest_anomaly": bool(iso_anomaly),
        "iso_forest_score": round(iso_score, 6),
        "lof_anomaly": bool(lof_anomaly),
        "lof_score": round(lof_score, 6),
        "hard_quality_issue": hard_issue,
        "out_of_range": quality_info["out_of_range"],
    }


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
    n_days = int(body.get("n_days", 30))
    n_days = max(3, min(n_days, 365))
    count = train_models(n_days=n_days)
    return jsonify({"ok": True, "training_rows": count, "trained_at": STATE["trained_at"]})


@app.post("/api/ingest")
def ingest():
    body = request.get_json(silent=True) or {}
    try:
        result = score_observation(body)

        # Hard physical-limit checks are preserved separately from ML flags.
        invalid_fields = []
        for field, (low, high) in PHYSICAL_LIMITS.items():
            if not low <= result[field] <= high:
                invalid_fields.append(field)

        if invalid_fields:
            result["hard_quality_issue"] = True
            result["status"] = "Anomalous/Requires Review"
            result["is_anomaly"] = True
            result["invalid_fields"] = invalid_fields

        save_reading(result)
        return jsonify({"ok": True, "observation": result}), 201

    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Ingestion failed")
        return jsonify({"ok": False, "error": "Processing failed. Check server logs."}), 500


@app.post("/api/manual")
def manual():
    body = request.get_json(silent=True) or {}
    body["source"] = "manual"
    return ingest()


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
    allowed = set(SENSOR_FIELDS)
    if variable not in allowed:
        return jsonify({"ok": False, "error": "Unsupported variable"}), 400
    rows = get_latest_readings(500)
    return jsonify({
        "ok": True,
        "variable": variable,
        "points": [
            {
                "timestamp": row["timestamp"],
                variable: row[variable],
                "is_anomaly": row["is_anomaly"],
                "status": row["status"],
            }
            for row in rows
        ]
    })


@app.get("/api/quality_summary")
def quality_summary():
    return jsonify({"ok": True, **get_summary()})


init_db()
try:
    train_models()
except Exception:
    app.logger.exception("Initial model training failed; call POST /api/train after fixing data.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=False)
