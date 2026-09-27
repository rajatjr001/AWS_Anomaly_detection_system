"""
Backend API for the AWS Anomaly Detection & Data Quality dashboard.

Wraps the pipeline in pipeline.py (data generation -> preprocessing ->
feature engineering -> Isolation Forest / LOF anomaly detection -> data
quality assessment -> real-time monitor) behind a small Flask JSON API,
so the frontend (static/index.html) can run the pipeline and display
results, and step through a simulated real-time observation stream.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, jsonify, request, send_from_directory
import numpy as np
import pandas as pd

import pipeline as pl

app = Flask(__name__, static_folder="static", static_url_path="")

# In-memory state for the current session (single-user demo app).
STATE = {
    "df": None,          # full processed dataframe (history)
    "metrics": None,
    "monitor": None,      # RealTimeMonitor fit on the "historical" portion
    "stream_queue": None,  # remaining rows to simulate as they "arrive"
    "stream_cursor": 0,
    "stream_log": [],      # results of processed streamed points
}


def _clean_for_json(value):
    """Replace NaN/inf with None so jsonify doesn't choke."""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def df_to_records(df: pd.DataFrame, columns) -> list:
    out = []
    for _, row in df.iterrows():
        record = {"timestamp": row["timestamp"].isoformat()}
        for c in columns:
            record[c] = _clean_for_json(row[c])
        out.append(record)
    return out


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/run", methods=["POST"])
def run():
    """
    Runs the full offline pipeline (Steps 1-6, 8, 9) on a fresh synthetic
    dataset, and sets aside the LAST N_STREAM rows to simulate as a live
    incoming stream via /api/stream/next. Returns a summary for the dashboard.
    """
    body = request.get_json(silent=True) or {}
    n_days = int(body.get("n_days", 21))
    anomaly_fraction = float(body.get("anomaly_fraction", 0.03))
    n_stream = int(body.get("n_stream", 40))

    df = pl.generate_synthetic_aws_data(n_days=n_days, anomaly_fraction=anomaly_fraction)
    df = pl.preprocess_data(df)
    df = pl.engineer_features(df)
    df, split_info = pl.detect_anomalies(df)
    df = pl.assess_data_quality(df)
    metrics = pl.evaluate_model(df)

    history_df = df.iloc[:-n_stream].reset_index(drop=True)
    stream_df = df.iloc[-n_stream:].reset_index(drop=True)

    monitor = pl.build_realtime_monitor(history_df)

    STATE["df"] = history_df
    STATE["metrics"] = metrics
    STATE["monitor"] = monitor
    STATE["stream_queue"] = stream_df
    STATE["stream_cursor"] = 0
    STATE["stream_log"] = []

    quality_counts = history_df["data_quality_status"].value_counts().to_dict()

    return jsonify({
        "ok": True,
        "n_points": len(history_df),
        "n_stream_pending": len(stream_df),
        "split_info": split_info,
        "quality_counts": quality_counts,
        "metrics": metrics,
        "missing_counts": {
            c: int(history_df[f"{c}_was_missing"].sum())
            for c in ["temperature", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction"]
        },
    })


@app.route("/api/timeseries")
def timeseries():
    """Returns the processed history as JSON for charting."""
    if STATE["df"] is None:
        return jsonify({"ok": False, "error": "Run the pipeline first via POST /api/run"}), 400

    variable = request.args.get("variable", "temperature")
    valid_vars = ["temperature", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction"]
    if variable not in valid_vars:
        return jsonify({"ok": False, "error": f"variable must be one of {valid_vars}"}), 400

    df = STATE["df"]
    cols = [variable, "ml_anomaly_flag", "data_quality_status", "iso_forest_score"]
    records = df_to_records(df, cols)
    return jsonify({"ok": True, "variable": variable, "points": records})


@app.route("/api/quality_summary")
def quality_summary():
    if STATE["df"] is None:
        return jsonify({"ok": False, "error": "Run the pipeline first via POST /api/run"}), 400
    counts = STATE["df"]["data_quality_status"].value_counts().to_dict()
    return jsonify({"ok": True, "quality_counts": counts, "metrics": STATE["metrics"]})


@app.route("/api/stream/next", methods=["POST"])
def stream_next():
    """
    Simulates the next observation "arriving" from an AWS station and scores
    it in real time using the pre-fitted RealTimeMonitor (Step 7).
    """
    if STATE["monitor"] is None or STATE["stream_queue"] is None:
        return jsonify({"ok": False, "error": "Run the pipeline first via POST /api/run"}), 400

    cursor = STATE["stream_cursor"]
    queue = STATE["stream_queue"]
    if cursor >= len(queue):
        return jsonify({"ok": True, "done": True, "message": "No more simulated observations."})

    row = queue.iloc[cursor]
    result = STATE["monitor"].score_observation(row)

    entry = {
        "timestamp": row["timestamp"].isoformat(),
        "temperature": _clean_for_json(row["temperature"]),
        "humidity": _clean_for_json(row["humidity"]),
        "pressure": _clean_for_json(row["pressure"]),
        "wind_speed": _clean_for_json(row["wind_speed"]),
        "is_anomaly": result["is_anomaly"],
        "anomaly_score": round(result["anomaly_score"], 4),
        "status": result["status"],
    }
    STATE["stream_cursor"] += 1
    STATE["stream_log"].append(entry)

    return jsonify({"ok": True, "done": False, "observation": entry,
                     "remaining": len(queue) - STATE["stream_cursor"]})


@app.route("/api/stream/log")
def stream_log():
    return jsonify({"ok": True, "log": STATE["stream_log"]})


@app.route("/api/stream/reset", methods=["POST"])
def stream_reset():
    STATE["stream_cursor"] = 0
    STATE["stream_log"] = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
