"""
Backend API for the weather station anomaly dashboard.

Run:
    python app.py
Then open http://127.0.0.1:5001
"""

import os
import tempfile

from flask import Flask, Response, jsonify, request, send_from_directory
import numpy as np
import pandas as pd

import pipeline as pl
import recommender as rec

app = Flask(__name__, static_folder="static", static_url_path="")

# In-memory state for the current session (single-user demo app).
STATE = {
    "df": None,
    "metrics": None,
    "monitor": None,
    "stream_queue": None,
    "stream_cursor": 0,
    "stream_log": [],
}

SENSORS = ["temperature", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction"]


def _clean_for_json(value):
    """Replace NaN/inf with None so jsonify doesn't choke."""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def df_to_records(df, columns):
    out = []
    for _, row in df.iterrows():
        record = {"timestamp": row["timestamp"].isoformat()}
        for c in columns:
            record[c] = _clean_for_json(row[c])
        out.append(record)
    return out


def _fit_state(df, n_stream):
    """Run the full pipeline on a DataFrame and store the results."""
    df = pl.preprocess_data(df)
    df = pl.engineer_features(df)
    df, split_info = pl.detect_anomalies(df)
    df = pl.assess_data_quality(df)
    metrics = pl.evaluate_model(df)
    history_df = df.iloc[:-n_stream].reset_index(drop=True)
    stream_df = df.iloc[-n_stream:].reset_index(drop=True)
    STATE.update(
        df=history_df,
        metrics=metrics,
        monitor=pl.build_realtime_monitor(history_df),
        stream_queue=stream_df,
        stream_cursor=0,
        stream_log=[],
    )
    return {
        "ok": True,
        "n_points": len(history_df),
        "n_stream_pending": len(stream_df),
        "split_info": split_info,
        "metrics": metrics,
        "quality_counts": history_df["data_quality_status"].value_counts().to_dict(),
    }


def _need_data():
    if STATE["df"] is None:
        return jsonify({"ok": False, "error": "Run the analysis first."}), 400
    return None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/run", methods=["POST"])
def run():
    body = request.get_json(silent=True) or {}
    n_days = int(body.get("n_days", 21))
    anomaly_fraction = float(body.get("anomaly_fraction", 0.03))
    n_stream = int(body.get("n_stream", 40))
    df = pl.generate_synthetic_aws_data(n_days=n_days, anomaly_fraction=anomaly_fraction)
    return jsonify(_fit_state(df, n_stream))


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    f = request.files.get("file")
    if f is None or not f.filename.lower().endswith(".csv"):
        return jsonify({"ok": False, "error": "Choose a .csv file."}), 400
    path = os.path.join(tempfile.gettempdir(), "upload_aws.csv")
    f.save(path)
    try:
        df = pl.load_aws_data(path)
        need = ["timestamp"] + SENSORS
        miss = [c for c in need if c not in df.columns]
        if miss:
            return jsonify({"ok": False, "error": "CSV is missing columns: " + ", ".join(miss)}), 400
        if len(df) < 200:
            return jsonify({"ok": False, "error": "Need at least 200 rows."}), 400
        return jsonify(_fit_state(df, min(40, len(df) // 5)))
    except Exception as exc:
        app.logger.exception("Upload failed")
        return jsonify({"ok": False, "error": f"Could not analyse this file: {exc}"}), 400


@app.route("/api/manual", methods=["POST"])
def manual():
    err = _need_data()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    required = ["temperature", "humidity", "pressure", "wind_speed"]
    missing = [k for k in required if k not in body or body[k] in (None, "")]
    if missing:
        return jsonify({"ok": False, "error": f"Missing required values: {', '.join(missing)}"}), 400
    try:
        values = {
            "temperature": float(body["temperature"]),
            "humidity": float(body["humidity"]),
            "pressure": float(body["pressure"]),
            "wind_speed": float(body["wind_speed"]),
            "rainfall": float(body.get("rainfall", 0.0)),
        }
        wd = body.get("wind_direction")
        wind_direction = None if wd in (None, "") else float(wd)

        result = pl.score_manual_observation(
            history_df=STATE["df"],
            temperature=values["temperature"],
            humidity=values["humidity"],
            pressure=values["pressure"],
            wind_speed=values["wind_speed"],
            rainfall=values["rainfall"],
            wind_direction=wind_direction,
        )
        hour = int(body.get("hour", pd.Timestamp.now().hour))
        result["why"] = rec.explain_values(rec.baseline(STATE["df"]), values, hour)

        entry = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            **{k: values[k] for k in ["temperature", "humidity", "pressure", "wind_speed"]},
            "is_anomaly": result["is_anomaly"],
            "anomaly_score": result["anomaly_score"],
            "status": result["status"],
        }
        STATE["stream_log"].append(entry)
        return jsonify({"ok": True, "observation": entry, "result": result})
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"Invalid sensor value: {exc}"}), 400
    except Exception as exc:
        app.logger.exception("Manual anomaly scoring failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/timeseries")
def timeseries():
    err = _need_data()
    if err:
        return err
    variable = request.args.get("variable", "temperature")
    if variable not in SENSORS:
        return jsonify({"ok": False, "error": f"variable must be one of {SENSORS}"}), 400
    cols = [variable, "ml_anomaly_flag", "data_quality_status", "iso_forest_score"]
    return jsonify({"ok": True, "variable": variable, "points": df_to_records(STATE["df"], cols)})


@app.route("/api/quality_summary")
def quality_summary():
    err = _need_data()
    if err:
        return err
    counts = STATE["df"]["data_quality_status"].value_counts().to_dict()
    return jsonify({"ok": True, "quality_counts": counts, "metrics": STATE["metrics"]})


@app.route("/api/incidents")
def incidents():
    err = _need_data()
    if err:
        return err
    incs = rec.find_incidents(STATE["df"], rec.baseline(STATE["df"]))
    return jsonify({"ok": True, "incidents": incs})


@app.route("/api/recommendations")
def recommendations():
    err = _need_data()
    if err:
        return err
    b = rec.baseline(STATE["df"])
    incs = rec.find_incidents(STATE["df"], b)
    return jsonify({"ok": True, **rec.recommendations(STATE["df"], b, incs)})


@app.route("/api/incidents.csv")
def incidents_csv():
    err = _need_data()
    if err:
        return err
    incs = rec.find_incidents(STATE["df"], rec.baseline(STATE["df"]))
    csv = pd.DataFrame(incs).drop(columns=["sensor_key", "hour"], errors="ignore").to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=incidents.csv"})


@app.route("/api/stream/next", methods=["POST"])
def stream_next():
    if STATE["monitor"] is None or STATE["stream_queue"] is None:
        return jsonify({"ok": False, "error": "Run the analysis first."}), 400
    cursor, queue = STATE["stream_cursor"], STATE["stream_queue"]
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
    app.run(debug=True, port=5001)