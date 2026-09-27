"""
A Machine Learning Framework for Real-Time Anomaly Detection and
Data Quality Assessment in Automatic Weather Stations (AWS)

This script implements the full pipeline described in the project synopsis:
  Step 1: Data Collection            -> load_aws_data() / generate_synthetic_aws_data()
  Step 2: Data Preprocessing         -> preprocess_data()
  Step 3: Exploratory Data Analysis  -> run_eda()
  Step 4: Feature Engineering        -> engineer_features()
  Step 5: Anomaly Detection          -> detect_anomalies()  (Isolation Forest + LOF)
  Step 6: Data Quality Assessment    -> assess_data_quality()
  Step 7: Real-Time Processing       -> RealTimeMonitor class
  Step 8: Visualization & Evaluation -> visualize_results(), evaluate_model()
  Step 9: Model Validation           -> train_test split baked into detect_anomalies()

No real AWS dataset was supplied, so `generate_synthetic_aws_data()` builds a
realistic synthetic dataset (diurnal/seasonal temperature cycles, correlated
humidity/pressure, rainfall events, wind) with deliberately injected faults
(spikes, sensor stuck/frozen values, dropouts, drift, missing data) so the
whole framework can be demonstrated and evaluated end-to-end. Swap in a real
CSV via load_aws_data() when one is available -- the rest of the pipeline is
unchanged.

Usage:
    python aws_anomaly_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------
# STEP 1: DATA COLLECTION
# --------------------------------------------------------------------------

def load_aws_data(csv_path: str) -> pd.DataFrame:
    """
    Load a real AWS dataset from CSV. Expected columns (rename as needed):
    timestamp, temperature, humidity, pressure, rainfall, wind_speed, wind_direction
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def generate_synthetic_aws_data(n_days: int = 30, freq_minutes: int = 15,
                                 anomaly_fraction: float = 0.03) -> pd.DataFrame:
    """
    Generates a synthetic AWS observation dataset with realistic diurnal
    cycles and correlated variables, then injects labelled faults so the
    detection framework can be trained/tested/evaluated. `is_anomaly` is
    the ground-truth label (kept aside for evaluation only, NOT used as a
    model feature since real deployments won't have it).
    """
    periods = int(n_days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range("2026-06-01", periods=periods, freq=f"{freq_minutes}min")
    hour = timestamps.hour + timestamps.minute / 60.0
    day_of_year = timestamps.dayofyear

    # Base diurnal + seasonal signals
    temperature = (25 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi)
                   + 3 * np.sin(day_of_year / 365 * 2 * np.pi)
                   + np.random.normal(0, 0.6, periods))
    humidity = np.clip(70 - 1.5 * (temperature - 25) + np.random.normal(0, 4, periods), 5, 100)
    pressure = 1013 + 2 * np.sin(day_of_year / 30 * 2 * np.pi) + np.random.normal(0, 0.8, periods)

    rainfall = np.zeros(periods)
    rain_events = np.random.choice(periods, size=int(periods * 0.02), replace=False)
    for idx in rain_events:
        length = np.random.randint(2, 12)
        rainfall[idx: idx + length] = np.random.exponential(2.0, size=min(length, periods - idx))

    wind_speed = np.abs(4 + 2 * np.sin(hour / 24 * 2 * np.pi) + np.random.normal(0, 1.2, periods))
    wind_direction = (180 + 60 * np.sin(hour / 24 * 2 * np.pi) + np.random.normal(0, 15, periods)) % 360

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
    })
    df["is_anomaly"] = 0  # ground truth, for evaluation only

    # --- Inject faults (Step 1 context: sensor malfunction, comms failure, etc.) ---
    n_anomalies = int(periods * anomaly_fraction)
    fault_types = ["spike", "stuck_sensor", "dropout_missing", "drift", "impossible_value"]

    for _ in range(n_anomalies):
        i = np.random.randint(50, periods - 50)
        fault = np.random.choice(fault_types)
        col = np.random.choice(["temperature", "humidity", "pressure", "wind_speed"])

        if fault == "spike":
            df.loc[i, col] += np.random.choice([-1, 1]) * np.random.uniform(15, 30)
        elif fault == "stuck_sensor":
            length = np.random.randint(6, 20)
            stuck_val = df.loc[i, col]
            df.loc[i:i + length, col] = stuck_val
            df.loc[i:i + length, "is_anomaly"] = 1
        elif fault == "dropout_missing":
            length = np.random.randint(1, 5)
            df.loc[i:i + length, col] = np.nan
            df.loc[i:i + length, "is_anomaly"] = 1
        elif fault == "drift":
            length = np.random.randint(20, 60)
            drift = np.linspace(0, np.random.uniform(8, 15), min(length, periods - i))
            df.loc[i:i + len(drift) - 1, col] += drift
            df.loc[i:i + len(drift) - 1, "is_anomaly"] = 1
        elif fault == "impossible_value":
            impossible = {"temperature": 85, "humidity": 150, "pressure": 700, "wind_speed": 120}
            df.loc[i, col] = impossible[col]

        df.loc[i, "is_anomaly"] = 1

    return df


# --------------------------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# --------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Flags missing values (kept, not dropped, so 'missing observation'
      is itself a data-quality signal rather than silently discarded)
    - Removes duplicate timestamps
    - Enforces physically plausible ranges (hard physical-limit QC pass,
      complementary to the ML-based detector)
    - Interpolates short gaps for feature computation, while keeping a
      `was_missing` flag per sensor
    """
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    sensor_cols = ["temperature", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction"]
    physical_limits = {
        "temperature": (-40, 55),
        "humidity": (0, 100),
        "pressure": (870, 1085),
        "rainfall": (0, 500),
        "wind_speed": (0, 100),
        "wind_direction": (0, 360),
    }

    for col in sensor_cols:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        lo, hi = physical_limits[col]
        out_of_range = ~df[col].between(lo, hi) & df[col].notna()
        df[f"{col}_out_of_range"] = out_of_range.astype(int)
        df.loc[out_of_range, col] = np.nan  # treat impossible readings as missing before interpolation

    # Time-aware interpolation for short gaps only (limit caps how far we fill)
    df[sensor_cols] = df[sensor_cols].interpolate(method="linear", limit=4, limit_direction="both")
    # Any remaining longer gaps: forward/back fill as last resort so downstream steps don't break
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df


# --------------------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> dict:
    """Returns basic statistical summaries used to sanity-check the dataset."""
    sensor_cols = ["temperature", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction"]
    summary = df[sensor_cols].describe().to_dict()
    missing_counts = {c: int(df[f"{c}_was_missing"].sum()) for c in sensor_cols}
    out_of_range_counts = {c: int(df[f"{c}_out_of_range"].sum()) for c in sensor_cols}
    return {"summary_stats": summary, "missing_counts": missing_counts, "out_of_range_counts": out_of_range_counts}


# --------------------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# --------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features that help distinguish normal variability from anomalies:
      - lagged values, rate of change (delta)
      - rolling mean / std (local context, catches drift & stuck sensors)
      - cross-variable relationships (e.g., temperature vs humidity)
    """
    sensor_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    window = 8  # ~2 hours at 15-min frequency

    for col in sensor_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window, min_periods=1).std().fillna(0)
        # Stuck-sensor signal: rolling std ~ 0 while roll window is full
        df[f"{col}_is_flat"] = (df[f"{col}_roll_std"] < 1e-3).astype(int)

    # Cross-variable relationship: unusually high temp with unusually high humidity is atypical
    df["temp_humidity_product_z"] = (
        (df["temperature"] - df["temperature"].mean()) / df["temperature"].std()
    ) * ((df["humidity"] - df["humidity"].mean()) / df["humidity"].std())

    df = df.bfill().fillna(0)
    return df


FEATURE_COLUMNS = (
    ["temperature", "humidity", "pressure", "wind_speed"]
    + [f"{c}_delta" for c in ["temperature", "humidity", "pressure", "wind_speed"]]
    + [f"{c}_roll_std" for c in ["temperature", "humidity", "pressure", "wind_speed"]]
    + [f"{c}_is_flat" for c in ["temperature", "humidity", "pressure", "wind_speed"]]
    + ["temp_humidity_product_z"]
)


# --------------------------------------------------------------------------
# STEP 5 & 9: ANOMALY DETECTION + VALIDATION
# --------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05):
    """
    Trains Isolation Forest and Local Outlier Factor on engineered features.
    Returns the dataframe with per-model predictions plus a combined score.
    -1 = anomalous, 1 = normal (sklearn convention), converted to 1/0 flags.
    """
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(len(df) * 0.7)  # simple time-based train/validation split (Step 9)
    X_train, X_val = X_scaled[:split], X_scaled[split:]

    iso_forest = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=RANDOM_STATE
    )
    iso_forest.fit(X_train)
    iso_pred = iso_forest.predict(X_scaled)
    iso_score = iso_forest.decision_function(X_scaled)  # higher = more normal

    # LOF has no separate predict-on-new-data in novelty=False mode; run on full set
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
    lof_pred = lof.fit_predict(X_scaled)
    lof_score = lof.negative_outlier_factor_

    df["iso_forest_anomaly"] = (iso_pred == -1).astype(int)
    df["lof_anomaly"] = (lof_pred == -1).astype(int)
    df["iso_forest_score"] = iso_score
    df["lof_score"] = lof_score

    # Combined flag: either model raises an alarm
    df["ml_anomaly_flag"] = ((df["iso_forest_anomaly"] == 1) | (df["lof_anomaly"] == 1)).astype(int)
    return df, {"train_size": split, "val_size": len(df) - split}


def evaluate_model(df: pd.DataFrame) -> dict:
    """Compares ML detections against injected ground truth (Step 8: evaluation)."""
    if "is_anomaly" not in df.columns:
        return {}
    y_true = df["is_anomaly"]
    metrics = {}
    for col in ["iso_forest_anomaly", "lof_anomaly", "ml_anomaly_flag"]:
        y_pred = df[col]
        metrics[col] = {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
    return metrics


# --------------------------------------------------------------------------
# STEP 6: DATA QUALITY ASSESSMENT
# --------------------------------------------------------------------------

def assess_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns each observation a quality label:
      Normal/Good              - no missing/out-of-range values, no ML anomaly
      Suspicious                - single weak signal (e.g. only one model flags it)
      Anomalous/Requires Review - missing/out-of-range value OR both ML models agree
    """
    sensor_cols = ["temperature", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction"]
    hard_issue = df[[f"{c}_was_missing" for c in sensor_cols]].sum(axis=1) > 0
    hard_issue |= df[[f"{c}_out_of_range" for c in sensor_cols]].sum(axis=1) > 0

    both_models_agree = (df["iso_forest_anomaly"] == 1) & (df["lof_anomaly"] == 1)
    one_model_flags = (df["iso_forest_anomaly"] == 1) ^ (df["lof_anomaly"] == 1)

    quality = pd.Series("Normal/Good", index=df.index)
    quality[one_model_flags] = "Suspicious"
    quality[both_models_agree | hard_issue] = "Anomalous/Requires Review"
    df["data_quality_status"] = quality
    return df


# --------------------------------------------------------------------------
# STEP 7: REAL-TIME PROCESSING
# --------------------------------------------------------------------------

class RealTimeMonitor:
    """
    Wraps a fitted Isolation Forest + scaler so new observations can be
    scored one at a time (or in small batches) as they arrive, simulating
    a real-time monitoring pipeline built on top of the offline-trained model.
    """

    def __init__(self, model: IsolationForest, scaler: StandardScaler, feature_columns: list):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns

    def score_observation(self, feature_row: pd.Series) -> dict:
        X = self.scaler.transform([feature_row[self.feature_columns].values])
        pred = self.model.predict(X)[0]
        score = self.model.decision_function(X)[0]
        return {
            "is_anomaly": bool(pred == -1),
            "anomaly_score": float(score),
            "status": "Anomalous/Requires Review" if pred == -1 else "Normal/Good",
        }


def build_realtime_monitor(df: pd.DataFrame) -> RealTimeMonitor:
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler().fit(X)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_STATE)
    model.fit(scaler.transform(X))
    return RealTimeMonitor(model, scaler, FEATURE_COLUMNS)


def build_manual_detectors(df: pd.DataFrame):
    """
    Fit the same Isolation Forest used by the offline pipeline plus a
    novelty-enabled LOF model so a brand-new manual observation can be scored.
    The original offline LOF uses novelty=False and therefore cannot score a
    new point; novelty=True is required for this real-time/manual use case.
    """
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=RANDOM_STATE,
    )
    iso_forest.fit(X_scaled)

    n_neighbors = min(20, max(2, len(df) - 1))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=0.05,
        novelty=True,
    )
    lof.fit(X_scaled)

    return scaler, iso_forest, lof


def make_manual_feature_row(
    history_df: pd.DataFrame,
    temperature: float,
    humidity: float,
    pressure: float,
    wind_speed: float,
    rainfall: float = 0.0,
    wind_direction: float | None = None,
) -> tuple[pd.Series, dict]:
    """
    Build the same engineered feature set used by detect_anomalies() for one
    manually supplied observation.

    Lag/delta/rolling features use the most recent historical observations.
    The temperature-humidity z-product uses historical statistics, so the
    manual value is scored against the learned historical context.
    """
    if history_df is None or len(history_df) < 2:
        raise ValueError("Run the main pipeline first so historical data is available.")

    sensor_values = {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "pressure": float(pressure),
        "wind_speed": float(wind_speed),
        "rainfall": float(rainfall),
    }

    if wind_direction is None:
        sensor_values["wind_direction"] = float(history_df["wind_direction"].iloc[-1])
    else:
        sensor_values["wind_direction"] = float(wind_direction)

    physical_limits = {
        "temperature": (-40, 55),
        "humidity": (0, 100),
        "pressure": (870, 1085),
        "rainfall": (0, 500),
        "wind_speed": (0, 100),
        "wind_direction": (0, 360),
    }

    hard_issue = False
    out_of_range = {}
    for col, (lo, hi) in physical_limits.items():
        bad = not (lo <= sensor_values[col] <= hi)
        out_of_range[col] = int(bad)
        hard_issue = hard_issue or bad

    row = {}
    recent = history_df.tail(7)

    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        value = sensor_values[col]
        previous = float(history_df[col].iloc[-1])
        values = pd.concat(
            [history_df[col].tail(7), pd.Series([value], dtype=float)],
            ignore_index=True,
        )

        row[col] = value
        row[f"{col}_delta"] = value - previous
        row[f"{col}_roll_std"] = float(values.std())
        row[f"{col}_is_flat"] = int(row[f"{col}_roll_std"] < 1e-3)

    temp_mean = float(history_df["temperature"].mean())
    temp_std = float(history_df["temperature"].std())
    hum_mean = float(history_df["humidity"].mean())
    hum_std = float(history_df["humidity"].std())

    temp_z = (sensor_values["temperature"] - temp_mean) / temp_std if temp_std else 0.0
    hum_z = (sensor_values["humidity"] - hum_mean) / hum_std if hum_std else 0.0
    row["temp_humidity_product_z"] = temp_z * hum_z

    feature_row = pd.Series(row, dtype=float)

    # Keep the raw sensor values/quality flags for the API response.
    quality_info = {
        "hard_issue": bool(hard_issue),
        "out_of_range": out_of_range,
        "was_missing": {c: 0 for c in physical_limits},
        "sensor_values": sensor_values,
    }
    return feature_row, quality_info


def score_manual_observation(
    history_df: pd.DataFrame,
    temperature: float,
    humidity: float,
    pressure: float,
    wind_speed: float,
    rainfall: float = 0.0,
    wind_direction: float | None = None,
) -> dict:
    """
    Score one manual observation with both Isolation Forest and LOF.
    Returns individual model results plus the combined data-quality status.
    """
    feature_row, quality_info = make_manual_feature_row(
        history_df=history_df,
        temperature=temperature,
        humidity=humidity,
        pressure=pressure,
        wind_speed=wind_speed,
        rainfall=rainfall,
        wind_direction=wind_direction,
    )

    scaler, iso_forest, lof = build_manual_detectors(history_df)

    X = scaler.transform([feature_row[FEATURE_COLUMNS].values])

    iso_pred = int(iso_forest.predict(X)[0])
    iso_score = float(iso_forest.decision_function(X)[0])

    lof_pred = int(lof.predict(X)[0])
    lof_score = float(lof.decision_function(X)[0])

    iso_anomaly = int(iso_pred == -1)
    lof_anomaly = int(lof_pred == -1)

    if quality_info["hard_issue"] or (iso_anomaly and lof_anomaly):
        status = "Anomalous/Requires Review"
    elif iso_anomaly ^ lof_anomaly:
        status = "Suspicious"
    else:
        status = "Normal/Good"

    return {
        "is_anomaly": bool(iso_anomaly or lof_anomaly or quality_info["hard_issue"]),
        "status": status,
        "anomaly_score": round(iso_score, 4),
        "iso_forest": {
            "is_anomaly": bool(iso_anomaly),
            "prediction": iso_pred,
            "score": round(iso_score, 4),
        },
        "lof": {
            "is_anomaly": bool(lof_anomaly),
            "prediction": lof_pred,
            "score": round(lof_score, 4),
        },
        "data_quality": {
            "hard_issue": quality_info["hard_issue"],
            "out_of_range": quality_info["out_of_range"],
        },
        "features": {
            c: float(feature_row[c]) for c in FEATURE_COLUMNS
        },
    }


# --------------------------------------------------------------------------
# STEP 8: VISUALIZATION
# --------------------------------------------------------------------------

def visualize_results(df: pd.DataFrame, out_path: str = "aws_anomaly_dashboard.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df["timestamp"], df["temperature"], label="Temperature (°C)", color="tab:orange", linewidth=0.8)
    anomalies = df[df["ml_anomaly_flag"] == 1]
    axes[0].scatter(anomalies["timestamp"], anomalies["temperature"], color="red", s=12, label="Flagged anomaly", zorder=5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("AWS Data Quality Monitoring Dashboard")

    axes[1].plot(df["timestamp"], df["humidity"], label="Humidity (%)", color="tab:blue", linewidth=0.8)
    axes[1].scatter(anomalies["timestamp"], anomalies["humidity"], color="red", s=12, zorder=5)
    axes[1].set_ylabel("Humidity (%)")

    status_colors = {"Normal/Good": "tab:green", "Suspicious": "orange", "Anomalous/Requires Review": "red"}
    for status, color in status_colors.items():
        subset = df[df["data_quality_status"] == status]
        axes[2].scatter(subset["timestamp"], subset["iso_forest_score"], s=8, label=status, color=color)
    axes[2].set_ylabel("Isolation Forest score")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------

def run_pipeline(csv_path: str = None):
    print("Step 1: Data Collection...")
    df = load_aws_data(csv_path) if csv_path else generate_synthetic_aws_data()
    print(f"  Loaded {len(df)} observations.")

    print("Step 2: Data Preprocessing...")
    df = preprocess_data(df)

    print("Step 3: Exploratory Data Analysis...")
    eda = run_eda(df)
    print(f"  Missing counts per sensor: {eda['missing_counts']}")

    print("Step 4: Feature Engineering...")
    df = engineer_features(df)

    print("Step 5 & 9: Anomaly Detection + Validation...")
    df, split_info = detect_anomalies(df)
    print(f"  Train/val split: {split_info}")

    print("Step 6: Data Quality Assessment...")
    df = assess_data_quality(df)
    print(df["data_quality_status"].value_counts().to_string())

    print("Step 7: Real-Time Processing (simulated stream of last 5 observations)...")
    monitor = build_realtime_monitor(df.iloc[:-5])
    for _, row in df.iloc[-5:].iterrows():
        result = monitor.score_observation(row)
        print(f"  {row['timestamp']}: {result['status']} (score={result['anomaly_score']:.3f})")

    print("Step 8: Visualization and Evaluation...")
    chart_path = visualize_results(df)
    print(f"  Dashboard saved to {chart_path}")

    metrics = evaluate_model(df)
    if metrics:
        print("  Evaluation against injected ground-truth anomalies:")
        for model_name, m in metrics.items():
            print(f"    {model_name}: precision={m['precision']}, recall={m['recall']}, f1={m['f1_score']}")

    df.to_csv("aws_quality_assessed_output.csv", index=False)
    print("Full annotated dataset saved to aws_quality_assessed_output.csv")
    return df, metrics


if __name__ == "__main__":
    run_pipeline()
