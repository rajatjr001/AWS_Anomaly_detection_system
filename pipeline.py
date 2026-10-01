"""Data preparation, features, and anomaly detectors used by app.py."""
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = ["temperature", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction"]
FEATURE_COLUMNS = SENSOR_COLS + [
    "temp_change_1h", "humidity_change_1h", "pressure_change_3h",
    "temp_roll_mean_6h", "pressure_roll_std_6h",
]
PHYSICAL_LIMITS = {
    "temperature": (-40, 55), "humidity": (0, 100), "pressure": (870, 1085),
    "rainfall": (0, 500), "wind_speed": (0, 100), "wind_direction": (0, 360),
}
STEP_MINUTES = 15  # Open-Meteo's current values update about every 15 minutes

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_VARS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "precipitation": "rainfall",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
}


def fetch_openmeteo_history(latitude, longitude, n_days=365, lag_days=7):
    """Real hourly weather from Open-Meteo's archive, resampled to the 15-minute grid used by live readings."""
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lag_days)).tz_localize(None).normalize()
    start = end - pd.Timedelta(days=n_days)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(ARCHIVE_VARS),
        "timezone": "UTC",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for api_name, col in ARCHIVE_VARS.items():
        frame[col] = pd.to_numeric(pd.Series(hourly[api_name]), errors="coerce")

    # Interpolate wind direction through sine/cosine so 350° and 10° blend to 0°, not 180°.
    radians = np.deg2rad(frame["wind_direction"])
    frame["_sin"], frame["_cos"] = np.sin(radians), np.cos(radians)
    frame = frame.dropna(subset=SENSOR_COLS + ["_sin", "_cos"]).set_index("timestamp")

    grid = frame[["temperature", "humidity", "pressure", "rainfall", "wind_speed", "_sin", "_cos"]]
    grid = grid.resample(f"{STEP_MINUTES}min").interpolate(method="time")
    grid["wind_direction"] = np.round(np.rad2deg(np.arctan2(grid["_sin"], grid["_cos"])), 3) % 360
    grid = grid.dropna().reset_index()
    grid[SENSOR_COLS] = grid[SENSOR_COLS].round(3)
    return grid[["timestamp"] + SENSOR_COLS]


def generate_synthetic_aws_data(n_days=30, freq_minutes=STEP_MINUTES, anomaly_fraction=0.02, seed=42):
    """Fallback baseline used only if the Open-Meteo history download fails."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor(f"{freq_minutes}min")
    ts = pd.date_range(end=end, periods=n_days * 24 * 60 // freq_minutes, freq=f"{freq_minutes}min")
    n = len(ts)
    doy = ts.dayofyear.values
    local_hour = (ts.hour.values + ts.minute.values / 60 + 5.5) % 24  # IST is UTC+5:30

    t_mean = 25 - 8 * np.cos(2 * np.pi * (doy - 15) / 365)
    temperature = t_mean + 6 * np.sin(2 * np.pi * (local_hour - 9) / 24) + rng.normal(0, 0.7, n)
    humidity = np.clip(85 - 1.2 * (temperature - 20) + rng.normal(0, 4, n), 15, 100)
    pressure = 990 + 4 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1.0, n)
    wind_speed = np.clip(rng.gamma(2.0, 4.0, n), 0, 60)
    rainfall = np.where(rng.random(n) < 0.02, rng.exponential(3.0, n), 0.0)
    wind_direction = rng.uniform(0, 360, n)

    is_anomaly = rng.random(n) < anomaly_fraction
    sign = rng.choice([-1.0, 1.0], n)
    temperature = np.where(is_anomaly, temperature + sign * rng.uniform(12, 20, n), temperature)
    pressure = np.where(is_anomaly, pressure - sign * rng.uniform(12, 25, n), pressure)
    humidity = np.where(is_anomaly, np.clip(humidity + sign * 30, 0, 100), humidity)

    return pd.DataFrame({
        "timestamp": ts,
        "temperature": temperature.round(2),
        "humidity": humidity.round(2),
        "pressure": pressure.round(2),
        "rainfall": rainfall.round(2),
        "wind_speed": wind_speed.round(2),
        "wind_direction": wind_direction.round(1),
        "is_anomaly": is_anomaly,
    })


def preprocess_data(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[SENSOR_COLS] = df[SENSOR_COLS].interpolate(limit_direction="both")
    return df.dropna(subset=SENSOR_COLS).reset_index(drop=True)


def _add_features(df):
    out = df.copy()
    out["temp_change_1h"] = out["temperature"].diff(4)
    out["humidity_change_1h"] = out["humidity"].diff(4)
    out["pressure_change_3h"] = out["pressure"].diff(12)
    out["temp_roll_mean_6h"] = out["temperature"].rolling(24, min_periods=1).mean()
    out["pressure_roll_std_6h"] = out["pressure"].rolling(24, min_periods=2).std().fillna(0)
    return out


def engineer_features(df):
    return _add_features(df).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def build_manual_detectors(history):
    X_raw = history[FEATURE_COLUMNS].values
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42).fit(X)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=0.02).fit(X)
    return scaler, iso, lof


def make_manual_feature_row(history_df, temperature, humidity, pressure, wind_speed, rainfall, wind_direction):
    """Feature row for one new observation, computed from the most recent history."""
    values = {"temperature": temperature, "humidity": humidity, "pressure": pressure,
              "wind_speed": wind_speed, "rainfall": rainfall, "wind_direction": wind_direction}
    last_ts = pd.Timestamp(history_df["timestamp"].iloc[-1])
    new = pd.DataFrame([{"timestamp": last_ts + pd.Timedelta(minutes=STEP_MINUTES), **values}])
    tail = history_df[["timestamp"] + SENSOR_COLS].tail(30)
    row = _add_features(pd.concat([tail, new], ignore_index=True)).tail(1)
    out_of_range = [f for f, (lo, hi) in PHYSICAL_LIMITS.items() if not lo <= values[f] <= hi]
    return row, {"hard_issue": bool(out_of_range), "out_of_range": out_of_range}


def update_history(history_df, observation):
    """Append a scored observation so later features see the most recent readings."""
    last_ts = pd.Timestamp(history_df["timestamp"].iloc[-1])
    ts = pd.to_datetime(observation.get("timestamp"), utc=True, errors="coerce")
    ts = pd.Timestamp.now(tz="UTC") if pd.isna(ts) else ts
    ts = ts.tz_convert(None)
    if ts <= last_ts:
        ts = last_ts + pd.Timedelta(seconds=1)
    row = {"timestamp": ts, **{c: observation[c] for c in SENSOR_COLS}}
    combined = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
    return combined.tail(50000).reset_index(drop=True)
