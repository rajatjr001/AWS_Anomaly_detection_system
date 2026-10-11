"""Turns raw anomaly flags into incidents, likely causes and recommended actions."""
import numpy as np
import pandas as pd

SENS = ["temperature", "humidity", "pressure", "wind_speed"]
SIX = SENS[:3] + ["rainfall", "wind_speed", "wind_direction"]
LABEL = {"temperature": "Temperature", "humidity": "Humidity", "pressure": "Pressure", "wind_speed": "Wind speed"}


def baseline(df):
    """What is 'usual' for each sensor at each hour of day, from readings no model flagged."""
    clean = df[df["ml_anomaly_flag"] == 0]
    b = {}
    for s in SENS:
        g = clean.groupby("hour")[s]
        mad = g.apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))
        floor = 0.25 * clean[s].std()
        d = clean[f"{s}_delta"].abs().median() * 1.4826
        b[s] = {"med": g.median().to_dict(), "scale": mad.clip(lower=floor).to_dict(), "dscale": max(d, 1e-6)}
    return b


def _z(df, b):
    d, h = df.copy(), df["hour"]
    for s in SENS:
        d["z_" + s] = (d[s] - h.map(b[s]["med"])) / h.map(b[s]["scale"])
        d["dz_" + s] = d[s + "_delta"] / b[s]["dscale"]
    return d.fillna(0)


def explain_values(b, values, hour):
    """Why is this reading unusual? Top sensors ranked by distance from the hourly norm."""
    out = []
    for s in SENS:
        exp = b[s]["med"].get(hour)
        if exp is None:
            continue
        z = (values[s] - exp) / b[s]["scale"][hour]
        out.append({"sensor": s, "z": round(float(z), 1),
                    "text": f"{LABEL[s]} is {abs(z):.1f}x the normal spread {'above' if z > 0 else 'below'} "
                            f"its usual {exp:.1f} at {hour:02d}:00 (reading {values[s]:.1f})."})
    out.sort(key=lambda r: -abs(r["z"]))
    return [r for r in out if abs(r["z"]) >= 2][:3]


def _any(seg, suffix):
    return any(f"{s}_{suffix}" in seg and seg[f"{s}_{suffix}"].sum() > 0 for s in SIX)


def find_incidents(df, b, gap=3):
    d = _z(df, b)
    flag = np.where(d["ml_anomaly_flag"].values == 1)[0]
    if len(flag) == 0:
        return []
    step = max(float(d["timestamp"].diff().dt.total_seconds().median() / 60), 1)
    out = []
    for i, g in enumerate(np.split(flag, np.where(np.diff(flag) > gap)[0] + 1)):
        seg, n = d.iloc[g[0]:g[-1] + 1], g[-1] - g[0] + 1
        peak = {s: float(max(seg["z_" + s].abs().max(), seg["dz_" + s].abs().max())) for s in SENS}
        ps = max(peak, key=peak.get)
        pz, nsens = peak[ps], sum(v > 3 for v in peak.values())
        rows = d.iloc[g]
        agree = float(((rows.iso_forest_anomaly == 1) & (rows.lof_anomaly == 1)).mean())
        flat = [s for s in SENS if g[0] >= 8 and seg[s + "_is_flat"].rolling(6).sum().max() >= 6]
        name = LABEL[ps].lower()
        if _any(seg, "was_missing"):
            cause, act = "Data gap", "Check station power and the data link; backfill from the on-site logger if you can."
        elif _any(seg, "out_of_range"):
            cause, act = "Reading outside physical limits", f"Inspect the {name} sensor and its calibration. Exclude these readings from reports."
        elif flat:
            cause, act = f"Stuck sensor ({LABEL[flat[0]].lower()} not changing)", f"Clean or replace the {LABEL[flat[0]].lower()} sensor and check its wiring."
        elif n <= 2 and peak[ps] > 6:
            cause, act = "Isolated spike", "Likely electrical noise or a transmission glitch. Exclude the reading; if it repeats, check cabling and shielding."
        elif nsens >= 3:
            cause, act = "Several sensors moved together", "Compare with nearby stations or the forecast. If neighbours agree it is a real weather event; if not, check the logger and power supply."
        elif n >= 6:
            cause, act = f"Sustained {name} deviation", f"Possible drift. Compare with a reference sensor and schedule a calibration check if it persists."
        else:
            cause, act = f"Short {name} deviation", "Review the readings. No action unless it recurs."
        score = 0.5 * min(pz / 10, 1) + 0.3 * min(n / 12, 1) + 0.2 * agree
        sev = "Critical" if score >= .65 else "High" if score >= .45 else "Medium" if score >= .3 else "Low"
        r = seg.loc[seg["z_" + ps].abs().idxmax()]
        out.append({"id": i + 1, "start": seg["timestamp"].iloc[0].isoformat(), "end": seg["timestamp"].iloc[-1].isoformat(),
                    "readings": int(n), "minutes": int(n * step), "sensor": LABEL[ps], "peak_z": round(pz, 1),
                    "value": round(float(r[ps]), 1), "expected": round(float(b[ps]["med"].get(int(r["hour"]), np.nan)), 1),
                    "models_agree": round(agree * 100), "severity": sev, "score": round(score, 2),
                    "cause": cause, "action": act, "hour": int(seg["hour"].iloc[0]), "sensor_key": ps})
    return sorted(out, key=lambda x: -x["score"])


def recommendations(df, b, incs):
    n, acts, sensors = len(df), [], []
    for s in SENS:
        mine = [x for x in incs if x["sensor_key"] == s]
        share = sum(x["readings"] for x in mine) / max(n, 1)
        bad = (df[f"{s}_was_missing"].mean() + df[f"{s}_out_of_range"].mean()) if f"{s}_was_missing" in df else 0
        flats = sum("Stuck" in x["cause"] and s in x["cause"].lower().replace(" ", "_") for x in mine)
        health = int(max(0, min(100, 100 - share * 400 - bad * 500 - flats * 10)))
        sensors.append({"sensor": LABEL[s], "health": health, "incidents": len(mine), "share_pct": round(share * 100, 1)})
        if health < 70:
            acts.append({"priority": "P1", "title": f"Inspect the {LABEL[s].lower()} sensor",
                         "detail": f"{len(mine)} incidents and {share*100:.1f}% of readings trace back to it. Check calibration, wiring and housing."})
        elif health < 90:
            acts.append({"priority": "P2", "title": f"Watch the {LABEL[s].lower()} sensor",
                         "detail": f"Health {health}/100. Compare against a reference sensor at the next visit."})
    crit = [x for x in incs if x["severity"] == "Critical"]
    if crit:
        acts.insert(0, {"priority": "P1", "title": f"Review {len(crit)} critical incident{'s' if len(crit) > 1 else ''}",
                        "detail": f"Largest: {crit[0]['sensor'].lower()} {crit[0]['peak_z']}x off normal. {crit[0]['action']}"})
    both, iso, lof = int(((df.iso_forest_anomaly == 1) & (df.lof_anomaly == 1)).sum()), int(df.iso_forest_anomaly.sum()), int(df.lof_anomaly.sum())
    single = (iso + lof - 2 * both)
    tot = max(iso + lof - both, 1)
    acts.append({"priority": "P3", "title": "Alert only when both models agree",
                 "detail": f"{single/tot*100:.0f}% of flagged readings come from a single model. Page on agreement; send the rest to a daily review list."})
    hours = pd.Series([x["hour"] for x in incs])
    peak_hour = int(hours.mode()[0]) if len(hours) else None
    if peak_hour is not None and (hours == peak_hour).mean() > 0.12:
        acts.append({"priority": "P3", "title": f"Incidents cluster around {peak_hour:02d}:00",
                     "detail": "Look for a recurring cause at that time, such as sun on the sensor housing, a generator or irrigation."})
    acts.sort(key=lambda a: a["priority"])
    return {"sensors": sensors, "actions": acts, "agreement": {"both": both, "iso_only": iso - both, "lof_only": lof - both}}
