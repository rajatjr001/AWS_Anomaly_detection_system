# AWS Anomaly Detection & Data Quality Framework 

*"A Machine Learning Framework for
Real-Time Anomaly Detection and Data Quality Assessment in Automatic Weather
Stations."*

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib
python aws_anomaly_detection.py
```

Outputs:
- `aws_quality_assessed_output.csv` — full dataset with per-observation quality labels
- `aws_anomaly_dashboard.png` — visualization (temperature/humidity time series with flagged points, anomaly score plot)
- Console output — EDA summary, train/val split info, simulated real-time scoring, precision/recall/F1 evaluation

## How the code maps to the synopsis's methodology (Section 6)

| Synopsis Step | Function |
|---|---|
| 1. Data Collection | `generate_synthetic_aws_data()` (swap for `load_aws_data("your_file.csv")` once you have a real dataset) |
| 2. Data Preprocessing | `preprocess_data()` — missing-value flags, physical-limit checks, interpolation |
| 3. Exploratory Data Analysis | `run_eda()` |
| 4. Feature Engineering | `engineer_features()` — lags, deltas, rolling stats, cross-variable features |
| 5. Anomaly Detection | `detect_anomalies()` — Isolation Forest + Local Outlier Factor |
| 6. Data Quality Assessment | `assess_data_quality()` — Normal/Good, Suspicious, Anomalous/Requires Review |
| 7. Real-Time Processing | `RealTimeMonitor` class + `build_realtime_monitor()` |
| 8. Visualization & Evaluation | `visualize_results()`, `evaluate_model()` (precision/recall/F1 vs. injected ground truth) |
| 9. Model Validation | Time-based train/validation split inside `detect_anomalies()` |

## Using a real dataset instead of synthetic data

Replace the call in `run_pipeline()`:

```python
df = load_aws_data("path/to/your_aws_data.csv")
```

Your CSV needs a `timestamp` column plus whichever of `temperature`, `humidity`,
`pressure`, `rainfall`, `wind_speed`, `wind_direction` you have. If your dataset
has labelled anomalies, keep that column named `is_anomaly` (0/1) so
`evaluate_model()` can compute precision/recall/F1; otherwise the framework
still runs, it just skips the evaluation step (real deployments won't have
ground-truth labels anyway — the synthetic-label evaluation here is only to
demonstrate/validate the approach for the project report).

## Extending toward the "Expected Outcomes" in the synopsis

- **Live AWS/IoT integration**: replace `generate_synthetic_aws_data()` with a
  reader for your station's API/serial feed, and call
  `RealTimeMonitor.score_observation()` on each new reading as it arrives.
- **Multiple stations**: key the pipeline by `station_id` and fit one model per
  station (weather baselines differ by location).
- **Dashboard**: the matplotlib PNG is a starting point; for something
  interactive, feed `aws_quality_assessed_output.csv` into a small
  Streamlit/Plotly Dash app or a Grafana panel.
- **Alerting**: hook `RealTimeMonitor.score_observation()`'s `is_anomaly`
  output into email/SMS/webhook alerts for the "Suspicious"/"Anomalous" cases.
