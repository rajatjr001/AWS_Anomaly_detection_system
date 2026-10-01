# 🚨 AWS Anomaly Detection & Data Quality Monitoring System

<p align="center">

**An end-to-end machine learning system for anomaly detection, data quality analysis, and real-time monitoring.**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20API-black?logo=flask)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 📌 Overview

The **AWS Anomaly Detection & Data Quality Monitoring System** is a machine-learning-based dashboard designed to identify unusual patterns in data and monitor data quality in real time.

The system combines two anomaly detection techniques:

* 🌲 **Isolation Forest**
* 🔍 **Local Outlier Factor (LOF)**

Along with anomaly detection, the application performs **data quality assessment** and provides a **real-time monitoring interface** through a Flask web application.

The project demonstrates how machine learning can be integrated into a practical monitoring pipeline rather than being used only as an offline model.

---

## 🎯 Key Features

### 🔎 Anomaly Detection

Detects unusual observations using multiple unsupervised ML algorithms.

* Isolation Forest
* Local Outlier Factor (LOF)
* Anomaly scores
* Normal vs anomalous data identification

### 📊 Data Quality Assessment

Analyzes incoming data for common quality problems such as:

* Missing values
* Duplicate records
* Invalid values
* Data consistency issues
* Basic statistical characteristics

### ⚡ Real-Time Monitoring

The system can process incoming observations and continuously monitor them for abnormal behavior.

### 🖥️ Interactive Dashboard

The Flask-based web interface provides:

* Dataset statistics
* Anomaly counts
* Model results
* Data-quality metrics
* Monitoring information
* Visual representation of results
* Manual data input

### 🔌 REST API

The backend exposes API endpoints that allow the frontend to communicate with the ML pipeline.

---

# 🧠 Machine Learning Pipeline

The complete workflow can be represented as:

```text
                    ┌─────────────────────┐
                    │    Data Generation  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               ↓
                 ┌─────────────┴─────────────┐
                 ↓                           ↓
       ┌──────────────────┐        ┌──────────────────┐
       │ Isolation Forest │        │       LOF        │
       └────────┬─────────┘        └────────┬─────────┘
                ↓                           ↓
                 └─────────────┬─────────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Anomaly Assessment  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Data Quality Check  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Real-Time Monitor   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Flask Dashboard/API │
                    └─────────────────────┘
```

---

# 🤖 Algorithms Used

## 1. Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm.

The basic idea is that **anomalous observations are easier to isolate than normal observations**.

It creates random decision trees and identifies observations that require fewer splits to become isolated.

### Advantages

* Works well with large datasets
* Does not require labeled anomalies
* Suitable for high-dimensional data
* Efficient for anomaly detection

---

## 2. Local Outlier Factor (LOF)

**Local Outlier Factor** detects anomalies by comparing the local density of an observation with the density of its neighboring observations.

An observation can be considered anomalous when it has significantly lower local density than its neighbors.

### Advantages

* Detects local anomalies
* Useful when data has different density regions
* Works well for identifying observations that differ from their surrounding neighborhood

---

# 📊 Data Quality Monitoring

Anomaly detection alone does not guarantee that the underlying data is reliable.

Therefore, the system also evaluates data quality.

Typical checks include:

| Check              | Purpose                               |
| ------------------ | ------------------------------------- |
| Missing Values     | Detect incomplete records             |
| Duplicate Records  | Identify repeated observations        |
| Data Types         | Verify expected data formats          |
| Invalid Values     | Detect values outside expected ranges |
| Statistical Checks | Understand unusual distributions      |
| Completeness       | Measure how much data is available    |

This provides a broader view of the dataset:

```text
Data Quality
      +
Anomaly Detection
      +
Real-Time Monitoring
      ↓
Reliable Data Monitoring
```

---

# 🖥️ Dashboard

The application provides a web-based dashboard for interacting with the anomaly detection pipeline.

### Dashboard capabilities

* View dataset information
* View anomaly statistics
* Compare anomaly detection results
* Monitor data quality
* Submit manual values
* Run anomaly detection on new observations
* Monitor incoming data

> 📸 Add your latest dashboard screenshot here if desired:

```markdown
![Dashboard](dashboard.png)
```

---

# 🛠️ Tech Stack

## Programming Language

* **Python**

## Machine Learning

* **Scikit-learn**
* Isolation Forest
* Local Outlier Factor

## Data Processing

* **Pandas**
* **NumPy**

## Backend

* **Flask**
* REST API

## Frontend

* HTML
* CSS
* JavaScript

## Visualization

* Chart-based dashboard components

## Development Environment

* Python virtual environment
* Git
* GitHub

---

# 📁 Project Structure

```text
AWS_Anomaly_detection_system/
│
├── app.py
│
├── pipeline.py
│
├── requirements.txt
│
├── README.md
│
├── data/
│   └── *.csv
│
├── static/
│   └── index.html
│
└── output/
    └── generated results
```

### Important Files

### `app.py`

Main Flask application.

Responsible for:

* Starting the web server
* Providing API endpoints
* Connecting the frontend with the ML pipeline
* Handling incoming requests
* Returning JSON responses

### `pipeline.py`

Contains the main machine-learning and data-processing pipeline.

The pipeline handles:

```text
Data Generation
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Isolation Forest
      ↓
LOF
      ↓
Data Quality
      ↓
Real-Time Monitoring
```

### `static/index.html`

Frontend dashboard containing:

* User interface
* Data input
* Dashboard components
* API communication
* Visualization

### `requirements.txt`

Contains the Python dependencies required to run the project.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/rajatjr001/AWS_Anomaly_detection_system.git
```

Move into the project directory:

```bash
cd AWS_Anomaly_detection_system
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

The application will run locally.

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

# 🔌 API Architecture

The frontend communicates with the Flask backend through REST API endpoints.

Conceptually:

```text
Browser
   │
   │ HTTP Request
   ↓
Flask API
   │
   ↓
ML Pipeline
   │
   ├── Isolation Forest
   │
   ├── LOF
   │
   ├── Data Quality
   │
   └── Real-Time Monitoring
   │
   ↓
JSON Response
   │
   ↓
Dashboard
```

The `/api/manual` functionality allows manually entered observations to be passed through the anomaly detection pipeline.

---

# 📥 Manual Data Detection

The dashboard supports manual data entry.

The workflow is:

```text
User enters values
       ↓
Frontend
       ↓
/api/manual
       ↓
Preprocessing
       ↓
Isolation Forest + LOF
       ↓
Anomaly Result
       ↓
Dashboard
```

This allows users to test individual observations without needing to modify the underlying dataset.

---

# 📈 Example Output

A processed observation can contain information such as:

```text
Observation
     ↓
Isolation Forest
     ↓
Normal
```
