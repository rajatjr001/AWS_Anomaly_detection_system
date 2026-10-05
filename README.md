# 🚨 AWS Anomaly Detection & Data Quality Monitoring System

An intelligent **machine learning-based anomaly detection and data quality monitoring system** built with Python and Flask. The application detects unusual data patterns using **Isolation Forest** and **Local Outlier Factor (LOF)** while continuously monitoring data quality through a web-based dashboard.

The project also provides **manual data analysis, real-time monitoring, anomaly statistics, and an interactive visualization dashboard**, making it useful for identifying abnormal or low-quality data in real-world data pipelines.

---

## 🌟 Key Features

### 🔍 Anomaly Detection

* Detects abnormal observations using:

  * **Isolation Forest**
  * **Local Outlier Factor (LOF)**
* Supports unsupervised anomaly detection without requiring labeled data.
* Provides anomaly labels and scores for individual records.
* Allows comparison between different anomaly detection techniques.

### 📊 Data Quality Monitoring

The system evaluates incoming data based on important quality metrics such as:

* Missing values
* Duplicate records
* Invalid values
* Data distribution
* Outliers
* Overall data quality score

### 📈 Interactive Dashboard

The web dashboard provides:

* Dataset overview
* Total records
* Number of anomalies
* Anomaly percentage
* Data quality metrics
* Visual representation of anomalies
* Real-time monitoring information
* Manual data input and analysis

### ⚡ Real-Time Monitoring

The system can simulate and monitor streaming data to identify anomalies as new observations arrive.

The monitoring component keeps track of:

* Incoming records
* Detected anomalies
* Anomaly rate
* Data quality
* Monitoring history

### ✍️ Manual Analysis

Users can manually enter feature values and send them to the backend for anomaly analysis.

The system processes the input using the trained anomaly detection models and returns the corresponding result.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      User / Browser  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Flask Web Server   │
                    │       app.py         │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
       │ Preprocess  │  │   Feature   │  │ Data Quality │
       │    Data     │  │ Engineering │  │   Analysis   │
       └──────┬──────┘  └──────┬──────┘  └──────────────┘
              │                │
              └────────┬───────┘
                       ▼
              ┌────────────────────┐
              │ Anomaly Detection  │
              ├────────────────────┤
              │ Isolation Forest   │
              │ LOF                │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Real-Time Monitor  │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Dashboard / JSON   │
              │ API Responses      │
              └────────────────────┘
```

---

# 🧠 Machine Learning Approach

## 1. Isolation Forest

Isolation Forest is an **unsupervised anomaly detection algorithm** that identifies unusual observations by isolating them from the rest of the data.

Anomalies generally require fewer random splits to become isolated.

### Advantages

* Works without labeled data
* Efficient for large datasets
* Suitable for high-dimensional data
* Effective for detecting unusual observations

---

## 2. Local Outlier Factor (LOF)

LOF identifies anomalies by comparing the **local density of a data point with the density of its neighboring points**.

A point that has significantly lower local density than its neighbors can be considered an anomaly.

### Advantages

* Detects local anomalies
* Useful when different regions of the dataset have different densities
* Works well for identifying unusual local patterns

---

# 🔄 Data Processing Pipeline

The project follows the following workflow:

```text
Data Generation / Input
          ↓
Data Preprocessing
          ↓
Feature Engineering
          ↓
Data Quality Assessment
          ↓
Isolation Forest
          ↓
LOF
          ↓
Anomaly Detection
          ↓
Real-Time Monitoring
          ↓
Dashboard Visualization
```

---

# 🛠️ Technology Stack

| Technology                               | Purpose                                    |
| ---------------------------------------- | ------------------------------------------ |
| **Python**                               | Core programming language                  |
| **Flask**                                | Backend web framework and REST API         |
| **Scikit-learn**                         | Machine learning and anomaly detection     |
| **Pandas**                               | Data processing and manipulation           |
| **NumPy**                                | Numerical computations                     |
| **Matplotlib / Visualization Libraries** | Data visualization                         |
| **HTML**                                 | Dashboard structure                        |
| **CSS**                                  | Dashboard styling                          |
| **JavaScript**                           | Frontend interaction and API communication |
| **Gunicorn**                             | Production WSGI server                     |
| **AWS Elastic Beanstalk**                | Cloud deployment                           |
| **Git & GitHub**                         | Version control and source-code management |

---

# 📁 Project Structure

```text
AWS_Anomaly_detection_system/
│
├── app.py
│   └── Flask application and API endpoints
│
├── pipeline.py
│   └── Data processing and machine learning pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── Procfile
│   └── Production server configuration
│
├── .ebextensions/
│   └── Elastic Beanstalk configuration
│
├── static/
│   ├── index.html
│   └── index_manual.html
│
├── README.md
│
└── .gitignore
```

---

# 🚀 Getting Started

## Prerequisites

Make sure you have the following installed:

* Python 3.x
* pip
* Git
* Web browser

---

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
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

```bash
python app.py
```

The application should start locally.

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

# ☁️ AWS Deployment

The application can be deployed using **AWS Elastic Beanstalk**.

### Install Elastic Beanstalk CLI

```bash
pip install awsebcli
```

Initialize the application:

```bash
eb init
```

Create an environment:

```bash
eb create aws-anomaly-production
```

Deploy the application:

```bash
eb deploy
```

Open the deployed application:

```bash
eb open
```

Check environment status:

```bash
eb status
```

---

# ⚙️ Production Server

For production deployment, the application uses **Gunicorn**.

The `Procfile` contains:

```text
web: gunicorn --bind :8000 app:app
```

Here:

* `app` → Python file `app.py`
* `app` → Flask application object
* `8000` → Port used by the production server

---

# 🔌 API Functionality

The Flask backend provides API endpoints for interacting with the anomaly detection pipeline.

Typical functionality includes:

| Functionality      | Description                            |
| ------------------ | -------------------------------------- |
| Dataset processing | Processes input data                   |
| Anomaly detection  | Runs Isolation Forest and LOF          |
| Data quality       | Calculates quality metrics             |
| Manual analysis    | Processes manually entered values      |
| Monitoring         | Tracks incoming data                   |
| Metrics            | Returns anomaly and quality statistics |

The APIs communicate with the frontend using **JSON**.

---

# 📊 Dashboard

The dashboard is designed to provide a quick overview of the system.

### Dashboard capabilities

* 📌 Dataset statistics
* 📌 Anomaly count
* 📌 Anomaly percentage
* 📌 Data quality score
* 📌 Model results
* 📌 Real-time monitoring
* 📌 Manual anomaly detection
* 📌 Visual anomaly representation

---

# 🔬 Example Workflow

Suppose the system receives a dataset containing:

```text
Feature 1    Feature 2    Feature 3
-----------------------------------
10           20           15
11           21           14
12           19           16
500          800          900
13           20           15
```

The first three and last records follow a similar pattern.

The record:

```text
500, 800, 900
```

is significantly different from the rest.

The anomaly detection models can identify this observation as an **outlier**.

---

# 📈 Model Output

Scikit-learn anomaly detection models generally produce labels such as:

```text
 1  → Normal
-1  → Anomaly
```

The dashboard converts these results into user-friendly anomaly statistics and visualizations.

---

# 🎯 Objectives

The main objectives of this project are:

1. Detect anomalies without requiring labeled datasets.
2. Compare different unsupervised anomaly detection techniques.
3. Monitor data quality automatically.
4. Provide real-time anomaly monitoring.
5. Create an easy-to-use visualization dashboard.
6. Provide APIs for integrating anomaly detection into applications.
7. Deploy the ML application on AWS.

---

# 💡 Real-World Applications

This system can be adapted for:

* 🏦 Financial fraud detection
* 🛒 E-commerce transaction monitoring
* 🏭 Industrial sensor monitoring
* ☁️ Cloud infrastructure monitoring
* 📡 IoT data monitoring
* 📊 Data pipeline quality monitoring
* 🔐 Cybersecurity anomaly detection
* 💳 Transaction anomaly detection
* 📈 Business data monitoring

---

# 🔮 Future Improvements

Possible improvements include:

* [ ] Add additional anomaly detection algorithms
* [ ] Add XGBoost-based classification for labeled anomaly datasets
* [ ] Implement automatic model selection
* [ ] Add persistent database storage
* [ ] Add user authentication
* [ ] Add email/SMS anomaly alerts
* [ ] Integrate AWS S3 for dataset storage
* [ ] Integrate AWS CloudWatch for monitoring
* [ ] Add Docker containerization
* [ ] Add CI/CD using GitHub Actions
* [ ] Add model performance tracking
* [ ] Add automated data drift detection
* [ ] Improve real-time streaming using AWS services

---

# 📚 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Machine Learning
* Unsupervised Learning
* Anomaly Detection
* Isolation Forest
* Local Outlier Factor
* Feature Engineering
* Data Preprocessing
* Data Quality Analysis
* REST APIs
* Flask
* Frontend–Backend Integration
* Real-Time Monitoring
* Cloud Deployment
* AWS Elastic Beanstalk
* Git & GitHub
* Production ML Application Deployment

---

# 👨‍💻 Author

**Rajat Kumar**

### Areas of Interest

* Machine Learning
* Artificial Intelligence
* Python
* Data Science
* Cloud Computing
* Software Development

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Repository:**
https://github.com/rajatjr001/AWS_Anomaly_detection_system

---

## 📜 License

This project is intended for educational and demonstration purposes.
