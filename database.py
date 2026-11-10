import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, DateTime, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///aws_monitor.db")

engine_options = {}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, **engine_options)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True)
    timestamp = Column(String(64), nullable=False, index=True)
    station_id = Column(String(100), nullable=False, default="station-001", index=True)
    source = Column(String(40), nullable=False, default="http")

    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False, default=0)
    wind_speed = Column(Float, nullable=False)
    wind_direction = Column(Float, nullable=False, default=0)

    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_score = Column(Float)
    status = Column(String(80), nullable=False)
    iso_forest_anomaly = Column(Boolean, default=False)
    iso_forest_score = Column(Float)
    lof_anomaly = Column(Boolean, default=False)
    lof_score = Column(Float)
    hard_quality_issue = Column(Boolean, default=False)


Index("ix_readings_station_timestamp", SensorReading.station_id, SensorReading.timestamp)


def init_db():
    Base.metadata.create_all(bind=engine)


def save_reading(data):
    allowed = {
        "timestamp", "station_id", "source", "temperature", "humidity",
        "pressure", "rainfall", "wind_speed", "wind_direction",
        "is_anomaly", "anomaly_score", "status", "iso_forest_anomaly",
        "iso_forest_score", "lof_anomaly", "lof_score", "hard_quality_issue"
    }
    clean = {key: data[key] for key in allowed if key in data}
    session = SessionLocal()
    try:
        record = SensorReading(**clean)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _serialize(row):
    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "station_id": row.station_id,
        "source": row.source,
        "temperature": row.temperature,
        "humidity": row.humidity,
        "pressure": row.pressure,
        "rainfall": row.rainfall,
        "wind_speed": row.wind_speed,
        "wind_direction": row.wind_direction,
        "is_anomaly": bool(row.is_anomaly),
        "anomaly_score": row.anomaly_score,
        "status": row.status,
        "iso_forest_anomaly": bool(row.iso_forest_anomaly),
        "lof_anomaly": bool(row.lof_anomaly),
        "hard_quality_issue": bool(row.hard_quality_issue),
    }


def get_latest_readings(limit=100):
    session = SessionLocal()
    try:
        rows = (
            session.query(SensorReading)
            .order_by(SensorReading.id.desc())
            .limit(limit)
            .all()
        )
        return [_serialize(row) for row in reversed(rows)]
    finally:
        session.close()


def get_summary():
    session = SessionLocal()
    try:
        total = session.query(SensorReading).count()
        anomalies = session.query(SensorReading).filter(
            SensorReading.is_anomaly.is_(True)
        ).count()
        suspicious = session.query(SensorReading).filter(
            SensorReading.status == "Suspicious"
        ).count()
        return {
            "total_readings": total,
            "anomalies": anomalies,
            "suspicious": suspicious,
            "normal": max(0, total - anomalies - suspicious),
            "anomaly_percentage": round((anomalies / total) * 100, 2) if total else 0,
        }
    finally:
        session.close()
