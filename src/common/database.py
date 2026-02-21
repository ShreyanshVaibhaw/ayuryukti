"""Database setup with PostgreSQL + SQLite fallback."""

from __future__ import annotations

from typing import Dict, Optional

from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import func

from config import POSTGRES_URI


def get_postgres_engine() -> Optional[Engine]:
    """Create Postgres engine if available."""
    try:
        engine = create_engine(POSTGRES_URI, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(func.now().select())
        return engine
    except Exception:
        return None


class DatabaseManager:
    """Manage persistence tables with automatic SQLite fallback."""

    def __init__(self, db_uri: str | None = None):
        self.db_uri = db_uri or POSTGRES_URI
        self.engine = self._connect(self.db_uri)
        self.metadata = MetaData()
        self._define_tables()
        self.create_tables()

    def _connect(self, db_uri: str) -> Engine:
        try:
            engine = create_engine(db_uri, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(func.now().select())
            return engine
        except Exception:
            return create_engine("sqlite:///ayuryukti_local.db", pool_pre_ping=True)

    def _define_tables(self):
        self.patients = Table(
            "patients",
            self.metadata,
            Column("patient_id", String, primary_key=True),
            Column("abha_id", String),
            Column("age", Integer),
            Column("sex", String),
            Column("prakriti_type", String),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.encounters = Table(
            "encounters",
            self.metadata,
            Column("encounter_id", String, primary_key=True),
            Column("patient_id", String),
            Column("doctor_id", String),
            Column("centre_id", String),
            Column("diagnosis_code", String),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.outcomes = Table(
            "outcomes",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("encounter_id", String),
            Column("patient_prakriti", String),
            Column("condition_code", String),
            Column("formulation_name", String),
            Column("outcome", String),
            Column("follow_up_days", Integer),
            Column("timestamp", String),
        )
        self.alerts = Table(
            "alerts",
            self.metadata,
            Column("alert_id", String, primary_key=True),
            Column("district", String),
            Column("state", String),
            Column("condition", String),
            Column("severity", String),
            Column("payload", JSON),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )

    def create_tables(self):
        """Create all tables."""
        self.metadata.create_all(self.engine)

    def insert_outcome(self, row: Dict):
        """Insert outcome row."""
        try:
            with self.engine.begin() as conn:
                conn.execute(self.outcomes.insert().values(**row))
        except SQLAlchemyError:
            # Keep tracker resilient; skip DB write on runtime failures.
            return

