"""Database migration helper for AyurYukti tables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.database import DatabaseManager


def main() -> None:
    db = DatabaseManager()
    db.create_tables()
    print(f"database_engine={db.engine.url}")
    print("tables=patients,encounters,outcomes,alerts")


if __name__ == "__main__":
    main()
