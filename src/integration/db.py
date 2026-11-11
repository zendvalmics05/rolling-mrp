"""Database connection and initialization utilities for ROLLINGMRP."""

from pathlib import Path
from typing import Optional
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "rollingmrp.db"
SCHEMA_SQL_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return an open SQLite connection with foreign keys enabled."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_engine(db_url: Optional[str] = None) -> Engine:
    """Return a SQLAlchemy Engine instance."""
    if db_url is None:
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
    else:
        url = db_url
    return create_engine(url, echo=False)


def init_db(db_path: Optional[Path] = None, schema_path: Optional[Path] = None) -> None:
    """Initialize database tables using schema.sql."""
    conn = get_connection(db_path)
    schema_file = schema_path or SCHEMA_SQL_PATH
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with conn:
        conn.executescript(schema_sql)
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DEFAULT_DB_PATH)

