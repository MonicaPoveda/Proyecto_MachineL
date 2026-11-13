"""Persistence layer for user accounts.

Uses PostgreSQL when a ``DATABASE_URL`` environment variable is set (Render),
otherwise falls back to a local SQLite file for development. Both backends
expose the same API, so the rest of the app does not change.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from flask import g

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG = True
except ImportError:  # pragma: no cover - psycopg2 only required in production
    psycopg2 = None
    RealDictCursor = None
    HAS_PSYCOPG = False


DB_PATH = Path(__file__).resolve().parent / "users.db"

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def using_postgres() -> bool:
    """Return whether the app should use PostgreSQL (DATABASE_URL is set)."""
    return bool(_database_url())


def _require_psycopg() -> None:
    if not HAS_PSYCOPG:
        raise RuntimeError(
            "DATABASE_URL is set but psycopg2 is not installed. "
            "Add psycopg2-binary to requirements.txt."
        )


def _sql(query: str) -> str:
    """Adapt ``%s`` placeholders to ``?`` when running on SQLite."""
    return query.replace("%s", "?") if not using_postgres() else query


def get_db():
    """Return the per-request connection stored on the app context."""
    if "db" not in g:
        if using_postgres():
            _require_psycopg()
            g.db = psycopg2.connect(_database_url())
        else:
            g.db = sqlite3.connect(DB_PATH)
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(exception=None):
    """Close the request-scoped connection (registered as teardown callback)."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create the users table if it does not exist yet (idempotent)."""
    if using_postgres():
        _require_psycopg()
        connection = psycopg2.connect(_database_url())
        try:
            with connection.cursor() as cursor:
                cursor.execute(POSTGRES_SCHEMA)
            connection.commit()
        finally:
            connection.close()
    else:
        connection = sqlite3.connect(DB_PATH)
        connection.executescript(SQLITE_SCHEMA)
        connection.commit()
        connection.close()


def _fetchone(query: str, params: tuple):
    db = get_db()
    if using_postgres():
        with db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
    return db.execute(_sql(query), params).fetchone()


def _execute(query: str, params: tuple) -> None:
    db = get_db()
    if using_postgres():
        with db.cursor() as cursor:
            cursor.execute(query, params)
    else:
        db.execute(_sql(query), params)
    db.commit()


def get_user_by_id(user_id):
    return _fetchone(
        "SELECT id, email, password_hash, created_at FROM users WHERE id = %s",
        (user_id,),
    )


def get_user_by_email(email):
    return _fetchone(
        "SELECT id, email, password_hash, created_at FROM users WHERE lower(email) = lower(%s)",
        (email,),
    )


def create_user(email, password_hash):
    _execute(
        "INSERT INTO users (email, password_hash) VALUES (lower(%s), %s)",
        (email, password_hash),
    )


def set_user_password(email, password_hash):
    _execute(
        "UPDATE users SET password_hash = %s WHERE lower(email) = lower(%s)",
        (password_hash, email),
    )
