"""SQLite persistence for user accounts.

Uses the standard-library ``sqlite3`` module, so no extra dependency is
required. The database file is created next to this module on first run.
"""

import sqlite3
from pathlib import Path

from flask import g

DB_PATH = Path(__file__).resolve().parent / "users.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def get_db():
    """Return the per-request SQLite connection stored on the app context."""
    if "db" not in g:
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
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(SCHEMA)
    connection.commit()
    connection.close()


def get_user_by_id(user_id):
    return get_db().execute(
        "SELECT id, email, password_hash, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()


def get_user_by_email(email):
    return get_db().execute(
        "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
        (email,),
    ).fetchone()


def create_user(email, password_hash):
    db = get_db()
    db.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (email, password_hash),
    )
    db.commit()


def set_user_password(email, password_hash):
    db = get_db()
    db.execute(
        "UPDATE users SET password_hash = ? WHERE email = ?",
        (password_hash, email),
    )
    db.commit()
