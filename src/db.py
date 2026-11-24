"""SQLite storage for runs and analyst labels.

Why SQLite: the app is single-user and local, labels must survive a restart, and
``sqlite3`` ships with Python, so there is nothing to install or host.

Design: a *run* is one uploaded file (identified by the SHA-256 of its bytes, so
re-uploading the same file finds the same run and its labels). An *alert* row is
created when the analyst labels a record, and holds everything needed to review it
later without the original file. All SQL uses ``?`` placeholders, never string
formatting, so file contents cannot inject SQL.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "threatlens.db"
VALID_LABELS = ("confirmed", "false_positive")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    file_sha256 TEXT NOT NULL UNIQUE,
    n_records   INTEGER NOT NULL,
    threshold   REAL NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    record_index  INTEGER NOT NULL,
    score         REAL NOT NULL,
    flagged       INTEGER NOT NULL,
    top_features  TEXT NOT NULL,
    record_json   TEXT NOT NULL,
    analyst_label TEXT NOT NULL CHECK (analyst_label IN ('confirmed', 'false_positive')),
    labeled_at    TEXT NOT NULL,
    UNIQUE (run_id, record_index)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the database, creating the file and tables on first use."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")   # SQLite leaves foreign keys off by default
    conn.executescript(SCHEMA)
    return conn


def get_or_create_run(conn: sqlite3.Connection, filename: str, file_sha256: str,
                      n_records: int, threshold: float) -> int:
    row = conn.execute("SELECT run_id FROM runs WHERE file_sha256 = ?", (file_sha256,)).fetchone()
    if row:
        return int(row["run_id"])
    cur = conn.execute(
        "INSERT INTO runs (filename, file_sha256, n_records, threshold, created_at) VALUES (?, ?, ?, ?, ?)",
        (filename, file_sha256, n_records, threshold, _now()))
    conn.commit()
    return int(cur.lastrowid)


def update_threshold(conn: sqlite3.Connection, run_id: int, threshold: float) -> None:
    conn.execute("UPDATE runs SET threshold = ? WHERE run_id = ?", (threshold, run_id))
    conn.commit()


def save_label(conn: sqlite3.Connection, run_id: int, record_index: int, score: float,
               flagged: bool, top_features: str, record: dict, label: str) -> None:
    """Insert or overwrite the analyst's label for one record."""
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
    conn.execute(
        """INSERT INTO alerts (run_id, record_index, score, flagged, top_features, record_json,
                               analyst_label, labeled_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (run_id, record_index) DO UPDATE SET
               score = excluded.score, flagged = excluded.flagged, top_features = excluded.top_features,
               analyst_label = excluded.analyst_label, labeled_at = excluded.labeled_at""",
        (run_id, int(record_index), float(score), int(flagged), top_features,
         json.dumps(record, default=str), label, _now()))
    conn.commit()


def clear_label(conn: sqlite3.Connection, run_id: int, record_index: int) -> None:
    conn.execute("DELETE FROM alerts WHERE run_id = ? AND record_index = ?", (run_id, int(record_index)))
    conn.commit()


def get_labels(conn: sqlite3.Connection, run_id: int) -> dict[int, str]:
    rows = conn.execute("SELECT record_index, analyst_label FROM alerts WHERE run_id = ?", (run_id,))
    return {int(r["record_index"]): r["analyst_label"] for r in rows}


def get_labeled_alerts(conn: sqlite3.Connection, run_id: int | None = None) -> pd.DataFrame:
    """Alert history (one run, or all runs) as a DataFrame, ready for display or CSV export."""
    query = """SELECT a.run_id, r.filename, a.record_index, a.score, a.flagged, a.analyst_label,
                      a.top_features, a.labeled_at, a.record_json
               FROM alerts a JOIN runs r USING (run_id)"""
    params: tuple = ()
    if run_id is not None:
        query += " WHERE a.run_id = ?"
        params = (run_id,)
    df = pd.read_sql_query(query + " ORDER BY a.labeled_at DESC, a.id DESC", conn, params=params)
    if df.empty:
        return df.drop(columns=["record_json"])
    records = pd.json_normalize(df.pop("record_json").map(json.loads))
    return pd.concat([df, records.add_prefix("rec_")], axis=1)
