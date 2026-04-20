"""SQLite database setup and schema management for BioTrace.

This module is responsible for:
- Opening / creating the SQLite database file.
- Creating all tables if they do not yet exist.
- Providing a connection factory for use by repository classes.

Usage::

    from app.storage.database import DatabaseManager
    db = DatabaseManager()          # call once at startup
    conn = db.get_connection()      # use in repositories
"""

import sqlite3
from pathlib import Path

from app.utils.config import DB_PATH
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,               -- custom user-defined session name
    started_at      DATETIME NOT NULL,
    ended_at        DATETIME,
    notes           TEXT,
    nasa_tlx_score  REAL,
    error_count     INTEGER,            -- surgical error count; NULL = not yet tracked
    video_path      TEXT                -- filesystem path to the MP4 recording
);

CREATE TABLE IF NOT EXISTS calibrations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    recorded_at         DATETIME NOT NULL,
    duration_seconds    INTEGER NOT NULL,       -- actual recording length
    baseline_rmssd      REAL,                   -- ms — resting HRV baseline
    baseline_rmssd_std  REAL,                   -- ms — RMSSD baseline standard deviation
    baseline_pupil_mm   REAL,                   -- camera units (px) — resting pupil diameter baseline
    baseline_pupil_std  REAL                    -- camera units (px) — pupil baseline standard deviation
);

CREATE TABLE IF NOT EXISTS hrv_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp     REAL    NOT NULL,   -- seconds since session start
    rr_interval   REAL    NOT NULL,   -- inter-beat interval in milliseconds
    bpm           REAL,               -- instantaneous BPM = 60 000 / rr_interval
    rmssd         REAL,               -- rolling RMSSD over 30-second sliding window (ms)
    delta_rmssd   REAL                -- rmssd − rmssd_previous (stress trend indicator)
);

CREATE TABLE IF NOT EXISTS pupil_samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp       REAL    NOT NULL,
    left_diameter   REAL,             -- camera units (px)
    right_diameter  REAL,             -- camera units (px)
    pdi             REAL,             -- (current_diameter − baseline) / baseline
    delta_pdi       REAL              -- pdi − pdi_previous (cognitive load trend)
);

CREATE TABLE IF NOT EXISTS cli_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp   REAL    NOT NULL,
    cli         REAL    NOT NULL    -- Cognitive Load Index 0.0–1.0
);

CREATE TABLE IF NOT EXISTS imported_datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    imported_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    filename        TEXT NOT NULL,
    participant     TEXT NOT NULL,   -- Login value from LapSim
    exercise        TEXT NOT NULL,   -- Task Name from LapSim
    trial_count     INTEGER NOT NULL,
    metric_used     TEXT NOT NULL    -- "Total Time (s)" | "Score" | "Tissue Damage (#)"
);

CREATE TABLE IF NOT EXISTS imported_trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES imported_datasets(id) ON DELETE CASCADE,
    trial_number    INTEGER NOT NULL,   -- 1-based chronological sequence
    start_time      TEXT,               -- ISO 8601 from LapSim
    raw_value       REAL NOT NULL,      -- value of the chosen metric
    score           REAL,               -- original Score column (always stored)
    total_time_s    REAL,               -- always stored regardless of chosen metric
    tissue_damage   INTEGER             -- always stored
);

-- Indexes on frequently queried foreign keys.
CREATE INDEX IF NOT EXISTS idx_calib_session  ON calibrations  (session_id);
CREATE INDEX IF NOT EXISTS idx_hrv_session    ON hrv_samples   (session_id);
CREATE INDEX IF NOT EXISTS idx_pupil_session  ON pupil_samples (session_id);
CREATE INDEX IF NOT EXISTS idx_cli_session    ON cli_samples   (session_id);
CREATE INDEX IF NOT EXISTS idx_import_trials  ON imported_trials (dataset_id);
"""

# ALTER TABLE migrations for databases created before Phase 6a.
# SQLite does not support IF NOT EXISTS on ALTER TABLE, so each statement is
# attempted individually and OperationalError (column already exists) is silenced.
_MIGRATIONS: list[str] = [
    "ALTER TABLE sessions     ADD COLUMN name         TEXT",
    "ALTER TABLE hrv_samples  ADD COLUMN bpm          REAL",
    "ALTER TABLE hrv_samples  ADD COLUMN delta_rmssd  REAL",
    "ALTER TABLE sessions     ADD COLUMN error_count  INTEGER",
    "ALTER TABLE sessions     ADD COLUMN video_path   TEXT",
    "ALTER TABLE pupil_samples ADD COLUMN delta_pdi   REAL",
    "ALTER TABLE calibrations ADD COLUMN baseline_rmssd_std REAL",
    "ALTER TABLE calibrations ADD COLUMN baseline_pupil_std REAL",
]



class DatabaseManager:
    """Manages the SQLite connection and schema lifecycle.

    A single instance of this class should be created at application startup
    and shared across all repository objects.

    Attributes:
        db_path: Resolved filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        """Initialise the database manager and ensure the schema exists.

        Args:
            db_path: Path to the SQLite file. Created if it does not exist.
        """
        self.db_path: Path = Path(db_path).resolve()
        self._connection: sqlite3.Connection | None = None
        self._initialise()

    def _initialise(self) -> None:
        """Open the database, create tables, and run any pending column migrations."""
        logger.info("Opening database at %s", self.db_path)
        conn = self._get_raw_connection()
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(_SCHEMA_SQL)

        # Apply additive migrations (ALTER TABLE ADD COLUMN).  SQLite raises
        # OperationalError when a column already exists, so we catch and ignore it.
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
                logger.debug("Migration applied: %s", migration)
            except sqlite3.OperationalError:
                pass  # column already exists — safe to ignore

        conn.commit()
        logger.info("Database schema verified / created.")

    def _get_raw_connection(self) -> sqlite3.Connection:
        """Return the shared connection, creating it if necessary."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # Repositories handle their own locking.
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._connection.row_factory = sqlite3.Row  # dict-like row access
        return self._connection

    def get_connection(self) -> sqlite3.Connection:
        """Return the active database connection for use by repositories.

        Returns:
            A configured :class:`sqlite3.Connection` with row_factory set.
        """
        return self._get_raw_connection()

    def close(self) -> None:
        """Close the database connection cleanly.

        Call this during application shutdown.
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed.")
