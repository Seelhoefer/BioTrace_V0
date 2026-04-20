"""Unit tests for the database and repository layer.

Uses an in-memory SQLite database so tests are fast and leave no files behind.

Run with:
    pytest tests/test_storage.py -v
"""

import pytest
from datetime import datetime, timezone

from app.storage.database import DatabaseManager
from app.storage.session_repository import SessionRepository
from app.storage.calibration_repository import CalibrationRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    """Return a fresh in-memory-backed DatabaseManager for each test."""
    return DatabaseManager(db_path=str(tmp_path / "test_biotrace.db"))


@pytest.fixture()
def session_repo(db: DatabaseManager) -> SessionRepository:
    return SessionRepository(db)


@pytest.fixture()
def cal_repo(db: DatabaseManager) -> CalibrationRepository:
    return CalibrationRepository(db)


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------


class TestDatabaseManager:
    def test_creates_schema(self, db: DatabaseManager) -> None:
        """All expected tables should be present after init."""
        conn = db.get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "sessions"      in tables
        assert "calibrations"  in tables
        assert "hrv_samples"   in tables
        assert "pupil_samples" in tables
        assert "cli_samples"   in tables

    def test_close_and_reopen(self, tmp_path) -> None:
        """Closing and re-opening the DB should not raise."""
        db = DatabaseManager(db_path=str(tmp_path / "reopen.db"))
        db.close()
        db2 = DatabaseManager(db_path=str(tmp_path / "reopen.db"))
        assert db2.get_connection() is not None
        db2.close()


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------


class TestSessionRepository:
    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def test_create_session_returns_id(self, session_repo: SessionRepository) -> None:
        sid = session_repo.create_session(self._now())
        assert isinstance(sid, int)
        assert sid >= 1

    def test_get_session_after_create(self, session_repo: SessionRepository) -> None:
        sid = session_repo.create_session(self._now())
        row = session_repo.get_session(sid)
        assert row is not None
        assert row["id"] == sid

    def test_end_session_records_end_time(self, session_repo: SessionRepository) -> None:
        sid = session_repo.create_session(self._now())
        session_repo.end_session(sid, self._now(), notes="test run")
        row = session_repo.get_session(sid)
        assert row["ended_at"] is not None
        assert row["notes"] == "test run"

    def test_save_nasa_tlx_score(self, session_repo: SessionRepository) -> None:
        sid = session_repo.create_session(self._now())
        session_repo.save_nasa_tlx_score(sid, 72.5)
        row = session_repo.get_session(sid)
        assert row["nasa_tlx_score"] == pytest.approx(72.5)

    def test_set_video_path(self, session_repo: SessionRepository) -> None:
        sid = session_repo.create_session(self._now())
        path = "/tmp/session_1.mp4"
        session_repo.set_video_path(sid, path)
        assert session_repo.get_video_path(sid) == path

    def test_get_all_sessions_ordered_newest_first(self, session_repo: SessionRepository) -> None:
        sid1 = session_repo.create_session(datetime(2024, 1, 1, tzinfo=timezone.utc))
        sid2 = session_repo.create_session(datetime(2024, 6, 1, tzinfo=timezone.utc))
        rows = session_repo.get_all_sessions()
        ids = [r["id"] for r in rows]
        # newest first: sid2 should appear before sid1
        assert ids.index(sid2) < ids.index(sid1)

    def test_delete_session(self, session_repo: SessionRepository) -> None:
        sid = session_repo.create_session(self._now())
        session_repo.delete_session(sid)
        assert session_repo.get_session(sid) is None

    def test_get_nonexistent_session_returns_none(self, session_repo: SessionRepository) -> None:
        assert session_repo.get_session(99999) is None

    def test_get_completed_sessions_excludes_abandoned(
        self, session_repo: SessionRepository
    ) -> None:
        """get_completed_sessions() must exclude sessions without ended_at."""
        sid_done = session_repo.create_session(datetime(2024, 1, 1, tzinfo=timezone.utc))
        session_repo.end_session(sid_done, datetime(2024, 1, 1, 1, tzinfo=timezone.utc))

        _sid_abandoned = session_repo.create_session(datetime(2024, 2, 1, tzinfo=timezone.utc))
        # never ended

        rows = session_repo.get_completed_sessions()
        ids = [r["id"] for r in rows]
        assert sid_done in ids
        assert _sid_abandoned not in ids

    def test_get_completed_sessions_ordered_newest_first(
        self, session_repo: SessionRepository
    ) -> None:
        """get_completed_sessions() orders by started_at descending."""
        sid1 = session_repo.create_session(datetime(2024, 1, 1, tzinfo=timezone.utc))
        session_repo.end_session(sid1, datetime(2024, 1, 1, 1, tzinfo=timezone.utc))
        sid2 = session_repo.create_session(datetime(2024, 6, 1, tzinfo=timezone.utc))
        session_repo.end_session(sid2, datetime(2024, 6, 1, 1, tzinfo=timezone.utc))

        rows = session_repo.get_completed_sessions()
        ids = [r["id"] for r in rows]
        assert ids.index(sid2) < ids.index(sid1)


# ---------------------------------------------------------------------------
# CalibrationRepository
# ---------------------------------------------------------------------------


class TestCalibrationRepository:
    def _make_session(self, session_repo: SessionRepository) -> int:
        return session_repo.create_session(datetime.now(tz=timezone.utc))

    def test_save_calibration_returns_id(
        self, db: DatabaseManager, session_repo: SessionRepository, cal_repo: CalibrationRepository
    ) -> None:
        sid = self._make_session(session_repo)
        cal_id = cal_repo.save_calibration(
            session_id=sid,
            baseline_rmssd=38.4,
            baseline_pupil_px=120.0,
            duration_seconds=60,
        )
        assert isinstance(cal_id, int)
        assert cal_id >= 1

    def test_get_latest_for_session(
        self, session_repo: SessionRepository, cal_repo: CalibrationRepository
    ) -> None:
        sid = self._make_session(session_repo)
        cal_repo.save_calibration(sid, 38.4, 120.0, 60)
        row = cal_repo.get_latest_for_session(sid)
        assert row is not None
        assert row["baseline_rmssd"] == pytest.approx(38.4)
        assert row["baseline_pupil_mm"] == pytest.approx(120.0)
        assert row["duration_seconds"] == 60

    def test_bulk_hrv_samples(
        self, session_repo: SessionRepository, cal_repo: CalibrationRepository, db: DatabaseManager
    ) -> None:
        sid = self._make_session(session_repo)
        # Tuples are (timestamp, rr_interval, rmssd, bpm, delta_rmssd).
        samples = [
            (0.0, 800.0, 35.2, 75.0, 0.0),
            (1.0, 820.0, 36.1, 73.2, 0.9),
            (2.0, 810.0, None, None, None),
        ]
        cal_repo.save_hrv_samples_bulk(sid, samples)
        conn = db.get_connection()
        count = conn.execute(
            "SELECT COUNT(*) FROM hrv_samples WHERE session_id=?", (sid,)
        ).fetchone()[0]
        assert count == 3

    def test_bulk_hrv_samples_persists_bpm_and_delta_rmssd(
        self, session_repo: SessionRepository, cal_repo: CalibrationRepository, db: DatabaseManager
    ) -> None:
        """bpm and delta_rmssd values round-trip through the database correctly."""
        sid = self._make_session(session_repo)
        samples = [(1.0, 600.0, 0.0, 100.0, 5.3)]
        cal_repo.save_hrv_samples_bulk(sid, samples)
        row = db.get_connection().execute(
            "SELECT bpm, delta_rmssd FROM hrv_samples WHERE session_id=?", (sid,)
        ).fetchone()
        assert row["bpm"] == pytest.approx(100.0)
        assert row["delta_rmssd"] == pytest.approx(5.3)

    def test_bulk_hrv_samples_accepts_null_bpm_and_delta_rmssd(
        self, session_repo: SessionRepository, cal_repo: CalibrationRepository, db: DatabaseManager
    ) -> None:
        """None values for bpm and delta_rmssd should persist as NULL."""
        sid = self._make_session(session_repo)
        samples = [(1.0, 800.0, 35.0, None, None)]
        cal_repo.save_hrv_samples_bulk(sid, samples)
        row = db.get_connection().execute(
            "SELECT bpm, delta_rmssd FROM hrv_samples WHERE session_id=?", (sid,)
        ).fetchone()
        assert row["bpm"] is None
        assert row["delta_rmssd"] is None

    def test_bulk_cli_samples(
        self, session_repo: SessionRepository, cal_repo: CalibrationRepository, db: DatabaseManager
    ) -> None:
        sid = self._make_session(session_repo)
        samples = [(0.5, 0.32), (1.5, 0.55), (2.5, 0.71)]
        cal_repo.save_cli_samples_bulk(sid, samples)
        conn = db.get_connection()
        count = conn.execute(
            "SELECT COUNT(*) FROM cli_samples WHERE session_id=?", (sid,)
        ).fetchone()[0]
        assert count == 3

    def test_no_calibration_for_missing_session(
        self, cal_repo: CalibrationRepository
    ) -> None:
        assert cal_repo.get_latest_for_session(99999) is None


# ---------------------------------------------------------------------------
# Phase 6a-3 — DataStore: HRVSample and add_hrv_sample
# ---------------------------------------------------------------------------


class TestHRVSampleAndDataStore:
    """HRVSample carries bpm + delta_rmssd; DataStore stores them."""

    def test_hrv_sample_has_bpm_field(self) -> None:
        from app.core.data_store import HRVSample
        s = HRVSample(timestamp=1.0, rr_interval=800.0, rmssd=35.0, bpm=75.0, delta_rmssd=1.2)
        assert s.bpm == pytest.approx(75.0)

    def test_hrv_sample_has_delta_rmssd_field(self) -> None:
        from app.core.data_store import HRVSample
        s = HRVSample(timestamp=1.0, rr_interval=800.0, rmssd=35.0, bpm=75.0, delta_rmssd=1.2)
        assert s.delta_rmssd == pytest.approx(1.2)

    def test_hrv_sample_bpm_defaults_to_none(self) -> None:
        from app.core.data_store import HRVSample
        s = HRVSample(timestamp=1.0, rr_interval=800.0, rmssd=35.0)
        assert s.bpm is None

    def test_hrv_sample_delta_rmssd_defaults_to_none(self) -> None:
        from app.core.data_store import HRVSample
        s = HRVSample(timestamp=1.0, rr_interval=800.0, rmssd=35.0)
        assert s.delta_rmssd is None

    def test_add_hrv_sample_stores_bpm_and_delta_rmssd(self) -> None:
        from app.core.data_store import DataStore
        store = DataStore()
        store.add_hrv_sample(1.0, 600.0, rmssd=0.0, bpm=100.0, delta_rmssd=2.5)
        sample = store.hrv_samples[0]
        assert sample.bpm == pytest.approx(100.0)
        assert sample.delta_rmssd == pytest.approx(2.5)

    def test_add_hrv_sample_bpm_defaults_to_none(self) -> None:
        from app.core.data_store import DataStore
        store = DataStore()
        store.add_hrv_sample(1.0, 800.0)
        assert store.hrv_samples[0].bpm is None
        assert store.hrv_samples[0].delta_rmssd is None


# ---------------------------------------------------------------------------
# Phase 6a-1 — Schema migration: new columns
# ---------------------------------------------------------------------------


class TestSchemaColumns:
    """Verify that Phase-6a columns exist on both fresh and migrated databases."""

    def _column_names(self, db: DatabaseManager, table: str) -> set[str]:
        conn = db.get_connection()
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return {row["name"] for row in cursor.fetchall()}

    def test_hrv_samples_has_bpm_and_delta_rmssd(self, db: DatabaseManager) -> None:
        cols = self._column_names(db, "hrv_samples")
        assert "bpm" in cols
        assert "delta_rmssd" in cols

    def test_sessions_has_error_count_and_video_path(self, db: DatabaseManager) -> None:
        cols = self._column_names(db, "sessions")
        assert "error_count" in cols
        assert "video_path" in cols

    def test_pupil_samples_has_delta_pdi(self, db: DatabaseManager) -> None:
        assert "delta_pdi" in self._column_names(db, "pupil_samples")

    def test_migration_adds_columns_to_existing_database(self, tmp_path) -> None:
        """A pre-Phase-6a database (no new columns) is updated when reopened."""
        import sqlite3 as _sqlite3

        db_path = str(tmp_path / "old_schema.db")

        # Build an old-style database that lacks the new columns.
        raw = _sqlite3.connect(db_path)
        raw.executescript("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at DATETIME NOT NULL,
                ended_at DATETIME,
                notes TEXT,
                nasa_tlx_score REAL
            );
            CREATE TABLE hrv_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                rr_interval REAL NOT NULL,
                rmssd REAL
            );
            CREATE TABLE pupil_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                left_diameter REAL,
                right_diameter REAL,
                pdi REAL
            );
        """)
        raw.commit()
        raw.close()

        # Re-opening via DatabaseManager should run migrations.
        db = DatabaseManager(db_path=db_path)
        conn = db.get_connection()

        hrv_cols     = {r["name"] for r in conn.execute("PRAGMA table_info(hrv_samples)").fetchall()}
        session_cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        pupil_cols   = {r["name"] for r in conn.execute("PRAGMA table_info(pupil_samples)").fetchall()}

        assert "bpm"          in hrv_cols
        assert "delta_rmssd"  in hrv_cols
        assert "error_count"  in session_cols
        assert "delta_pdi"    in pupil_cols
        db.close()

    def test_new_hrv_columns_accept_null(
        self, db: DatabaseManager, session_repo: SessionRepository
    ) -> None:
        """Inserting an hrv_samples row without bpm/delta_rmssd should leave them NULL."""
        sid = session_repo.create_session(datetime.now(tz=timezone.utc))
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO hrv_samples (session_id, timestamp, rr_interval) VALUES (?, ?, ?)",
            (sid, 1.0, 800.0),
        )
        conn.commit()
        row = conn.execute(
            "SELECT bpm, delta_rmssd FROM hrv_samples WHERE session_id = ?", (sid,)
        ).fetchone()
        assert row["bpm"] is None
        assert row["delta_rmssd"] is None
