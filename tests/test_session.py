"""Unit tests for session wiring changes introduced in Phase 6a-4.

Covers sensor selection and the new _store_hrv slot.  No QApplication required
because we only exercise signal/slot logic and in-memory data, never the UI.

Run with:
    pytest tests/test_session.py -v
"""

import pytest
from datetime import datetime, timezone
from PyQt6.QtWidgets import QApplication

from app.storage.database import DatabaseManager
from app.storage.session_repository import SessionRepository
from app.core.session import SessionManager, SessionState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


@pytest.fixture()
def manager(db: DatabaseManager, qapp) -> SessionManager:
    return SessionManager(db)


# ---------------------------------------------------------------------------
# Sensor selection
# ---------------------------------------------------------------------------


class TestSensorSelection:
    def test_mock_sensor_used_when_use_pico_ecg_is_false(
        self, db: DatabaseManager, monkeypatch
    ) -> None:
        """With USE_PICO_ECG=False, SessionManager uses MockHRVSensor."""
        monkeypatch.setattr("app.core.session.USE_PICO_ECG", False)
        from app.core.session import SessionManager as SM
        mgr = SM(db)
        from app.hardware.mock_sensors import MockHRVSensor
        assert isinstance(mgr._hrv_sensor, MockHRVSensor)

    def test_pico_sensor_used_when_use_pico_ecg_is_true(
        self, db: DatabaseManager, monkeypatch
    ) -> None:
        """With USE_PICO_ECG=True, SessionManager uses PicoECGSensor."""
        # Patch the name as it is bound in session.py (imported at module level).
        monkeypatch.setattr("app.core.session.USE_PICO_ECG", True)
        from app.core.session import SessionManager as SM
        mgr = SM(db)
        from app.hardware.pico_ecg_sensor import PicoECGSensor
        assert isinstance(mgr._hrv_sensor, PicoECGSensor)


# ---------------------------------------------------------------------------
# _store_hrv slot
# ---------------------------------------------------------------------------


class TestStoreHRV:
    def test_stores_all_hrv_fields_when_running(
        self, manager: SessionManager
    ) -> None:
        manager._state = SessionState.RUNNING
        manager._session_start_time = 0.0

        manager._store_hrv(rr_ms=800.0, bpm=75.0, rmssd=35.0, delta_rmssd=1.5, timestamp_s=1.0)

        samples = manager._data_store.hrv_samples
        assert len(samples) == 1
        s = samples[0]
        assert s.rr_interval  == pytest.approx(800.0)
        assert s.bpm          == pytest.approx(75.0)
        assert s.rmssd        == pytest.approx(35.0)
        assert s.delta_rmssd  == pytest.approx(1.5)
        assert s.timestamp    == pytest.approx(1.0)   # elapsed = 1.0 - 0.0

    def test_does_not_store_when_not_running(
        self, manager: SessionManager
    ) -> None:
        manager._state = SessionState.IDLE
        manager._store_hrv(800.0, 75.0, 35.0, 1.5, 1.0)
        assert len(manager._data_store.hrv_samples) == 0

    def test_elapsed_time_is_relative_to_session_start(
        self, manager: SessionManager
    ) -> None:
        manager._state = SessionState.RUNNING
        manager._session_start_time = 100.0  # session started at t=100 s

        manager._store_hrv(800.0, 75.0, 35.0, 0.0, 105.0)  # timestamp_s=105

        assert manager._data_store.hrv_samples[0].timestamp == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _persist_samples — includes bpm and delta_rmssd in the bulk INSERT
# ---------------------------------------------------------------------------


class TestPersistSamples:
    def test_bpm_and_delta_rmssd_persisted_to_database(
        self, manager: SessionManager, db: DatabaseManager
    ) -> None:
        repo = SessionRepository(db)
        sid = repo.create_session(datetime.now(tz=timezone.utc))

        manager._session_id = sid
        manager._data_store.session_id = sid
        manager._data_store.add_hrv_sample(
            1.0, 600.0, rmssd=35.0, bpm=100.0, delta_rmssd=2.5
        )

        manager._persist_samples()

        row = db.get_connection().execute(
            "SELECT bpm, delta_rmssd FROM hrv_samples WHERE session_id=?", (sid,)
        ).fetchone()
        assert row["bpm"]         == pytest.approx(100.0)
        assert row["delta_rmssd"] == pytest.approx(2.5)

    def test_none_bpm_persisted_as_null(
        self, manager: SessionManager, db: DatabaseManager
    ) -> None:
        repo = SessionRepository(db)
        sid = repo.create_session(datetime.now(tz=timezone.utc))

        manager._session_id = sid
        manager._data_store.session_id = sid
        manager._data_store.add_hrv_sample(1.0, 800.0)   # no bpm / delta_rmssd

        manager._persist_samples()

        row = db.get_connection().execute(
            "SELECT bpm, delta_rmssd FROM hrv_samples WHERE session_id=?", (sid,)
        ).fetchone()
        assert row["bpm"] is None
        assert row["delta_rmssd"] is None


# ---------------------------------------------------------------------------
# bpm_updated signal — forwarded from HRVProcessor regardless of state
# ---------------------------------------------------------------------------


class TestBPMForwarding:
    def test_bpm_updated_emitted_when_hrv_proc_fires(
        self, manager: SessionManager
    ) -> None:
        received: list[float] = []
        manager.bpm_updated.connect(lambda bpm, _ts: received.append(bpm))

        manager._hrv_proc.hrv_updated.emit(800.0, 75.0, 35.0, 1.0, 1.0)

        assert len(received) == 1
        assert received[0] == pytest.approx(75.0)

    def test_bpm_updated_forwarded_regardless_of_session_state(
        self, manager: SessionManager
    ) -> None:
        """BPM display should update even before a session is RUNNING."""
        received: list[float] = []
        manager.bpm_updated.connect(lambda bpm, _ts: received.append(bpm))

        assert manager.state.name == "IDLE"
        manager._hrv_proc.hrv_updated.emit(850.0, 70.6, 30.0, 0.5, 2.0)

        assert len(received) == 1


# ---------------------------------------------------------------------------
# hrv_connection_changed — forwarded from the HRV sensor
# ---------------------------------------------------------------------------


class TestHRVConnectionForwarding:
    def test_hrv_connection_changed_forwarded_on_connect(
        self, manager: SessionManager
    ) -> None:
        received: list[tuple[bool, str]] = []
        manager.hrv_connection_changed.connect(
            lambda ok, msg: received.append((ok, msg))
        )

        manager._hrv_sensor.connection_status_changed.emit(True, "Connected")

        assert received == [(True, "Connected")]

    def test_hrv_connection_changed_forwarded_on_disconnect(
        self, manager: SessionManager
    ) -> None:
        received: list[bool] = []
        manager.hrv_connection_changed.connect(lambda ok, _msg: received.append(ok))

        manager._hrv_sensor.connection_status_changed.emit(False, "Disconnected")

        assert received == [False]


# ---------------------------------------------------------------------------
# Eye tracker — disabled when USE_EYE_TRACKER=False
# ---------------------------------------------------------------------------


class TestEyeTrackerDisabled:
    def test_eye_tracker_not_started_in_session_when_disabled(
        self, db: DatabaseManager, monkeypatch
    ) -> None:
        """start_session() must not call eye_tracker.start() when USE_EYE_TRACKER=False."""
        monkeypatch.setattr("app.core.session.USE_EYE_TRACKER", False)
        from app.core.session import SessionManager as SM
        mgr = SM(db)

        mgr._state = SessionState.READY
        mgr.start_session()

        assert not mgr._eye_tracker._running

    def test_eye_tracker_not_started_in_calibration_when_disabled(
        self, db: DatabaseManager, monkeypatch
    ) -> None:
        """start_calibration() must not call eye_tracker.start() when USE_EYE_TRACKER=False."""
        monkeypatch.setattr("app.core.session.USE_EYE_TRACKER", False)
        from app.core.session import SessionManager as SM
        mgr = SM(db)

        mgr.start_calibration()

        assert not mgr._eye_tracker._running


# ---------------------------------------------------------------------------
# Error counting
# ---------------------------------------------------------------------------


class TestErrorCounting:
    def test_increment_error_count_emits_signal(self, manager: SessionManager) -> None:
        """increment_error_count() increments and emits signal when RUNNING."""
        manager._state = SessionState.RUNNING
        received: list[int] = []
        manager.error_count_updated.connect(received.append)

        manager.increment_error_count()
        assert manager._error_count == 1
        assert received == [1]

    def test_decrement_error_count_floored_at_zero(self, manager: SessionManager) -> None:
        """decrement_error_count() stays at 0, never negative."""
        manager._state = SessionState.RUNNING
        manager._error_count = 1

        manager.decrement_error_count()
        assert manager._error_count == 0

        manager.decrement_error_count()
        assert manager._error_count == 0

    def test_error_count_reset_on_start_session(self, manager: SessionManager) -> None:
        """start_session() resets error count to zero."""
        manager._error_count = 5
        manager._state = SessionState.READY
        manager.start_session()
        assert manager._error_count == 0

    def test_error_count_persisted_on_end_session(
        self, manager: SessionManager, db: DatabaseManager
    ) -> None:
        """end_session() writes the final error count to the database."""
        manager._state = SessionState.READY
        sid = manager.start_session()
        manager._error_count = 3
        manager.end_session()

        row = db.get_connection().execute(
            "SELECT error_count FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        assert row["error_count"] == 3

    def test_hardware_error_increments_count(self, manager: SessionManager) -> None:
        """Signal from ErrorCounter increments the count."""
        manager._state = SessionState.RUNNING
        manager._error_counter.error_detected.emit()
        assert manager._error_count == 1

    def test_pico_wall_contact_signal_increments_count(
        self, db: DatabaseManager, monkeypatch
    ) -> None:
        """Wall-contact events from the Pico serial stream increment the count."""
        monkeypatch.setattr("app.core.session.USE_PICO_ECG", True)
        from app.core.session import SessionManager as SM

        manager = SM(db)
        manager._state = SessionState.RUNNING

        manager._hrv_sensor.wall_contact_detected.emit()

        assert manager._error_count == 1
