import pytest
from PyQt6.QtWidgets import QApplication

from app.core.session import SessionManager
from app.storage.database import DatabaseManager
from app.ui.views.live_view import LiveView


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    return DatabaseManager(db_path=str(tmp_path / "test_live_view.db"))


@pytest.fixture()
def manager(db: DatabaseManager, qapp) -> SessionManager:
    return SessionManager(db)


class TestLiveView:
    def test_bind_session_manager_restores_pupil_percentage_mode(
        self, qapp, manager: SessionManager
    ) -> None:
        view = LiveView()
        manager.set_pupil_baseline(120.0)

        view.bind_session_manager(manager)

        assert view._has_pupil_baseline is True
        assert view._pupil_card._unit == "%"

    def test_session_start_keeps_pupil_percentage_mode_when_baseline_exists(
        self, qapp, manager: SessionManager, monkeypatch
    ) -> None:
        view = LiveView()
        manager.set_pupil_baseline(120.0)
        view.bind_session_manager(manager)

        monkeypatch.setattr(view._video_feed, "start", lambda *args, **kwargs: None)
        monkeypatch.setattr(view, "_start_camera_recording", lambda: None)

        view._on_session_started(1)

        assert view._has_pupil_baseline is True
        assert view._pupil_card._unit == "%"

    def test_rmssd_updates_feed_hrv_series(
        self, qapp
    ) -> None:
        view = LiveView()

        view.on_rmssd_updated(50.0, 100.0)

        assert "HRV" in view._timeline_chart._curves
        assert "STRESS" not in view._timeline_chart._curves
        assert list(view._timeline_chart._timestamps["HRV"]) == [100.0]
        assert list(view._timeline_chart._values["HRV"]) == [pytest.approx(50.0)]
