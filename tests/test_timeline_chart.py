"""Unit tests for the TimelineChart widget.

Focuses on empty state and DB data loading logic.
"""

import pytest
from PyQt6.QtWidgets import QApplication
from app.ui.widgets.timeline_chart import TimelineChart
from app.storage.database import DatabaseManager
from app.storage.session_repository import SessionRepository
from app.storage.calibration_repository import CalibrationRepository
from datetime import datetime, timezone

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])

@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    return DatabaseManager(db_path=str(tmp_path / "test_timeline.db"))

@pytest.fixture()
def chart(qapp) -> TimelineChart:
    return TimelineChart()

class TestTimelineChart:
    def test_initial_state_is_empty(self, chart: TimelineChart) -> None:
        assert not chart._empty_label.isHidden()
        assert chart._plot_widget.isHidden()

    def test_load_session_with_data(self, chart: TimelineChart, db: DatabaseManager) -> None:
        # Create session and data
        repo = SessionRepository(db)
        cal_repo = CalibrationRepository(db)
        sid = repo.create_session(datetime.now(tz=timezone.utc))
        
        # Add some samples
        cal_repo.save_hrv_samples_bulk(sid, [(0.0, 800.0, 40.0, 75.0, 0.0), (1.0, 820.0, 42.0, 73.0, 2.0)])
        cal_repo.save_pupil_samples_bulk(sid, [(0.5, 120.0, 0.0, 0.10), (1.5, 121.0, 0.0, 0.20)])
        
        chart.load_session(db, sid)
        
        assert chart._empty_label.isHidden()
        assert not chart._plot_widget.isHidden()
        
        # Verify curves have data
        assert len(chart._stress_curve.xData) == 2
        assert len(chart._pupil_curve.xData) == 2

    def test_load_session_empty_data_shows_placeholder(self, chart: TimelineChart, db: DatabaseManager) -> None:
        repo = SessionRepository(db)
        sid = repo.create_session(datetime.now(tz=timezone.utc))
        
        chart.load_session(db, sid)
        assert not chart._empty_label.isHidden()
        assert chart._plot_widget.isHidden()
