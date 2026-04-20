"""Tests for dashboard stress KPI card aggregation."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from app.storage.database import DatabaseManager
from app.ui.views.dashboard_view import DashboardView

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


def test_dashboard_stress_card_shows_session_average(
    db: DatabaseManager, qapp: QApplication
) -> None:
    conn = db.get_connection()
    start_a = datetime(2026, 1, 1, 10, 0, 0)
    start_b = datetime(2026, 1, 2, 10, 0, 0)

    cur_a = conn.execute(
        "INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)",
        (start_a.isoformat(sep=" "), (start_a + timedelta(minutes=3)).isoformat(sep=" "), 0),
    )
    sid_a = int(cur_a.lastrowid)

    cur_b = conn.execute(
        "INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)",
        (start_b.isoformat(sep=" "), (start_b + timedelta(minutes=2)).isoformat(sep=" "), 0),
    )
    sid_b = int(cur_b.lastrowid)

    conn.executemany(
        "INSERT INTO calibrations (session_id, recorded_at, duration_seconds, baseline_rmssd, baseline_pupil_mm) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (sid_a, start_a.isoformat(sep=" "), 30, 50.0, 120.0),
            (sid_b, start_b.isoformat(sep=" "), 30, 50.0, 120.0),
        ],
    )

    # Session A stress events: 2 crossings below -10%.
    conn.executemany(
        "INSERT INTO hrv_samples (session_id, timestamp, rr_interval, rmssd) VALUES (?, ?, ?, ?)",
        [
            (sid_a, 1.0, 800.0, 50.0),
            (sid_a, 2.0, 800.0, 44.0),
            (sid_a, 3.0, 800.0, 43.0),
            (sid_a, 4.0, 800.0, 47.0),
            (sid_a, 5.0, 800.0, 29.0),
            (sid_a, 6.0, 800.0, 25.0),
            (sid_a, 7.0, 800.0, 45.0),
        ],
    )
    # Session B stress events: 1 crossing below -10%.
    conn.executemany(
        "INSERT INTO hrv_samples (session_id, timestamp, rr_interval, rmssd) VALUES (?, ?, ?, ?)",
        [
            (sid_b, 1.0, 800.0, 50.0),
            (sid_b, 2.0, 800.0, 44.0),
            (sid_b, 3.0, 800.0, 46.0),
            (sid_b, 4.0, 800.0, 49.0),
        ],
    )

    # Session A average |PDI| = 20%; Session B average |PDI| = 30%.
    conn.executemany(
        "INSERT INTO pupil_samples (session_id, timestamp, left_diameter, right_diameter, pdi) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (sid_a, 1.0, None, None, 0.10),
            (sid_a, 2.0, None, None, 0.30),
            (sid_b, 1.0, None, None, 0.40),
            (sid_b, 2.0, None, None, 0.20),
        ],
    )
    conn.commit()

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    assert view._stress_gauge is not None
    assert view._stress_gauge._center_text == "1.5"

    view.close()
