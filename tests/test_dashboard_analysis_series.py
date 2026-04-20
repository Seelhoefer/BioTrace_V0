"""Session Trend Analysis on the main dashboard uses persisted session samples."""

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


def test_analysis_series_empty_when_no_sessions(
    db: DatabaseManager, qapp: QApplication
) -> None:
    view = DashboardView(db=db)
    x, stress, workload, labels = view._build_analysis_series()
    assert x == []
    assert stress == []
    assert workload == []
    assert labels == []


def test_analysis_series_matches_db_avg_rmssd_and_cli(
    db: DatabaseManager, qapp: QApplication
) -> None:
    """Series values match AVG(rmssd) / AVG(cli) from hrv_samples and cli_samples."""
    conn = db.get_connection()
    t0 = datetime(2026, 1, 1, 10, 0, 0)
    t1 = datetime(2026, 1, 2, 10, 0, 0)
    cur0 = conn.execute(
        "INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)",
        (t0.isoformat(sep=" "), (t0 + timedelta(minutes=10)).isoformat(sep=" "), 0),
    )
    sid0 = int(cur0.lastrowid)
    cur1 = conn.execute(
        "INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)",
        (t1.isoformat(sep=" "), (t1 + timedelta(minutes=10)).isoformat(sep=" "), 0),
    )
    sid1 = int(cur1.lastrowid)

    conn.executemany(
        "INSERT INTO hrv_samples (session_id, timestamp, rr_interval, rmssd) VALUES (?, ?, ?, ?)",
        [
            (sid0, 1.0, 800.0, 40.0),
            (sid0, 2.0, 800.0, 40.0),
        ],
    )
    conn.executemany(
        "INSERT INTO cli_samples (session_id, timestamp, cli) VALUES (?, ?, ?)",
        [
            (sid0, 1.0, 0.2),
            (sid0, 2.0, 0.3),
        ],
    )
    conn.executemany(
        "INSERT INTO hrv_samples (session_id, timestamp, rr_interval, rmssd) VALUES (?, ?, ?, ?)",
        [
            (sid1, 1.0, 800.0, 60.0),
            (sid1, 2.0, 800.0, 60.0),
        ],
    )
    conn.executemany(
        "INSERT INTO cli_samples (session_id, timestamp, cli) VALUES (?, ?, ?)",
        [
            (sid1, 1.0, 0.7),
            (sid1, 2.0, 0.8),
        ],
    )
    conn.commit()

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    x, stress, workload, labels = view._build_analysis_series()
    assert x == [0.0, 1.0]
    assert labels == ["S1", "S2"]
    # Cohort min RMSSD=40, max=60 → inverted stress %: low HRV → high %
    assert stress[0] == pytest.approx(100.0)
    assert stress[1] == pytest.approx(0.0)
    assert workload[0] == pytest.approx(25.0)
    assert workload[1] == pytest.approx(75.0)

    view.close()


def test_analysis_series_neutral_stress_when_no_hrv_but_cli_present(
    db: DatabaseManager, qapp: QApplication
) -> None:
    """If a session has CLI but no HRV rows, stress point is neutral (50 %)."""
    conn = db.get_connection()
    t0 = datetime(2026, 1, 1, 10, 0, 0)
    cur = conn.execute(
        "INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)",
        (t0.isoformat(sep=" "), (t0 + timedelta(minutes=5)).isoformat(sep=" "), 0),
    )
    sid = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO cli_samples (session_id, timestamp, cli) VALUES (?, ?, ?)",
        (sid, 1.0, 0.5),
    )
    conn.commit()

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    _x, stress, workload, _labels = view._build_analysis_series()
    assert len(stress) == 1
    assert stress[0] == pytest.approx(50.0)
    assert workload[0] == pytest.approx(50.0)

    view.close()
