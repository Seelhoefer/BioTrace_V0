"""Tests for dashboard wall-contact error-rate aggregation."""

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


def test_compute_error_rate_per_minute() -> None:
    """Error frequency should be derived from session duration."""
    rate = DashboardView._compute_error_rate_per_minute(
        error_count=6,
        started_at="2026-01-01 10:00:00",
        ended_at="2026-01-01 10:03:00",
    )
    assert rate == pytest.approx(2.0)


def test_dashboard_error_gauge_shows_average_errors_per_session(
    db: DatabaseManager, qapp: QApplication
) -> None:
    """The KPI should display average wall-contact count per session."""
    conn = db.get_connection()
    start_a = datetime(2026, 1, 1, 10, 0, 0)
    start_b = datetime(2026, 1, 2, 10, 0, 0)

    conn.execute(
        "INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)",
        (start_a.isoformat(sep=" "), (start_a + timedelta(minutes=3)).isoformat(sep=" "), 6),
    )
    conn.execute(
        "INSERT INTO sessions (started_at, ended_at, error_count) VALUES (?, ?, ?)",
        (start_b.isoformat(sep=" "), (start_b + timedelta(minutes=1)).isoformat(sep=" "), 1),
    )
    conn.commit()

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    assert view._error_gauge is not None
    assert view._error_gauge._center_text == "3.5/session"

    view.close()

