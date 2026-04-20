"""Tests for dashboard progress chart learning-curve series generation."""

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


def _insert_session(
    db: DatabaseManager,
    started_at: datetime,
    duration_minutes: int,
    error_count: int | None,
    nasa_tlx_score: float | None = None,
) -> None:
    conn = db.get_connection()
    ended_at = started_at + timedelta(minutes=duration_minutes)
    conn.execute(
        "INSERT INTO sessions (started_at, ended_at, nasa_tlx_score, error_count) VALUES (?, ?, ?, ?)",
        (
            started_at.isoformat(sep=" "),
            ended_at.isoformat(sep=" "),
            nasa_tlx_score,
            error_count,
        ),
    )
    conn.commit()


def test_progress_chart_uses_normalized_error_history_for_actual_series(
    db: DatabaseManager, qapp: QApplication
) -> None:
    """Observed points should be SCORE_MAX-normalized inverse error scores."""
    start = datetime(2026, 1, 1, 9, 0, 0)
    # Raw errors decline over sessions (improvement).
    for i, err in enumerate([10, 8, 6, 5, 4]):
        _insert_session(db, start + timedelta(days=i), 3, err)

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    _, actual_values, estimate_values, labels = view._build_chart_series()

    # min=4, max=10 -> normalized errors = [100, 66.7, 33.3, 16.7, 0]
    # performance = 100 - normalized_error
    assert labels == ["S1", "S2", "S3", "S4", "S5"]
    assert actual_values[0] == pytest.approx(0.0, abs=0.2)
    assert actual_values[1] == pytest.approx(33.3, abs=0.5)
    assert actual_values[2] == pytest.approx(66.7, abs=0.5)
    assert actual_values[3] == pytest.approx(83.3, abs=0.5)
    assert actual_values[4] == pytest.approx(100.0, abs=0.2)
    assert len(estimate_values) == len(actual_values)

    view.close()


def test_progress_chart_returns_empty_when_no_valid_error_history(
    db: DatabaseManager, qapp: QApplication
) -> None:
    """Without valid error history, progress chart should render no fake data."""
    start = datetime(2026, 2, 1, 9, 0, 0)
    _insert_session(db, start, 3, None, nasa_tlx_score=30.0)
    _insert_session(db, start + timedelta(days=1), 3, None, nasa_tlx_score=70.0)

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    _, actual_values, estimate_values, labels = view._build_chart_series()
    assert labels == []
    assert actual_values == []
    assert estimate_values == []

    view.close()


def test_progress_chart_single_valid_session_shows_single_point(
    db: DatabaseManager, qapp: QApplication
) -> None:
    """One valid session should render one real point (no synthetic trend)."""
    start = datetime(2026, 2, 10, 9, 0, 0)
    _insert_session(db, start, 2, 3)

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    x_values, actual_values, estimate_values, labels = view._build_chart_series()
    assert x_values == [0.0]
    assert labels == ["S1"]
    assert len(actual_values) == 1
    # With a single session, min=max rate -> normalized error 0 -> score 100.
    assert actual_values[0] == pytest.approx(100.0, abs=0.2)
    # Trend line is optional and model-driven; should be absent for one point.
    assert estimate_values == []

    view.close()


def test_progress_chart_prefers_shorter_session_with_same_wall_contacts(
    db: DatabaseManager, qapp: QApplication
) -> None:
    """Same wall contacts but shorter duration → lower effective_time → better score.

    The learning curve uses ``duration_s + error_count * WALL_CONTACT_PENALTY_S``,
    not errors-per-minute.
    """
    start = datetime(2026, 3, 1, 9, 0, 0)
    _insert_session(db, start, 3, 6)
    _insert_session(db, start + timedelta(days=1), 1, 6)

    view = DashboardView(db=db)
    view.show()
    qapp.processEvents()

    _, actual_values, _, labels = view._build_chart_series()

    assert labels == ["S1", "S2"]
    assert actual_values[0] == pytest.approx(0.0, abs=0.2)
    assert actual_values[1] == pytest.approx(100.0, abs=0.2)
    assert actual_values[0] < actual_values[1]

    view.close()
