"""SessionTable — sortable table widget for past sessions.

Displays all sessions from the database in a styled, sortable
:class:`QTableWidget`.  Clicking a row emits ``session_selected``.

Usage::

    repo = SessionRepository(db)
    table = SessionTable()
    table.load_sessions(repo.get_all_sessions())
    table.session_selected.connect(on_session_selected)
"""

import sqlite3
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHeaderView,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from app.ui.theme import COLOR_CARD, COLOR_PRIMARY
from app.utils.logger import get_logger

logger = get_logger(__name__)

_HEADERS = ["Session", "Duration", "Avg CLI", "NASA-TLX", "Notes"]


class SessionTable(QTableWidget):
    """Sortable table displaying the BioTrace session history.

    Signals:
        session_selected (int):
            Emitted with the database session ``id`` when the user clicks a row.
    """

    session_selected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(_HEADERS), parent)
        self._session_ids: list[int] = []
        self._configure_table()

    def _configure_table(self) -> None:
        """Apply visual settings that remain constant."""
        self.setHorizontalHeaderLabels(_HEADERS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setSortingEnabled(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(
            f"alternate-background-color: #F4F7FF; background-color: {COLOR_CARD};"
        )

        # Column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self.cellClicked.connect(self._on_cell_clicked)

    def load_sessions(self, sessions: list[sqlite3.Row]) -> None:
        """Populate the table from a list of session rows.

        Args:
            sessions: List of :class:`sqlite3.Row` objects (from
                      ``SessionRepository.get_all_sessions()``).
        """
        self.setSortingEnabled(False)
        self.setRowCount(0)
        self._session_ids = []

        if not sessions:
            self.setRowCount(1)
            placeholder = QTableWidgetItem("No sessions recorded yet.")
            placeholder.setForeground(Qt.GlobalColor.gray)
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.setItem(0, 0, placeholder)
            self.setSpan(0, 0, 1, len(_HEADERS))
            return

        for row_data in sessions:
            row_idx = self.rowCount()
            self.insertRow(row_idx)
            self._session_ids.append(row_data["id"])

            # Session Name / Date
            name = row_data["name"] if "name" in row_data.keys() else None
            if name:
                display_name = name
            else:
                started = row_data["started_at"] or "—"
                display_name = str(started)[:16]
            self.setItem(row_idx, 0, QTableWidgetItem(display_name))

            # Duration
            duration = _compute_duration(row_data["started_at"], row_data["ended_at"])
            self.setItem(row_idx, 1, QTableWidgetItem(duration))

            # Avg CLI (not stored per-session yet — placeholder)
            self.setItem(row_idx, 2, QTableWidgetItem("—"))

            # NASA-TLX
            tlx = row_data["nasa_tlx_score"]
            tlx_str = f"{tlx:.1f}" if tlx is not None else "—"
            self.setItem(row_idx, 3, QTableWidgetItem(tlx_str))

            # Notes
            notes = row_data["notes"] or ""
            self.setItem(row_idx, 4, QTableWidgetItem(notes))

        self.setSortingEnabled(True)
        logger.info("SessionTable loaded %d sessions.", len(sessions))

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        """Emit the session ID for the clicked row."""
        if row < len(self._session_ids):
            session_id = self._session_ids[row]
            self.session_selected.emit(session_id)
            logger.debug("Session selected: id=%d", session_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_duration(started_at: str | None, ended_at: str | None) -> str:
    """Compute a human-readable session duration string.

    Args:
        started_at: ISO datetime string of session start.
        ended_at: ISO datetime string of session end, or ``None``.

    Returns:
        Duration as ``"MM:SS"`` or ``"—"`` if data is unavailable.
    """
    if not started_at or not ended_at:
        return "—"
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        start = datetime.strptime(started_at[:19], fmt)
        end = datetime.strptime(ended_at[:19], fmt)
        total_seconds = int((end - start).total_seconds())
        if total_seconds < 0:
            return "—"
        m, s = divmod(total_seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    except (ValueError, TypeError):
        return "—"
