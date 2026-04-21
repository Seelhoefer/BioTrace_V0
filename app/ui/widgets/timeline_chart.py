"""TimelineChart — pyqtgraph-based session analysis chart.

Displays Stress (RMSSD) and Pupil Dilation (PDI) over the course of a session
on a unified ``% Change from Baseline`` Y axis.

Both series are normalised before plotting:

- **Stress (RMSSD)**: ``-((rmssd − baseline_rmssd) / baseline_rmssd × 100)``
    RMSSD drops below baseline therefore move upward in the chart.
    This keeps directional comparability with pupil-dilation delta percent.

- **Pupil Dilation (PDI)**: ``pdi × 100``
  PDI is already ``(diameter − baseline) / baseline``, so multiplying by 100
  yields a direct percentage change.

A persistent grey playhead line tracks the current video position.
A hover hairline follows the mouse cursor.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from app.storage.database import DatabaseManager
from app.ui.theme import (
    COLOR_BORDER,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_WARNING,
    COLOR_CARD,
    FONT_SMALL,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TimelineChart(QWidget):
    """Interactive session analysis chart with Stress and Pupil Dilation series.

    Both series share a single Y axis scaled to ``% Change from Baseline``.

    Signals:
        timestamp_clicked (float): Emitted when the user clicks on the chart,
                                   in milliseconds from session start.
    """

    timestamp_clicked = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_session(
        self,
        db: DatabaseManager,
        session_id: int,
        expected_duration_s: int | None = None,
    ) -> None:
        """Query the database and plot normalised series for the given session.

        RMSSD is expressed as an inverted percent change from the calibration
        baseline so stress-relevant drops go up in the plot.
        PDI is expressed as percent change (pdi × 100) — it is already a
        fractional deviation from the resting pupil diameter.

        Args:
            db: Database manager instance.
            session_id: ID of the session to load.
        """
        self.clear()

        conn = db.get_connection()

        # ── Calibration baseline for RMSSD normalisation ──────────────
        cal_row = conn.execute(
            "SELECT baseline_rmssd FROM calibrations "
            "WHERE session_id = ? AND baseline_rmssd IS NOT NULL ORDER BY id ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        baseline_rmssd: float | None = float(cal_row[0]) if cal_row else None

        # ── Stress (RMSSD) as inverted % change from baseline ─────────
        hrv_rows = conn.execute(
            "SELECT timestamp, rmssd FROM hrv_samples "
            "WHERE session_id = ? AND rmssd IS NOT NULL ORDER BY timestamp",
            (session_id,),
        ).fetchall()

        stress_x: list[float] = []
        stress_y: list[float] = []

        if hrv_rows:
            rmssd_vals = [float(r["rmssd"]) for r in hrv_rows]
            # Fall back to session mean when no calibration baseline is available.
            ref = baseline_rmssd if (baseline_rmssd and baseline_rmssd > 0) else (
                float(np.mean(rmssd_vals)) if rmssd_vals else None
            )
            if ref and ref > 0:
                stress_x = [float(r["timestamp"]) for r in hrv_rows]
                # Invert sign so RMSSD drops (stress increase) trend upward.
                stress_y = [-(v - ref) / ref * 100.0 for v in rmssd_vals]

        # ── Pupil Dilation (PDI) as % change ─────────────────────────
        pupil_rows = conn.execute(
            "SELECT timestamp, pdi FROM pupil_samples "
            "WHERE session_id = ? AND pdi IS NOT NULL ORDER BY timestamp",
            (session_id,),
        ).fetchall()

        pupil_x: list[float] = []
        pupil_y: list[float] = []

        if pupil_rows:
            pupil_x = [float(r["timestamp"]) for r in pupil_rows]
            pupil_y = [float(r["pdi"]) * 100.0 for r in pupil_rows]

        if not stress_x and not pupil_x:
            self._show_empty_state(True)
            return

        self._show_empty_state(False)

        if stress_x:
            self._stress_curve.setData(stress_x, stress_y)

        if pupil_x:
            self._pupil_curve.setData(pupil_x, pupil_y)

        max_data_time = 0.0
        if stress_x:
            max_data_time = max(max_data_time, max(stress_x))
        if pupil_x:
            max_data_time = max(max_data_time, max(pupil_x))

        if expected_duration_s is not None and expected_duration_s > 0:
            x_max = float(expected_duration_s)
            if max_data_time > 0.0 and abs(max_data_time - x_max) > 1.0:
                logger.warning(
                    "Timeline duration mismatch for session %d: data=%.2fs, expected=%.2fs",
                    session_id,
                    max_data_time,
                    x_max,
                )
        else:
            x_max = max_data_time if max_data_time > 0.0 else 1.0

        self._plot_widget.setXRange(0.0, x_max, padding=0.0)
        self._plot_widget.setYRange(-50.0, 150.0, padding=0)
        logger.info(
            "TimelineChart loaded session %d: %d RMSSD %%, %d PDI %% points",
            session_id, len(stress_x), len(pupil_x),
        )

    def set_series_visibility(self, name: str, visible: bool) -> None:
        """Toggle visibility of a specific series.

        Args:
            name: ``"STRESS"`` or ``"WORKLOAD"``.
            visible: Whether the series should be shown.
        """
        if name == "STRESS":
            self._stress_curve.setVisible(visible)
        elif name == "WORKLOAD":
            self._pupil_curve.setVisible(visible)

    def set_playhead_ms(self, position_ms: float) -> None:
        """Move the persistent playhead line to a video timestamp.

        Args:
            position_ms: Video position in milliseconds.
        """
        self._playhead_line.setPos(position_ms / 1000.0)

    def clear(self) -> None:
        """Reset the chart to an empty state."""
        self._stress_curve.setData([], [])
        self._pupil_curve.setData([], [])
        self._playhead_line.setPos(0)
        self._show_empty_state(True)

    # ------------------------------------------------------------------
    # Internal Logic
    # ------------------------------------------------------------------

    def _show_empty_state(self, empty: bool) -> None:
        """Toggle between the chart and an 'empty' placeholder message."""
        if empty:
            self._empty_label.show()
            self._plot_widget.hide()
        else:
            self._empty_label.hide()
            self._plot_widget.show()

    def _on_mouse_clicked(self, event) -> None:
        """Handle clicks on the plot to emit the timestamp."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self._plot_item.sceneBoundingRect().contains(pos):
                self._seek_to_scene_pos(pos)

    def _seek_to_scene_pos(self, scene_pos) -> None:
        """Map a scene position to a timestamp, update playhead, and emit seek signal."""
        mouse_point = self._plot_item.vb.mapSceneToView(scene_pos)
        timestamp_ms = max(0.0, float(mouse_point.x()) * 1000.0)
        self.set_playhead_ms(timestamp_ms)
        self.timestamp_clicked.emit(timestamp_ms)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Empty State Label ──────────────────────────────────────────
        import PyQt6.QtWidgets as QtWidgets
        self._empty_label = QtWidgets.QLabel("No timeline data available for this session.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._empty_label.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px;"
        )
        layout.addWidget(self._empty_label)

        # ── Plot Widget ────────────────────────────────────────────────
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(COLOR_CARD)
        self._plot_widget.setAntialiasing(True)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)

        # Disable all mouse-driven panning, zooming, and wheel scrolling.
        self._plot_widget.setMouseEnabled(x=False, y=False)

        self._plot_item = self._plot_widget.getPlotItem()
        self._plot_item.setLabel("bottom", "Time", units="s", color=COLOR_FONT_MUTED)
        self._plot_item.setLabel(
            "left", "% Change from Baseline", color=COLOR_FONT_MUTED
        )
        self._plot_widget.setYRange(-50.0, 150.0, padding=0)

        # Add a zero-baseline reference line.
        zero_line = pg.InfiniteLine(
            angle=0, pos=0, movable=False,
            pen=pg.mkPen(color=COLOR_BORDER, width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_item.addItem(zero_line, ignoreBounds=True)

        # Style axes.
        for ax in ("left", "bottom"):
            axis = self._plot_item.getAxis(ax)
            axis.setPen(pg.mkPen(color=COLOR_BORDER, width=1))
            axis.setTextPen(pg.mkPen(color=COLOR_FONT_MUTED))

        # ── Data curves ────────────────────────────────────────────────
        self._stress_curve = self._plot_item.plot(
            pen=pg.mkPen(color=COLOR_PRIMARY, width=2.5),
            name="Stress (RMSSD %)",
        )
        self._pupil_curve = self._plot_item.plot(
            pen=pg.mkPen(color=COLOR_WARNING, width=2.5),
            name="Pupil Dilation (%)",
        )

        # ── Persistent playhead line (follows video position) ──────────
        self._playhead_line = pg.InfiniteLine(
            angle=90, pos=0, movable=False,
            pen=pg.mkPen(color="#AAAAAA", width=2),
        )
        self._plot_item.addItem(self._playhead_line, ignoreBounds=True)

        # ── Hover hairline (follows mouse) ─────────────────────────────
        self._v_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=COLOR_BORDER, width=1, style=Qt.PenStyle.DotLine),
        )
        self._plot_item.addItem(self._v_line, ignoreBounds=True)

        def _mouse_moved(evt):
            pos = evt[0]
            if self._plot_item.sceneBoundingRect().contains(pos):
                mouse_point = self._plot_item.vb.mapSceneToView(pos)
                self._v_line.setPos(mouse_point.x())
                # Click-and-drag seek support.
                if QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
                    self._seek_to_scene_pos(pos)

        self._proxy = pg.SignalProxy(
            self._plot_item.scene().sigMouseMoved, rateLimit=60, slot=_mouse_moved
        )
        self._plot_item.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        layout.addWidget(self._plot_widget)
        self._show_empty_state(True)
