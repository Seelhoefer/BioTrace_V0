"""LiveChart — real-time scrolling pyqtgraph chart widget.

Renders one or more named data series as continuously scrolling time-series
plots.  The x-axis always shows the last ``window_seconds`` of data.

Usage::

    chart = LiveChart(
        series=["RMSSD", "PDI"],
        colours=["#3B579F", "#E74C3C"],
        y_label="Value",
        window_seconds=120,
    )
    chart.append("RMSSD", timestamp, 42.5)
    chart.append("PDI", timestamp, 0.12)

The widget can host any number of named series and is thread-safe for
appending data (since Qt signal slots are used).
"""

from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from app.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_CARD,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of data points kept per series (60 s × 10 Hz = 600 per min).
_MAX_POINTS: int = 10_000


def _configure_graph_style() -> None:
    """Apply BioTrace theme defaults to pyqtgraph once at module import."""
    pg.setConfigOptions(
        antialias=True,
        foreground=COLOR_FONT,
        background=COLOR_CARD,
    )


_configure_graph_style()


class LiveChart(QWidget):
    """Scrolling real-time chart with one or more named series.

    Args:
        series: List of series names, e.g. ``["RMSSD", "CLI"]``.
        colours: Hex colour strings matching the series list length.
        y_label: Label for the y-axis.
        y_range: Optional fixed ``(min, max)`` tuple for the y-axis.
                 If ``None``, the axis auto-scales.
        window_seconds: Width of the visible time window in seconds.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        series: list[str],
        colours: list[str],
        y_label: str = "",
        y_range: tuple[float, float] | None = None,
        window_seconds: int = 120,
        pen_styles: list[Qt.PenStyle] | None = None,
        pen_width: float = 2.5,
        allow_interaction: bool = True,
        transparent: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        if len(colours) < len(series):
            raise ValueError("colours list must be at least as long as series list")

        self._series_names = series
        self._colours = colours
        self._window_seconds = window_seconds
        self._y_range = y_range
        self._pen_styles = pen_styles or [Qt.PenStyle.SolidLine] * len(series)
        self._pen_width = pen_width
        self._allow_interaction = allow_interaction
        self._transparent = transparent

        # Data buffers — deques of (timestamp, value) pairs.
        self._timestamps: dict[str, deque[float]] = {s: deque(maxlen=_MAX_POINTS) for s in series}
        self._values: dict[str, deque[float]] = {s: deque(maxlen=_MAX_POINTS) for s in series}

        self._build_ui(y_label)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, y_label: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground((0,0,0,0) if self._transparent else COLOR_CARD)
        self._plot_widget.showGrid(x=False, y=True, alpha=0.3)
        self._plot_widget.getAxis("bottom").setLabel("Time (s)", color=COLOR_FONT_MUTED)
        self._plot_widget.getAxis("left").setLabel(y_label, color=COLOR_FONT_MUTED)
        if not self._allow_interaction:
            self._plot_widget.setMouseEnabled(x=False, y=False)
            self._plot_widget.setMenuEnabled(False)
            self._plot_widget.hideButtons()

        # Style both axes.
        for axis_name in ("bottom", "left"):
            axis = self._plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen(color=COLOR_BORDER, width=1))
            axis.setTextPen(pg.mkPen(color=COLOR_FONT_MUTED))

        if self._y_range is not None:
            self._plot_widget.setYRange(*self._y_range, padding=0.05)

        # Create one PlotDataItem per series.
        self._curves: dict[str, pg.PlotDataItem] = {}
        for name, colour, pen_style in zip(self._series_names, self._colours, self._pen_styles):
            pen = pg.mkPen(color=colour, width=self._pen_width, style=pen_style)
            curve = self._plot_widget.plot([], [], pen=pen, name=name)
            self._curves[name] = curve

        # Legend (only shown if more than one series).
        if len(self._series_names) > 1:
            legend = self._plot_widget.addLegend(offset=(10, 10))
            legend.setLabelTextColor(COLOR_FONT)

        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def append(self, series_name: str, timestamp: float, value: float) -> None:
        """Append a data point to a named series and refresh the chart.

        Args:
            series_name: The series to update (must exist in the ``series`` list).
            timestamp: Absolute Unix timestamp of the sample (seconds).
            value: The numeric value of the sample.
        """
        if series_name not in self._curves:
            logger.warning("LiveChart.append: unknown series '%s'", series_name)
            return

        self._timestamps[series_name].append(timestamp)
        self._values[series_name].append(value)
        self._refresh_curve(series_name)

    def _refresh_curve(self, series_name: str) -> None:
        """Redraw a single curve with the current buffer contents.

        The x-axis is re-expressed as elapsed seconds from the most recent
        sample so the chart always reads "0 = now, −120 = 2 min ago".
        """
        ts = list(self._timestamps[series_name])
        vals = list(self._values[series_name])

        if not ts:
            return

        latest = ts[-1]
        relative_ts = [t - latest for t in ts]

        # Restrict x-axis to the visible window.
        self._plot_widget.setXRange(-self._window_seconds, 0, padding=0)

        self._curves[series_name].setData(relative_ts, vals)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def clear_series(self, series_name: str) -> None:
        """Clear all data for a single series.

        Args:
            series_name: The series to clear.
        """
        if series_name in self._timestamps:
            self._timestamps[series_name].clear()
            self._values[series_name].clear()
            self._curves[series_name].setData([], [])

    def clear_all(self) -> None:
        """Clear all series data (call at session start)."""
        for name in self._series_names:
            self.clear_series(name)
        logger.info("LiveChart cleared.")

    def set_window_seconds(self, seconds: int) -> None:
        """Update the visible time window and refresh the view.

        Args:
            seconds: New width of the visible window in seconds.
        """
        self._window_seconds = seconds
        self._plot_widget.setXRange(-self._window_seconds, 0, padding=0)
        logger.info("LiveChart window updated to %d s", seconds)
