"""MetricCard — reusable single-metric display widget.

A styled card (QFrame) that shows a label, a large current value, a unit
string, and an optional small subtitle.  The value animates smoothly on
update via a QPropertyAnimation on a custom ``_display_value`` property.

Usage::

    card = MetricCard(name="RMSSD", unit="ms", subtitle="HRV / Stress proxy")
    card.set_value(42.5)
"""

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import time
from collections import deque

from app.ui.theme import (
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_DANGER,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    FONT_METRIC_XL,
    FONT_SMALL,
    FONT_SUBTITLE,
    SPACE_1,
    get_icon,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class _MiniSparkline(QWidget):
    """Lightweight sparkline renderer for metric cards.
    
    Shows a trend of values over a fixed time window (e.g. 10 seconds).
    """

    def __init__(self, line_color: str, window_seconds: float = 10.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._line_color = line_color
        self._data: deque[tuple[float, float]] = deque(maxlen=1000)  # (timestamp, value)
        self._window_seconds = window_seconds
        self.setMinimumSize(120, 88)

    def append(self, value: float, timestamp: float | None = None) -> None:
        """Append a new value with timestamp and redraw."""
        now = timestamp if timestamp is not None else time.time()
        self._data.append((now, float(value)))
        self._prune(now)
        self.update()

    def _prune(self, now: float) -> None:
        """Remove samples older than the window duration."""
        cutoff = now - self._window_seconds
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def clear(self) -> None:
        """Clear plotted values."""
        self._data.clear()
        self.update()

    def set_line_color(self, color: str) -> None:
        """Update line colour for the sparkline."""
        self._line_color = color
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(4, 8, -4, -8)

        baseline_pen = QPen(QColor(COLOR_BORDER), 1)
        baseline_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(baseline_pen)
        baseline_y = rect.top() + rect.height() * 0.55
        painter.drawLine(rect.left(), int(baseline_y), rect.right(), int(baseline_y))

        if len(self._data) < 2:
            return

        vals = [d[1] for d in self._data]
        min_v = min(vals)
        max_v = max(vals)
        span = max(max_v - min_v, 1e-6)

        path = QPainterPath()
        count = len(self._data)
        
        # Express x-axis as relative to the most recent timestamp
        latest_ts = self._data[-1][0]
        
        for i, (ts, value) in enumerate(self._data):
            # X coordinate based on time relative to window
            # Most recent sample is at rect.right()
            # Sample exactly window_seconds ago is at rect.left()
            rel_time = ts - latest_ts  # will be 0 to -window_seconds
            x = rect.right() + (rel_time / self._window_seconds) * rect.width()
            x = max(rect.left(), x) # clamp to left edge if data is old

            norm = (value - min_v) / span
            y = rect.bottom() - norm * rect.height()
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        line_pen = QPen(QColor(self._line_color), 2)
        painter.setPen(line_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)


class MetricCard(QFrame):
    """A polished metric display card with smooth value transitions.

    Args:
        name: Short metric name displayed at the top (e.g. ``"RMSSD"``).
        unit: Unit string appended to the value (e.g. ``"ms"``).
        subtitle: Small explanatory text below the value.
        decimals: Number of decimal places to display (default 1).
        show_sparkline: Whether to show the trend sparkline on the right.
        parent: Optional parent widget.

    Signals:
        value_changed (float): Emitted when ``set_value()`` is called.
    """

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        name: str,
        unit: str = "",
        subtitle: str = "",
        decimals: int = 1,
        show_sparkline: bool = True,
        window_seconds: float = 10.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(220)
        self.setMinimumWidth(200)

        self._name = name
        self._unit = unit
        self._subtitle = subtitle
        self._decimals = decimals
        self._show_sparkline = show_sparkline
        self._window_seconds = window_seconds
        self._raw_value: float = 0.0
        self._display_value: float = 0.0
        self._has_data: bool = False
        self._accent_color: str = COLOR_PRIMARY

        self._build_ui()
        self._animation = QPropertyAnimation(self, b"display_value", self)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.setDuration(400)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        self._name_label = QLabel(self._name)
        self._name_label.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px; font-weight: 700; letter-spacing: 2px;"
        )
        header_row.addWidget(self._name_label)
        header_row.addStretch(1)

        icon_label = QLabel()
        icon_label.setPixmap(get_icon(self._icon_for_metric(self._name), color="#89A6DA").pixmap(14, 14))
        header_row.addWidget(icon_label)
        layout.addLayout(header_row)

        value_col = QVBoxLayout()
        value_col.setContentsMargins(0, 0, 0, 0)
        value_col.setSpacing(2)

        self._value_label = QLabel("—")
        self._value_label.setStyleSheet(
            f"color: {COLOR_PRIMARY}; font-size: {FONT_METRIC_XL}px; font-weight: 700;"
        )
        value_col.addWidget(self._value_label)

        self._subtitle_label = QLabel(self._subtitle)
        self._subtitle_label.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px;"
        )
        self._subtitle_label.setVisible(bool(self._subtitle))
        value_col.addWidget(self._subtitle_label)

        layout.addLayout(value_col)

        if self._show_sparkline:
            self._sparkline = _MiniSparkline(line_color=self._accent_color, window_seconds=self._window_seconds)
            layout.addWidget(self._sparkline, stretch=1)
        else:
            self._sparkline = None
            layout.addStretch(1)

    @staticmethod
    def _icon_for_metric(name: str) -> str:
        """Map metric title to a Phosphor icon name."""
        upper = name.upper()
        if "PUPIL" in upper:
            return "ph.eye-fill"
        if "HRV" in upper or "STRESS" in upper:
            return "ph.heartbeat-fill"
        if "SPEED" in upper:
            return "ph.lightning-fill"
        if "ACCURACY" in upper:
            return "ph.target-fill"
        if "ERROR" in upper:
            return "ph.warning-fill"
        if "WORKLOAD" in upper:
            return "ph.brain-fill"
        return "ph.activity"

    def paintEvent(self, event) -> None:
        """Paint event override."""
        super().paintEvent(event)

    # ------------------------------------------------------------------
    # Animated property
    # ------------------------------------------------------------------

    @pyqtProperty(float)
    def display_value(self) -> float:  # type: ignore[override]
        """The currently rendered (interpolated) value."""
        return self._display_value

    @display_value.setter  # type: ignore[override]
    def display_value(self, value: float) -> None:
        self._display_value = value
        self._refresh_label()

    def _refresh_label(self) -> None:
        """Re-render the value label with current display_value."""
        formatted = f"{self._display_value:.{self._decimals}f}"
        if self._unit:
            formatted = f"{formatted} {self._unit}"
        self._value_label.setText(formatted)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_value(self, value: float, timestamp: float | None = None) -> None:
        """Update the displayed metric value with a smooth animation.

        Args:
            value: New numeric value to display.
            timestamp: Optional Unix timestamp for the sparkline trend.
        """
        if self._sparkline is not None:
            self._sparkline.append(value, timestamp)

        if not self._has_data:
            self._has_data = True
            self._display_value = value
            self._raw_value = value
            self._refresh_label()
        else:
            self._animation.stop()
            self._animation.setStartValue(self._display_value)
            self._animation.setEndValue(value)
            self._animation.start()
            self._raw_value = value

        self.value_changed.emit(value)

    def set_unit(self, unit: str) -> None:
        """Change the unit string appended to the displayed value.

        Args:
            unit: New unit label (e.g. ``"%"`` or ``"px"``).
        """
        self._unit = unit
        self._refresh_label()

    def set_colour(self, colour: str) -> None:
        """Change the value label colour (e.g. for CLI alert zones).

        Args:
            colour: Hex colour string (e.g. ``"#E74C3C"``).
        """
        self._accent_color = colour
        self._value_label.setStyleSheet(
            f"color: {colour}; font-size: {FONT_METRIC_XL}px; font-weight: 700;"
        )
        if self._sparkline is not None:
            self._sparkline.set_line_color(colour)
        self.update()  # Trigger repaint

    def set_alert_colour_from_cli(self, cli: float) -> None:
        """Automatically colour the value based on CLI alert thresholds.

        Args:
            cli: Current CLI value in [0.0, 1.0].
        """
        if cli < 0.33:
            self.set_colour(COLOR_SUCCESS)
        elif cli < 0.66:
            self.set_colour(COLOR_WARNING)
        else:
            self.set_colour(COLOR_DANGER)

    def reset(self) -> None:
        """Reset the card to its initial placeholder state."""
        self._animation.stop()
        self._has_data = False
        self._display_value = 0.0
        self._raw_value = 0.0
        self._value_label.setText("—")
        self._accent_color = COLOR_PRIMARY
        self._value_label.setStyleSheet(
            f"color: {COLOR_PRIMARY}; font-size: {FONT_METRIC_XL}px; font-weight: 700;"
        )
        if self._sparkline is not None:
            self._sparkline.set_line_color(COLOR_PRIMARY)
            self._sparkline.clear()
        self.update()
