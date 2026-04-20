"""Compact semicircular gauge with a needle pointer."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from app.ui.theme import COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING
from app.utils.config import STRESS_RMSSD_PCT_MILD_THRESHOLD, STRESS_RMSSD_PCT_STABLE_THRESHOLD


class NeedleGauge(QWidget):
    """Semicircle gauge: needle position in [0, 1] (0 = low stress, 1 = high).

    Coloured arcs match RMSSD drop bands (same as ``STRESS_RMSSD_PCT_*``):
    stable 0–10 %, mild 10–40 %, high 40–100 % of full-scale drop (100 % = needle right).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value: float = 0.0
        self.setMinimumSize(96, 56)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h - 6.0
        radius = min(w * 0.44, h * 0.88)

        rect_x = cx - radius
        rect_y = cy - radius
        rect_w = radius * 2.0
        rect_h = radius * 2.0

        # Arc degrees: stable = 0..-10% → 10% of scale; mild = next 30%; high = rest 60%.
        stable_mag = -STRESS_RMSSD_PCT_STABLE_THRESHOLD
        mild_mag = -STRESS_RMSSD_PCT_MILD_THRESHOLD - stable_mag
        high_mag = 100.0 + STRESS_RMSSD_PCT_MILD_THRESHOLD
        total = stable_mag + mild_mag + high_mag
        stable_deg = 180.0 * stable_mag / total
        mild_deg = 180.0 * mild_mag / total
        high_deg = 180.0 - stable_deg - mild_deg

        # Track
        track_pen = QPen(QColor("#E5EAF5"), 8)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(int(rect_x), int(rect_y), int(rect_w), int(rect_h), 0, 180 * 16)

        # Right → left: high (red), mild (amber), stable (green); matches needle 0=left, 1=right.
        zone_pen = QPen(QColor(COLOR_DANGER), 8)
        zone_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(zone_pen)
        painter.drawArc(int(rect_x), int(rect_y), int(rect_w), int(rect_h), 0, int(high_deg * 16))

        zone_pen.setColor(QColor(COLOR_WARNING))
        painter.setPen(zone_pen)
        painter.drawArc(
            int(rect_x), int(rect_y), int(rect_w), int(rect_h), int(high_deg * 16), int(mild_deg * 16)
        )

        zone_pen.setColor(QColor(COLOR_SUCCESS))
        painter.setPen(zone_pen)
        painter.drawArc(
            int(rect_x),
            int(rect_y),
            int(rect_w),
            int(rect_h),
            int((high_deg + mild_deg) * 16),
            int(stable_deg * 16),
        )

        angle_deg = 180.0 - (self._value * 180.0)
        angle_rad = math.radians(angle_deg)
        needle_len = radius * 0.80
        nx = cx + needle_len * math.cos(angle_rad)
        ny = cy - needle_len * math.sin(angle_rad)

        needle_pen = QPen(QColor("#142970"), 3)
        needle_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(needle_pen)
        painter.drawLine(int(cx), int(cy), int(nx), int(ny))

        hub_pen = QPen(QColor("#142970"), 1)
        painter.setPen(hub_pen)
        painter.setBrush(QColor("#142970"))
        painter.drawEllipse(int(cx - 4), int(cy - 4), 8, 8)
