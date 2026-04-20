from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget
from app.ui.theme import COLOR_FONT, FONT_HEADING_2

class DonutGauge(QWidget):
    """Circular or semi-circular gauge with centered text.
    
    Args:
        value: Normalized value [0.0, 1.0].
        accent_color: Hex color for the active arc.
        track_color: Hex color for the background track.
        center_text: Label displayed in the middle.
        size: Diameter of the widget.
        half_circle: If True, renders as a 180-degree top arc.
    """

    def __init__(
        self,
        value: float,
        accent_color: str,
        track_color: str,
        center_text: str,
        size: int = 146,
        half_circle: bool = False,
        text_color: str | None = None,
        center_text_pixel_size: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self._accent_color = accent_color
        self._track_color = track_color
        self._center_text = center_text
        self._half_circle = half_circle
        self._text_color = text_color or COLOR_FONT
        self._center_text_pixel_size = center_text_pixel_size or FONT_HEADING_2
        
        if half_circle:
            # Semi-circle plus space for the text which is placed near the baseline
            self.setFixedSize(size, size // 2 + 30)
        else:
            self.setFixedSize(size, size)

    def set_value(self, value: float, center_text: str) -> None:
        """Set normalized value and center label text."""
        self._value = max(0.0, min(1.0, value))
        self._center_text = center_text
        self.update()

    def set_accent_color(self, accent_color: str) -> None:
        """Update active arc color at runtime."""
        self._accent_color = accent_color
        self.update()

    def set_center_text_pixel_size(self, pixel_size: int) -> None:
        """Set center-label font size in pixels."""
        self._center_text_pixel_size = max(8, int(pixel_size))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        """Render track and arc with anti-aliased painting."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Helper to handle rgba(r, g, b, a) strings safely
        def get_qcolor(color_str: str) -> QColor:
            if color_str.startswith("rgba"):
                # Simple parser for rgba(255, 255, 255, 0.5) or rgba(255, 255, 255, 50)
                parts = color_str.replace("rgba(", "").replace(")", "").split(",")
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                a_str = parts[3].strip()
                if "." in a_str or a_str == "0" or a_str == "1":
                    a = int(float(a_str) * 255)
                else:
                    a = int(a_str)
                return QColor(r, g, b, a)
            return QColor(color_str)

        # Padding for the stroke width
        padding = 15
        
        if self._half_circle:
            width = self.width()
            # Draw the arc in the top half of a square area
            arc_rect = QRectF(padding, padding, width - 2*padding, width - 2*padding)
            
            start_angle = 180 * 16
            span_total = -180 * 16
            
            track_pen = QPen(get_qcolor(self._track_color), 18)
            track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(track_pen)
            painter.drawArc(arc_rect, start_angle, span_total)

            arc_pen = QPen(get_qcolor(self._accent_color), 18)
            arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(arc_pen)
            span_value = int(self._value * span_total)
            painter.drawArc(arc_rect, start_angle, span_value)

            # Center text relative to the baseline of the semi-circle
            painter.setPen(get_qcolor(self._text_color))
            center_font = painter.font()
            center_font.setPixelSize(self._center_text_pixel_size)
            center_font.setBold(True)
            painter.setFont(center_font)
            
            # The baseline of the arc is at y = padding + (width - 2*padding)/2
            baseline_y = padding + (width - 2*padding) / 2
            text_rect = QRectF(0, baseline_y - 20, width, 40)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._center_text)
            
        else:
            rect = self.rect().adjusted(padding, padding, -padding, -padding)

            track_pen = QPen(get_qcolor(self._track_color), 18)
            track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(track_pen)
            painter.drawArc(rect, 0, 360 * 16)

            arc_pen = QPen(get_qcolor(self._accent_color), 18)
            arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(arc_pen)
            span = int(self._value * 360 * 16)
            painter.drawArc(rect, 90 * 16, -span)

            painter.setPen(get_qcolor(self._text_color))
            center_font = painter.font()
            center_font.setPixelSize(self._center_text_pixel_size)
            center_font.setBold(True)
            painter.setFont(center_font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._center_text)
