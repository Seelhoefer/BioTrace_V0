from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget, QSizePolicy

class LevelBar(QWidget):
    """A vertical bar widget that fills up from bottom to top based on a percentage."""

    def __init__(
        self,
        accent_color: str,
        track_color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value: float = 0.0  # 0.0 to 1.0
        self._accent_color = accent_color
        self._track_color = track_color
        
        # Default sizing for a vertical bar
        self.setFixedWidth(32)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def set_value(self, value: float) -> None:
        """Update the fill level (normalized 0.0 to 1.0)."""
        self._value = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        r = rect.width() / 2

        # Draw track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._track_color))
        painter.drawRoundedRect(rect, r, r)

        # Draw fill (from bottom up)
        if self._value > 0.001:
            fill_height = rect.height() * self._value
            # Ensure the fill is at least as high as the width for visual consistency at low values
            fill_height = max(fill_height, rect.width())
            
            fill_rect = QRectF(
                0, 
                rect.height() - fill_height, 
                rect.width(), 
                fill_height
            )
            painter.setBrush(QColor(self._accent_color))
            painter.drawRoundedRect(fill_rect, r, r)
