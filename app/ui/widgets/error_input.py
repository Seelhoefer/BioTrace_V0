"""Manual error counter widget for Live View.

Allows trainers to manually record wire-touch errors using plus/minus buttons.
Acts as a pure view: emits increment/decrement requests and displays the
current count from an external source (SessionManager).
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt

from app.ui.theme import (
    COLOR_PRIMARY,
    COLOR_BORDER,
    COLOR_CARD,
    FONT_HEADING_2,
    WEIGHT_BOLD,
    SPACE_1,
    RADIUS_LG,
)


class ErrorInputWidget(QWidget):
    """Widget for manual error (wire-touch) counting in Live View.

    Signals:
        plus_requested: Emitted when the '+' button is clicked.
        minus_requested: Emitted when the '−' button is clicked.
    """

    plus_requested = pyqtSignal()
    minus_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the error input widget."""
        super().__init__(parent)
        self._count = 0
        self._init_ui()

    def _init_ui(self) -> None:
        """Build the plus/minus counter UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_1)

        # Minus button
        self._minus_btn = QPushButton("−")
        self._minus_btn.setFixedSize(44, 44)
        self._minus_btn.setStyleSheet(
            f"QPushButton {{ background: {COLOR_CARD}; color: {COLOR_PRIMARY}; "
            f"border: 2px solid {COLOR_BORDER}; border-radius: {RADIUS_LG}px; "
            f"font-size: 20px; font-weight: {WEIGHT_BOLD}; padding-bottom: 2px; }}"
            f"QPushButton:hover {{ background: {COLOR_BORDER}; }}"
            f"QPushButton:disabled {{ color: {COLOR_BORDER}; }}"
        )
        self._minus_btn.clicked.connect(self.minus_requested)

        # Count label
        self._count_label = QLabel("0")
        self._count_label.setFixedWidth(44)
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_label.setStyleSheet(
            f"font-size: {FONT_HEADING_2}px; font-weight: {WEIGHT_BOLD}; "
            f"color: {COLOR_PRIMARY};"
        )

        # Plus button
        self._plus_btn = QPushButton("+")
        self._plus_btn.setFixedSize(44, 44)
        self._plus_btn.setStyleSheet(
            f"QPushButton {{ background: {COLOR_CARD}; color: {COLOR_PRIMARY}; "
            f"border: 2px solid {COLOR_BORDER}; border-radius: {RADIUS_LG}px; "
            f"font-size: 20px; font-weight: {WEIGHT_BOLD}; padding-bottom: 2px; }}"
            f"QPushButton:hover {{ background: {COLOR_BORDER}; }}"
        )
        self._plus_btn.clicked.connect(self.plus_requested)

        layout.addWidget(self._minus_btn)
        layout.addWidget(self._count_label)
        layout.addWidget(self._plus_btn)

        # Initial state
        self._update_display()

    @pyqtSlot(int)
    def set_count(self, count: int) -> None:
        """Update the displayed count.

        Args:
            count: The current error count from SessionManager.
        """
        self._count = count
        self._update_display()

    @pyqtSlot()
    def increment_from_hardware(self) -> None:
        """Slot for hardware signals (Phase 6b hook).

        Simply emits plus_requested to keep the SessionManager as the source of truth.
        """
        self.plus_requested.emit()

    def reset(self) -> None:
        """Reset the display to zero."""
        self._count = 0
        self._update_display()

    def _update_display(self) -> None:
        """Update the label text and enable/disable the minus button."""
        self._count_label.setText(str(self._count))
        self._minus_btn.setEnabled(self._count > 0)
