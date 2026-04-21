"""Settings view for BioTrace.

Allows users to configure session preferences like calibration duration
and manage local data storage.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.storage.database import DatabaseManager
from app.ui.theme import (
    CARD_PADDING,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_DANGER_BG,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_PRIMARY_SUBTLE,
    COLOR_SUCCESS,
    CONTENT_PADDING_H,
    CONTENT_PADDING_V,
    FONT_BODY,
    FONT_CAPTION,
    FONT_SMALL,
    FONT_TITLE,
    ICON_SIZE_DEFAULT,
    RADIUS_LG,
    RADIUS_MD,
    SPACE_1,
    SPACE_2,
    SPACE_3,
    SPACE_4,
    WEIGHT_SEMIBOLD,
    get_icon,
)
from app.utils.config import get_calibration_duration, set_calibration_duration
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── _DataManagementCard ───────────────────────────────────────────────────


class _DataManagementCard(QFrame):
    """Calibration settings and data management controls.

    Signals:
        data_cleared: Emitted after all sessions have been deleted.
    """

    data_cleared = pyqtSignal()

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._db = db

        layout = QVBoxLayout(self)
        layout.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        layout.setSpacing(SPACE_4)

        # ── Calibration block ─────────────────────────────────────────
        cal_block = QFrame()
        cal_block.setStyleSheet(
            f"QFrame {{ background-color: {COLOR_PRIMARY_SUBTLE}; "
            f"border: 1px solid {COLOR_BORDER}; border-radius: {RADIUS_LG}px; }}"
        )
        cal_layout = QVBoxLayout(cal_block)
        cal_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        cal_layout.setSpacing(SPACE_2)

        cal_title = QLabel("Calibration Duration")
        cal_title.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_TITLE}px; font-weight: {WEIGHT_SEMIBOLD};"
        )
        cal_layout.addWidget(cal_title)

        cal_help = QLabel(
            "Set the duration (in seconds) for the resting baseline measurement "
            "before each session. A longer duration (e.g., 30–60s) improves HRV "
            "accuracy but takes more time."
        )
        cal_help.setObjectName("muted")
        cal_help.setWordWrap(True)
        cal_help.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px; line-height: 1.45;"
        )
        cal_layout.addWidget(cal_help)

        input_row = QHBoxLayout()
        input_row.setSpacing(SPACE_2)

        self._cal_spin = QSpinBox()
        self._cal_spin.setRange(60, 300)
        self._cal_spin.setValue(get_calibration_duration())
        self._cal_spin.setSuffix(" seconds")
        self._cal_spin.setMinimumHeight(40)
        self._cal_spin.setMinimumWidth(120)
        self._cal_spin.setStyleSheet(
            f"QSpinBox {{ "
            f"  background: #FFFFFF; "
            f"  border: 1px solid {COLOR_BORDER}; "
            f"  border-radius: {RADIUS_MD}px; "
            f"  padding: 0 8px; "
            f"  font-size: {FONT_BODY}px; "
            f"  color: {COLOR_FONT}; "
            f"}} "
        )
        input_row.addWidget(self._cal_spin)

        save_cal_btn = QPushButton("Save")
        save_cal_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_cal_btn.setMinimumHeight(40)
        save_cal_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_PRIMARY}; color: #FFFFFF; "
            f"border: none; border-radius: {RADIUS_MD}px; padding: 0 20px; "
            f"font-size: {FONT_BODY}px; font-weight: {WEIGHT_SEMIBOLD}; }}"
            f"QPushButton:hover {{ background-color: {COLOR_PRIMARY_HOVER}; }}"
        )
        save_cal_btn.clicked.connect(self._on_save_calibration)
        input_row.addWidget(save_cal_btn)
        input_row.addStretch(1)

        cal_layout.addLayout(input_row)

        self._cal_status = QLabel()
        self._cal_status.setVisible(False)
        cal_layout.addWidget(self._cal_status)

        layout.addWidget(cal_block)

        # ── Danger zone ──────────────────────────────────────────────
        danger_title = QLabel("Delete data")
        danger_title.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_TITLE}px; font-weight: {WEIGHT_SEMIBOLD};"
        )
        layout.addWidget(danger_title)

        delete_intro = QLabel(
            "Permanently removes every BioTrace session and all linked samples and "
            "calibrations from this device. This cannot be undone."
        )
        delete_intro.setWordWrap(True)
        delete_intro.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px; line-height: 1.45;"
        )
        layout.addWidget(delete_intro)

        self._delete_btn = QPushButton("  Delete all data")
        try:
            self._delete_btn.setIcon(get_icon("ph.trash-fill", color="#FFFFFF"))
        except Exception:
            pass
        self._delete_btn.setIconSize(QSize(ICON_SIZE_DEFAULT, ICON_SIZE_DEFAULT))
        self._delete_btn.setMinimumHeight(48)
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_DANGER}; color: #FFFFFF; "
            f"border: none; border-radius: {RADIUS_MD}px; padding: 10px 20px; "
            f"font-size: {FONT_BODY}px; font-weight: {WEIGHT_SEMIBOLD}; }}"
            f"QPushButton:hover {{ background-color: #DC2626; }}"
        )
        self._delete_btn.clicked.connect(self._show_confirm)
        layout.addWidget(self._delete_btn)

        # ── Confirmation banner (hidden by default) ───────────────────
        self._confirm_banner = QFrame()
        self._confirm_banner.setStyleSheet(
            f"QFrame {{ background-color: {COLOR_DANGER_BG}; "
            f"border: 1px solid {COLOR_DANGER}; border-radius: {RADIUS_MD}px; }}"
        )
        banner_layout = QHBoxLayout(self._confirm_banner)
        banner_layout.setContentsMargins(SPACE_2, SPACE_1, SPACE_2, SPACE_1)

        self._confirm_label = QLabel()
        self._confirm_label.setWordWrap(True)
        self._confirm_label.setStyleSheet(
            f"color: {COLOR_DANGER}; font-size: {FONT_BODY}px; "
            f"background: transparent; border: none;"
        )
        banner_layout.addWidget(self._confirm_label, stretch=1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self._hide_confirm)
        banner_layout.addWidget(cancel_btn)

        confirm_btn = QPushButton("Delete everything")
        confirm_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_DANGER}; color: #FFFFFF; "
            f"border-radius: {RADIUS_MD}px; padding: 10px 16px; }}"
            f"QPushButton:hover {{ background-color: #DC2626; }}"
        )
        confirm_btn.clicked.connect(self._on_confirm_delete)
        banner_layout.addWidget(confirm_btn)

        self._confirm_banner.setVisible(False)
        layout.addWidget(self._confirm_banner)

    def _on_save_calibration(self) -> None:
        """Save the new calibration duration and show a success message."""
        seconds = self._cal_spin.value()
        set_calibration_duration(seconds)
        self._cal_status.setText(f"Calibration duration saved as {seconds} seconds.")
        self._cal_status.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: {FONT_SMALL}px;")
        self._cal_status.setVisible(True)
        logger.info("Calibration duration updated to %d s.", seconds)

    def _session_count(self) -> int:
        from app.storage.session_repository import SessionRepository

        return len(SessionRepository(self._db).get_all_sessions())

    def _show_confirm(self) -> None:
        """Show the inline confirmation banner with session count."""
        n = self._session_count()
        self._confirm_label.setText(
            f"This permanently deletes all {n} session(s) on this computer. "
            "This action cannot be undone."
        )
        self._confirm_banner.setVisible(True)

    def _hide_confirm(self) -> None:
        """Collapse the confirmation banner."""
        self._confirm_banner.setVisible(False)

    def _on_confirm_delete(self) -> None:
        """Execute the delete via the repository layer and emit data_cleared."""
        from app.storage.session_repository import SessionRepository

        SessionRepository(self._db).delete_all_sessions()
        self._confirm_banner.setVisible(False)
        logger.info("All session data deleted by user.")
        self.data_cleared.emit()


# ── SettingsView ──────────────────────────────────────────────────────────


class SettingsView(QWidget):
    """Root settings page widget.

    Sections:
    - Data management (export all via system save dialog, delete all).

    Signals:
        data_cleared: Forwarded from _DataManagementCard after deletion.

    Args:
        db: Shared database manager (dependency injected from MainWindow).
        parent: Optional parent widget.
    """

    data_cleared = pyqtSignal()

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui(db)

    def _build_ui(self, db: DatabaseManager) -> None:
        """Construct all child widgets."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(
            CONTENT_PADDING_H,
            CONTENT_PADDING_V,
            CONTENT_PADDING_H,
            CONTENT_PADDING_V,
        )
        layout.setSpacing(SPACE_3)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        heading = QLabel("Settings")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        intro = QLabel(
            "Configure your session preferences and manage local data. Calibration "
            "settings affect the baseline recording duration before a session starts."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        intro.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px; max-width: 560px;"
        )
        layout.addWidget(intro)

        data_card = _DataManagementCard(db)
        data_card.data_cleared.connect(self.data_cleared)
        layout.addWidget(data_card)

        layout.addStretch(1)
