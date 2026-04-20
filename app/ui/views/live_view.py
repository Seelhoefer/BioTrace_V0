"""Live feedback view — real-time biofeedback during a training session.

Layout & user flow
------------------
The user arrives here directly from CalibrationView (after baseline is done
and MainWindow calls ``session_manager.start_session()``).  There is no
"Start Session" button — the session is already running on arrival.

Toolbar (always visible):
    ● LIVE SESSION  |  [Biofeedback]  [Camera + Bio]  |  PAUSE  END SESSION

Mode Biofeedback (default, index 0 in the mode stack):
    Top row: CORE STATE SYNTHESIS (circular gauges, 40 %) +
             SYNCHRONIZED STATE TIMELINE (live chart, 60 %)
    Bottom row: 4 metric cards — PUPIL DILATION · HRV (RMSSD) · TASK SPEED · ERROR RATE

Mode Camera + Bio (index 1 in the mode stack):
    Full-screen camera feed with a minimal recording control bar.

Dependency injection: call ``bind_session_manager()`` once after construction.
"""

import time
from collections import deque
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import (
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_DANGER,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_PRIMARY_SUBTLE,
    COLOR_SUCCESS,
    COLOR_WARNING,
    FONT_BODY,
    FONT_CAPTION,
    FONT_HEADING_2,
    FONT_METRIC_XL,
    FONT_SMALL,
    FONT_SUBTITLE,
    FONT_TITLE,
    FONT_DISPLAY,
    RADIUS_LG,
    SPACE_4,
    SPACE_3,
    SPACE_2,
    SPACE_1,
    SPACE_MICRO,
    CHART_HEIGHT_TIMELINE,
    get_icon,
)

# Sensor status badge colours
_SENSOR_CONNECTED_BG    = COLOR_SUCCESS   # #22C55E green fill
_SENSOR_DISCONNECTED_BG = "#E5E7EB"       # light gray fill
from app.ui.widgets.donut_gauge import DonutGauge
from app.ui.widgets.live_chart import LiveChart
from app.ui.widgets.metric_card import MetricCard
from app.ui.widgets.needle_gauge import NeedleGauge
from app.ui.widgets.video_feed import VideoFeed
from app.utils.config import (
    CAMERA_INDEX,
    USE_EYE_TRACKER,
    WORKLOAD_BASELINE_SECONDS,
    WORKLOAD_PUPIL_ROLLING_SECONDS,
    WORKLOAD_PUPIL_SMOOTHING_SECONDS,
    WORKLOAD_STATE_PERSIST_SECONDS,
    WORKLOAD_THRESHOLD_FACTOR,
)
from app.utils.config import VIDEO_RECORDINGS_DIR
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Chart trace colours (from design.md)
_COLOR_RMSSD = COLOR_SUCCESS
_COLOR_PDI   = COLOR_PRIMARY
_COLOR_CLI   = "#3B579F"
_COLOR_THRESHOLD = COLOR_WARNING

# Mode indices in the QStackedWidget
_MODE_BIOFEEDBACK = 0   # data-only (default)
_MODE_CAMERA      = 1   # camera + data overlay


class ModeSwitcher(QWidget):
    """Integrated mode switcher with icons and a sliding highlight.
    
    Shows both 'Biofeedback' and 'Camera' positions as distinct targets
    to feel more like a physical switch between two modes than a binary toggle.
    """
    toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checked: bool = False
        
        # Theme properties
        self._active_icon_color = "#FFFFFF"
        self._inactive_icon_color = COLOR_PRIMARY
        
        self.setFixedSize(100, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Background track
        self._track = QFrame(self)
        self._track.setGeometry(0, 0, 100, 36)
        
        # Static Icons (Visible when not covered by thumb)
        self._bio_icon = QLabel(self)
        self._bio_icon.move(12, 9)
        
        self._cam_icon = QLabel(self)
        self._cam_icon.move(70, 9)

        # Sliding thumb
        self._thumb = QFrame(self)
        self._thumb.setFixedSize(44, 28)
        
        # Active icon on top of the thumb
        self._active_icon = QLabel(self._thumb)
        self._active_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._active_icon.setFixedSize(44, 28)
        
        # Set default theme (will be updated by set_theme)
        self.set_theme(False)
        self._update_visuals()

    def isChecked(self) -> bool:
        """Return current checked state (True = Camera mode)."""
        return self._checked

    def setChecked(self, checked: bool) -> None:
        """Set checked state and update visuals without emitting signal."""
        if self._checked != checked:
            self._checked = checked
            self._update_visuals()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            # Click detection: if on left half, set bio; if on right half, set cam
            new_state = event.pos().x() > self.width() // 2
            if new_state != self._checked:
                self._checked = new_state
                self._update_visuals()
                self.toggled.emit(self._checked)
            event.accept()
            return
        super().mousePressEvent(event)

    def _update_visuals(self) -> None:
        """Move the thumb and update icon visibility/color."""
        if self._checked:
            self._thumb.move(52, 4)
            self._active_icon.setPixmap(get_icon("ph.video-camera", color=self._active_icon_color).pixmap(18, 18))
            self._cam_icon.hide()
            self._bio_icon.show()
        else:
            self._thumb.move(4, 4)
            self._active_icon.setPixmap(get_icon("ph.activity", color=self._active_icon_color).pixmap(18, 18))
            self._bio_icon.hide()
            self._cam_icon.show()

    def set_theme(self, is_video: bool) -> None:
        """Update track and icon colors for the current theme.

        is_video: True if white theme (for camera), False for standard theme.
        """
        if is_video:
            # Video View Style: White track, Blue thumb, White active icon, Blue inactive icons
            track_bg = "#FFFFFF"
            track_border = "transparent"
            thumb_bg = COLOR_PRIMARY
            self._active_icon_color = "#FFFFFF"
            self._inactive_icon_color = COLOR_PRIMARY
        else:
            # Dashboard View Style: Light track, Blue thumb, White active icon, Muted inactive icons
            track_bg = COLOR_PRIMARY_SUBTLE
            track_border = COLOR_BORDER
            thumb_bg = COLOR_PRIMARY
            self._active_icon_color = "#FFFFFF"
            self._inactive_icon_color = COLOR_FONT_MUTED

        self._track.setStyleSheet(
            f"background-color: {track_bg}; border: 1px solid {track_border}; border-radius: 18px;"
        )
        self._bio_icon.setPixmap(get_icon("ph.activity", color=self._inactive_icon_color).pixmap(18, 18))
        self._cam_icon.setPixmap(get_icon("ph.video-camera", color=self._inactive_icon_color).pixmap(18, 18))
        self._thumb.setStyleSheet(f"background-color: {thumb_bg}; border-radius: 14px;")
        
        self._update_visuals()



class LiveView(QWidget):
    """Real-time biofeedback view during an active training session.

    This view does NOT own a ``SessionManager``.  After construction,
    ``MainWindow`` calls ``bind_session_manager()`` to wire signals.

    The session always starts externally (via ``MainWindow._on_proceed_to_live``).
    The only session control available here is END SESSION.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_active: bool = False
        self._elapsed_seconds: int = 0
        self._session_manager = None
        self._session_id: int | None = None
        self._recording_path: Path | None = None
        self._current_camera_index: int = CAMERA_INDEX
        self._baseline_rmssd: float = 0.0
        self._baseline_pupil_px: float = 0.0
        self._has_pupil_baseline: bool = False  # True once calibration sets a baseline
        self._bootstrap_pupil_samples: deque[tuple[float, float]] = deque()
        self._pupil_smoothing_window: deque[tuple[float, float]] = deque()
        self._pupil_rolling_window: deque[tuple[float, float]] = deque()
        self._workload_state_is_elevated: bool = False
        self._pending_workload_state: bool | None = None
        self._pending_workload_since: float | None = None
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._tick_clock)

        self._build_ui()
        self._setup_shortcuts()

    def _setup_shortcuts(self) -> None:
        """Add keyboard shortcuts for easy view and source switching."""
        # Toggle between Biofeedback and Camera + Bio modes (Hotkey: C)
        self._mode_toggle_shortcut = QShortcut(QKeySequence("C"), self)
        self._mode_toggle_shortcut.activated.connect(
            self._toggle_mode_via_shortcut
        )
        
        # Switch camera source (Hotkey: S)
        self._source_toggle_shortcut = QShortcut(QKeySequence("S"), self)
        self._source_toggle_shortcut.activated.connect(self._on_switch_camera)

    def _toggle_mode_via_shortcut(self) -> None:
        """Cycle mode when shortcut is pressed."""
        new_state = not self._mode_switcher.isChecked()
        self._mode_switcher.setChecked(new_state)
        self._set_mode(_MODE_CAMERA if new_state else _MODE_BIOFEEDBACK)

    # ------------------------------------------------------------------
    # SessionManager binding (called by MainWindow after construction)
    # ------------------------------------------------------------------

    def bind_session_manager(self, session_manager) -> None:
        """Inject the SessionManager and wire its signals to this view.

        Args:
            session_manager: A :class:`~app.core.session.SessionManager` instance.
        """
        self._session_manager = session_manager
        session_manager.rmssd_updated.connect(self.on_rmssd_updated)
        session_manager.pdi_updated.connect(self.on_pdi_updated)
        session_manager.cli_updated.connect(self.on_cli_updated)
        session_manager.bpm_updated.connect(self.on_bpm_updated)
        session_manager.hrv_connection_changed.connect(self._on_hrv_connection_changed)
        session_manager.eye_connection_changed.connect(self._on_eye_connection_changed)
        session_manager.calibration_complete.connect(self._on_calibration_complete)
        session_manager.session_started.connect(self._on_session_started)
        session_manager.session_ended.connect(self._on_session_ended)
        session_manager.session_paused.connect(self._on_session_paused)
        session_manager.session_resumed.connect(self._on_session_resumed)
        session_manager.error_count_updated.connect(self.on_error_count_updated)
        self._sync_pupil_baseline_state()
        logger.info(
            "Live timeline normalization: percent change from baseline mean "
            "(percent_delta=((value-baseline_mean)/baseline_mean)*100)."
        )
        logger.info("LiveView bound to SessionManager.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Layered layout: Toolbar floats over the Mode Stack
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Base: Mode stack
        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._build_mode_biofeedback())  # index 0
        self._mode_stack.addWidget(self._build_mode_camera())       # index 1
        self._mode_stack.setCurrentIndex(_MODE_BIOFEEDBACK)
        layout.addWidget(self._mode_stack, 0, 0)

        # Overlay: Top toolbar
        toolbar_container = QWidget()
        toolbar_container.setObjectName("live_toolbar")
        toolbar_container.setFixedHeight(80)
        toolbar_container.setStyleSheet(
            f"#live_toolbar {{ background: transparent; border: none; }}"
        )
        toolbar_container.setLayout(self._build_toolbar())
        layout.addWidget(toolbar_container, 0, 0, Qt.AlignmentFlag.AlignTop)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(SPACE_3, 0, SPACE_3, 0)
        row.setSpacing(SPACE_2)

        # ── Left: title + sensor status badges ────────────────────────────
        self._live_label = QLabel("LIVE SESSION")
        self._live_label.setStyleSheet(
            f"font-size: {FONT_SMALL}px; font-weight: 700; letter-spacing: 1.5px;"
        )
        row.addWidget(self._live_label)

        row.addSpacing(SPACE_2)

        self._ecg_badge = self._make_status_badge("ph.heartbeat-fill")
        row.addWidget(self._ecg_badge)

        self._eye_badge = self._make_status_badge("ph.eye-fill")
        row.addWidget(self._eye_badge)

        row.addSpacing(SPACE_1)

        row.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        # ── Center: dashboard/camera toggle switch ───────────────────────
        self._mode_switcher = ModeSwitcher(self)
        self._mode_switcher.toggled.connect(self._on_mode_switch_toggled)
        row.addWidget(self._mode_switcher)

        row.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        # ── Right: session controls ─────────────────────────────────────
        # Pause button (fully round)
        self._pause_btn = QPushButton()
        self._pause_btn.setIconSize(QSize(18, 18))
        self._pause_btn.setFixedSize(44, 44)
        self._pause_btn.setCheckable(True)
        self._pause_btn.clicked.connect(self._toggle_pause)
        row.addWidget(self._pause_btn)

        # END SESSION button
        self._end_btn = QPushButton(" END SESSION")
        self._end_btn.setIconSize(QSize(16, 16))
        self._end_btn.setFixedHeight(40)
        self._end_btn.setMinimumWidth(130)
        self._end_btn.clicked.connect(self._end_session)
        row.addWidget(self._end_btn)

        # Initialize with bio theme (dark blue)
        self._update_toolbar_theme(is_video=False)

        return row

    def _update_toolbar_theme(self, is_video: bool) -> None:
        """Update toolbar colors based on current mode.
        
        is_video: True if Camera view (white), False if Bio view (dark blue).
        """
        if is_video:
            text_color = "#FFFFFF"
            border_color = "rgba(255, 255, 255, 0.4)"
            hover_bg = "rgba(255, 255, 255, 0.1)"
            btn_bg = "#FFFFFF"
            btn_text = COLOR_PRIMARY
        else:
            text_color = COLOR_PRIMARY
            border_color = COLOR_BORDER
            hover_bg = COLOR_PRIMARY_SUBTLE
            btn_bg = COLOR_PRIMARY
            btn_text = "#FFFFFF"

        # 1. Update Title
        self._live_label.setStyleSheet(
            f"color: {text_color}; font-size: {FONT_SMALL}px; "
            "font-weight: 700; letter-spacing: 1.5px;"
        )

        # 2. Update Mode Switcher
        self._mode_switcher.set_theme(is_video)

        # 3. Update Pause Button
        if is_video:
            pause_bg = "#FFFFFF"
            pause_icon_color = COLOR_PRIMARY
            pause_border = "none"
            pause_hover = COLOR_PRIMARY_SUBTLE
            pause_checked_bg = COLOR_PRIMARY
            pause_checked_icon = "#FFFFFF"
        else:
            pause_bg = "transparent"
            pause_icon_color = COLOR_PRIMARY
            pause_border = f"2px solid {COLOR_BORDER}"
            pause_hover = COLOR_PRIMARY_SUBTLE
            pause_checked_bg = COLOR_WARNING
            pause_checked_icon = "#FFFFFF"

        is_paused = self._pause_btn.isChecked()
        current_icon = "ph.play-fill" if is_paused else "ph.pause-fill"
        current_color = pause_checked_icon if is_paused else pause_icon_color
        
        self._pause_btn.setIcon(get_icon(current_icon, color=current_color))
        self._pause_btn.setStyleSheet(
            f"QPushButton {{ background-color: {pause_bg}; border: {pause_border}; border-radius: 22px; outline: none; }}"
            f"QPushButton:hover {{ background-color: {pause_hover}; }}"
            f"QPushButton:checked {{ background-color: {pause_checked_bg}; border-color: {pause_checked_bg}; }}"
        )

        # 4. Update End Button
        self._end_btn.setIcon(get_icon("ph.stop-circle-fill", color=btn_text))
        self._end_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {btn_bg};
                color: {btn_text};
                border: none;
                border-radius: 20px;
                padding: 0 18px;
                font-size: 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: {hover_bg if not is_video else COLOR_PRIMARY_SUBTLE};
            }}
            """
        )

    def _on_mode_switch_toggled(self, checked: bool) -> None:
        """Switch view mode when the center toggle changes."""
        self._set_mode(_MODE_CAMERA if checked else _MODE_BIOFEEDBACK)

    # ------------------------------------------------------------------
    # Mode Biofeedback — data-only (default)
    # ------------------------------------------------------------------

    def _build_mode_biofeedback(self) -> QWidget:
        """Build the Biofeedback (data-only) mode layout.

        Top row:
            - CORE STATE SYNTHESIS: two circular/gauge metric cards (40 %)
            - SYNCHRONIZED STATE TIMELINE: scrolling line chart (60 %)
        Bottom row:
            - 4 metric cards: PUPIL DILATION · HRV (RMSSD) · TASK SPEED · ERROR RATE
        """
        widget = QWidget()
        outer = QVBoxLayout(widget)
        # Add 80px margin to top to clear the floating toolbar
        outer.setContentsMargins(SPACE_3, 80, SPACE_3, SPACE_2)
        outer.setSpacing(SPACE_2)

        # ── Top row (Single Card) ───────────────────────────────────────
        top_card = self._make_card()
        top_row_layout = QHBoxLayout(top_card)
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        top_row_layout.setSpacing(0)

        # Left: Core State Synthesis panel (2/5 width)
        core_widget = QWidget()
        core_layout = QVBoxLayout(core_widget)
        core_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        core_layout.setSpacing(SPACE_1)

        core_header = QHBoxLayout()
        core_title = QLabel("CORE STATE SYNTHESIS")
        core_title.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px; "
            "font-weight: 700; letter-spacing: 2px;"
        )
        core_header.addWidget(core_title)
        core_header.addStretch()
        info_icon = QLabel()
        info_icon.setPixmap(get_icon("ph.info-fill", color=COLOR_FONT_MUTED).pixmap(14, 14))
        core_header.addWidget(info_icon)
        core_layout.addLayout(core_header)

        gauges_col = QVBoxLayout()
        gauges_col.setSpacing(SPACE_2)

        # Cognitive Workload Gauge
        self._workload_gauge, workload_panel = self._make_gauge_panel(
            title="COGNITIVE WORKLOAD",
            accent=COLOR_PRIMARY,
            track=COLOR_PRIMARY_SUBTLE
        )
        gauges_col.addWidget(workload_panel)

        # Physical Stress Gauge
        self._stress_gauge, stress_panel = self._make_gauge_panel(
            title="PHYSICAL STRESS",
            accent=_COLOR_CLI,
            track="#E5E7EB"
        )
        gauges_col.addWidget(stress_panel)
        core_layout.addLayout(gauges_col, stretch=1)

        top_row_layout.addWidget(core_widget, stretch=1)

        # Vertical Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet(f"color: {COLOR_BORDER};")
        top_row_layout.addWidget(divider)

        # Right: Synchronized State Timeline panel (3/5 width)
        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        timeline_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        timeline_layout.setSpacing(SPACE_1)

        tl_header = QHBoxLayout()
        tl_title = QLabel("Timeline")
        tl_title.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px; "
            "font-weight: 700; letter-spacing: 2px;"
        )
        tl_header.addWidget(tl_title)

        tl_header.addStretch()

        # Legend dots
        for label_text, color in [
            ("PUPIL %", _COLOR_PDI),
            ("THRESHOLD %", _COLOR_THRESHOLD),
            ("HRV %", _COLOR_RMSSD),
        ]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 9px;")
            tl_header.addWidget(dot)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px; font-weight: 600;"
            )
            tl_header.addWidget(lbl)

        timeline_layout.addLayout(tl_header)

        self._timeline_chart = LiveChart(
            series=["PUPIL", "THRESHOLD", "HRV"],
            colours=[_COLOR_PDI, _COLOR_THRESHOLD, _COLOR_RMSSD],
            y_label="Change from baseline (%)",
            y_range=(-50.0, 150.0),
            window_seconds=180,
            pen_styles=[
                Qt.PenStyle.SolidLine,
                Qt.PenStyle.DashLine,
                Qt.PenStyle.SolidLine,
            ],
            pen_width=3.5,
            allow_interaction=False,
            transparent=True,
        )
        self._timeline_chart.setMinimumHeight(CHART_HEIGHT_TIMELINE)
        timeline_layout.addWidget(self._timeline_chart, stretch=1)

        top_row_layout.addWidget(timeline_widget, stretch=2)
        outer.addWidget(top_card, stretch=1)

        # ── Bottom row: 4 metric cards ──────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(SPACE_2)

        self._pupil_card = MetricCard(name="PUPIL DILATION", unit="%", decimals=1)
        self._bpm_card = MetricCard(
            name="HEART RATE",
            unit="bpm",
            decimals=0,
            window_seconds=60.0,
        )
        self._rmssd_card = MetricCard(
            name="HRV (RMSSD)",
            unit="ms",
            decimals=1,
            window_seconds=60.0,
        )
        self._speed_card = MetricCard(
            name="TASK SPEED",
            unit="s",
            decimals=0,
            show_sparkline=False,
        )
        self._accuracy_card = MetricCard(
            name="ERROR RATE",
            unit="errors",
            decimals=0,
            show_sparkline=False,
        )

        for card in (self._pupil_card, self._bpm_card, self._rmssd_card,
                     self._speed_card, self._accuracy_card):
            card.setObjectName("card")
            card.setStyleSheet("background-color: transparent;")
            bottom_row.addWidget(card, stretch=1)

        outer.addLayout(bottom_row, stretch=1)
        return widget

    # ------------------------------------------------------------------
    # Mode Camera + Bio — full-screen camera with overlay
    # ------------------------------------------------------------------

    def _build_mode_camera(self) -> QWidget:
        """Build the Camera + Bio mode layout.

        Camera feed fills the area with a transparent HUD overlay.
        Metrics are positioned in a bottom row of equally sized cards.
        """
        widget = QWidget()
        widget.setStyleSheet("background: #000000;")

        # Layering: use QGridLayout to stack HUD on top of VideoFeed
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Base: Video feed
        self._video_feed = VideoFeed()
        layout.addWidget(self._video_feed, 0, 0)

        # HUD Overlay
        hud = QWidget()
        hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        hud.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        hud.setStyleSheet("background: transparent;")
        hud_layout = QVBoxLayout(hud)
        # Leave room for floating toolbar; cards occupy four video corners.
        hud_layout.setContentsMargins(SPACE_3, 100, SPACE_3, SPACE_3)
        hud_layout.setSpacing(0)

        title_style = (
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_BODY}px; "
            "font-weight: 700; letter-spacing: 1px;"
        )
        value_style = f"color: {COLOR_PRIMARY}; font-size: {FONT_TITLE}px; font-weight: 800;"

        def make_corner_card(title: str) -> tuple[QFrame, QVBoxLayout]:
            card = QFrame()
            card.setFixedWidth(170)
            card.setStyleSheet(
                f"background-color: {COLOR_CARD}; border: none; "
                f"border-radius: {RADIUS_LG}px;"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(SPACE_2, SPACE_2, SPACE_2, SPACE_2)
            card_layout.setSpacing(SPACE_1)

            label = QLabel(title)
            label.setStyleSheet(title_style)
            card_layout.addWidget(label)
            return card, card_layout

        # Top row: TIME (left) | BPM (right)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(0)

        time_card, time_layout = make_corner_card("TIME")
        self._cam_timer_lbl = QLabel("00:00")
        self._cam_timer_lbl.setStyleSheet(value_style)
        time_layout.addWidget(self._cam_timer_lbl)
        top_row.addWidget(time_card, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        top_row.addStretch(1)

        bpm_card, bpm_layout = make_corner_card("BPM")
        bpm_row = QHBoxLayout()
        bpm_row.setSpacing(SPACE_1)
        heart_icon = QLabel()
        heart_icon.setPixmap(get_icon("ph.heartbeat-fill", color=COLOR_PRIMARY).pixmap(16, 16))
        bpm_row.addWidget(heart_icon)
        self._cam_bpm_lbl = QLabel("—")
        self._cam_bpm_lbl.setStyleSheet(value_style)
        self._cam_bpm_lbl.setMinimumWidth(56)
        bpm_row.addWidget(self._cam_bpm_lbl)
        bpm_row.addStretch(1)
        bpm_layout.addLayout(bpm_row)
        top_row.addWidget(bpm_card, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        hud_layout.addLayout(top_row)

        hud_layout.addStretch(1)

        # Bottom row: LOAD (left) | STRESS (right)
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(0)

        load_card, load_layout = make_corner_card("LOAD")
        load_row = QHBoxLayout()
        load_row.setSpacing(SPACE_1)
        load_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._cam_workload_dot = QFrame()
        self._cam_workload_dot.setFixedSize(16, 16)
        self._cam_workload_dot.setStyleSheet(
            f"background-color: {COLOR_SUCCESS}; border-radius: 8px;"
        )
        load_row.addWidget(self._cam_workload_dot)
        self._cam_workload_value = QLabel("—")
        self._cam_workload_value.setStyleSheet(value_style)
        self._cam_workload_value.setMinimumWidth(56)
        load_row.addWidget(self._cam_workload_value)
        load_row.addStretch(1)
        load_layout.addLayout(load_row)
        bottom_row.addWidget(load_card, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        bottom_row.addStretch(1)

        stress_card, stress_layout = make_corner_card("STRESS")
        stress_row = QHBoxLayout()
        stress_row.setSpacing(SPACE_1)
        stress_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._cam_stress_bar = NeedleGauge()
        self._cam_stress_bar.setFixedSize(96, 56)
        stress_row.addWidget(self._cam_stress_bar)
        self._cam_stress_value = QLabel("—")
        self._cam_stress_value.setStyleSheet(value_style)
        self._cam_stress_value.setMinimumWidth(56)
        stress_row.addWidget(self._cam_stress_value)
        stress_row.addStretch(1)
        stress_layout.addLayout(stress_row)
        bottom_row.addWidget(stress_card, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        hud_layout.addLayout(bottom_row)

        layout.addWidget(hud, 0, 0)
        return widget

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _end_session(self) -> None:
        """End the current session via SessionManager."""
        if self._session_manager is not None:
            # Stop recording and capture the path.
            path = self._stop_camera_recording()
            self._session_manager.set_recording_path(path)

            self._session_manager.end_session()

    def _toggle_pause(self) -> None:
        """Toggle session pause state via SessionManager."""
        if self._session_manager is None:
            return

        if self._pause_btn.isChecked():
            self._session_manager.pause_session()
        else:
            self._session_manager.resume_session()

    @pyqtSlot(int)
    def _on_session_paused(self, _session_id: int) -> None:
        """Called when the session is paused."""
        self._clock_timer.stop()
        self._pause_btn.setChecked(True)
        is_video = self._mode_stack.currentIndex() == _MODE_CAMERA
        self._update_toolbar_theme(is_video)
        logger.info("LiveView: session paused UI updated.")

    @pyqtSlot(int)
    def _on_session_resumed(self, _session_id: int) -> None:
        """Called when the session is resumed."""
        self._clock_timer.start()
        self._pause_btn.setChecked(False)
        is_video = self._mode_stack.currentIndex() == _MODE_CAMERA
        self._update_toolbar_theme(is_video)
        logger.info("LiveView: session resumed UI updated.")

    def cleanup(self) -> None:
        """Stop background activity before the view is destroyed."""
        self._video_feed.stop()

    @pyqtSlot(int)
    def _on_session_started(self, session_id: int) -> None:
        """Called when SessionManager emits ``session_started``."""
        logger.info("LiveView: session %d started.", session_id)
        self._session_id = session_id
        self._session_active = True
        self._sync_pupil_baseline_state()
        self._reset_widgets()
        self.on_error_count_updated(0)
        self._start_clock()

        # Always start the feed first (feed must be active before recording can begin),
        # then start recording — the camera runs in the background regardless of which
        # display mode (Biofeedback vs Camera+Bio) is currently selected.
        self._video_feed.start(camera_index=self._current_camera_index)
        self._start_camera_recording()

    @pyqtSlot(int)
    def _on_session_ended(self, session_id: int) -> None:
        """Called when SessionManager emits ``session_ended``.

        MainWindow also listens to this signal and navigates to the
        Post-Session view — no navigation logic belongs here.
        """
        logger.info("LiveView: session %d ended.", session_id)
        self._stop_camera_recording()
        self._session_active = False
        self._stop_clock()
        self._video_feed.stop()
        self._session_id = None

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def _start_clock(self) -> None:
        self._elapsed_seconds = 0
        self._cam_timer_lbl.setText("00:00")
        self._clock_timer.start()

    def _stop_clock(self) -> None:
        self._clock_timer.stop()

    def _tick_clock(self) -> None:
        self._elapsed_seconds += 1
        m = self._elapsed_seconds // 60
        s = self._elapsed_seconds % 60

        # Update TASK SPEED card with duration in seconds
        self._speed_card.set_value(float(self._elapsed_seconds))

        # Update camera overlay timer label
        time_str = f"{m:02d}:{s:02d}"
        self._cam_timer_lbl.setText(time_str)

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _set_mode(self, index: int) -> None:
        """Switch between Biofeedback and Camera+Bio modes.

        Args:
            index: ``_MODE_BIOFEEDBACK`` (0) or ``_MODE_CAMERA`` (1).
        """
        self._mode_stack.setCurrentIndex(index)
        self._mode_switcher.setChecked(index == _MODE_CAMERA)
        
        # Update toolbar colors for the new mode
        self._update_toolbar_theme(is_video=(index == _MODE_CAMERA))

        # The camera feed and recording run continuously for the whole session
        # regardless of display mode.  Only the *display panel* switches here —
        # never touch the feed or recording lifecycle on a mode switch.

        logger.info(
            "Live view mode: %s",
            "Biofeedback" if index == _MODE_BIOFEEDBACK else "Camera + Bio",
        )

    def _on_switch_camera(self) -> None:
        """Cycle through external USB camera indices (1 -> 2 -> 1).

        Index 0 is intentionally excluded — it maps to the built-in laptop
        camera, which is never used by BioTrace.
        """
        self._current_camera_index = 2 if self._current_camera_index == 1 else 1

        if self._session_active:
            # Re-start feed and recording on new camera index (feed always runs during session).
            self._stop_camera_recording()
            self._video_feed.start(camera_index=self._current_camera_index)
            self._start_camera_recording()
            
        logger.info("Switched to camera index %d", self._current_camera_index)

    # ------------------------------------------------------------------
    # Public data-update slots (called via SessionManager signals)
    # ------------------------------------------------------------------

    @pyqtSlot(float, float)
    def on_rmssd_updated(self, rmssd: float, timestamp: float) -> None:
        """Receive RMSSD update and refresh all relevant widgets.

        Args:
            rmssd: RMSSD in milliseconds.
            timestamp: Unix timestamp of the sample.
        """
        self._rmssd_card.set_value(rmssd, timestamp)

        baseline_rmssd = self._baseline_rmssd
        if baseline_rmssd <= 0.0 and self._session_manager is not None:
            baseline_rmssd = float(getattr(self._session_manager, "baseline_rmssd", 0.0) or 0.0)
            self._baseline_rmssd = baseline_rmssd

        if baseline_rmssd > 0.0:
            # RMSSD trend as % change from calibration baseline (same bands as
            # ``STRESS_RMSSD_PCT_*`` in config and ``NeedleGauge`` arc widths).
            hrv_percent_change = ((rmssd - baseline_rmssd) / baseline_rmssd) * 100.0
            # Higher stress when RMSSD drops below baseline; stress index 0–100 = −Δ%.
            stress_pct = max(0.0, min(100.0, -hrv_percent_change))
        else:
            # Fallback range when no calibration baseline exists yet.
            hrv_percent_change = max(0.0, min(100.0, ((rmssd - 20.0) / 60.0) * 100.0))
            stress_pct = 100.0 - hrv_percent_change

        self._stress_gauge.set_value(stress_pct / 100.0, f"{stress_pct:.0f}%")
        self._cam_stress_bar.set_value(stress_pct / 100.0)
        self._cam_stress_value.setText(f"{stress_pct:.0f}%")

        # Feed timeline using percent change from baseline mean.
        self._timeline_chart.append("HRV", timestamp, hrv_percent_change)

    @pyqtSlot(float, float)
    def on_pdi_updated(self, pdi: float, timestamp: float) -> None:
        """Receive PDI update (or raw diameter) and refresh pupil dilation card.

        When a calibration baseline exists, ``pdi`` is the Pupil Dilation Index
        (dimensionless) and is displayed scaled to a %-like index.
        Before calibration, ``pdi`` carries the raw diameter in px so that the
        live view shows sensor activity even without a baseline.

        Args:
            pdi: PDI ratio (with baseline) or raw diameter in px (without baseline).
            timestamp: Unix timestamp of the sample.
        """
        # If calibration baseline is missing, bootstrap from first ~5s of raw pupil.
        if not self._has_pupil_baseline:
            if pdi > 0.0:
                self._bootstrap_pupil_samples.append((timestamp, pdi))
                cutoff = timestamp - WORKLOAD_BASELINE_SECONDS
                while self._bootstrap_pupil_samples and self._bootstrap_pupil_samples[0][0] < cutoff:
                    self._bootstrap_pupil_samples.popleft()
            if (
                self._bootstrap_pupil_samples
                and self._bootstrap_pupil_samples[-1][0] - self._bootstrap_pupil_samples[0][0]
                >= WORKLOAD_BASELINE_SECONDS
            ):
                baseline_px = sum(v for _, v in self._bootstrap_pupil_samples) / len(self._bootstrap_pupil_samples)
                if self._session_manager is not None and baseline_px > 0.0:
                    self._session_manager.set_pupil_baseline(float(baseline_px))
                    self._has_pupil_baseline = True
                    self._pupil_card.set_unit("%")
                    logger.info("LiveView bootstrap pupil baseline set to %.3f px", baseline_px)
            return

        # With baseline: pdi is ratio, convert to percent for display and chart.
        pdi_pct = pdi * 100.0
        self._pupil_card.set_value(pdi_pct, timestamp)

        smoothed_pdi = self._append_window_value(
            self._pupil_smoothing_window,
            timestamp,
            pdi,
            WORKLOAD_PUPIL_SMOOTHING_SECONDS,
        )
        rolling_mean = self._append_window_value(
            self._pupil_rolling_window,
            timestamp,
            smoothed_pdi,
            WORKLOAD_PUPIL_ROLLING_SECONDS,
        )
        threshold = rolling_mean * WORKLOAD_THRESHOLD_FACTOR
        self._update_adaptive_workload_state(smoothed_pdi, threshold, timestamp)

        # PDI is already (value-baseline_mean)/baseline_mean; convert to percent.
        self._timeline_chart.append("PUPIL", timestamp, smoothed_pdi * 100.0)
        self._timeline_chart.append("THRESHOLD", timestamp, threshold * 100.0)

    @pyqtSlot(float, float)
    def _on_calibration_complete(self, _baseline_rmssd: float, baseline_pupil_px: float) -> None:
        """Record whether a valid pupil baseline was established."""
        self._baseline_rmssd = float(_baseline_rmssd or 0.0)
        self._baseline_pupil_px = float(baseline_pupil_px or 0.0)
        self._has_pupil_baseline = baseline_pupil_px > 0.0
        self._pupil_card.set_unit("%")
        logger.info(
            "LiveView: calibration complete — pupil baseline %.2f px (has_baseline=%s).",
            baseline_pupil_px, self._has_pupil_baseline,
        )

    @pyqtSlot(float, float)
    def on_cli_updated(self, cli: float, timestamp: float) -> None:
        """Receive CLI update and refresh all cognitive load displays.

        Args:
            cli: Cognitive Load Index in [0.0, 1.0].
            timestamp: Unix timestamp of the sample.
        """
        _ = (cli, timestamp)

    @pyqtSlot(float, float)
    def on_bpm_updated(self, bpm: float, _timestamp: float) -> None:
        """Receive instantaneous BPM and update the heart rate card.

        Args:
            bpm: Beats per minute derived from the latest RR interval.
            _timestamp: Unix timestamp (unused in the display layer).
        """
        self._bpm_card.set_value(bpm, _timestamp)
        self._cam_bpm_lbl.setText(f"{bpm:.0f}")

    @pyqtSlot(int)
    def on_error_count_updated(self, count: int) -> None:
        """Receive wall-contact count and update the ERROR RATE card."""
        self._accuracy_card.set_value(float(max(0, count)))

    @pyqtSlot(bool, str)
    def _on_hrv_connection_changed(self, connected: bool, _message: str) -> None:
        """Update ECG badge and clear stale heart-rate values on disconnect."""
        self._set_badge_status(self._ecg_badge, "ph.heartbeat-fill", connected)
        if not connected:
            self._bpm_card.reset()
            self._rmssd_card.reset()
            self._stress_gauge.set_value(0.0, "—")
            self._cam_stress_bar.set_value(0.0)
            self._cam_stress_value.setText("—")
            self._cam_bpm_lbl.setText("—")

    @pyqtSlot(bool, str)
    def _on_eye_connection_changed(self, connected: bool, _message: str) -> None:
        """Update eye tracker badge and clear stale pupil values on disconnect."""
        self._set_badge_status(self._eye_badge, "ph.eye-fill", connected)
        if not connected:
            self._pupil_card.reset()
            self._reset_workload_state()

    # ------------------------------------------------------------------
    # Widget reset
    # ------------------------------------------------------------------

    def _reset_widgets(self) -> None:
        """Clear all metric cards and charts before a new session."""
        for card in (self._pupil_card, self._bpm_card, self._rmssd_card,
                     self._speed_card, self._accuracy_card):
            card.reset()
        self._pupil_card.set_unit("%")

        self._reset_workload_state()
        self._stress_gauge.set_value(0.0, "—")
        self._cam_stress_bar.set_value(0.0)
        self._cam_stress_value.setText("—")
        self._cam_bpm_lbl.setText("—")

        self._timeline_chart.clear_all()

    def _start_camera_recording(self) -> None:
        """Start recording automatically for the active session."""
        if not self._session_active:
            return
        if self._video_feed.is_recording:
            return

        # Use the session-specific folder if available
        if self._session_manager and self._session_manager.current_session_dir:
            recordings_dir = self._session_manager.current_session_dir
        else:
            recordings_dir = Path(VIDEO_RECORDINGS_DIR)
            recordings_dir.mkdir(parents=True, exist_ok=True)

        session_label = self._session_id if self._session_id is not None else "unknown"
        file_name = f"session_{session_label}_{int(time.time())}.mp4"
        output_path = recordings_dir / file_name
        self._recording_path = str(output_path.absolute())

        if self._video_feed.start_recording(self._recording_path):
            logger.info("Video recording started automatically: %s", self._recording_path)
        else:
            self._recording_path = None
            logger.warning("Could not start automatic video recording.")

    def _stop_camera_recording(self) -> str | None:
        """Stop camera recording and return the absolute path to the file.

        Returns:
            The absolute path of the recording, or None if no recording was made.
        """
        path = None
        if self._video_feed.is_recording:
            self._video_feed.stop_recording()
            path = self._recording_path
            logger.info("Video recording stopped: %s", path)

        self._recording_path = None
        return path

    def _make_status_badge(self, icon_name: str) -> QFrame:
        """Create a rounded status indicator frame."""
        badge = QFrame()
        badge.setFixedSize(28, 28)
        layout = QHBoxLayout(badge)
        layout.setContentsMargins(0, 0, 0, 0)

        icon_lbl = QLabel()
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_lbl)

        # Initial state: disconnected (transparent)
        self._set_badge_status(badge, icon_name, connected=False)
        return badge

    def _set_badge_status(self, badge: QFrame, icon_name: str, connected: bool) -> None:
        """Apply the connected/disconnected style to a status badge."""
        if connected:
            bg = COLOR_SUCCESS
            border = "none"
            color = "#FFFFFF"
        else:
            bg = "#E5E7EB"
            border = "none"
            color = COLOR_PRIMARY

        badge.setStyleSheet(
            f"background: {bg}; border: {border}; border-radius: 14px;"
        )

        # Find the icon label within the layout
        icon_lbl = badge.findChild(QLabel)
        if icon_lbl:
            icon_lbl.setPixmap(get_icon(icon_name, color=color).pixmap(14, 14))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_window_value(
        buffer: deque[tuple[float, float]],
        timestamp: float,
        value: float,
        window_seconds: float,
    ) -> float:
        """Append value to rolling window and return current mean."""
        buffer.append((timestamp, value))
        cutoff = timestamp - window_seconds
        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()
        return float(sum(sample for _, sample in buffer) / len(buffer))

    def _update_adaptive_workload_state(
        self,
        smoothed_pdi: float,
        threshold: float,
        timestamp: float,
    ) -> None:
        """Classify workload against adaptive threshold with persistence."""
        target_state = smoothed_pdi > threshold
        if self._pending_workload_state != target_state:
            self._pending_workload_state = target_state
            self._pending_workload_since = timestamp

        if (
            self._pending_workload_since is not None
            and timestamp - self._pending_workload_since >= WORKLOAD_STATE_PERSIST_SECONDS
        ):
            self._workload_state_is_elevated = target_state

        is_high = self._workload_state_is_elevated
        gauge_label = "HIGH" if is_high else "LOW"
        gauge_color = COLOR_DANGER if is_high else COLOR_SUCCESS
        self._workload_gauge.set_accent_color(gauge_color)
        self._workload_gauge.set_value(1.0, gauge_label)
        self._cam_workload_dot.setStyleSheet(
            f"background-color: {gauge_color}; border-radius: 8px;"
        )
        self._cam_workload_value.setText(gauge_label)

    def _reset_workload_state(self) -> None:
        """Reset adaptive workload windows/state."""
        self._bootstrap_pupil_samples.clear()
        self._pupil_smoothing_window.clear()
        self._pupil_rolling_window.clear()
        self._workload_state_is_elevated = False
        self._pending_workload_state = None
        self._pending_workload_since = None
        self._workload_gauge.set_accent_color(COLOR_SUCCESS)
        self._workload_gauge.set_value(1.0, "LOW")
        self._cam_workload_dot.setStyleSheet(
            f"background-color: {COLOR_SUCCESS}; border-radius: 8px;"
        )
        self._cam_workload_value.setText("LOW")

    def _make_gauge_panel(
        self, 
        title: str, 
        accent: str, 
        track: str
    ) -> tuple[DonutGauge, QWidget]:
        """Create a vertical panel containing a donut gauge and a title.
        
        This panel has no background or border as requested.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_1)

        gauge = DonutGauge(value=0.0, accent_color=accent, track_color=track, center_text="—", size=148)
        layout.addWidget(gauge, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_SMALL}px; font-weight: 700; letter-spacing: 1.5px;"
        )
        layout.addWidget(title_lbl)
        layout.addStretch(1)

        return gauge, panel

    def _sync_pupil_baseline_state(self) -> None:
        """Mirror the current calibration baseline from SessionManager into the UI state."""
        baseline_pupil_px = 0.0
        if self._session_manager is not None:
            baseline_pupil_px = float(getattr(self._session_manager, "baseline_pupil_px", 0.0) or 0.0)

        self._has_pupil_baseline = baseline_pupil_px > 0.0
        self._baseline_pupil_px = baseline_pupil_px
        self._baseline_rmssd = float(getattr(self._session_manager, "baseline_rmssd", 0.0) or 0.0)
        self._pupil_card.set_unit("%")

    @staticmethod
    def _make_card() -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("background-color: transparent;")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return card
