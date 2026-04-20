"""Post-session analysis view for BioTrace.

Displays session statistics, a biometric timeline chart, and a video
playback area for post-session review.
"""

from datetime import datetime

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import (
    CARD_PADDING,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_DANGER,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_PRIMARY_SUBTLE,
    COLOR_WARNING,
    FONT_BODY,
    FONT_CAPTION,
    FONT_HEADING_2,
    FONT_SMALL,
    RADIUS_LG,
    SPACE_1,
    SPACE_2,
    SPACE_3,
    GRID_GUTTER,
    CONTENT_PADDING_H,
    CONTENT_PADDING_V,
    CHART_HEIGHT_TIMELINE,
    WEIGHT_BOLD,
    WEIGHT_SEMIBOLD,
    get_icon,
)
from app.utils.config import (
    VIDEO_RECORDING_FPS_FALLBACK,
    WORKLOAD_PUPIL_ROLLING_SECONDS,
    WORKLOAD_PUPIL_SMOOTHING_SECONDS,
    WORKLOAD_THRESHOLD_FACTOR,
)
from app.storage.database import DatabaseManager
from app.storage.export import SessionExporter
from app.storage.session_repository import SessionRepository
from app.ui.widgets.video_player import VideoPlayer
from app.ui.widgets.timeline_chart import TimelineChart
from app.utils.logger import get_logger

logger = get_logger(__name__)

class PostSessionView(QWidget):
    """Individual session dashboard shown after a session ends or when opened from history.

    Shows session date/title, summary metric cards (session duration,
    errors, error rate/min, stress events, high workload events), a biometric timeline chart with series toggle, and a video
    playback area.

    User flow
    ---------
    LiveView END SESSION → MainWindow._on_session_ended →
    ``load_session(session_id)`` is called → this view is shown.
    """

    back_to_dashboard = pyqtSignal()
    new_session_requested = pyqtSignal()
    session_renamed = pyqtSignal(int, str)

    def __init__(
        self,
        db: DatabaseManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._db = db
        self._session_id: int | None = None
        self._exporter = SessionExporter(db)
        self._session_repo = SessionRepository(db)
        
        self._metric_value_labels: dict[str, QLabel] = {}
        self._metric_subtitle_labels: dict[str, QLabel] = {}
        self._start_session_btn: QPushButton | None = None
        self._export_btn: QPushButton | None = None

        self._timeline_chart = TimelineChart()
        self._video_player = VideoPlayer()
        
        self._build_ui()
        self._wire_signals()

    def _wire_signals(self) -> None:
        """Connect internal widget signals."""
        # Clicking on the chart seeks the video.
        self._timeline_chart.timestamp_clicked.connect(self._video_player.seek_to)
        # Video playback moves the chart playhead.
        self._video_player.playback_position_changed.connect(self._timeline_chart.set_playhead_ms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _on_export_clicked(self) -> None:
        """Open a save-file dialog and export the current session to Excel."""
        if self._session_id is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session Data",
            f"session_{self._session_id}.xlsx",
            "Excel Workbook (*.xlsx)",
        )
        if not path:
            return  # user cancelled

        if not path.endswith(".xlsx"):
            path += ".xlsx"

        self._exporter.export_excel(self._session_id, path)
        logger.info("Session %d exported by user to %s", self._session_id, path)

    def load_session(self, session_id: int) -> None:
        """Load session metadata and data into the dashboard.

        Args:
            session_id: The database ID of the session to display.
        """
        self._session_id = session_id

        session = self._session_repo.get_session(session_id)
        if session is None:
            logger.warning("PostSessionView: session %d not found in DB.", session_id)
            self._title_label.setText("Session —")
            return

        # ── Title: custom name OR actual session date ─────────────────
        if session["name"]:
            self._title_label.setText(session["name"])
        else:
            try:
                started_at = datetime.fromisoformat(str(session["started_at"]))
                self._title_label.setText(f"Session {started_at.strftime('%-d.%-m.%Y')}")
            except Exception:
                self._title_label.setText("Session —")

        duration_s = self._compute_session_duration_seconds(session["started_at"], session["ended_at"])
        video_path = self._session_repo.get_video_path(session_id)
        recording_duration_s = self._compute_recording_duration_seconds(video_path)
        # Session duration reflects active session runtime in DB:
        # from post-calibration session start until user presses End Session.
        self._set_metric_cards(session_id, duration_s, session["error_count"])

        # ── Timeline Data ──────────────────────────────────────────────
        self._timeline_chart.load_session(
            self._db,
            session_id,
            expected_duration_s=duration_s,
        )

        # ── Video Recording ────────────────────────────────────────────
        self._video_player.load(video_path)

        if (
            duration_s is not None
            and recording_duration_s is not None
            and abs(duration_s - recording_duration_s) > 1
        ):
            logger.warning(
                "Session %d duration mismatch: DB=%ds, video=%ds",
                session_id,
                duration_s,
                recording_duration_s,
            )

        logger.info("PostSessionView loaded session_id=%d", session_id)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header (Static at top) ─────────────────────────────────────
        header_widget = QWidget()
        header_widget.setStyleSheet(f"background: {COLOR_BACKGROUND};")
        header_v = QVBoxLayout(header_widget)
        header_v.setContentsMargins(CONTENT_PADDING_H, CONTENT_PADDING_V,
                                    CONTENT_PADDING_H, SPACE_2)
        header_v.addLayout(self._build_header())
        outer.addWidget(header_widget)

        # ── Scroll area for content ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        content = QWidget()
        content.setStyleSheet(f"background: {COLOR_BACKGROUND};")
        content_layout = QVBoxLayout(content)
        # Large margins and even larger vertical spacing for "breathing room"
        content_layout.setContentsMargins(CONTENT_PADDING_H, SPACE_2,
                                          CONTENT_PADDING_H, CONTENT_PADDING_V * 2)
        content_layout.setSpacing(int(GRID_GUTTER * 1.5))

        # Row 1: Summary Metric Cards
        content_layout.addLayout(self._build_metric_cards())
        
        # Row 2: Session Analysis (Timeline Chart)
        analysis_card = self._build_analysis_card()
        # Ensure it has enough space in the scroll view
        analysis_card.setMinimumHeight(520)
        content_layout.addWidget(analysis_card)

        # Row 3: Video Player
        video_area = self._build_video_area()
        video_area.setMinimumHeight(520) # Large playback area
        content_layout.addWidget(video_area)

        content_layout.addStretch(1)
        
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_header(self) -> QHBoxLayout:
        """Build the header row: title plus primary session actions."""
        header = QHBoxLayout()
        header.setSpacing(SPACE_2)

        # Keep icon controls visually proportional to the shared heading size.
        status_button_size = FONT_HEADING_2 * 2
        status_icon_size = FONT_HEADING_2 - 8

        # Container for the title and rename edit
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(SPACE_1)

        self._title_label = QLabel("Session —")
        self._title_label.setObjectName("heading")
        title_layout.addWidget(self._title_label)

        self._title_edit = QLineEdit()
        self._title_edit.setObjectName("heading")
        self._title_edit.setStyleSheet(
            f"font-size: {FONT_HEADING_2}px; font-weight: {WEIGHT_BOLD};"
        )
        self._title_edit.setMinimumWidth(250)
        self._title_edit.hide()
        self._title_edit.returnPressed.connect(self._on_name_saved)
        title_layout.addWidget(self._title_edit)

        self._rename_btn = QToolButton()
        self._rename_btn.setIcon(get_icon("ph.pencil-simple", color=COLOR_FONT_MUTED))
        self._rename_btn.setIconSize(QSize(18, 18))
        self._rename_btn.setFixedSize(32, 32)
        self._rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rename_btn.setStyleSheet("background: transparent; border: none;")
        self._rename_btn.clicked.connect(self._on_rename_clicked)
        title_layout.addWidget(self._rename_btn)

        header.addWidget(title_container)
        header.addStretch(1)

        primary_action_button_stylesheet = (
            f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: {FONT_HEADING_2}px;
                padding: 0px {SPACE_2}px;
                font-size: {FONT_BODY}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLOR_PRIMARY_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {COLOR_PRIMARY};
                padding-top: 1px;
                padding-bottom: 0px;
            }}
            """
        )

        ghost_action_button_stylesheet = (
            f"""
            QPushButton#secondary {{
                background-color: {COLOR_CARD};
                color: {COLOR_PRIMARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: {FONT_HEADING_2}px;
                padding: 0px {SPACE_2}px;
                font-size: {FONT_BODY}px;
                font-weight: 600;
            }}
            QPushButton#secondary:hover {{
                background-color: {COLOR_PRIMARY_SUBTLE};
            }}
            QPushButton#secondary:pressed {{
                background-color: {COLOR_PRIMARY_SUBTLE};
                padding-top: 1px;
                padding-bottom: 0px;
            }}
            """
        )

        self._export_btn = QPushButton("Export Data ")
        self._export_btn.setObjectName("secondary")
        self._export_btn.setIcon(get_icon("ph.arrow-right", color=COLOR_PRIMARY))
        self._export_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._export_btn.setIconSize(QSize(FONT_BODY + 2, FONT_BODY + 2))
        self._export_btn.setFixedHeight(FONT_HEADING_2 * 2)
        self._export_btn.setMinimumWidth(170)
        self._export_btn.setStyleSheet(ghost_action_button_stylesheet)
        self._export_btn.clicked.connect(self._on_export_clicked)
        header.addWidget(self._export_btn)

        self._start_session_btn = QPushButton("Start Session")
        self._start_session_btn.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        self._start_session_btn.setIconSize(QSize(FONT_BODY + 2, FONT_BODY + 2))
        self._start_session_btn.setFixedHeight(FONT_HEADING_2 * 2)
        self._start_session_btn.setMinimumWidth(170)
        self._start_session_btn.setStyleSheet(primary_action_button_stylesheet)
        self._start_session_btn.clicked.connect(self.new_session_requested.emit)
        header.addWidget(self._start_session_btn)

        return header

    def _on_rename_clicked(self) -> None:
        """Switch title label to edit mode."""
        self._title_label.hide()
        self._rename_btn.hide()
        self._title_edit.setText(self._title_label.text())
        self._title_edit.show()
        self._title_edit.setFocus()
        self._title_edit.selectAll()

    def _on_name_saved(self) -> None:
        """Save the new session name and revert to label mode."""
        new_name = self._title_edit.text().strip()
        if self._session_id is not None and new_name:
            self._session_repo.set_session_name(self._session_id, new_name)
            self._title_label.setText(new_name)
            self.session_renamed.emit(self._session_id, new_name)

        self._title_edit.hide()
        self._title_label.show()
        self._rename_btn.show()

    def _build_metric_cards(self) -> QHBoxLayout:
        """Build the summary metric cards for this individual session."""
        row = QHBoxLayout()
        row.setSpacing(SPACE_2)

        self._add_metric_card(row, key="duration", title="SESSION DURATION")
        self._add_metric_card(row, key="errors", title="NUMBER OF ERRORS")
        self._add_metric_card(row, key="error_rate_per_min", title="ERROR RATE / MIN")
        self._add_metric_card(row, key="stress_events", title="STRESS EVENTS")
        self._add_metric_card(row, key="workload_events", title="HIGH WORKLOAD EVENTS")

        return row

    def _add_metric_card(
        self,
        layout: QHBoxLayout,
        key: str,
        title: str,
    ) -> None:
        """Create one numeric metric card and register value/subtitle labels."""
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(
            f"QFrame#card {{ background-color: transparent; border: 1px solid {COLOR_BORDER};"
            f" border-radius: {RADIUS_LG}px; }}"
        )
        card.setMinimumHeight(190)
        card.setMinimumWidth(0)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(SPACE_3, SPACE_2, SPACE_3, SPACE_3)
        card_layout.setSpacing(SPACE_1)

        icon_by_key = {
            "duration": ("ph.timer-fill", COLOR_PRIMARY),
            "errors": ("ph.warning-circle-fill", COLOR_DANGER),
            "error_rate_per_min": ("ph.speedometer-fill", COLOR_DANGER),
            "stress_events": ("ph.heartbeat-fill", COLOR_PRIMARY),
            "workload_events": ("ph.brain-fill", COLOR_WARNING),
        }
        icon_name, icon_color = icon_by_key.get(key, ("ph.circle-fill", COLOR_FONT_MUTED))

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_row.addStretch(1)

        icon_label = QLabel()
        icon_label.setPixmap(get_icon(icon_name, color=icon_color).pixmap(14, 14))
        title_row.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_CAPTION}px; font-weight: {WEIGHT_BOLD};"
            "letter-spacing: 0.5px;"
        )
        title_row.addWidget(title_label)
        title_row.addStretch(1)
        card_layout.addLayout(title_row)

        card_layout.addStretch(1)

        value_label = QLabel("—")
        value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        value_label.setStyleSheet(
            f"color: {COLOR_PRIMARY}; font-size: {FONT_HEADING_2 + 10}px; font-weight: {WEIGHT_BOLD};"
        )
        card_layout.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignCenter)

        subtitle_label = QLabel("")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        subtitle_label.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px; font-weight: {WEIGHT_SEMIBOLD};"
        )
        subtitle_label.hide()
        card_layout.addWidget(subtitle_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        card_layout.addStretch(1)

        self._metric_value_labels[key] = value_label
        self._metric_subtitle_labels[key] = subtitle_label
        layout.addWidget(card, stretch=1)

    def _build_analysis_card(self) -> QFrame:
        """Build the session analysis card with series toggle buttons."""
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("background-color: transparent;")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(card)
        outer.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        outer.setSpacing(SPACE_2)

        # ── Title + toggle row ─────────────────────────────────────
        title_row = QHBoxLayout()

        title_lbl = QLabel("SESSION OVERVIEW")
        title_lbl.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px;"
            f" font-weight: {WEIGHT_BOLD}; letter-spacing: 1px;"
        )
        title_row.addWidget(title_lbl)

        title_row.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        _toggle_base = (
            f"border: 1px solid; border-radius: {RADIUS_LG}px;"
            f" padding: 4px 12px; font-size: {FONT_SMALL}px; font-weight: {WEIGHT_SEMIBOLD};"
        )
        cog_btn = QPushButton(" PUPIL DILATION")
        cog_btn.setIcon(get_icon("ph.brain-fill", color=COLOR_WARNING))
        cog_btn.setCheckable(True)
        cog_btn.setChecked(True)
        cog_btn.setFixedHeight(30)
        
        def _update_cog_style(checked):
            color = "#FFFFFF" if checked else COLOR_WARNING
            bg = COLOR_WARNING if checked else "transparent"
            cog_btn.setIcon(get_icon("ph.brain-fill", color=color))
            cog_btn.setStyleSheet(
                f"QPushButton {{ {_toggle_base} background: {bg}; color: {color}; border-color: {COLOR_WARNING}; }}"
            )
        
        cog_btn.toggled.connect(_update_cog_style)
        cog_btn.toggled.connect(lambda v: self._timeline_chart.set_series_visibility("WORKLOAD", v))
        _update_cog_style(True)
        title_row.addWidget(cog_btn)

        hrv_btn = QPushButton(" STRESS (RMSSD)")
        hrv_btn.setIcon(get_icon("ph.heartbeat-fill", color=COLOR_PRIMARY))
        hrv_btn.setCheckable(True)
        hrv_btn.setChecked(True)
        hrv_btn.setFixedHeight(30)

        def _update_hrv_style(checked):
            color = "#FFFFFF" if checked else COLOR_PRIMARY
            bg = COLOR_PRIMARY if checked else "transparent"
            hrv_btn.setIcon(get_icon("ph.heartbeat-fill", color=color))
            hrv_btn.setStyleSheet(
                f"QPushButton {{ {_toggle_base} background: {bg}; color: {color}; border-color: {COLOR_PRIMARY}; }}"
            )

        hrv_btn.toggled.connect(_update_hrv_style)
        hrv_btn.toggled.connect(lambda v: self._timeline_chart.set_series_visibility("STRESS", v))
        _update_hrv_style(True)
        title_row.addWidget(hrv_btn)

        outer.addLayout(title_row)
        self._timeline_chart.setMinimumHeight(CHART_HEIGHT_TIMELINE + 80)
        outer.addWidget(self._timeline_chart, stretch=1)

        return card

    def _build_video_area(self) -> QWidget:
        """Return the video player widget."""
        return self._video_player

    def _compute_session_duration_seconds(
        self, started_at_raw: str | None, ended_at_raw: str | None
    ) -> int | None:
        """Return non-negative session duration in seconds, or None when unavailable."""
        if not started_at_raw or not ended_at_raw:
            return None
        try:
            started_at = datetime.fromisoformat(str(started_at_raw))
            ended_at = datetime.fromisoformat(str(ended_at_raw))
        except Exception:
            return None
        duration_s = int((ended_at - started_at).total_seconds())
        if duration_s < 0:
            return None
        return duration_s

    def _compute_recording_duration_seconds(self, video_path: str | None) -> int | None:
        """Return video recording duration in seconds, or None if unavailable."""
        if not video_path:
            return None
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0.0:
                return None
            # Match VideoFeed / VideoPlayer: we encode at VIDEO_RECORDING_FPS_FALLBACK;
            # container-reported FPS is unreliable for mp4v on some platforms.
            fps = float(VIDEO_RECORDING_FPS_FALLBACK)
            duration_s = int(round(frame_count / fps))
            return duration_s if duration_s >= 0 else None
        except Exception:
            logger.warning("Could not compute recording duration for %s", video_path)
            return None

    def _format_duration(self, seconds: int) -> str:
        """Format seconds as M:SS (or H:MM:SS when >= 1h)."""
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @staticmethod
    def _compute_error_rate_per_minute(
        error_count: int | None,
        duration_s: int | None,
    ) -> float | None:
        """Return wall-contact frequency as errors/minute for one session."""
        if error_count is None or duration_s is None or duration_s <= 0:
            return None
        return float(max(0, error_count)) / (float(duration_s) / 60.0)

    def _query_stress_event_counts(self, session_id: int) -> tuple[int, int]:
        """Return (stress_events, severe_events) using crossing-based RMSSD logic."""
        conn = self._db.get_connection()
        cal_row = conn.execute(
            "SELECT baseline_rmssd FROM calibrations "
            "WHERE session_id = ? AND baseline_rmssd IS NOT NULL "
            "ORDER BY id ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        baseline_rmssd = float(cal_row[0]) if cal_row and cal_row[0] is not None else None

        rows = conn.execute(
            "SELECT rmssd FROM hrv_samples "
            "WHERE session_id = ? AND rmssd IS NOT NULL ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        if not rows:
            return (0, 0)

        rmssd_vals = [float(r["rmssd"]) for r in rows]
        if baseline_rmssd is not None and baseline_rmssd > 0:
            ref = baseline_rmssd
        else:
            ref = float(sum(rmssd_vals) / len(rmssd_vals))
            if ref <= 0:
                return (0, 0)

        stress_events = 0
        severe_events = 0
        prev_below_stress = False
        prev_below_severe = False

        for value in rmssd_vals:
            rmssd_pct_change = ((value - ref) / ref) * 100.0
            is_below_stress = rmssd_pct_change < -10.0
            is_below_severe = rmssd_pct_change < -40.0

            if is_below_stress and not prev_below_stress:
                stress_events += 1
            if is_below_severe and not prev_below_severe:
                severe_events += 1

            prev_below_stress = is_below_stress
            prev_below_severe = is_below_severe

        return (stress_events, severe_events)

    def _query_high_workload_events(self, session_id: int) -> int:
        """Return count of pupil-vs-threshold crossings (same logic as live view)."""
        conn = self._db.get_connection()
        rows = conn.execute(
            "SELECT timestamp, pdi FROM pupil_samples "
            "WHERE session_id = ? AND pdi IS NOT NULL ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        if not rows:
            return 0

        from collections import deque

        smooth_window: deque[tuple[float, float]] = deque()
        rolling_window: deque[tuple[float, float]] = deque()

        def append_window_mean(
            buffer: deque[tuple[float, float]],
            timestamp: float,
            value: float,
            window_seconds: float,
        ) -> float:
            buffer.append((timestamp, value))
            cutoff = timestamp - window_seconds
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()
            return float(sum(sample for _, sample in buffer) / len(buffer))

        events = 0
        prev_above = False
        for row in rows:
            timestamp = float(row["timestamp"])
            pdi = float(row["pdi"])
            smoothed_pdi = append_window_mean(
                smooth_window,
                timestamp,
                pdi,
                WORKLOAD_PUPIL_SMOOTHING_SECONDS,
            )
            rolling_mean = append_window_mean(
                rolling_window,
                timestamp,
                smoothed_pdi,
                WORKLOAD_PUPIL_ROLLING_SECONDS,
            )
            threshold = rolling_mean * WORKLOAD_THRESHOLD_FACTOR
            above = smoothed_pdi > threshold
            if above and not prev_above:
                events += 1
            prev_above = above
        return events

    def _set_card_value(self, key: str, value: str, subtitle: str = "") -> None:
        """Set value/subtitle on one summary metric card."""
        value_label = self._metric_value_labels.get(key)
        subtitle_label = self._metric_subtitle_labels.get(key)
        if value_label is not None:
            value_label.setText(value)
        if subtitle_label is not None:
            subtitle_label.setText(subtitle)
            subtitle_label.setVisible(bool(subtitle))

    def _set_metric_cards(self, session_id: int, duration_s: int | None, error_count: int | None) -> None:
        """Populate summary cards with duration/error/stress/workload event counts."""
        self._set_card_value(
            key="duration",
            value="—" if duration_s is None else self._format_duration(duration_s),
        )

        err_val = int(error_count) if error_count is not None else 0
        self._set_card_value(key="errors", value=str(err_val))
        error_rate_per_min = self._compute_error_rate_per_minute(error_count, duration_s)
        self._set_card_value(
            key="error_rate_per_min",
            value="—" if error_rate_per_min is None else f"{error_rate_per_min:.1f}/min",
        )

        stress_events, _severe_events = self._query_stress_event_counts(session_id)
        self._set_card_value(
            key="stress_events",
            value=str(stress_events),
            subtitle="",
        )

        workload_events = self._query_high_workload_events(session_id)
        self._set_card_value(key="workload_events", value=str(workload_events))
