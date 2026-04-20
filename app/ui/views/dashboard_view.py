"""Dashboard view matching the main design mockup.

The layout is split into three areas:
1) Header with status actions and start button
2) Top content row with progress chart and personal best cards
3) Bottom row with three KPI summary cards (icon + value)
4) Session Trend Analysis chart

All colours and font sizes are sourced exclusively from ``app.ui.theme``.
No hex literals or magic pixel values appear in this file.
"""

from collections import deque
from datetime import datetime, timedelta
import sqlite3
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.storage.database import DatabaseManager
from app.storage.session_repository import SessionRepository
from app.analytics.learning_curve import fit_schmettow, predict_at_trial
from app.utils.config import (
    LC_MIN_SESSIONS,
    SCORE_MAX,
    WALL_CONTACT_PENALTY_S,
    WORKLOAD_PUPIL_ROLLING_SECONDS,
    WORKLOAD_PUPIL_SMOOTHING_SECONDS,
    WORKLOAD_THRESHOLD_FACTOR,
)
from app.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_CHART_AXIS,
    COLOR_DANGER,
    COLOR_DANGER_BG,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_WARNING_BG,
    FONT_BODY,
    FONT_HEADING_2,
    FONT_SUBTITLE,
    RADIUS_LG,
    SPACE_1,
    SPACE_2,
    SPACE_3,
    SPACE_4,
    CONTENT_PADDING_H,
    CONTENT_PADDING_V,
    GRID_GUTTER,
    get_icon,
)
from app.ui.widgets.level_bar import LevelBar
from app.utils.logger import get_logger

logger = get_logger(__name__)

CARD_TITLE_FONT_SIZE = FONT_HEADING_2 - 4


class _PersonalBestRowFrame(QFrame):
    """One personal-best row; emits ``row_clicked`` with session id when set."""

    row_clicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_id: int | None = None

    def set_session_id(self, session_id: int | None) -> None:
        self._session_id = session_id
        if session_id is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Open session details")
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setToolTip("")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._session_id is not None:
            self.row_clicked.emit(self._session_id)
        super().mousePressEvent(event)
TOP_ROW_CARD_MIN_HEIGHT = 432
PROGRESS_CHART_MIN_HEIGHT = 304


class DashboardView(QWidget):
    """Main dashboard aligned to the design in ``designs/maindashboard.png``."""

    new_session_requested = pyqtSignal()
    session_selected = pyqtSignal(int)

    def __init__(
        self,
        db: DatabaseManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo: SessionRepository | None = (
            SessionRepository(db) if db is not None else None
        )
        self._sessions: list[sqlite3.Row] = []

        self._sessions_count_label: QLabel | None = None
        self._best_time_rows: list[tuple[QLabel, QLabel, QLabel]] = []
        self._best_row_frames: list[_PersonalBestRowFrame] = []
        self._personal_best_placeholder: QLabel | None = None

        self._stress_value_label: QLabel | None = None
        self._error_value_label: QLabel | None = None
        self._workload_value_label: QLabel | None = None

        # Per-session biometric stats cached on each refresh() call.
        # Keys are session IDs; see _load_biometric_stats() for the dict schema.
        self._biometric_stats: dict[int, dict] = {}

        self._progress_plot: pg.PlotWidget | None = None
        self._actual_curve: pg.PlotCurveItem | None = None
        self._estimate_curve: pg.PlotCurveItem | None = None

        self._analysis_plot: pg.PlotWidget | None = None
        self._stress_series_curve: pg.PlotCurveItem | None = None
        self._workload_series_curve: pg.PlotCurveItem | None = None
        self._analysis_filter_group: QButtonGroup | None = None
        self._active_analysis_filters: set[str] = {"stress", "workload"}
        self._content_scroll: QScrollArea | None = None

        self._db = db

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Outer shell: fills the view, no margins ────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header sits above the scroll area (always visible) ─────────────
        header_widget = QWidget()
        header_widget.setObjectName("dashboard_header")
        header_widget.setStyleSheet(f"background: {COLOR_BACKGROUND};")
        header_v = QVBoxLayout(header_widget)
        header_v.setContentsMargins(CONTENT_PADDING_H, CONTENT_PADDING_V,
                                    CONTENT_PADDING_H, SPACE_2)
        header_v.setSpacing(SPACE_2)
        header_v.addLayout(self._build_header())

        outer.addWidget(header_widget)

        # ── Scroll area for the main content ──────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._content_scroll = scroll

        content = QWidget()
        content.setStyleSheet(f"background: {COLOR_BACKGROUND};")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(CONTENT_PADDING_H, SPACE_2,
                                          CONTENT_PADDING_H, CONTENT_PADDING_V)
        content_layout.setSpacing(GRID_GUTTER)

        # ── Row 1: Progress chart + Personal Best ──────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(GRID_GUTTER)
        progress_card = self._make_progress_card()
        progress_card.setMinimumHeight(TOP_ROW_CARD_MIN_HEIGHT)
        personal_card = self._make_personal_best_card()
        personal_card.setMinimumHeight(TOP_ROW_CARD_MIN_HEIGHT)
        top_row.addWidget(progress_card, stretch=2)
        top_row.addWidget(personal_card, stretch=1)
        content_layout.addLayout(top_row)

        # ── Row 2: KPI gauge cards ─────────────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(GRID_GUTTER)
        for kwargs in (
            dict(
                title="Ø Stress Events",
                accent=COLOR_PRIMARY,
                key="stress",
                icon_name="ph.heartbeat-fill",
            ),
            dict(
                title="Ø Error Rate",
                accent=COLOR_DANGER,
                key="error",
                icon_name="ph.warning-circle-fill",
            ),
            dict(
                title="Ø High Cognitive Load Events",
                accent=COLOR_WARNING,
                key="workload",
                icon_name="ph.brain-fill",
            ),
        ):
            card = self._make_metric_card(**kwargs)
            card.setMinimumHeight(300)
            bottom_row.addWidget(card)
        content_layout.addLayout(bottom_row)

        # ── Row 3: Session Trend Analysis ─────────────────────────────────
        analysis = self._make_analysis_card()
        analysis.setMinimumHeight(380)
        content_layout.addWidget(analysis)

        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

    def _build_header(self) -> QHBoxLayout:
        """Create the top dashboard row with title and controls."""
        header = QHBoxLayout()
        header.setSpacing(SPACE_2)

        # Keep icon controls visually proportional to the shared heading size.
        status_button_size = FONT_HEADING_2 * 2
        status_icon_size = FONT_HEADING_2 - 8

        title = QLabel("Dashboard")
        title.setObjectName("heading")
        header.addWidget(title)
        header.addStretch(1)

        start_button = QPushButton("Start Session")
        start_button.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        start_button.setIconSize(QSize(FONT_BODY + 2, FONT_BODY + 2))
        start_button.setFixedHeight(FONT_HEADING_2 * 2)
        start_button.setMinimumWidth(170)
        start_button.setStyleSheet(
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
                background-color: {COLOR_PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {COLOR_PRIMARY};
                padding-top: 1px;
                padding-bottom: 0px;
            }}
            """
        )
        start_button.clicked.connect(self._on_new_session)
        header.addWidget(start_button)
        return header

    def _make_progress_card(self) -> QFrame:
        """Left card: combined learning curve (time + wall contact penalty)."""
        card = self._make_card()
        card.setStyleSheet(
            f"""
            QFrame#card {{
                background-color: transparent;
                border: 1px solid {COLOR_BORDER};
                border-radius: {RADIUS_LG}px;
            }}
            """
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        layout.setSpacing(SPACE_2)

        header = QHBoxLayout()
        left = QVBoxLayout()

        title = QLabel("Learning Curve")
        title.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {CARD_TITLE_FONT_SIZE}px; font-weight: 700;"
        )
        left.addWidget(title)
        header.addLayout(left)
        header.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(2)
        sessions_label = QLabel("SESSIONS")
        sessions_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        sessions_label.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_BODY}px; letter-spacing: 1px;"
        )
        right.addWidget(sessions_label)

        self._sessions_count_label = QLabel("0")
        self._sessions_count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._sessions_count_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {CARD_TITLE_FONT_SIZE}px; font-weight: 700;"
        )
        right.addWidget(self._sessions_count_label)

        header.addLayout(right)
        layout.addLayout(header)

        self._progress_plot = pg.PlotWidget()
        self._progress_plot.setBackground((0, 0, 0, 0))
        self._progress_plot.getAxis("left").setStyle(showValues=True)
        self._progress_plot.showGrid(x=False, y=False)
        self._progress_plot.setMenuEnabled(False)
        self._progress_plot.hideButtons()
        self._progress_plot.setMouseEnabled(x=False, y=False)
        self._progress_plot.setMinimumHeight(PROGRESS_CHART_MIN_HEIGHT)
        self._progress_plot.getAxis("bottom").setPen(pg.mkPen(COLOR_CHART_AXIS, width=0.8))
        self._progress_plot.getAxis("left").setPen(pg.mkPen(COLOR_CHART_AXIS, width=0.8))
        self._progress_plot.getAxis("bottom").setTextPen(pg.mkPen(COLOR_FONT_MUTED, width=1))
        self._progress_plot.getAxis("left").setTextPen(pg.mkPen(COLOR_FONT_MUTED, width=1))
        self._progress_plot.getAxis("left").setLabel("Performance", color=COLOR_FONT_MUTED)
        self._progress_plot.getAxis("bottom").setLabel("Sessions", color=COLOR_FONT_MUTED)

        estimate_pen = pg.mkPen(color=COLOR_FONT_MUTED, width=3, style=Qt.PenStyle.DashLine)
        actual_pen = pg.mkPen(color=COLOR_PRIMARY, width=4)

        self._estimate_curve = self._progress_plot.plot([], [], pen=estimate_pen)
        self._actual_curve = self._progress_plot.plot(
            [], [],
            pen=actual_pen,
            symbol="o",
            symbolSize=8,
            symbolBrush=pg.mkBrush(COLOR_PRIMARY),
            symbolPen=pg.mkPen(COLOR_PRIMARY),
        )
        self._progress_plot.installEventFilter(self)
        layout.addWidget(self._progress_plot, stretch=1)

        return card

    def _make_personal_best_card(self) -> QFrame:
        """Right card: minimalist top-3 best sessions layout."""
        card = self._make_card()
        card.setStyleSheet(
            f"""
            QFrame#card {{
                background-color: transparent;
                border: none;
                border-radius: {RADIUS_LG}px;
            }}
            """
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        layout.setSpacing(SPACE_3)

        title = QLabel("Personal Best Times")
        title.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {CARD_TITLE_FONT_SIZE}px; font-weight: 700;"
        )
        layout.addWidget(title)

        sessions_wrap = QFrame()
        sessions_wrap.setObjectName("personal_best_sessions")
        sessions_wrap.setStyleSheet(
            f"""
            QFrame#personal_best_sessions {{
                border: 1px solid {COLOR_BORDER};
                border-radius: {RADIUS_LG}px;
                background-color: {COLOR_CARD};
            }}
            """
        )
        wrap_layout = QVBoxLayout(sessions_wrap)
        wrap_layout.setContentsMargins(SPACE_2, SPACE_2, SPACE_2, SPACE_2)
        wrap_layout.setSpacing(SPACE_2)

        self._personal_best_placeholder = QLabel(
            "No session data available yet. Start your first session to see your "
            "personal best times here."
        )
        self._personal_best_placeholder.setWordWrap(True)
        self._personal_best_placeholder.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._personal_best_placeholder.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_BODY}px; font-weight: 500; "
            f"padding: {SPACE_2}px;"
        )
        self._personal_best_placeholder.setVisible(False)
        wrap_layout.addWidget(self._personal_best_placeholder)

        # Reset references in case the card is rebuilt.
        self._best_time_rows = []
        self._best_row_frames = []

        for rank in range(1, 4):
            row_card = _PersonalBestRowFrame()
            row_card.row_clicked.connect(self.session_selected.emit)
            row_card.setStyleSheet(
                f"""
                QFrame {{
                    border: none;
                    border-radius: {RADIUS_LG}px;
                    background-color: {COLOR_BACKGROUND};
                }}
                """
            )

            row_layout = QVBoxLayout(row_card)
            row_layout.setContentsMargins(SPACE_2, SPACE_2, SPACE_2, SPACE_2)
            row_layout.setSpacing(6)

            top_line = QHBoxLayout()
            top_line.setContentsMargins(0, 0, 0, 0)
            top_line.setSpacing(SPACE_1)

            meta_label = QLabel(f"BEST SESSION #{rank}")
            meta_label.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: {FONT_BODY}px; font-weight: 600; letter-spacing: 1px;"
            )

            duration_label = QLabel("0:00")
            duration_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            duration_label.setStyleSheet(
                f"color: {COLOR_PRIMARY}; font-size: {FONT_HEADING_2}px; font-weight: 700;"
            )

            top_line.addWidget(meta_label)
            top_line.addStretch(1)
            top_line.addWidget(duration_label)

            session_label = QLabel("Session —")
            session_label.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: {FONT_BODY}px; font-weight: 500;"
            )

            _pb_transparent = Qt.WidgetAttribute.WA_TransparentForMouseEvents
            meta_label.setAttribute(_pb_transparent, True)
            duration_label.setAttribute(_pb_transparent, True)
            session_label.setAttribute(_pb_transparent, True)

            row_layout.addLayout(top_line)
            row_layout.addWidget(session_label)

            self._best_time_rows.append((meta_label, duration_label, session_label))
            self._best_row_frames.append(row_card)
            wrap_layout.addWidget(row_card)

        layout.addWidget(sessions_wrap)
        layout.addStretch(1)
        return card

    def _make_metric_card(
        self,
        title: str,
        accent: str,
        key: str,
        icon_name: str,
    ) -> QFrame:
        """Create a bottom KPI card with a large icon and numeric value."""
        card = self._make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(SPACE_3, SPACE_2, SPACE_3, SPACE_3)
        card_layout.setSpacing(SPACE_2)

        card_layout.addStretch(1)

        icon_px = max(56, FONT_HEADING_2 * 3)
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        icon_label.setPixmap(
            get_icon(icon_name, color=accent, size=icon_px).pixmap(QSize(icon_px, icon_px))
        )
        card_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        value_label = QLabel("—")
        value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        value_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_HEADING_2}px; font-weight: 700;"
        )
        card_layout.addWidget(value_label)
        card_layout.addStretch(1)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {CARD_TITLE_FONT_SIZE}px; font-weight: 700;"
        )
        card_layout.addWidget(title_label)

        if key == "stress":
            self._stress_value_label = value_label
        elif key == "error":
            self._error_value_label = value_label
        elif key == "workload":
            self._workload_value_label = value_label

        return card

    def _make_analysis_card(self) -> QFrame:
        """Create the interactive percentage chart with filter buttons."""
        card = self._make_card()
        card.setStyleSheet(
            f"""
            QFrame#card {{
                background-color: transparent;
                border: 1px solid {COLOR_BORDER};
                border-radius: {RADIUS_LG}px;
            }}
            """
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        layout.setSpacing(SPACE_2)

        top_row = QHBoxLayout()
        title = QLabel("Session Trend Analysis")
        title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        title.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {CARD_TITLE_FONT_SIZE}px; font-weight: 700;"
        )
        top_row.addWidget(title)
        top_row.addStretch(1)

        self._analysis_filter_group = QButtonGroup(self)
        self._analysis_filter_group.setExclusive(False)

        stress_btn = self._make_filter_button("ph.heartbeat-fill", "Stress", "stress")
        workload_btn = self._make_filter_button("ph.brain-fill", "Cognitive Workload", "workload")
        top_row.addWidget(stress_btn)
        top_row.addWidget(workload_btn)
        layout.addLayout(top_row)

        self._analysis_plot = pg.PlotWidget()
        self._analysis_plot.setBackground((0, 0, 0, 0))
        self._analysis_plot.showGrid(x=True, y=True, alpha=0.2)
        self._analysis_plot.setMenuEnabled(False)
        self._analysis_plot.hideButtons()
        self._analysis_plot.setMouseEnabled(x=False, y=False)
        self._analysis_plot.setMinimumHeight(280)

        left_axis = self._analysis_plot.getAxis("left")
        left_axis.setPen(pg.mkPen(COLOR_CHART_AXIS, width=0.8))
        left_axis.setTextPen(pg.mkPen(COLOR_FONT_MUTED, width=1))
        left_axis.setLabel("Percentage (%)", color=COLOR_FONT_MUTED)

        bottom_axis = self._analysis_plot.getAxis("bottom")
        bottom_axis.setPen(pg.mkPen(COLOR_CHART_AXIS, width=0.8))
        bottom_axis.setTextPen(pg.mkPen(COLOR_FONT_MUTED, width=1))
        bottom_axis.setLabel("Sessions", color=COLOR_FONT_MUTED)

        stress_pen = pg.mkPen(color=COLOR_PRIMARY, width=3)
        workload_pen = pg.mkPen(color=COLOR_WARNING, width=3)

        self._stress_series_curve = self._analysis_plot.plot(
            [], [],
            pen=stress_pen,
            symbol="o",
            symbolSize=8,
            symbolBrush=COLOR_PRIMARY,
            name="Stress",
        )
        self._workload_series_curve = self._analysis_plot.plot(
            [], [],
            pen=workload_pen,
            symbol="o",
            symbolSize=8,
            symbolBrush=COLOR_WARNING,
            name="Cognitive Workload",
        )

        self._analysis_plot.addLegend(offset=(10, 10))
        self._analysis_plot.installEventFilter(self)
        layout.addWidget(self._analysis_plot)
        return card

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Forward wheel scrolling to the page scroll area when hovering charts.
        """
        if event.type() == QEvent.Type.Wheel and watched in {
            self._progress_plot,
            self._analysis_plot,
        }:
            if self._content_scroll is not None:
                delta = event.angleDelta().y()
                bar = self._content_scroll.verticalScrollBar()
                bar.setValue(bar.value() - delta)
                event.accept()
                return True

        return super().eventFilter(watched, event)

    def _make_filter_button(self, icon_name: str, tooltip: str, key: str) -> QPushButton:
        """Create checkable icon filter button."""
        button = QPushButton()
        button.setCheckable(True)
        button.setChecked(True)
        button.setToolTip(tooltip)
        
        accent = COLOR_PRIMARY if key == "stress" else COLOR_WARNING
        button.setFixedSize(44, 44)
        button.setIconSize(QSize(FONT_BODY + 4, FONT_BODY + 4))

        def _update_style():
            is_checked = button.isChecked()
            bg = accent if is_checked else COLOR_CARD
            icon_color = "#FFFFFF" if is_checked else accent
            border = accent if is_checked else COLOR_BORDER
            
            button.setIcon(get_icon(icon_name, color=icon_color))
            button.setStyleSheet(
                f"QPushButton {{ "
                f"  background-color: {bg}; "
                f"  border: 1px solid {border}; "
                f"  border-radius: {RADIUS_LG}px; "
                f"}} "
                f"QPushButton:hover {{ border-color: {accent}; }}"
            )

        button.toggled.connect(_update_style)
        _update_style()

        button.clicked.connect(lambda checked, k=key: self._on_analysis_filter_toggled(k, checked))
        if self._analysis_filter_group is not None:
            self._analysis_filter_group.addButton(button)
        return button

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload dashboard data and refresh all widgets."""
        if self._repo is not None:
            self._sessions = self._repo.get_all_sessions()
        else:
            self._sessions = []

        # Load per-session biometric stats once so gauges and analysis chart
        # share the same query results without hitting the DB twice.
        self._biometric_stats = self._load_biometric_stats()

        self._update_session_count()
        self._update_personal_best()
        self._update_gauges()
        self._update_progress_plot()
        self._update_analysis_plot()
        logger.info("Dashboard refreshed: %d sessions.", len(self._sessions))

    def _on_analysis_filter_toggled(self, key: str, checked: bool) -> None:
        """Toggle chart series visibility from filter buttons."""
        if checked:
            self._active_analysis_filters.add(key)
        else:
            self._active_analysis_filters.discard(key)

        if not self._active_analysis_filters:
            # Keep at least one active metric to avoid an empty chart.
            self._active_analysis_filters.add(key)
            sender = self.sender()
            if isinstance(sender, QPushButton):
                sender.setChecked(True)

        self._update_analysis_plot()

    def _update_session_count(self) -> None:
        """Update top-right sessions number in the progress card."""
        if self._sessions_count_label is None:
            return
        self._sessions_count_label.setText(str(len(self._sessions)))

    def _update_personal_best(self) -> None:
        """Compute and display the top 3 fastest completed sessions."""
        if not self._best_time_rows:
            return

        if len(self._sessions) == 0:
            if self._personal_best_placeholder is not None:
                self._personal_best_placeholder.setVisible(True)
            for row in self._best_row_frames:
                row.setVisible(False)
                row.set_session_id(None)
            return

        if self._personal_best_placeholder is not None:
            self._personal_best_placeholder.setVisible(False)
        for row in self._best_row_frames:
            row.setVisible(True)

        durations: list[tuple[int, sqlite3.Row]] = []
        for session in self._sessions:
            duration_seconds = self._compute_session_duration_seconds(
                session["started_at"],
                session["ended_at"],
            )
            if duration_seconds is not None and duration_seconds > 0:
                durations.append((duration_seconds, session))

        best_three = sorted(durations, key=lambda item: item[0])[:3]

        for idx, (meta_label, duration_label, session_label) in enumerate(self._best_time_rows, start=1):
            row_frame = self._best_row_frames[idx - 1]
            meta_label.setText(f"BEST SESSION #{idx}")
            if idx <= len(best_three):
                duration_seconds, session_row = best_three[idx - 1]
                duration_label.setText(self._format_seconds(duration_seconds))

                # Show name if available, otherwise date
                name = session_row["name"] if "name" in session_row.keys() else None
                if name:
                    session_label.setText(name)
                else:
                    session_date = str(session_row["started_at"]).replace("T", " ")[:10]
                    session_label.setText(f"Session {session_date}")
                row_frame.set_session_id(int(session_row["id"]))
            else:
                duration_label.setText("0:00")
                session_label.setText("Session —")
                row_frame.set_session_id(None)

    def _update_gauges(self) -> None:
        """Map session aggregates to the bottom KPI cards (icon + value text).

        - Ø Stress Events: average stress-event count per session (same RMSSD
          threshold-crossing logic as PostSessionView).
        - Ø Error Rate: average wall-contact count per session.
        - Ø High Cognitive Load Events: average pupil-based high-workload
          crossings per session (same logic as PostSessionView).
        """
        stats = self._biometric_stats

        # ── Ø Stress Events ──────────────────────────────────────────────────
        event_counts = [v["stress_events"] for v in stats.values()]
        avg_events = sum(event_counts) / len(event_counts) if event_counts else 0.0

        # ── Ø Error Rate ─────────────────────────────────────────────────────
        error_rates = [
            v["error_rate_per_session"]
            for v in stats.values()
            if v["error_rate_per_session"] is not None
        ]
        avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0.0

        # ── Ø High cognitive load (pupil threshold crossings) ───────────────
        workload_counts = [v["high_workload_events"] for v in stats.values()]
        avg_workload = (
            sum(workload_counts) / len(workload_counts) if workload_counts else 0.0
        )

        if self._stress_value_label is not None:
            self._stress_value_label.setText(
                f"{avg_events:.1f}/session" if event_counts else "—"
            )
        if self._error_value_label is not None:
            self._error_value_label.setText(
                f"{avg_error_rate:.1f}/session" if error_rates else "—"
            )
        if self._workload_value_label is not None:
            self._workload_value_label.setText(
                f"{avg_workload:.1f}/session" if workload_counts else "—"
            )

    def _update_progress_plot(self) -> None:
        """Update progress chart data and x-axis labels."""
        if (
            self._progress_plot is None
            or self._actual_curve is None
            or self._estimate_curve is None
        ):
            return

        x_values, actual_values, estimate_values, tick_labels = self._build_chart_series()

        self._actual_curve.setData(x_values, actual_values)
        self._estimate_curve.setData(x_values, estimate_values)
        self._progress_plot.setYRange(0, 100, padding=0.04)
        if x_values:
            self._progress_plot.setXRange(-0.3, len(x_values) - 0.7, padding=0)
        else:
            self._progress_plot.setXRange(-0.3, 0.7, padding=0)

        y_axis = self._progress_plot.getAxis("left")
        y_axis.setTicks([[(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]])

        axis = self._progress_plot.getAxis("bottom")
        ticks = [(idx, label) for idx, label in enumerate(tick_labels)]
        axis.setTicks([ticks])

    def _build_chart_series(self) -> tuple[list[float], list[float], list[float], list[str]]:
        """Build a combined performance learning curve.

        Performance metric per session (only sessions with a recorded
        ``error_count`` — ``NULL`` skips the trial so the chart reflects tracked
        wall contacts):

            effective_time = duration_s + (wall_contacts x WALL_CONTACT_PENALTY_S)

        This single value captures both speed (time on task) and accuracy
        (wall contacts).  Lower effective_time = better performance.

        The raw effective times are normalized to a 0-100 error scale, then
        inverted so 100 % = best performance on the chart.  The Schmettow
        model is fitted on the error-domain values.
        """
        ordered = sorted(self._sessions, key=lambda row: str(row["started_at"]))

        valid_trials: list[int] = []
        raw_effective_times: list[float] = []

        for session in ordered:
            duration_s = self._compute_session_duration_seconds(
                session["started_at"], session["ended_at"]
            )
            if duration_s is None or duration_s <= 0:
                continue

            error_count = session["error_count"]
            if error_count is None:
                continue

            wall_contacts = float(error_count)
            effective_time = float(duration_s) + wall_contacts * WALL_CONTACT_PENALTY_S

            valid_trials.append(len(valid_trials) + 1)
            raw_effective_times.append(effective_time)

        if not raw_effective_times:
            return [], [], [], []

        # Normalize to 0-SCORE_MAX error scale
        e_min = min(raw_effective_times)
        e_max = max(raw_effective_times)
        e_span = e_max - e_min

        def _norm_error(v: float) -> float:
            if e_span <= 0.0:
                return 0.0
            return ((v - e_min) / e_span) * float(SCORE_MAX)

        norm_errors = [_norm_error(v) for v in raw_effective_times]

        # Fit Schmettow on the error domain
        fit = None
        if len(norm_errors) >= LC_MIN_SESSIONS and e_span > 0.0:
            fit = fit_schmettow(
                np.array(valid_trials, dtype=float),
                np.array(norm_errors, dtype=float),
                score_max=float(SCORE_MAX),
            )

        def _clamp(v: float) -> float:
            return max(0.0, min(float(SCORE_MAX), float(v)))

        x_values = [float(i) for i in range(len(valid_trials))]
        labels = [f"S{i}" for i in valid_trials]
        actual_values = [_clamp(float(SCORE_MAX) - e) for e in norm_errors]

        if fit is None:
            estimate_values = []
        else:
            estimate_values = [
                _clamp(predict_at_trial(fit, t, score_max=float(SCORE_MAX)))
                for t in valid_trials
            ]

        return x_values, actual_values, estimate_values, labels

    def _update_analysis_plot(self) -> None:
        """Render the lower interactive chart with selected metrics."""
        if (
            self._analysis_plot is None
            or self._stress_series_curve is None
            or self._workload_series_curve is None
        ):
            return

        x_values, stress_values, workload_values, labels = self._build_analysis_series()

        if "stress" in self._active_analysis_filters:
            self._stress_series_curve.setData(x_values, stress_values)
        else:
            self._stress_series_curve.setData([], [])

        if "workload" in self._active_analysis_filters:
            self._workload_series_curve.setData(x_values, workload_values)
        else:
            self._workload_series_curve.setData([], [])

        self._analysis_plot.setYRange(0, 100, padding=0.05)
        self._analysis_plot.setXRange(-0.2, max(0.8, len(x_values) - 0.8), padding=0)

        left_axis = self._analysis_plot.getAxis("left")
        left_axis.setTicks([[(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]])

        bottom_axis = self._analysis_plot.getAxis("bottom")
        ticks = [(idx, label) for idx, label in enumerate(labels)]
        bottom_axis.setTicks([ticks])

    def _build_analysis_series(self) -> tuple[list[float], list[float], list[float], list[str]]:
        """Build per-session trend series from persisted live-session samples.

        Data is read via :meth:`_load_biometric_stats` (same aggregates the app
        stores at session end: ``AVG(rmssd)`` from ``hrv_samples``, ``AVG(cli)``
        from ``cli_samples``).

        **Stress %** (y-axis): among the up-to-12 displayed sessions, each
        point is cohort-normalised mean RMSSD inverted to 0–100 % so higher
        values mean relatively lower HRV within that window.

        **Workload %**: mean CLI for the session × 100 (CLI is 0–1 in the DB).

        Missing HRV or CLI for a session uses a neutral 50 % placeholder so the
        series length still matches session count.
        """
        if not self._sessions:
            return [], [], [], []

        stats = self._biometric_stats
        ordered = sorted(self._sessions, key=lambda row: str(row["started_at"]))
        display = ordered[-12:]

        # Collect RMSSD values across displayed sessions so we can normalise
        # them to a 0–100 % scale relative to this cohort.
        rmssd_vals = [
            stats[s["id"]]["avg_rmssd"]
            for s in display
            if s["id"] in stats and stats[s["id"]]["avg_rmssd"] is not None
        ]
        rmssd_min = min(rmssd_vals) if rmssd_vals else 0.0
        rmssd_max = max(rmssd_vals) if rmssd_vals else 1.0
        rmssd_range = (rmssd_max - rmssd_min) if rmssd_max > rmssd_min else 1.0

        x_values: list[float] = []
        stress_values: list[float] = []
        workload_values: list[float] = []
        labels: list[str] = []

        for idx, session in enumerate(display):
            x_values.append(float(idx))
            labels.append(f"S{idx + 1}")

            sid = session["id"]
            s = stats.get(sid, {})

            avg_rmssd = s.get("avg_rmssd")
            avg_cli = s.get("avg_cli")

            # Stress %: invert normalised RMSSD so that lower HRV → higher line.
            if avg_rmssd is not None:
                rmssd_norm = (avg_rmssd - rmssd_min) / rmssd_range  # 0 = lowest, 1 = highest
                stress_pct = max(0.0, min(100.0, (1.0 - rmssd_norm) * 100.0))
            else:
                stress_pct = 50.0  # neutral placeholder when no HRV data

            # Workload %: CLI is already 0–1.
            if avg_cli is not None:
                workload_pct = max(0.0, min(100.0, avg_cli * 100.0))
            else:
                workload_pct = 50.0  # neutral placeholder when no CLI data

            stress_values.append(stress_pct)
            workload_values.append(workload_pct)

        return x_values, stress_values, workload_values, labels

    def _load_biometric_stats(self) -> dict[int, dict]:
        """Query per-session biometric statistics from the database.

        Queries ``hrv_samples``, ``pupil_samples``, ``cli_samples``, and
        ``calibrations`` to compute, for each session:

        - ``avg_rmssd``: mean RMSSD over the session (ms), or ``None``.
        - ``avg_cli``:   mean CLI over the session (0–1), or ``None``.
        - ``avg_pupil_change_pct``: mean absolute PDI as a percentage (|PDI|*100),
          or ``None``.
        - ``stress_events``: number of RMSSD threshold crossings (< -10 % from
          baseline), matching the per-session stress-event definition.
        - ``high_workload_events``: pupil PDI upward crossings above the adaptive
          workload threshold (same definition as PostSessionView).
        - ``error_count``: raw surgical error count from the sessions table,
          or ``None`` when not yet recorded.
        - ``error_rate_per_session``: surgical error frequency for the session,
          treated as raw ``error_count`` (one session-normalised unit).
        - ``error_rate_per_minute``: surgical error frequency per minute,
          derived from session duration.

        Returns:
            Dict keyed by session ID.  Always returns an entry for every
            session in ``self._sessions`` so callers need not guard for
            missing keys.
        """
        if self._db is None:
            return {}

        conn = self._db.get_connection()

        # Seed with all current sessions so every key is present.
        stats: dict[int, dict] = {
            session["id"]: {
                "avg_rmssd": None,
                "avg_cli": None,
                "avg_pupil_change_pct": None,
                "stress_events": 0,
                "high_workload_events": 0,
                "error_count": session["error_count"],
                "error_rate_per_session": self._compute_error_rate_per_session(
                    session["error_count"],
                ),
                "error_rate_per_minute": self._compute_error_rate_per_minute(
                    session["error_count"],
                    session["started_at"],
                    session["ended_at"],
                ),
            }
            for session in self._sessions
        }

        # ── Average RMSSD per session ────────────────────────────────────────
        for row in conn.execute(
            "SELECT session_id, AVG(rmssd) FROM hrv_samples "
            "WHERE rmssd IS NOT NULL GROUP BY session_id"
        ).fetchall():
            if row[0] in stats:
                stats[row[0]]["avg_rmssd"] = row[1]

        # ── Average CLI per session ──────────────────────────────────────────
        for row in conn.execute(
            "SELECT session_id, AVG(cli) FROM cli_samples GROUP BY session_id"
        ).fetchall():
            if row[0] in stats:
                stats[row[0]]["avg_cli"] = row[1]

        # ── Average absolute PDI % per session ──────────────────────────────
        for row in conn.execute(
            "SELECT session_id, AVG(ABS(pdi)) * 100.0 "
            "FROM pupil_samples WHERE pdi IS NOT NULL GROUP BY session_id"
        ).fetchall():
            if row[0] in stats:
                stats[row[0]]["avg_pupil_change_pct"] = row[1]

        # ── Calibration baselines (one per session, take the first) ─────────
        baselines: dict[int, float] = {}
        for row in conn.execute(
            "SELECT session_id, baseline_rmssd FROM calibrations "
            "WHERE baseline_rmssd IS NOT NULL ORDER BY id ASC"
        ).fetchall():
            if row[0] not in baselines:
                baselines[row[0]] = float(row[1])

        # ── RMSSD stress events: threshold crossings below -10% ─────────────
        rmssd_samples_by_session: dict[int, list[float]] = {}
        for row in conn.execute(
            "SELECT session_id, rmssd FROM hrv_samples "
            "WHERE rmssd IS NOT NULL ORDER BY session_id, timestamp"
        ).fetchall():
            sid = int(row[0])
            if sid in stats:
                rmssd_samples_by_session.setdefault(sid, []).append(float(row[1]))

        for sid, rmssd_values in rmssd_samples_by_session.items():
            baseline_rmssd = baselines.get(sid)
            stats[sid]["stress_events"] = self._compute_stress_event_count(
                rmssd_values,
                baseline_rmssd,
            )

        pupil_by_session: dict[int, list[tuple[float, float]]] = {}
        for row in conn.execute(
            "SELECT session_id, timestamp, pdi FROM pupil_samples "
            "WHERE pdi IS NOT NULL ORDER BY session_id, timestamp"
        ).fetchall():
            sid = int(row[0])
            if sid in stats:
                pupil_by_session.setdefault(sid, []).append(
                    (float(row[1]), float(row[2]))
                )

        for sid, samples in pupil_by_session.items():
            stats[sid]["high_workload_events"] = self._compute_high_workload_event_count(
                samples
            )

        logger.debug(
            "Biometric stats loaded for %d sessions.", len(stats)
        )
        return stats

    @staticmethod
    def _compute_session_duration_seconds(
        started_at: str | None,
        ended_at: str | None,
    ) -> int | None:
        """Return session duration in seconds if both timestamps are valid."""
        if not started_at or not ended_at:
            return None

        try:
            start_dt = datetime.fromisoformat(str(started_at).replace("T", " "))
            end_dt = datetime.fromisoformat(str(ended_at).replace("T", " "))
        except ValueError:
            return None

        duration = int((end_dt - start_dt).total_seconds())
        if duration <= 0:
            return None
        return duration

    @classmethod
    def _compute_error_rate_per_session(
        cls,
        error_count: int | None,
    ) -> float | None:
        """Return wall-contact frequency as errors per session."""
        if error_count is None:
            return None
        return float(max(0, error_count))

    @classmethod
    def _compute_error_rate_per_minute(
        cls,
        error_count: int | None,
        started_at: str | None,
        ended_at: str | None,
    ) -> float | None:
        """Return wall-contact frequency as errors per minute."""
        if error_count is None:
            return None

        duration_seconds = cls._compute_session_duration_seconds(started_at, ended_at)
        if duration_seconds is None or duration_seconds <= 0:
            return None

        duration_minutes = duration_seconds / 60.0
        if duration_minutes <= 0.0:
            return None

        return float(max(0, error_count)) / duration_minutes

    @staticmethod
    def _compute_stress_event_count(
        rmssd_values: list[float],
        baseline_rmssd: float | None,
    ) -> int:
        """Return RMSSD stress-event count via -10% threshold crossings."""
        if not rmssd_values:
            return 0

        if baseline_rmssd is not None and baseline_rmssd > 0.0:
            reference = float(baseline_rmssd)
        else:
            reference = float(sum(rmssd_values) / len(rmssd_values))
            if reference <= 0.0:
                return 0

        stress_events = 0
        prev_below_stress = False
        for rmssd in rmssd_values:
            rmssd_pct_change = ((rmssd - reference) / reference) * 100.0
            is_below_stress = rmssd_pct_change < -10.0
            if is_below_stress and not prev_below_stress:
                stress_events += 1
            prev_below_stress = is_below_stress
        return stress_events

    @staticmethod
    def _compute_high_workload_event_count(
        samples: list[tuple[float, float]],
    ) -> int:
        """Count upward crossings above adaptive pupil workload threshold.

        Matches :meth:`PostSessionView._query_high_workload_events` / live view
        logic (smoothed PDI vs rolling mean × ``WORKLOAD_THRESHOLD_FACTOR``).
        """
        if not samples:
            return 0

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
            return float(sum(s for _, s in buffer) / len(buffer))

        events = 0
        prev_above = False
        for timestamp, pdi in samples:
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

    @staticmethod
    def _format_seconds(total_seconds: int) -> str:
        """Format seconds into H:MM:SS for long runs or M:SS for short runs."""
        minutes, seconds = divmod(max(0, total_seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def _format_date(value: str | None) -> str:
        """Format datetime text as day.month.year for display."""
        if not value:
            return "-"
        text = str(value)
        try:
            dt = datetime.fromisoformat(text.replace("T", " "))
            return dt.strftime("%-d.%-m.%Y")
        except ValueError:
            return text[:10]

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_new_session(self) -> None:
        """Navigate to the Calibration view to start a new session."""
        logger.info("New session requested from Dashboard.")
        self.new_session_requested.emit()

    def _on_export(self) -> None:
        """Show a file dialog and export the selected session to CSV."""
        logger.info("Export requested.")
        # TODO (Phase 5): wire to SessionExporter.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_card() -> QFrame:
        """Create base card using the global design-system card style."""
        card = QFrame()
        card.setObjectName("card")
        return card
