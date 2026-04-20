"""ExcelImportView — interactive page for importing and analysing LapSim data.

Allows researchers to upload LapSim Excel exports, select a metric, fit the
learning curve model, and save datasets to a local history for later review.
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from app.analytics.lapsim_parser import LapSimParser, ParsedDataset, TrialRecord
from app.analytics.lapsim_metrics import extract_metric_series, compute_performance_series
from app.analytics.learning_curve import SessionDataPoint, SchmettowFit
from app.storage.database import DatabaseManager
from app.ui.widgets.learning_curve_chart import LearningCurveChart
from app.ui.workers.analytics_worker import LearningCurveWorker
from app.ui.theme import (
    CARD_PADDING,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_DANGER,
    COLOR_DANGER_BG,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_PRIMARY_SUBTLE,
    COLOR_SUCCESS,
    COLOR_WARNING,
    CONTENT_PADDING_H,
    CONTENT_PADDING_V,
    FONT_BODY,
    FONT_CAPTION,
    FONT_HEADING_2,
    FONT_SMALL,
    RADIUS_LG,
    RADIUS_MD,
    SPACE_1,
    SPACE_2,
    SPACE_3,
    WEIGHT_BOLD,
    WEIGHT_SEMIBOLD,
    get_icon,
)
from app.utils.config import LC_MIN_SESSIONS
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ExcelImportView(QWidget):
    """View for simplified importing and analysis of historical LapSim learning curves."""

    close_requested = pyqtSignal()

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._parser = LapSimParser()
        
        self._current_dataset: Optional[ParsedDataset] = None
        self._current_path: Optional[str] = None
        self._current_fit: Optional[SchmettowFit] = None
        
        self._lc_thread: Optional[QThread] = None
        self._lc_worker: Optional[LearningCurveWorker] = None

        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header_widget = QWidget()
        header_widget.setObjectName("header")
        header_widget.setStyleSheet(f"background: {COLOR_BACKGROUND};")
        header_v = QVBoxLayout(header_widget)
        header_v.setContentsMargins(CONTENT_PADDING_H, CONTENT_PADDING_V,
                                    CONTENT_PADDING_H, SPACE_2)
        header_v.setSpacing(SPACE_2)
        header_v.addLayout(self._build_header())
        outer.addWidget(header_widget)

        # ── Content Area ───────────────────────────────────────────
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(CONTENT_PADDING_H, 0, CONTENT_PADDING_H, CONTENT_PADDING_V)
        body_layout.setSpacing(SPACE_3)

        # 1. Top Control Bar (Drop Zone + Exercise + Participant)
        controls_row = QHBoxLayout()
        controls_row.setSpacing(SPACE_3)

        # Compact Drop Zone / Import Button
        self._drop_zone = QPushButton(" Click or Drop Excel File")
        self._drop_zone.setIcon(get_icon("ph.file-arrow-up", color=COLOR_PRIMARY))
        self._drop_zone.setIconSize(QSize(24, 24))
        self._drop_zone.setFixedHeight(112)
        self._drop_zone.setMaximumWidth(720)
        self._drop_zone.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {COLOR_PRIMARY}; "
            f"border: 2px dashed {COLOR_BORDER}; border-radius: {RADIUS_LG}px; "
            f"font-size: {FONT_BODY}px; font-weight: 600; padding: 0 24px; }}"
            f"QPushButton:hover {{ background-color: {COLOR_PRIMARY_SUBTLE}; border-color: {COLOR_PRIMARY}; }}"
        )
        self._drop_zone.clicked.connect(self._on_browse_clicked)
        controls_row.addWidget(self._drop_zone, stretch=1)

        # Participant Selector
        self._participant_card = QWidget()
        self._participant_card.setVisible(False)
        part_layout = QVBoxLayout(self._participant_card)
        part_layout.setContentsMargins(0, SPACE_2, 0, SPACE_2)
        part_layout.setSpacing(SPACE_1)
        
        part_header = QHBoxLayout()
        part_icon = QLabel()
        part_icon.setPixmap(get_icon("ph.user-circle", color=COLOR_FONT_MUTED).pixmap(16, 16))
        part_header.addWidget(part_icon)
        part_label = QLabel("Participant")
        part_label.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px; font-weight: {WEIGHT_SEMIBOLD};")
        part_header.addWidget(part_label)
        part_header.addStretch(1)
        part_layout.addLayout(part_header)
        
        self._participant_combo = QComboBox()
        self._participant_combo.setMinimumWidth(240)
        self._participant_combo.currentIndexChanged.connect(self._on_participant_changed)
        part_layout.addWidget(self._participant_combo)
        controls_row.addWidget(self._participant_card)

        # Exercise Selector
        self._exercise_card = QWidget()
        self._exercise_card.setVisible(False)
        ex_layout = QVBoxLayout(self._exercise_card)
        ex_layout.setContentsMargins(0, SPACE_2, 0, SPACE_2)
        ex_layout.setSpacing(SPACE_1)
        
        ex_header = QHBoxLayout()
        ex_icon = QLabel()
        ex_icon.setPixmap(get_icon("ph.activity", color=COLOR_FONT_MUTED).pixmap(16, 16))
        ex_header.addWidget(ex_icon)
        ex_label = QLabel("Exercise")
        ex_label.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px; font-weight: {WEIGHT_SEMIBOLD};")
        ex_header.addWidget(ex_label)
        ex_header.addStretch(1)
        ex_layout.addLayout(ex_header)
        
        self._exercise_combo = QComboBox()
        self._exercise_combo.setMinimumWidth(240)
        self._exercise_combo.currentIndexChanged.connect(self._on_exercise_changed)
        ex_layout.addWidget(self._exercise_combo)
        controls_row.addWidget(self._exercise_card)

        body_layout.addLayout(controls_row)

        # Error/Warning Panel
        self._info_panel = QFrame()
        self._info_panel.setFixedHeight(50)
        self._info_panel.setVisible(False)
        info_layout = QHBoxLayout(self._info_panel)
        self._info_icon = QLabel()
        info_layout.addWidget(self._info_icon)
        self._info_text = QLabel()
        self._info_text.setWordWrap(True)
        info_layout.addWidget(self._info_text, stretch=1)
        body_layout.addWidget(self._info_panel)

        # 2. Main Analysis Area
        self._analysis_card = QFrame()
        self._analysis_card.setObjectName("card")
        self._analysis_card.setStyleSheet(
            f"QFrame#card {{ background-color: transparent; border: 1px solid {COLOR_BORDER};"
            f" border-radius: {RADIUS_LG}px; }}"
        )
        self._analysis_card.setVisible(False)
        analysis_layout = QVBoxLayout(self._analysis_card)
        analysis_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        
        self._chart = LearningCurveChart(show_position_marker=False, transparent=True)
        analysis_layout.addWidget(self._chart, stretch=1)
        
        # Simple stats footer
        self._stats_lbl = QLabel("Ready to import performance data")
        self._stats_lbl.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px;")
        analysis_layout.addWidget(self._stats_lbl)
        
        body_layout.addWidget(self._analysis_card, stretch=1)
        outer.addWidget(body)

    def _build_header(self) -> QHBoxLayout:
        """Create the top row with the page title."""
        header = QHBoxLayout()
        header.setSpacing(SPACE_2)

        title = QLabel("Import")
        title.setObjectName("heading")
        title.setFixedHeight(FONT_HEADING_2 * 2)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(title)
        header.addStretch(1)
        
        self._header_import_btn = QPushButton("Import Data")
        self._header_import_btn.setIcon(get_icon("ph.file-arrow-down", color="#FFFFFF"))
        self._header_import_btn.setIconSize(QSize(FONT_BODY + 2, FONT_BODY + 2))
        self._header_import_btn.setFixedHeight(FONT_HEADING_2 * 2)
        self._header_import_btn.setMinimumWidth(170)
        self._header_import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header_import_btn.setStyleSheet(
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
        self._header_import_btn.clicked.connect(self._on_browse_clicked)
        header.addWidget(self._header_import_btn)
        
        return header

    def _make_card(self, title_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(
            f"QFrame#card {{ background-color: {COLOR_CARD}; border: 1px solid {COLOR_BORDER};"
            f" border-radius: {RADIUS_LG}px; }}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        layout.setSpacing(SPACE_2)
        title = QLabel(title_text)
        title.setStyleSheet(f"color: {COLOR_FONT}; font-size: {FONT_BODY}px; font-weight: {WEIGHT_BOLD};")
        layout.addWidget(title)
        return card

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().endswith(".xlsx") for url in urls):
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.endswith(".xlsx"):
                self._handle_file_selected(path)
                break
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _on_browse_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select LapSim Export", "", "Excel Files (*.xlsx)")
        if path:
            self._handle_file_selected(path)

    def _handle_file_selected(self, path: str) -> None:
        self._current_path = path
        self._info_panel.setVisible(False)
        self._exercise_card.setVisible(False)
        self._participant_card.setVisible(False)
        self._analysis_card.setVisible(False)

        sheets = self._parser.list_sheets(path)
        if not sheets:
            self._show_error("No sheets found in file.")
            return

        # Only include sheets with at least LC_MIN_SESSIONS
        valid_sheets = []
        for s in sheets:
            if self._parser.get_data_row_count(path, s) >= LC_MIN_SESSIONS:
                valid_sheets.append(s)

        if not valid_sheets:
            self._show_error(f"No valid exercises found (at least {LC_MIN_SESSIONS} sessions required).")
            return

        # Populate exercise dropdown
        self._exercise_combo.blockSignals(True)
        self._exercise_combo.clear()
        for s in valid_sheets:
            self._exercise_combo.addItem(s, s)
        self._exercise_combo.blockSignals(False)

        self._exercise_card.setVisible(True)
        self._drop_zone.setVisible(False)

        # Select first sheet → triggers participant population + fit
        self._exercise_combo.setCurrentIndex(0)
        self._on_exercise_changed(0)

    def _on_exercise_changed(self, index: int) -> None:
        if index < 0 or not self._current_path:
            return

        sheet = self._exercise_combo.currentData()
        participants = self._parser.get_participants(self._current_path, sheet)
        if not participants:
            self._show_error(f"No valid participant data found in sheet '{sheet}'.")
            self._participant_card.setVisible(False)
            return

        self._participant_combo.blockSignals(True)
        self._participant_combo.clear()
        for p in participants:
            name = f"{p['firstname']} {p['lastname']}".strip()
            label = f"{name} ({p['login']})" if name else p['login']
            self._participant_combo.addItem(label, p['login'])
        self._participant_combo.blockSignals(False)

        self._participant_card.setVisible(True)
        self._participant_combo.setCurrentIndex(0)
        self._on_participant_changed(0)

    def _on_participant_changed(self, index: int) -> None:
        if index < 0 or not self._current_path:
            return

        login = self._participant_combo.currentData()
        sheet = self._exercise_combo.currentData()

        try:
            dataset = self._parser.parse(self._current_path, sheet, login=login)
            self._current_dataset = dataset
            self._on_fit_requested()
        except ValueError as e:
            self._show_error(str(e))

    def _on_fit_requested(self) -> None:
        if not self._current_dataset:
            return

        trials, errors, score_max = compute_performance_series(self._current_dataset.trials)

        if len(trials) == 0:
            self._show_error("No time data found — cannot compute performance metric.")
            return

        if len(trials) < 5:
            self._show_warning(
                f"Only {len(trials)} sessions available — need at least 5 to fit a learning curve."
            )
            return

        data_points = [
            SessionDataPoint(trial=int(t), error_count=float(e), performance_score=score_max - float(e))
            for t, e in zip(trials, errors)
        ]

        self._start_fit_worker(trials, errors, score_max, data_points, "Performance (Time + Tissue Damage)")

    def _start_fit_worker(self, trials, errors, score_max, data_points, metric):
        self._drop_zone.setEnabled(False)
        
        self._lc_thread = QThread()
        self._lc_worker = LearningCurveWorker(trials, errors, score_max)
        self._lc_worker.moveToThread(self._lc_thread)
        
        self._lc_thread.started.connect(self._lc_worker.run)
        self._lc_worker.finished.connect(lambda fit: self._on_fit_finished(fit, data_points, metric))
        self._lc_worker.finished.connect(self._lc_thread.quit)
        self._lc_worker.finished.connect(self._lc_worker.deleteLater)
        self._lc_thread.finished.connect(self._lc_thread.quit) # Ensure quit
        
        self._lc_thread.start()

    def _on_fit_finished(self, fit: Optional[SchmettowFit], data_points: List[SessionDataPoint], metric: str) -> None:
        self._drop_zone.setEnabled(True)
        self._current_fit = fit
        
        self._analysis_card.setVisible(True)
        self._chart.update_data(data_points, fit, metric_label=metric, y_axis_inverted=False)
        
        if fit:
            self._stats_lbl.setText("")
            self._drop_zone.setVisible(False)
        else:
            self._stats_lbl.setText("Could not fit learning curve model to this data.")
            self._drop_zone.setVisible(True)
            self._drop_zone.setText(" Click or Drop Excel File")

    def _show_error(self, msg: str) -> None:
        self._info_panel.setVisible(True)
        self._info_panel.setStyleSheet(f"background: {COLOR_DANGER_BG}; border-radius: 8px;")
        self._info_text.setText(msg)
        self._info_text.setStyleSheet(f"color: {COLOR_DANGER}; font-size: {FONT_SMALL}px; font-weight: 600;")
        self._info_icon.setPixmap(get_icon("ph.x-circle-fill", color=COLOR_DANGER).pixmap(16, 16))

    def _show_warning(self, msg: str) -> None:
        self._info_panel.setVisible(True)
        self._info_panel.setStyleSheet(f"background: transparent; border-radius: 8px;")
        self._info_text.setText(msg)
        self._info_text.setStyleSheet(f"color: {COLOR_WARNING}; font-size: {FONT_SMALL}px; font-weight: 600;")
        self._info_icon.setPixmap(get_icon("ph.warning-fill", color=COLOR_WARNING).pixmap(16, 16))
