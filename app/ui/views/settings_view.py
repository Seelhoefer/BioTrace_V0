"""Settings view for BioTrace.

Contains two sections:
1. Data Management — export all sessions to Excel, delete all data.
2. How It Works — three animated flowchart cards explaining the scientific
   pipeline behind Stress, Cognitive Load, and Learning Curves.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QHideEvent, QShowEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.storage.database import DatabaseManager
from app.ui.theme import (
    CARD_PADDING,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_DANGER_BG,
    COLOR_SUCCESS,
    CONTENT_PADDING_H,
    CONTENT_PADDING_V,
    FONT_BODY,
    FONT_SMALL,
    GRID_GUTTER,
    ICON_SIZE_DEFAULT,
    RADIUS_LG,
    RADIUS_MD,
    SPACE_1,
    SPACE_2,
    SPACE_4,
    get_icon,
)
from app.ui.widgets.flowchart_card import FlowchartCard, NodeDef
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Node definitions ──────────────────────────────────────────────────────

STRESS_NODES: list[NodeDef] = [
    NodeDef(
        icon="ph.timer-fill",
        label="Baseline",
        formula="RMSSD_baseline: 60s resting state",
        description=(
            "A one-minute resting measurement taken during calibration to establish "
            "your personal physiological baseline."
        ),
        reference="",
    ),
    NodeDef(
        icon="ph.heartbeat-fill",
        label="Heart Sensor",
        formula="ECG sampled at 250 Hz",
        description=(
            "Raw heart signals captured via the HRV sensor. The system detects "
            "individual heartbeats (R-peaks) in real-time."
        ),
        reference="",
    ),
    NodeDef(
        icon="ph.calculator-fill",
        label="RMSSD",
        formula="RMSSD = \u221a( mean( (RR[i+1] \u2212 RR[i])\u00b2 ) )",
        description=(
            "Root Mean Square of Successive Differences. A robust measure of heart "
            "rate variability reflecting parasympathetic activity."
        ),
        reference=(
            "Task Force of ESC & NASPE (1996). Heart rate variability standards. "
            "Circulation, 93(5), 1043\u20131065."
        ),
    ),
    NodeDef(
        icon="ph.arrows-down-up-fill",
        label="\u0394 Baseline",
        formula="\u0394RMSSD = RMSSD_t \u2212 RMSSD_baseline",
        description=(
            "We calculate the deviation from your resting baseline. Significant "
            "drops in RMSSD indicate increased physiological stress."
        ),
        reference="",
        is_threshold=True,
    ),
    NodeDef(
        icon="ph.gauge-fill",
        label="Stress Score",
        formula="Score = norm(1 / RMSSD), range 0\u20131",
        description=(
            "The final stress index shown on your dashboard. Higher values "
            "represent higher levels of physiological tension."
        ),
        reference="",
    ),
]

COGNITIVE_LOAD_NODES: list[NodeDef] = [
    NodeDef(
        icon="ph.timer-fill",
        label="Baseline",
        formula="d_baseline: 60s pupil diameter",
        description=(
            "Average pupil diameter recorded during the resting baseline phase "
            "of calibration."
        ),
        reference="",
    ),
    NodeDef(
        icon="ph.eye-fill",
        label="Eye Camera",
        formula="d(t): pupil diameter in pixels",
        description=(
            "High-speed infrared camera tracking pupil diameter at 30 Hz. "
            "Artefacts like blinks are filtered out automatically."
        ),
        reference="",
    ),
    NodeDef(
        icon="ph.drop-fill",
        label="PDI",
        formula="PDI = (d_t \u2212 d_baseline) / d_baseline",
        description=(
            "Pupil Dilation Index \u2014 the percentage change from your resting "
            "pupil size baseline."
        ),
        reference=(
            "Beatty, J. (1982). Task-evoked pupillary responses. "
            "Psychological Bulletin, 91(2), 276\u2013292."
        ),
    ),
    NodeDef(
        icon="ph.brain-fill",
        label="CLI",
        formula="CLI = 0.5 \u00b7 Stress + 0.5 \u00b7 PDI",
        description=(
            "Cognitive Load Index \u2014 a composite metric combining physical stress "
            "and mental effort into a single 0\u20131 scale."
        ),
        reference="",
    ),
]

LEARNING_CURVE_NODES: list[NodeDef] = [
    NodeDef(
        icon="ph.file-arrow-up-fill",
        label="Import",
        formula="Excel (.xlsx) from LapSim",
        description=(
            "Import historical training data exported from the surgical simulator. "
            "The system automatically identifies participants and exercises."
        ),
        reference="",
    ),
    NodeDef(
        icon="ph.chart-line-up-fill",
        label="Model Fit",
        formula="\u0177(t) = scale \u00b7 (1-leff)^(t+pexp) + maxp",
        description=(
            "Schmettow parametric model fitted to your performance history. "
            "It predicts your future potential and learning speed."
        ),
        reference=(
            "Schmettow, M. (2026). BioTrace Learning Curve Analysis."
        ),
    ),
    NodeDef(
        icon="ph.trend-up-fill",
        label="Efficiency",
        formula="leff \u2208 (0, 1)",
        description=(
            "A numerical score of your skill acquisition speed. Higher values "
            "mean you master new tasks more quickly."
        ),
        reference="",
    ),
]


# ── Export worker ─────────────────────────────────────────────────────────

class _ExportWorker(QThread):
    """Background thread that runs export_all_sessions() off the UI thread.

    Signals:
        finished: Emitted on successful completion.
        error (str): Emitted with the error message on failure.
    """

    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, db: DatabaseManager, path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._path = path

    def run(self) -> None:
        """Execute the export."""
        from app.storage.export import SessionExporter
        try:
            SessionExporter(self._db).export_all_sessions(self._path)
            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            logger.error("Export failed: %s", exc)
            self.error.emit(str(exc))


# ── _DataManagementCard ───────────────────────────────────────────────────

class _DataManagementCard(QFrame):
    """Card containing Export All and Delete All controls.

    Signals:
        data_cleared: Emitted after all sessions have been deleted.
    """

    data_cleared = pyqtSignal()

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(
            f"QFrame#card {{ background-color: transparent; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: {RADIUS_LG}px; }}"
        )
        self._db = db
        self._export_worker: _ExportWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        layout.setSpacing(SPACE_2)

        # ── Heading ──────────────────────────────────────────────────
        heading = QLabel("Data Management")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        # ── Export row ───────────────────────────────────────────────
        export_row = QHBoxLayout()
        self._export_btn = QPushButton("  Export All Sessions")
        try:
            self._export_btn.setIcon(
                get_icon("ph.file-spreadsheet-fill", color="#FFFFFF")
            )
        except Exception:
            try:
                self._export_btn.setIcon(get_icon("ph.export-fill", color="#FFFFFF"))
            except Exception:
                pass
        self._export_btn.setIconSize(QSize(ICON_SIZE_DEFAULT, ICON_SIZE_DEFAULT))
        self._export_btn.clicked.connect(self._on_export)
        export_row.addWidget(self._export_btn)

        self._export_status = QLabel()
        self._export_status.setVisible(False)
        export_row.addWidget(self._export_status)
        export_row.addStretch()
        layout.addLayout(export_row)

        export_desc = QLabel("Download all sessions as a multi-sheet Excel (.xlsx) file.")
        export_desc.setObjectName("muted")
        layout.addWidget(export_desc)

        # ── Divider ──────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        # ── Delete row ───────────────────────────────────────────────
        delete_row = QHBoxLayout()
        self._delete_btn = QPushButton("  Delete All Data")
        try:
            self._delete_btn.setIcon(get_icon("ph.trash-fill", color="#FFFFFF"))
        except Exception:
            pass
        self._delete_btn.setIconSize(QSize(ICON_SIZE_DEFAULT, ICON_SIZE_DEFAULT))
        self._delete_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_DANGER}; color: #FFFFFF; "
            f"border-radius: {RADIUS_MD}px; padding: 10px 16px; }}"
            f"QPushButton:hover {{ background-color: #DC2626; }}"
        )
        self._delete_btn.clicked.connect(self._show_confirm)
        delete_row.addWidget(self._delete_btn)
        delete_row.addStretch()
        layout.addLayout(delete_row)

        delete_desc = QLabel("Permanently delete all session data. This cannot be undone.")
        delete_desc.setObjectName("muted")
        layout.addWidget(delete_desc)

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

        confirm_btn = QPushButton("Confirm Delete")
        confirm_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_DANGER}; color: #FFFFFF; "
            f"border-radius: {RADIUS_MD}px; padding: 10px 16px; }}"
            f"QPushButton:hover {{ background-color: #DC2626; }}"
        )
        confirm_btn.clicked.connect(self._on_confirm_delete)
        banner_layout.addWidget(confirm_btn)

        self._confirm_banner.setVisible(False)
        layout.addWidget(self._confirm_banner)

    def _show_confirm(self) -> None:
        """Show the inline confirmation banner with session count."""
        from app.storage.session_repository import SessionRepository
        n = len(SessionRepository(self._db).get_all_sessions())
        self._confirm_label.setText(
            f"This will permanently delete all {n} session(s) and cannot be undone."
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

    def _on_export(self) -> None:
        """Open a save dialog and start the background export thread."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export All Sessions",
            "biotrace_export.xlsx",
            "Excel Files (*.xlsx)",
        )
        if not path:
            return

        self._export_btn.setEnabled(False)
        self._export_btn.setText("  Exporting\u2026")
        self._export_status.setVisible(False)

        self._export_worker = _ExportWorker(self._db, path, parent=self)
        self._export_worker.finished.connect(self._on_export_done)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_done(self) -> None:
        """Re-enable the button and show a success status label."""
        self._export_btn.setEnabled(True)
        self._export_btn.setText("  Export All Sessions")
        self._export_status.setText("Exported successfully")
        self._export_status.setStyleSheet(
            f"color: {COLOR_SUCCESS}; font-size: {FONT_SMALL}px;"
        )
        self._export_status.setVisible(True)
        self._export_worker = None

    def _on_export_error(self, message: str) -> None:
        """Re-enable the button and show an error status label."""
        self._export_btn.setEnabled(True)
        self._export_btn.setText("  Export All Sessions")
        self._export_status.setText(f"Export failed: {message}")
        self._export_status.setStyleSheet(
            f"color: {COLOR_DANGER}; font-size: {FONT_SMALL}px;"
        )
        self._export_status.setVisible(True)
        self._export_worker = None


# ── SettingsView ──────────────────────────────────────────────────────────

class SettingsView(QWidget):
    """Root settings page widget.

    Sections:
    - Data Management card (export + delete).
    - How It Works section with three animated flowchart cards.

    Signals:
        data_cleared: Forwarded from _DataManagementCard after deletion.

    Args:
        db: Shared database manager (dependency injected from MainWindow).
        parent: Optional parent widget.
    """

    data_cleared = pyqtSignal()

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._flowchart_cards: list[FlowchartCard] = []
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
            CONTENT_PADDING_H, CONTENT_PADDING_V,
            CONTENT_PADDING_H, CONTENT_PADDING_V,
        )
        layout.setSpacing(SPACE_4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Page heading ─────────────────────────────────────────────
        heading = QLabel("Settings")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        # ── Data Management card ─────────────────────────────────────
        data_card = _DataManagementCard(db)
        data_card.data_cleared.connect(self.data_cleared)
        layout.addWidget(data_card)

        # ── How It Works section ─────────────────────────────────────
        how_heading = QLabel("Scientific Methodology")
        how_heading.setObjectName("heading")
        layout.addWidget(how_heading)

        how_sub = QLabel(
            "Explore the scientific pipelines used to calculate BioTrace metrics. "
            "Click any node to see detailed formulas and references."
        )
        how_sub.setObjectName("muted")
        how_sub.setWordWrap(True)
        layout.addWidget(how_sub)

        # Vertical stack of cards
        stress_card = FlowchartCard(
            "Stress Analysis Pipeline",
            "Measuring physiological tension via Heart Rate Variability (RMSSD)",
            STRESS_NODES,
        )
        cload_card = FlowchartCard(
            "Cognitive Load Pipeline",
            "Combining Pupil Dilation Index (PDI) and Stress into a unified index",
            COGNITIVE_LOAD_NODES,
        )
        lcurve_card = FlowchartCard(
            "Learning Curve Modeling",
            "Non-linear least squares fit of the Schmettow parametric model",
            LEARNING_CURVE_NODES,
        )

        for card in (stress_card, cload_card, lcurve_card):
            layout.addWidget(card)
            self._flowchart_cards.append(card)

        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Animation lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Start all flowchart animations when the page becomes visible."""
        super().showEvent(event)
        for card in self._flowchart_cards:
            card.start_animation()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        """Stop all flowchart animations when the page is hidden."""
        super().hideEvent(event)
        for card in self._flowchart_cards:
            card.stop_animation()
