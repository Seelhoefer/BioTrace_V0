"""BioTrace root window and view navigation controller.

Phase 3 update:
- Owns a ``DatabaseManager`` and ``SessionManager``.
- Binds the ``SessionManager`` to ``LiveView`` after construction.
- Passes the ``DatabaseManager`` to ``DashboardView`` for session listing.
"""

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.session import SessionManager
from app.storage.database import DatabaseManager
from app.ui.theme import (
    COLOR_FONT,
    COLOR_FONT_MUTED,
    FONT_SUBTITLE,
    FONT_SMALL,
    FONT_CAPTION,
    GLOBAL_STYLESHEET,
    ICON_SIZE_INLINE,
    ICON_SIZE_NAV,
    ICON_SIZE_NAV_LOGO,
    SIDEBAR_LOGO_HEIGHT,
    SIDEBAR_PADDING_X,
    SIDEBAR_PADDING_Y,
    SIDEBAR_WIDTH,
    get_icon,
)
from app.ui.views.calibration_view import CalibrationView
from app.ui.views.dashboard_view import DashboardView
from app.ui.views.excel_import_view import ExcelImportView
from app.ui.views.live_view import LiveView
from app.ui.views.post_session_view import PostSessionView
from app.ui.views.settings_view import SettingsView
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """Root application window with sidebar navigation.

    Owns the ``DatabaseManager`` and ``SessionManager`` for the application
    lifetime.  After all views are constructed, it injects dependencies into
    them (e.g. ``LiveView.bind_session_manager()``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BioTrace — Surgical Training Biofeedback")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 700)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # ── Infrastructure (created before views) ──────────────────────
        self._db = DatabaseManager()
        self._session_manager = SessionManager(self._db, parent=self)
        self._cleanup_done = False

        self._build_ui()

        # ── Dependency injection ───────────────────────────────────────
        self._live_view.bind_session_manager(self._session_manager)
        self._calibration_view.bind_session_manager(self._session_manager)

        # Calibration: baseline done → proceed to live view
        self._calibration_view.proceed_to_live.connect(self._on_proceed_to_live)

        # Calibration: X button → back to Dashboard
        self._calibration_view.close_requested.connect(lambda: self.navigate_to(0))

        # Import: X button → back to Dashboard
        self._excel_import_view.close_requested.connect(lambda: self.navigate_to(0))

        # Settings: data cleared → refresh dashboard + sidebar, navigate home
        self._settings_view.data_cleared.connect(self._on_data_cleared)

        # Dashboard "Start Session" button → conditional flow.
        # If sensors are not ready, route to Sensors; otherwise to Calibration.
        self._dashboard_view.new_session_requested.connect(self._on_new_session_requested)
        self._dashboard_view.session_selected.connect(self._on_view_session)

        # Session end → Post-Session view (index 4)
        self._session_manager.session_ended.connect(self._on_session_ended)

        # Post-Session "Back to Dashboard" button
        self._post_session_view.back_to_dashboard.connect(lambda: self.navigate_to(0))
        self._post_session_view.new_session_requested.connect(self._on_new_session_requested)
        self._post_session_view.session_renamed.connect(lambda sid, name: self._populate_recent_sessions())

        self.navigate_to(0)
        logger.info("MainWindow initialised.")

    def _on_view_session(self, session_id: int) -> None:
        """Navigate to the individual session dashboard."""
        self._post_session_view.load_session(session_id)
        self.navigate_to(4)

        # Ensure correct sidebar state: uncheck main buttons, check this session
        for btn in self._nav_buttons:
            btn.setChecked(False)

        for btn in self._recent_session_buttons:
            sid = btn.property("session_id")
            btn.setChecked(sid == session_id)

    def closeEvent(self, event) -> None:
        """Ensure the database is closed cleanly on exit."""
        self.cleanup()
        super().closeEvent(event)

    def cleanup(self) -> None:
        """Stop background activity before the application exits."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        if hasattr(self, "_live_view"):
            self._live_view.cleanup()

        if self._session_manager.state.name == "RUNNING":
            self._session_manager.end_session()
        elif self._session_manager.state.name == "CALIBRATING":
            self._session_manager.end_calibration(0)

        self._db.close()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root_widget = QWidget()
        root_layout = QHBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root_widget)

        self._sidebar = self._build_sidebar()
        root_layout.addWidget(self._sidebar)

        self._stack = QStackedWidget()

        self._dashboard_view    = DashboardView(db=self._db)
        self._calibration_view  = CalibrationView()
        self._live_view         = LiveView()
        self._post_session_view = PostSessionView(db=self._db)
        self._excel_import_view = ExcelImportView(db=self._db)

        self._stack.addWidget(self._dashboard_view)    # 0
        self._stack.addWidget(QWidget())               # 1 (Placeholder for removed SensorsView)
        self._stack.addWidget(self._calibration_view)  # 2
        self._stack.addWidget(self._live_view)          # 3
        self._stack.addWidget(self._post_session_view)  # 4
        self._stack.addWidget(self._excel_import_view)  # 5

        self._settings_view = SettingsView(db=self._db)
        self._stack.addWidget(self._settings_view)   # 6

        root_layout.addWidget(self._stack, stretch=1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(
            SIDEBAR_PADDING_X,
            SIDEBAR_PADDING_Y,
            SIDEBAR_PADDING_X,
            SIDEBAR_PADDING_Y,
        )
        layout.setSpacing(8)

        logo_container = QHBoxLayout()
        logo_container.setContentsMargins(8, 0, 8, 0)
        logo_container.setSpacing(12)
        logo_container_widget = QWidget()
        logo_container_widget.setFixedHeight(SIDEBAR_LOGO_HEIGHT)
        logo_container_widget.setLayout(logo_container)

        logo_icon = QLabel()
        logo_icon.setPixmap(
            get_icon("ph.brain-fill", color=COLOR_FONT).pixmap(ICON_SIZE_NAV_LOGO, ICON_SIZE_NAV_LOGO)
        )
        logo_container.addWidget(logo_icon)

        logo_label = QLabel("BioTrace")
        logo_label.setObjectName("sidebar_section_label")
        logo_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_SUBTITLE}px; font-weight: 700;"
        )
        logo_container.addWidget(logo_label)
        logo_container.addStretch()
        layout.addWidget(logo_container_widget)
        layout.addSpacing(8)

        nav_items = [
            ("ph.layout-fill",             "Dashboard",   0),
            ("ph.chart-line-up-fill",      "Import",      5),
        ]

        self._nav_buttons: list[QPushButton] = []
        self._recent_session_buttons: list[QPushButton] = []
        for icon_name, label, index in nav_items:
            btn = QPushButton(f"  {label}")
            btn.setIcon(get_icon(icon_name, color=COLOR_FONT))
            btn.setIconSize(QSize(ICON_SIZE_NAV, ICON_SIZE_NAV))
            btn.setObjectName("nav_button")
            btn.setCheckable(True)
            btn.setProperty("target_index", index)
            btn.clicked.connect(lambda checked, i=index: self.navigate_to(i))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        # Recent sessions area
        layout.addSpacing(24)
        recent_label = QLabel("RECENT SESSIONS")
        recent_label.setObjectName("sidebar_section_label")
        recent_label.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px; font-weight: 700; padding: 0px 12px 8px 12px; letter-spacing: 1px;")
        layout.addWidget(recent_label)

        self._recent_sessions_layout = QVBoxLayout()
        self._recent_sessions_layout.setSpacing(8)
        layout.addLayout(self._recent_sessions_layout)

        self._populate_recent_sessions()

        layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

        # Settings and Log out nav
        layout.addSpacing(16)
        settings_btn = QPushButton("  Settings")
        settings_btn.setIcon(get_icon("ph.gear-six-fill", color=COLOR_FONT))
        settings_btn.setIconSize(QSize(ICON_SIZE_NAV, ICON_SIZE_NAV))
        settings_btn.setObjectName("nav_button")
        settings_btn.setCheckable(True)
        settings_btn.clicked.connect(lambda: self.navigate_to(6))
        layout.addWidget(settings_btn)
        self._settings_btn = settings_btn

        logout_btn = QPushButton("  Log Out")
        logout_btn.setIcon(get_icon("ph.sign-out-fill", color=COLOR_FONT))
        logout_btn.setIconSize(QSize(ICON_SIZE_NAV, ICON_SIZE_NAV))
        logout_btn.setObjectName("nav_button")
        layout.addWidget(logout_btn)

        return sidebar

    def _populate_recent_sessions(self) -> None:
        from app.storage.session_repository import SessionRepository
        from datetime import datetime

        # clear existing
        self._recent_session_buttons = []
        while self._recent_sessions_layout.count():
            item = self._recent_sessions_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        repo = SessionRepository(self._db)
        sessions = repo.get_all_sessions()

        if not sessions:
            empty_lbl = QLabel("No recent sessions")
            empty_lbl.setStyleSheet(f"color: {COLOR_FONT_MUTED}; font-size: {FONT_SMALL}px; padding-left: 12px;")
            self._recent_sessions_layout.addWidget(empty_lbl)
            return

        for i, session in enumerate(sessions[:5]):  # Top 5
            name = session["name"]
            if name:
                label = name
            else:
                try:
                    dt = datetime.fromisoformat(str(session["started_at"]))
                    label = dt.strftime("%b %d, %H:%M")
                except Exception:
                    label = str(session["started_at"])[:16]

            btn = QPushButton(f"  {label}")
            btn.setIcon(get_icon("ph.activity-fill", color=COLOR_FONT))
            btn.setIconSize(QSize(ICON_SIZE_NAV, ICON_SIZE_NAV))
            btn.setObjectName("nav_button")
            btn.setCheckable(True)

            # Store session ID in the button for easier identification
            btn.setProperty("session_id", session["id"])

            btn.clicked.connect(lambda checked, sid=session["id"]: self._on_view_session(sid))
            self._recent_sessions_layout.addWidget(btn)
            self._recent_session_buttons.append(btn)

            # Keep the button checked if we are currently viewing this session
            if hasattr(self, "_stack") and hasattr(self, "_post_session_view"):
                if self._stack.currentIndex() == 4 and self._post_session_view._session_id == session["id"]:
                    btn.setChecked(True)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate_to(self, index: int) -> None:
        """Switch the active view and update sidebar button state.

        Args:
            index: 0=Dashboard, 1=Sensors, 2=Calibration, 3=Live, 4=Post-Session,
                   5=Import, 6=Settings.
        """
        if index < 0 or index >= self._stack.count():
            logger.warning("navigate_to(%d) out of range.", index)
            return

        # Reset calibration wizard whenever we enter it.
        if index == 2:
            self._calibration_view.reset()

        # Calibration and Live Session are shown in full-width mode without sidebar.
        self._sidebar.setVisible(index not in (2, 3))

        self._stack.setCurrentIndex(index)

        # Update main nav buttons
        for btn in self._nav_buttons:
            btn.setChecked(btn.property("target_index") == index)
        if hasattr(self, "_settings_btn"):
            self._settings_btn.setChecked(index == 6)

        # Uncheck all recent session buttons when navigating to a main view
        # (Post-session view handles its own checked state in _on_view_session)
        if index != 4:
            for btn in self._recent_session_buttons:
                btn.setChecked(False)

        # Refresh dashboard whenever we navigate to it.
        if index == 0:
            self._dashboard_view.refresh()

        view_names = ["Dashboard", "Placeholder", "Calibration", "Live Session", "Post-Session", "Import", "Settings"]
        logger.info("Navigated to: %s", view_names[index])

    def _on_proceed_to_live(self) -> None:
        """Called when CalibrationView emits proceed_to_live."""
        self.navigate_to(3)
        self._session_manager.start_session()
        logger.info("Proceeding to live session after calibration.")

    def _on_new_session_requested(self) -> None:
        """Route Start Session directly to Calibration."""
        logger.info("New session requested. Navigating to Calibration.")
        self.navigate_to(2)

    def _on_data_cleared(self) -> None:
        """Refresh all data-dependent views after the user deletes all sessions."""
        self._dashboard_view.refresh()
        self._populate_recent_sessions()
        self.navigate_to(0)
        logger.info("All data cleared — navigated to Dashboard.")

    def _on_session_ended(self, session_id: int) -> None:
        """Handle session end: navigate to the individual Post-Session dashboard.

        Also refreshes the main dashboard and sidebar recent-sessions list in
        the background so they are up-to-date when the user navigates back.

        Args:
            session_id: The database ID of the session that just ended.
        """
        self._post_session_view.load_session(session_id)
        self.navigate_to(4)
        self._dashboard_view.refresh()
        self._populate_recent_sessions()
        logger.info("Session %d ended — navigated to Post-Session view.", session_id)
