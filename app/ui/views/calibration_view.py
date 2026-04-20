"""Calibration view — eye alignment + breathing baseline calibration.

User flow
---------
New Session (Dashboard header) → CalibrationView opens →
step 1 shows a live eye feed with a circular guide →
user aligns their eye and presses Space/Next (manual override always available) →
step 2 shows the breathing-guided HRV + pupil baseline screen →
baseline completes → ``proceed_to_live`` → LiveView

Dependency injection: call ``bind_session_manager()`` once after
the view is added to the QStackedWidget.
"""

import numpy as np
import cv2

from PyQt6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QSize,
    QTimer,
    Qt,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
    QThread,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QRadialGradient,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import (
    CALIBRATION_CTA_HEIGHT,
    CALIBRATION_CTA_WIDTH,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    FONT_BODY,
    FONT_BODY_LARGE,
    FONT_CAPTION,
    FONT_SMALL,
    FONT_TITLE,
    RADIUS_MD,
    SPACE_1,
    SPACE_2,
    SPACE_3,
    SPACE_4,
    get_icon,
)
from app.utils.config import (
    get_calibration_duration,
    EYE_TRACKER_CAMERA_INDEX,
    USE_EYE_TRACKER,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_INHALE_SECONDS: int = 4
_EXHALE_SECONDS: int = 4


# ---------------------------------------------------------------------------
# CountdownRing
# ---------------------------------------------------------------------------

class CountdownRing(QWidget):
    """Compact circular countdown indicator used in calibration header."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._progress: float = 1.0
        self.setFixedSize(36, 36)

    def set_progress(self, progress: float) -> None:
        """Set normalized remaining progress (1.0 full to 0.0 empty)."""
        self._progress = max(0.0, min(1.0, progress))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLOR_BORDER))
        painter.drawEllipse(rect)

        span = int(360 * 16 * self._progress)
        if span > 0:
            painter.setBrush(QColor(COLOR_PRIMARY))
            painter.drawPie(rect, 90 * 16, -span)


# ---------------------------------------------------------------------------
# BreathingOrb — animated sphere widget
# ---------------------------------------------------------------------------

class BreathingOrb(QWidget):
    """Animated 3D-looking breathing orb.

    Pulses between a resting size and an expanded size using a
    QPropertyAnimation on the custom ``orb_radius`` property, guiding the
    user's breathing rhythm (inhale = expand, exhale = contract).

    The sphere uses a QRadialGradient to simulate a light source in the
    top-left quadrant, giving a glossy 3D appearance.
    """

    _FIXED_SIZE: int = 280   # widget canvas size in pixels
    _PREVIEW_RADIUS_MIN: float = 84.0
    _PREVIEW_RADIUS_MAX: float = 116.0
    _ACTIVE_RADIUS_MIN: float = 72.0
    _ACTIVE_RADIUS_MAX: float = 128.0

    phase_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._radius: float = self._PREVIEW_RADIUS_MIN
        self._preview_mode: bool = True
        self.setFixedSize(self._FIXED_SIZE, self._FIXED_SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        # Inhale: expand over 4 s
        self._inhale = QPropertyAnimation(self, b"orb_radius", self)
        self._inhale.setDuration(_INHALE_SECONDS * 1000)
        self._inhale.setEasingCurve(QEasingCurve.Type.Linear)

        # Exhale: contract over 4 s
        self._exhale = QPropertyAnimation(self, b"orb_radius", self)
        self._exhale.setDuration(_EXHALE_SECONDS * 1000)
        self._exhale.setEasingCurve(QEasingCurve.Type.Linear)

        self._apply_animation_profile()

        # Chain: inhale finished → start exhale, exhale finished → start inhale
        self._inhale.finished.connect(self._start_exhale)
        self._exhale.finished.connect(self._start_inhale)

    # ── Public API ──────────────────────────────────────────────────────

    def start_animation(self) -> None:
        """Begin the breathing pulse animation."""
        self._start_inhale()

    def restart_from_inhale(self) -> None:
        """Reset orb to initial size and restart from inhale phase."""
        self._inhale.stop()
        self._exhale.stop()
        start_radius = self._PREVIEW_RADIUS_MIN if self._preview_mode else self._ACTIVE_RADIUS_MIN
        self._set_orb_radius(start_radius)
        self._start_inhale()

    def stop_animation(self) -> None:
        """Stop the breathing pulse animation and reset radius."""
        self._inhale.stop()
        self._exhale.stop()

    def set_preview_mode(self, enabled: bool) -> None:
        """Toggle simplified preview rendering (no glow/shadow)."""
        self._preview_mode = enabled
        self._apply_animation_profile()
        self.update()

    def _apply_animation_profile(self) -> None:
        """Apply mode-specific radius range while keeping 4s inhale/exhale timing."""
        if self._preview_mode:
            radius_min = self._PREVIEW_RADIUS_MIN
            radius_max = self._PREVIEW_RADIUS_MAX
        else:
            radius_min = self._ACTIVE_RADIUS_MIN
            radius_max = self._ACTIVE_RADIUS_MAX

        self._inhale.setDuration(_INHALE_SECONDS * 1000)
        self._inhale.setStartValue(radius_min)
        self._inhale.setEndValue(radius_max)

        self._exhale.setDuration(_EXHALE_SECONDS * 1000)
        self._exhale.setStartValue(radius_max)
        self._exhale.setEndValue(radius_min)

    def _start_inhale(self) -> None:
        """Start inhale phase and notify listeners."""
        self.phase_changed.emit("inhale")
        self._inhale.start()

    def _start_exhale(self) -> None:
        """Start exhale phase and notify listeners."""
        self.phase_changed.emit("exhale")
        self._exhale.start()

    # ── Qt property (required for QPropertyAnimation) ───────────────────

    def _get_orb_radius(self) -> float:
        return self._radius

    def _set_orb_radius(self, r: float) -> None:
        self._radius = r
        self.update()  # triggers paintEvent

    orb_radius = pyqtProperty(float, _get_orb_radius, _set_orb_radius)

    # ── Painting ─────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        cx: float = self.width() / 2
        cy: float = self.height() / 2
        r: float = self._radius

        if self._preview_mode:
            sphere = QRadialGradient(QPointF(cx - r * 0.15, cy - r * 0.18), r * 1.35)
            sphere.setColorAt(0.00, QColor(178, 198, 238))
            sphere.setColorAt(0.55, QColor(126, 154, 215))
            sphere.setColorAt(1.00, QColor(88, 118, 188))
            painter.setBrush(QBrush(sphere))
            painter.drawEllipse(QPointF(cx, cy), r, r)
        else:
            highlight_x = cx - r * 0.32
            highlight_y = cy - r * 0.38
            sphere = QRadialGradient(QPointF(highlight_x, highlight_y), r * 1.6)
            sphere.setColorAt(0.00, QColor(214, 224, 248))
            sphere.setColorAt(0.18, QColor(162, 184, 232))
            sphere.setColorAt(0.42, QColor(103, 133, 202))
            sphere.setColorAt(0.72, QColor(59, 87, 159))
            sphere.setColorAt(1.00, QColor(18, 38, 104))

            painter.setBrush(QBrush(sphere))
            painter.drawEllipse(QPointF(cx, cy), r, r)

        painter.end()


# ---------------------------------------------------------------------------
# Eye Camera Preview — alignment guide widget
# ---------------------------------------------------------------------------

class _EyePreviewWorker(QThread):
    """Background thread: captures frames from the eye tracker camera,
    runs a simple pupil detector, and annotates each frame with a guide overlay.

    Signals:
        frame_ready (object): Emits an annotated BGR numpy array at ~30 fps.
        quality_changed (str): Emits one of "none" | "offcenter" | "good"
            for each frame processed.
        camera_unavailable (): Emitted once if the camera cannot be opened.
    """

    frame_ready         = pyqtSignal(object)  # annotated BGR numpy array
    quality_changed     = pyqtSignal(str)
    camera_unavailable  = pyqtSignal()

    # Absolute minimum blob area (m00 moment) to consider as a potential pupil.
    # Kept intentionally low — Otsu thresholding handles the hard work.
    _MIN_BLOB_AREA: float = 30.0

    # Acceptable diameter range (px) — deliberately wide because the actual size
    # depends on camera resolution and distance, which vary across setups.
    _MIN_DIAMETER_PX: float = 8.0
    _MAX_DIAMETER_PX: float = 400.0

    # Pupil centre must fall within this fraction of the frame dimensions
    # to be accepted as "good".  0.72 = central 72 % of each axis
    # (deliberately lenient to reduce positioning frustration).
    _CENTER_ZONE: float = 0.72

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._running = True

    def run(self) -> None:
        """Capture loop: open camera, detect pupil, emit annotated frames."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            logger.warning(
                "EyePreviewWorker: camera index %d not available.", self._camera_index
            )
            self.camera_unavailable.emit()
            return

        logger.info("EyePreviewWorker started on camera %d.", self._camera_index)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("EyePreviewWorker: failed to read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            quality, cx, cy, diameter = self._detect_pupil(gray)
            self._draw_guide(frame, gray.shape[0], gray.shape[1], quality, cx, cy, diameter)

            self.quality_changed.emit(quality)
            self.frame_ready.emit(frame.copy())
            self.msleep(33)  # ~30 fps

        cap.release()
        logger.info("EyePreviewWorker stopped.")

    def _detect_pupil(
        self, gray: np.ndarray
    ) -> tuple[str, float, float, float]:
        """Detect the pupil in a grayscale frame using Otsu thresholding + moments.

        Uses Otsu's method so the threshold adapts automatically to the ambient
        lighting and IR intensity of the specific eye-tracker camera in use.
        Quality is determined by whether the detected blob is roughly centred in
        the frame, not by its absolute pixel diameter (which depends on the
        camera–eye distance and resolution).

        Returns:
            Tuple of (quality, center_x, center_y, diameter_px).
            quality is "none" | "offcenter" | "good".
        """
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Otsu threshold — inverted so the dark pupil becomes foreground.
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        M = cv2.moments(thresh)

        if M["m00"] < self._MIN_BLOB_AREA:
            return "none", 0.0, 0.0, 0.0

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        diameter = 2.0 * float(np.sqrt(M["m00"] / np.pi))

        if diameter < self._MIN_DIAMETER_PX or diameter > self._MAX_DIAMETER_PX:
            return "none", cx, cy, diameter

        # Check that the pupil centre is in the central zone of the frame.
        margin_x = w * (1.0 - self._CENTER_ZONE) / 2.0
        margin_y = h * (1.0 - self._CENTER_ZONE) / 2.0
        if (cx < margin_x or cx > w - margin_x or
                cy < margin_y or cy > h - margin_y):
            return "offcenter", cx, cy, diameter

        return "good", cx, cy, diameter

    def _draw_guide(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        quality: str,
        cx: float,
        cy: float,
        diameter: float,
    ) -> None:
        """Draw targeting overlay and detected pupil onto the frame in-place.

        The guide ring is proportional to the frame dimensions so it renders
        correctly regardless of camera resolution or eye-to-camera distance.
        """
        center = (w // 2, h // 2)

        # Guide ring radius: 35 % of the shorter frame edge — large enough to
        # be clearly visible at any camera resolution.
        guide_r = max(12, int(min(h, w) * 0.35))
        # Crosshair arm length: 6 % of shorter edge.
        arm = max(8, int(min(h, w) * 0.06))

        if quality == "good":
            guide_rgb = (50, 200, 80)
        elif quality == "offcenter":
            guide_rgb = (50, 160, 255)
        else:
            guide_rgb = (140, 140, 160)

        cv2.circle(frame, center, guide_r, guide_rgb, 2, cv2.LINE_AA)
        cv2.line(frame, (center[0] - arm, center[1]), (center[0] + arm, center[1]), guide_rgb, 2)
        cv2.line(frame, (center[0], center[1] - arm), (center[0], center[1] + arm), guide_rgb, 2)

        # Detected pupil circle
        if diameter > 0.0:
            pupil_color = (50, 200, 80) if quality == "good" else (50, 160, 255)
            cv2.circle(
                frame,
                (int(cx), int(cy)),
                max(1, int(diameter / 2)),
                pupil_color,
                2,
                cv2.LINE_AA,
            )

    def stop(self) -> None:
        """Signal the capture loop to exit and wait for the thread."""
        self._running = False
        self.wait()


class EyeCameraPreview(QFrame):
    """Small eye-tracker camera preview with real-time alignment feedback.

    Displays a live camera feed with a targeting overlay.  The border colour
    and status label reflect the current alignment quality:

    - Gray border  : no pupil detected
    - Amber border : pupil detected but distance not ideal
    - Green border : pupil in the correct zone (``eye_locked`` emitted)

    Signals:
        eye_locked ():       Emitted once when the pupil is stably detected
                             in the ideal zone for ``_LOCK_FRAMES_REQUIRED``
                             consecutive frames.
        eye_unavailable ():  Emitted once if the camera cannot be opened
                             (e.g. missing macOS permission).
    """

    eye_locked      = pyqtSignal()
    eye_unavailable = pyqtSignal()

    _PREVIEW_W:            int = 224
    _PREVIEW_H:            int = 180
    _VIDEO_W:              int = 214
    _VIDEO_H:              int = 140
    _LOCK_FRAMES_REQUIRED: int = 10   # ~0.33 s at 30 fps — lenient lock for smoother UX

    def __init__(
        self,
        camera_index: int,
        parent: QWidget | None = None,
        *,
        compact: bool = True,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._worker: _EyePreviewWorker | None = None
        self._lock_counter: int = 0
        self._locked: bool = False
        self._compact = compact

        preview_w = self._PREVIEW_W if compact else 420
        preview_h = self._PREVIEW_H if compact else 360
        video_w = self._VIDEO_W if compact else 400
        video_h = self._VIDEO_H if compact else 300
        status_size = 9 if compact else 12

        self.setFixedSize(preview_w, preview_h)
        self.setObjectName("eye_preview_frame")
        self._apply_border("neutral")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Video frame label
        self._video_lbl = QLabel()
        self._video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_lbl.setFixedSize(video_w, video_h)
        self._video_lbl.setStyleSheet("background: #0D0D12; border-radius: 4px;")

        # Placeholder eye icon shown before camera opens
        placeholder_pixmap = get_icon("ph.eye", color=COLOR_FONT_MUTED).pixmap(32, 32)
        self._video_lbl.setPixmap(placeholder_pixmap)
        layout.addWidget(self._video_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status label
        self._status_lbl = QLabel("Waiting for camera…")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {status_size}px; background: transparent;"
        )
        layout.addWidget(self._status_lbl)

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the camera capture and alignment detection."""
        if self._worker and self._worker.isRunning():
            return
        self._lock_counter = 0
        self._locked = False
        self._apply_border("neutral")
        self._status_lbl.setText("Waiting for camera…")

        self._worker = _EyePreviewWorker(self._camera_index)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.quality_changed.connect(self._on_quality)
        self._worker.camera_unavailable.connect(self._on_camera_unavailable)
        self._worker.start()

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        if self._worker:
            self._worker.stop()
            self._worker = None

    # ── Private slots ─────────────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_frame(self, frame: np.ndarray) -> None:
        """Convert a BGR numpy array to QPixmap and display it."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._video_lbl.width(),
            self._video_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_lbl.setPixmap(pixmap)

    @pyqtSlot(str)
    def _on_quality(self, quality: str) -> None:
        """Update border and status based on detection quality."""
        if self._locked:
            return  # already locked — keep green state

        if quality == "none":
            self._lock_counter = 0
            self._apply_border("neutral")
            self._status_lbl.setText("Position your eye in front of the camera")

        elif quality == "offcenter":
            self._lock_counter = 0
            self._apply_border("warning")
            self._status_lbl.setText("Centre your eye in the frame")

        elif quality == "good":
            self._lock_counter += 1
            progress = self._lock_counter / self._LOCK_FRAMES_REQUIRED
            if progress < 1.0:
                self._apply_border("acquiring")
                self._status_lbl.setText(f"Hold still… {int(progress * 100)}%")
            else:
                self._locked = True
                self._apply_border("locked")
                self._status_lbl.setText("Eye aligned. Press Space to continue")
                self.eye_locked.emit()

    @pyqtSlot()
    def _on_camera_unavailable(self) -> None:
        """Handle the case where the camera could not be opened."""
        self._apply_border("neutral")
        self._status_lbl.setText(
            "Camera unavailable.\n"
            "Check macOS permissions:\n"
            "System Settings → Privacy → Camera"
        )
        self.eye_unavailable.emit()

    def _apply_border(self, state: str) -> None:
        """Update the frame's border color to reflect alignment state."""
        colors = {
            "neutral":   COLOR_BORDER,
            "warning":   COLOR_WARNING,
            "acquiring": "#86EFAC",  # light green — nearly locked
            "locked":    COLOR_SUCCESS,
        }
        border_color = colors.get(state, COLOR_BORDER)
        self.setStyleSheet(
            f"#eye_preview_frame {{"
            f"  background: white;"
            f"  border: 2px solid {border_color};"
            f"  border-radius: {RADIUS_MD}px;"
            f"}}"
        )


# ---------------------------------------------------------------------------
# CalibrationView
# ---------------------------------------------------------------------------

class CalibrationView(QWidget):
    """Breathing-guided baseline calibration screen.

    Presents a calming, minimal UI that guides the user through a
    resting baseline measurement before a live session.

    Signals:
        proceed_to_live:
            Emitted when the user clicks "Start Session" after a successful
            baseline recording.  ``MainWindow`` listens and navigates to
            ``LiveView``, then calls ``session_manager.start_session()``.
        close_requested:
            Emitted when the user clicks the back arrow (top-left).
            ``MainWindow`` listens and navigates back to the Dashboard.
    """

    proceed_to_live = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_manager = None
        self._step: str = "pupil_alignment"
        self._eye_ready: bool = not USE_EYE_TRACKER
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._baseline_remaining: int = get_calibration_duration()
        self._recording: bool = False
        self._complete: bool = False
        self._computed_rmssd: float = 0.0
        self._computed_pupil: float = 0.0

        # 1-second tick during baseline recording
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        self._prestart_timer = QTimer(self)
        self._prestart_timer.setInterval(1000)
        self._prestart_timer.timeout.connect(self._tick_prestart_countdown)
        self._prestart_remaining: int = 0
        self._prestart_active: bool = False

        self._build_ui()
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_shortcut.activated.connect(self._on_space_pressed)
        self._orb.start_animation()

    # ------------------------------------------------------------------
    # Dependency injection
    # ------------------------------------------------------------------

    def bind_session_manager(self, session_manager) -> None:
        """Inject the SessionManager.

        Args:
            session_manager: A :class:`~app.core.session.SessionManager` instance.
        """
        self._session_manager = session_manager
        session_manager.calibration_complete.connect(self._on_calibration_complete)
        logger.info("CalibrationView bound to SessionManager.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        root.setSpacing(SPACE_2)

        # ── Header ──────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.addStretch()

        skip_btn = QPushButton()
        skip_btn.setIcon(get_icon("ph.fast-forward-fill", color=COLOR_FONT_MUTED))
        skip_btn.setIconSize(QSize(16, 16))
        skip_btn.setFixedSize(36, 36)
        skip_btn.setToolTip("Skip calibration")
        skip_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {COLOR_BORDER};
                border-radius: 18px;
            }}
            QPushButton:hover {{
                background: {COLOR_BORDER};
            }}
            """
        )
        skip_btn.clicked.connect(self._on_skip_calibration)
        header_row.addWidget(skip_btn)

        self._restart_btn = QPushButton()
        self._restart_btn.setIcon(
            get_icon("ph.arrow-counter-clockwise-fill", color=COLOR_FONT_MUTED)
        )
        self._restart_btn.setIconSize(QSize(16, 16))
        self._restart_btn.setFixedSize(36, 36)
        self._restart_btn.setToolTip("Restart calibration")
        self._restart_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {COLOR_BORDER};
                border-radius: 18px;
            }}
            QPushButton:hover {{
                background: {COLOR_BORDER};
            }}
            """
        )
        self._restart_btn.clicked.connect(self._restart_baseline)
        self._restart_btn.hide()
        header_row.addWidget(self._restart_btn)

        self._countdown_ring = CountdownRing()
        self._countdown_ring.setToolTip("Calibration timer")
        self._countdown_ring.hide()
        header_row.addWidget(self._countdown_ring)
        root.addLayout(header_row)

        root.addStretch(1)

        self._content_stack = QStackedWidget()
        self._content_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self._content_stack, stretch=1)

        self._build_pupil_step()
        self._build_breathing_step()

        root.addStretch(1)

        self._show_pupil_alignment_step()

    def _build_pupil_step(self) -> None:
        """Build the pupil-dilation alignment step."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(SPACE_3)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        if USE_EYE_TRACKER:
            self._eye_preview = EyeCameraPreview(EYE_TRACKER_CAMERA_INDEX, compact=False)
            self._eye_preview.eye_locked.connect(self._on_eye_locked)
            self._eye_preview.eye_unavailable.connect(self._on_eye_cam_unavailable)
            layout.addWidget(self._eye_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            self._eye_preview = None
            placeholder = QLabel("Eye tracker preview disabled in settings")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                f"color: {COLOR_FONT_MUTED}; font-size: {FONT_BODY_LARGE}px;"
            )
            layout.addWidget(placeholder)

        self._pupil_status_label = QLabel("")
        self._pupil_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pupil_status_label.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px;"
        )
        layout.addWidget(self._pupil_status_label)

        self._pupil_hint_label = QLabel("Press Space or Next to continue")
        self._pupil_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pupil_hint_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_SMALL}px; font-weight: 600;"
        )
        layout.addWidget(self._pupil_hint_label)

        self._pupil_next_btn = QPushButton("Next")
        self._pupil_next_btn.setFixedSize(120, 40)
        self._pupil_next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pupil_next_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {COLOR_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: 20px;
                font-size: {FONT_BODY}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLOR_PRIMARY_HOVER};
            }}
            """
        )
        self._pupil_next_btn.clicked.connect(lambda: self._continue_from_pupil_step("next_button"))
        layout.addWidget(self._pupil_next_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._content_stack.addWidget(page)

    def _build_breathing_step(self) -> None:
        """Build the HRV breathing baseline step."""
        page = QWidget()
        center_col = QVBoxLayout(page)
        center_col.setSpacing(0)
        center_col.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        center_col.addSpacing(SPACE_2)

        # Breathing orb
        self._orb = BreathingOrb()
        self._orb.phase_changed.connect(self._on_orb_phase_changed)
        center_col.addWidget(self._orb, alignment=Qt.AlignmentFlag.AlignHCenter)
        center_col.addSpacing(SPACE_4)

        # Breath guidance text
        self._breath_label = QLabel("Press Start")
        self._breath_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._breath_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_TITLE}px; "
            "font-weight: 600; letter-spacing: 1px;"
        )
        center_col.addWidget(self._breath_label)
        center_col.addSpacing(SPACE_2)

        # Status / instruction text
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px;"
        )
        center_col.addWidget(self._status_label)
        center_col.addSpacing(SPACE_3)

        # CTA button — dark pill style
        self._cta_btn = QPushButton("  Start")
        self._cta_btn.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        self._cta_btn.setIconSize(QSize(16, 16))
        self._cta_btn.setFixedSize(CALIBRATION_CTA_WIDTH, CALIBRATION_CTA_HEIGHT)
        self._cta_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cta_btn.clicked.connect(self._on_cta_clicked)
        self._apply_cta_style_start()
        center_col.addWidget(self._cta_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._content_stack.addWidget(page)

    # ------------------------------------------------------------------
    # Eye alignment slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_eye_locked(self) -> None:
        """Unlock the handoff into the breathing step once alignment is stable."""
        self._eye_ready = True
        self._pupil_status_label.setText("Alignment confirmed.")
        self._pupil_hint_label.setText("Press Space or Next")
        logger.info("Eye alignment locked — waiting for user to continue.")

    @pyqtSlot()
    def _on_eye_cam_unavailable(self) -> None:
        """If the camera cannot open, allow the user to continue anyway."""
        self._eye_ready = True
        self._pupil_status_label.setText(
            "Camera unavailable — grant permission in System Settings → Privacy → Camera"
        )
        self._pupil_hint_label.setText("Press Space or Next to continue without the preview")
        logger.warning("Eye camera unavailable; step 1 can still continue.")

    def _show_pupil_alignment_step(self) -> None:
        """Activate the eye-alignment screen."""
        self._step = "pupil_alignment"
        self._content_stack.setCurrentIndex(0)
        self._countdown_ring.hide()
        self._restart_btn.hide()
        if USE_EYE_TRACKER:
            self._eye_ready = False
            self._pupil_status_label.setText("Auto-detecting pupil alignment...")
            self._pupil_hint_label.setText("Press Space or Next to continue")
            if self._eye_preview:
                self._eye_preview.start()
        else:
            self._eye_ready = True
            self._pupil_status_label.setText("Eye tracker disabled")
            self._pupil_hint_label.setText("Press Space or Next")
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _show_breathing_step(self) -> None:
        """Activate the breathing baseline screen after eye alignment."""
        self._step = "breathing"
        self._content_stack.setCurrentIndex(1)
        self._breath_label.setText("Press Start")
        self._status_label.setText("Follow the orb to record your HRV and pupil baseline")
        self._cta_btn.show()
        self._cta_btn.setEnabled(True)
        self._cta_btn.setText("  Start")
        self._cta_btn.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        self._apply_cta_style_start()
        if self._eye_preview:
            self._eye_preview.stop()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    @pyqtSlot()
    def _on_space_pressed(self) -> None:
        """Advance from pupil alignment to breathing when Space is pressed."""
        if self._step != "pupil_alignment":
            return
        self._continue_from_pupil_step("spacebar")

    def _continue_from_pupil_step(self, source: str) -> None:
        """Advance to breathing step, with manual override when eye lock is missing."""
        if self._step != "pupil_alignment":
            return
        if not self._eye_ready:
            self._pupil_status_label.setText("Proceeding without confirmed alignment.")
            logger.info("Pupil alignment manually overridden via %s.", source)
        self._show_breathing_step()

    # ------------------------------------------------------------------
    # Baseline popup
    # ------------------------------------------------------------------

    def _format_baseline_value_lines(self) -> tuple[str, str]:
        """Return two formatted baseline value lines without dummy values."""
        if self._computed_rmssd > 0.0:
            rmssd_line = f"Baseline RMSSD: {self._computed_rmssd:.2f} ms"
        else:
            rmssd_line = "Baseline RMSSD: Not available"

        if self._computed_pupil > 0.0:
            pupil_line = f"Baseline pupil: {self._computed_pupil:.3f} px"
        else:
            pupil_line = "Baseline pupil: Not available"

        return rmssd_line, pupil_line

    def _show_baseline_popup(self) -> None:
        """Show a popup card with computed baseline values after calibration."""
        rmssd_line, pupil_line = self._format_baseline_value_lines()
        calibration_ok = self._computed_rmssd > 0.0 or self._computed_pupil > 0.0

        popup = QDialog(self)
        popup.setModal(True)
        popup.setWindowTitle("Calibration Complete")
        popup.setMinimumWidth(360)

        layout = QVBoxLayout(popup)
        layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        layout.setSpacing(SPACE_2)

        if calibration_ok:
            headline = QLabel("Calibration successful")
            headline.setStyleSheet(
                f"color: {COLOR_SUCCESS}; font-size: {FONT_TITLE}px; font-weight: 700;"
            )
        else:
            headline = QLabel("Calibration not successful")
            headline.setStyleSheet(
                f"color: {COLOR_DANGER}; font-size: {FONT_TITLE}px; font-weight: 700;"
            )
            note = QLabel(
                "No sensor data was received during the calibration window. "
                "Check that your ECG and eye tracker are connected and try again."
            )
            note.setWordWrap(True)
            note.setStyleSheet(
                f"color: {COLOR_FONT}; font-size: {FONT_BODY_LARGE}px; font-weight: 400;"
            )

        layout.addWidget(headline)

        rmssd_label = QLabel(rmssd_line)
        rmssd_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_BODY_LARGE}px; font-weight: 500;"
        )
        layout.addWidget(rmssd_label)

        pupil_label = QLabel(pupil_line)
        pupil_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_BODY_LARGE}px; font-weight: 500;"
        )
        layout.addWidget(pupil_label)

        if not calibration_ok:
            layout.addWidget(note)

        next_btn = QPushButton("Start Session Anyway" if not calibration_ok else "Next")
        next_btn.clicked.connect(popup.accept)
        layout.addWidget(next_btn, alignment=Qt.AlignmentFlag.AlignRight)

        if popup.exec() == QDialog.DialogCode.Accepted:
            self.proceed_to_live.emit()

    # ------------------------------------------------------------------
    # CTA button styling
    # ------------------------------------------------------------------

    def _apply_cta_style_start(self) -> None:
        """Dark navy pill — used for the initial 'Start' state."""
        cta_radius = CALIBRATION_CTA_HEIGHT // 2
        self._cta_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {COLOR_FONT};
                color: #FFFFFF;
                border: none;
                border-radius: {cta_radius}px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLOR_PRIMARY};
            }}
            QPushButton:disabled {{
                background: #A8B4CE;
            }}
            """
        )

    def _apply_cta_style_complete(self) -> None:
        """Primary blue pill — used after baseline is complete."""
        cta_radius = CALIBRATION_CTA_HEIGHT // 2
        self._cta_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {COLOR_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: {cta_radius}px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLOR_PRIMARY_HOVER};
            }}
            """
        )

    # ------------------------------------------------------------------
    # Close / skip
    # ------------------------------------------------------------------

    def _on_skip_calibration(self) -> None:
        """Skip calibration and proceed directly to live session."""
        self._prestart_timer.stop()
        self._prestart_active = False
        self._countdown_timer.stop()
        if self._eye_preview:
            self._eye_preview.stop()

        if self._session_manager is not None and self._recording:
            elapsed = get_calibration_duration() - self._baseline_remaining
            try:
                rmssd, pupil = self._session_manager.end_calibration(duration_seconds=elapsed)
                self._computed_rmssd = rmssd
                self._computed_pupil = pupil
            except Exception:
                logger.exception("Failed to end calibration during skip.")

        self._recording = False
        self._complete = False
        self._countdown_ring.hide()
        self._countdown_ring.set_progress(1.0)
        self.proceed_to_live.emit()

    def _on_close(self) -> None:
        """Stop all timers and emit close_requested → back to Dashboard."""
        self._prestart_timer.stop()
        self._prestart_active = False
        self._countdown_timer.stop()
        if self._eye_preview:
            self._eye_preview.stop()
        if self._session_manager and self._recording:
            try:
                self._session_manager.end_calibration(
                    duration_seconds=get_calibration_duration() - self._baseline_remaining
                )
            except Exception:
                pass
        self.close_requested.emit()

    # ------------------------------------------------------------------
    # CTA handler
    # ------------------------------------------------------------------

    def _on_cta_clicked(self) -> None:
        if self._complete:
            self.proceed_to_live.emit()
        else:
            if self._recording or self._prestart_active:
                return
            self._start_prestart_countdown()

    def _start_prestart_countdown(self) -> None:
        """Show a 3-2-1 countdown before baseline recording starts."""
        self._prestart_active = True
        self._prestart_remaining = 3
        self._cta_btn.setEnabled(False)
        self._breath_label.setText(f"Starting in {self._prestart_remaining}…")
        self._status_label.setText("Relax and prepare to breathe with the orb")
        self._prestart_timer.start()

    def _tick_prestart_countdown(self) -> None:
        """Advance the pre-start countdown every second."""
        self._prestart_remaining -= 1
        if self._prestart_remaining > 0:
            self._breath_label.setText(f"Starting in {self._prestart_remaining}…")
            return

        self._prestart_timer.stop()
        self._prestart_active = False
        self._start_baseline()

    # ------------------------------------------------------------------
    # Baseline recording
    # ------------------------------------------------------------------

    def _start_baseline(self) -> None:
        """Begin the 60-second resting baseline measurement."""
        self._prestart_timer.stop()
        self._prestart_active = False
        self._recording = True
        self._complete = False
        self._baseline_remaining = get_calibration_duration()
        self._countdown_ring.show()
        self._countdown_ring.set_progress(1.0)
        self._restart_btn.show()
        self._orb.set_preview_mode(False)
        self._orb.restart_from_inhale()

        # Stop the preview camera — the eye tracker itself will take over.
        if self._eye_preview:
            self._eye_preview.stop()

        self._cta_btn.hide()
        self._breath_label.setText("Breathe in…")
        self._status_label.setText(
            f"Recording HRV + pupil baseline — {get_calibration_duration()} s remaining"
        )

        if self._session_manager is not None:
            self._session_manager.start_calibration()

        self._countdown_timer.start()
        logger.info("Calibration baseline recording started.")

    def _restart_baseline(self) -> None:
        """Restart baseline recording from 0 when user clicks Restart."""
        if not self._recording:
            return

        self._countdown_timer.stop()
        self._baseline_remaining = get_calibration_duration()
        self._countdown_ring.show()
        self._countdown_ring.set_progress(1.0)
        self._restart_btn.show()
        self._orb.set_preview_mode(False)
        self._orb.restart_from_inhale()
        self._status_label.setText("")

        if self._session_manager is not None:
            self._session_manager.start_calibration()

        self._countdown_timer.start()
        logger.info("Calibration baseline restarted by user.")

    def _tick_countdown(self) -> None:
        """Called every second during baseline recording."""
        self._baseline_remaining -= 1
        progress = self._baseline_remaining / get_calibration_duration()
        self._countdown_ring.set_progress(progress)
        if self._baseline_remaining > 0:
            self._status_label.setText(
                f"Recording HRV + pupil baseline — {self._baseline_remaining} s remaining"
            )
        else:
            self._countdown_timer.stop()
            self._finish_baseline()

    def _finish_baseline(self) -> None:
        """Compute baseline values and transition to 'Start Session' state."""
        if self._session_manager is not None:
            rmssd, pupil = self._session_manager.end_calibration(
                duration_seconds=get_calibration_duration()
            )
            self._computed_rmssd = rmssd
            self._computed_pupil = pupil

        self._recording = False
        self._complete = True
        self._step = "complete"
        self._breath_label.setText("Press Start")
        self._status_label.setText("Ready to begin")
        self._restart_btn.hide()
        self._orb.set_preview_mode(True)
        self._countdown_ring.hide()
        self._countdown_ring.set_progress(1.0)
        self._show_baseline_popup()

        self._cta_btn.show()
        self._cta_btn.setEnabled(True)
        self._cta_btn.setText("  Start Session")
        self._cta_btn.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        self._apply_cta_style_complete()
        logger.info(
            "Calibration complete. RMSSD=%.2f ms, pupil=%.3f px",
            self._computed_rmssd, self._computed_pupil,
        )

    # ------------------------------------------------------------------
    # SessionManager signal slot
    # ------------------------------------------------------------------

    @pyqtSlot(float, float)
    def _on_calibration_complete(self, rmssd: float, pupil_px: float) -> None:
        """Receive baseline values pushed by SessionManager after computation.

        Args:
            rmssd: Resting RMSSD baseline in milliseconds.
            pupil_px: Resting pupil diameter baseline in pixels.
        """
        self._computed_rmssd = rmssd
        self._computed_pupil = pupil_px
        logger.info(
            "CalibrationView received baseline: RMSSD=%.2f ms, pupil=%.3f px",
            rmssd, pupil_px,
        )

    # ------------------------------------------------------------------
    # Reset (called by MainWindow before each new session)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return to initial state for a fresh calibration run."""
        self._prestart_timer.stop()
        self._prestart_active = False
        self._countdown_timer.stop()
        self._step = "pupil_alignment"
        self._eye_ready = not USE_EYE_TRACKER
        self._baseline_remaining = get_calibration_duration()
        self._recording = False
        self._complete = False
        self._computed_rmssd = 0.0
        self._computed_pupil = 0.0
        self._restart_btn.hide()
        self._orb.set_preview_mode(True)
        self._countdown_ring.hide()
        self._countdown_ring.set_progress(1.0)
        self._breath_label.setText("Press Start")
        self._cta_btn.show()
        self._cta_btn.setText("  Start")
        self._cta_btn.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        self._apply_cta_style_start()
        logger.debug("CalibrationView reset.")

        self._status_label.setText("")
        self._show_pupil_alignment_step()

    def _on_orb_phase_changed(self, phase: str) -> None:
        """Update breath guidance text to match orb animation phase.

        Runs during both the pre-start countdown and the active recording so
        users can start syncing their breathing before the baseline begins.
        """
        if not self._recording and not self._prestart_active:
            return

        if phase == "inhale":
            self._breath_label.setText("Breathe in…")
        elif phase == "exhale":
            self._breath_label.setText("Breathe out…")
