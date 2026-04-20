"""VideoPlayer — OpenCV-based MP4 playback widget for BioTrace.

Provides a self-contained player for session recordings with play/pause,
seeking, and time display.
"""

from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import (
    COLOR_BORDER,
    COLOR_FONT,
    COLOR_FONT_MUTED,
    COLOR_PRIMARY,
    FONT_CAPTION,
    RADIUS_LG,
    SPACE_1,
    SPACE_2,
    CHART_HEIGHT_TIMELINE,
    get_icon,
)
from app.utils.config import VIDEO_RECORDING_FPS_FALLBACK
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoPlayer(QWidget):
    """A self-contained video playback widget using OpenCV and QTimer.

    Signals:
        timestamp_clicked (float): Emitted when the user seeks, in milliseconds.
    """

    timestamp_clicked = pyqtSignal(float)
    # Emitted continuously during playback and on scrubbing — in milliseconds.
    playback_position_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cap: cv2.VideoCapture | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(int(round(1000.0 / VIDEO_RECORDING_FPS_FALLBACK)))
        self._timer.timeout.connect(self._advance_frame)

        self._duration_ms: float = 0.0
        self._source_fps: float = VIDEO_RECORDING_FPS_FALLBACK
        self._frame_count: int = 0
        self._current_frame_index: int = 0
        self._is_playing: bool = False
        self._is_scrubbing: bool = False

        self._build_ui()

    def load(self, file_path: str | None, session_duration_s: float | None = None) -> None:
        """Open a video file for playback.

        Args:
            file_path: Path to the MP4 file, or None.
            session_duration_s: True wall-clock session duration in seconds
                (from the database).  When provided the effective playback FPS
                is derived from ``frame_count / session_duration_s`` so that
                playback matches real time even when the camera delivered fewer
                frames than the declared encoder FPS.
        """
        self.stop()
        if self._cap:
            self._cap.release()
            self._cap = None

        if not file_path or not Path(file_path).exists():
            self._show_placeholder("No recording available for this session.")
            self._controls_container.hide()
            return

        self._cap = cv2.VideoCapture(file_path)
        if not self._cap.isOpened():
            logger.warning("Failed to open video file: %s", file_path)
            self._show_placeholder("Error: Could not load video file.")
            self._controls_container.hide()
            return

        count = float(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = max(0, int(round(count)))

        # Derive the effective playback FPS from the true session duration so
        # that frame_count / fps == wall-clock time.  Falls back to the
        # declared encoder FPS when no duration is available.
        if session_duration_s and session_duration_s > 0 and self._frame_count > 0:
            self._source_fps = self._frame_count / session_duration_s
        else:
            self._source_fps = float(VIDEO_RECORDING_FPS_FALLBACK)

        if self._frame_count > 0:
            self._duration_ms = (self._frame_count / self._source_fps) * 1000.0
        else:
            self._duration_ms = 0.0
        self._timer.setInterval(max(1, int(round(1000.0 / self._source_fps))))

        self._slider.setRange(0, int(self._duration_ms))
        self._slider.setValue(0)
        self._current_frame_index = 0
        self._update_time_label(0)

        self._controls_container.show()
        self._advance_frame()  # Show first frame
        logger.info("Loaded video: %s (%.1f s)", file_path, self._duration_ms / 1000.0)

    def seek_to(self, position_ms: float) -> None:
        """Jump to a specific point in the video.

        Args:
            position_ms: Target timestamp in milliseconds.
        """
        if not self._cap or not self._cap.isOpened():
            return

        pos = max(0.0, min(position_ms, self._duration_ms))
        f = int(round((pos / 1000.0) * self._source_fps))
        if self._frame_count > 0:
            f = max(0, min(f, self._frame_count - 1))
        else:
            f = 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        self._current_frame_index = f

        if not self._is_scrubbing:
            self._slider.setValue(int(pos))
            self._update_time_label(int(pos))

        if not self._is_playing:
            self._advance_frame()

    def play(self) -> None:
        """Start or resume playback."""
        if self._cap and self._cap.isOpened():
            self._is_playing = True
            self._play_btn.setIcon(get_icon("ph.pause-fill", color="#FFFFFF"))
            self._timer.start()

    def pause(self) -> None:
        """Pause playback."""
        self._is_playing = False
        self._play_btn.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        self._timer.stop()

    def stop(self) -> None:
        """Stop playback and reset to start."""
        self.pause()
        if self._cap:
            self.seek_to(0)

    # ------------------------------------------------------------------
    # Internal Logic
    # ------------------------------------------------------------------

    def _display_frame_rgb(self, frame) -> None:
        """Scale and show an RGB numpy frame (caller converts BGR→RGB)."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled_pixmap)

    def _advance_frame(self) -> None:
        """Read the next frame from OpenCV and display it."""
        if not self._cap or not self._cap.isOpened():
            return

        next_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self._cap.read()
        if not ret:
            if self._is_playing:
                self.pause()
            return

        self._current_frame_index = next_idx
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_frame_rgb(frame)

        curr_ms = min(
            ((self._current_frame_index + 1) / self._source_fps) * 1000.0,
            self._duration_ms,
        )

        if self._is_playing:
            self._slider.blockSignals(True)
            self._slider.setValue(int(curr_ms))
            self._slider.blockSignals(False)
            self._update_time_label(int(curr_ms))
            self.playback_position_changed.emit(curr_ms)

    def _update_time_label(self, current_ms: int) -> None:
        """Update the MM:SS / MM:SS text."""
        def fmt(ms: int) -> str:
            s = ms // 1000
            m, s = divmod(s, 60)
            return f"{m}:{s:02d}"

        curr_str = fmt(current_ms)
        total_str = fmt(int(self._duration_ms))
        self._time_label.setText(f"{curr_str} / {total_str}")

    def _show_placeholder(self, text: str) -> None:
        """Display a message in the video area instead of a frame."""
        self._display.setPixmap(QPixmap())
        self._display.setText(text)
        self._display.setStyleSheet(
            f"background-color: #0A0F1E; border-radius: {RADIUS_LG}px;"
            f"color: {COLOR_FONT_MUTED}; font-size: {FONT_CAPTION}px;"
        )

    @pyqtSlot()
    def _on_play_toggle(self) -> None:
        if self._is_playing:
            self.pause()
        else:
            self.play()

    @pyqtSlot()
    def _on_slider_pressed(self) -> None:
        self._is_scrubbing = True

    @pyqtSlot(int)
    def _on_slider_moved(self, value: int) -> None:
        self.seek_to(float(value))
        self.playback_position_changed.emit(float(value))

    @pyqtSlot()
    def _on_slider_released(self) -> None:
        self._is_scrubbing = False
        val = self._slider.value()
        self.seek_to(float(val))
        self.timestamp_clicked.emit(float(val))

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_1)

        # ── Display Area ───────────────────────────────────────────
        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display.setMinimumHeight(CHART_HEIGHT_TIMELINE)
        self._display.setStyleSheet(
            f"background-color: #0A0F1E; border-radius: {RADIUS_LG}px;"
        )
        layout.addWidget(self._display, stretch=1)

        # ── Controls ───────────────────────────────────────────────
        self._controls_container = QWidget()
        ctrl_layout = QHBoxLayout(self._controls_container)
        ctrl_layout.setContentsMargins(SPACE_1, 0, SPACE_1, 0)
        ctrl_layout.setSpacing(SPACE_2)

        self._play_btn = QToolButton()
        self._play_btn.setFixedSize(32, 32)
        self._play_btn.setIcon(get_icon("ph.play-fill", color="#FFFFFF"))
        self._play_btn.setIconSize(QSize(16, 16))
        self._play_btn.setStyleSheet(
            f"QToolButton {{ background: {COLOR_PRIMARY}; border: none; border-radius: 16px; }}"
            f"QToolButton:hover {{ background: #1c3a8c; }}"
        )
        self._play_btn.clicked.connect(self._on_play_toggle)
        ctrl_layout.addWidget(self._play_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)
        ctrl_layout.addWidget(self._slider, stretch=1)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setStyleSheet(
            f"color: {COLOR_FONT}; font-size: {FONT_CAPTION}px; font-weight: 600;"
        )
        ctrl_layout.addWidget(self._time_label)

        layout.addWidget(self._controls_container)
        self._show_placeholder("Select a session to view recording")
        self._controls_container.hide()

    def resizeEvent(self, event) -> None:
        """Handle resize to ensure scaled pixmap fills the available space correctly."""
        super().resizeEvent(event)
        if not self._is_playing and self._cap and self._cap.isOpened() and self._frame_count > 0:
            f = max(0, min(self._current_frame_index, self._frame_count - 1))
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._display_frame_rgb(frame)
