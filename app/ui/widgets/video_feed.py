"""VideoFeed — OpenCV camera capture widget for BioTrace.

Captures frames from a camera in a dedicated ``QThread`` worker and
displays them in a ``QLabel``.  The main UI thread is never blocked.

Architecture:
    _CameraWorker (QThread) → frame_ready signal → VideoFeed (QLabel)

Usage::

    feed = VideoFeed(camera_index=0)
    feed.start()
    # show feed inside a layout
    # …
    feed.stop()

Recording support::

    feed.start_recording("recordings/session_1.mp4")
    # …
    feed.stop_recording()
"""

import sys
import threading
import time
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

from app.utils.config import CAMERA_INDEX, CAMERA_WARMUP_FRAMES
from app.utils.config import VIDEO_RECORDING_FOURCC, VIDEO_RECORDING_FPS_FALLBACK
from app.utils.logger import get_logger

logger = get_logger(__name__)


class _CameraWorker(QThread):
    """Background thread that reads frames from an OpenCV VideoCapture.

    Signals:
        frame_ready (QImage): Emitted with a pre-processed RGB QImage.
        error_occurred (str): Emitted when the camera cannot be opened or
                              a frame cannot be read.
    """

    frame_ready = pyqtSignal(object)   # QImage (RGB)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        super().__init__()
        self._camera_index = camera_index
        # _stop_requested is set by stop() and checked throughout run() so that
        # calling stop() during the warmup phase terminates the thread cleanly
        # instead of letting warmup finish and then re-arming the capture loop.
        self._stop_requested: bool = False
        self._record_target_path: str | None = None
        self._recording_enabled: bool = False
        self._writer: cv2.VideoWriter | None = None
        self._camera_ready: bool = False
        self._record_lock = threading.Lock()
        # Pace recording writes to VIDEO_RECORDING_FPS_FALLBACK so wall-clock
        # session time matches frame_count / fps (avoids double-speed playback
        # when the camera delivers faster than the declared encoder FPS).
        self._next_record_deadline: float | None = None

    def run(self) -> None:
        """Capture loop — runs until ``stop()`` is called."""
        logger.debug("Camera worker thread %s started.", self.objectName())

        # On macOS the default backend is already AVFoundation, but specifying it
        # explicitly avoids OpenCV picking an incompatible backend on mixed setups.
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._camera_index, backend)

        if not cap.isOpened():
            msg = f"Cannot open camera index {self._camera_index}."
            logger.error(msg)
            self.error_occurred.emit(msg)
            return
        # Reduce the internal buffer to 1 frame — on macOS/AVFoundation this
        # prevents the driver from delivering stale frames after initialization
        # and avoids the "opens but read() returns False" failure mode.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warmup phase: some USB cameras report isOpened()=True before they are
        # actually ready to stream.  Read and discard up to CAMERA_WARMUP_FRAMES
        # frames; the first successful read marks the camera as ready.
        # Each iteration checks _stop_requested so stop() terminates the thread
        # immediately rather than waiting up to 2 s for warmup to finish.
        warmed_up = False
        for _ in range(CAMERA_WARMUP_FRAMES):
            if self._stop_requested:
                cap.release()
                return
            ret, _ = cap.read()
            if ret:
                warmed_up = True
                break
            QThread.msleep(33)  # ~30 fps cadence

        if not warmed_up:
            msg = (
                f"Camera index {self._camera_index} opened but produced no frames "
                f"after {CAMERA_WARMUP_FRAMES} warmup attempts. "
                "Check that the camera is connected and not in use by another application."
            )
            logger.error(msg)
            self.error_occurred.emit(msg)
            cap.release()
            return

        if self._stop_requested:
            cap.release()
            return

        self._camera_ready = True
        logger.info("Camera %d opened successfully.", self._camera_index)

        _consecutive_failures = 0
        _FAILURE_LOG_THRESHOLD = 30  # log a warning only every N consecutive failures

        while not self._stop_requested:
            ret, frame = cap.read()
            if not ret:
                _consecutive_failures += 1
                if _consecutive_failures % _FAILURE_LOG_THRESHOLD == 1:
                    logger.warning(
                        "Frame capture failed (camera %d) — %d consecutive miss(es).",
                        self._camera_index,
                        _consecutive_failures,
                    )
                QThread.msleep(10)
                continue

            _consecutive_failures = 0

            self._sync_recording_state(cap, frame)
            with self._record_lock:
                if self._writer is not None:
                    now = time.monotonic()
                    interval = 1.0 / float(VIDEO_RECORDING_FPS_FALLBACK)
                    if self._next_record_deadline is None:
                        self._next_record_deadline = now
                    if now >= self._next_record_deadline - 1e-4:
                        try:
                            self._writer.write(frame)
                        except cv2.error as exc:
                            logger.warning("Video encoder write failed, stopping recording: %s", exc)
                            self._recording_enabled = False
                            self._release_writer()
                        else:
                            self._next_record_deadline += interval
                            # If we fell behind (UI load, GC), skip bursts — stay near real time.
                            if self._next_record_deadline < now - interval:
                                self._next_record_deadline = now

            # Offload BGR → RGB conversion to this worker thread.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            # .tobytes() creates an owned copy so the QImage does not hold a
            # reference to the numpy array.  .copy() on the QImage ensures it
            # owns its pixel buffer independently of the bytes object.
            qt_image = QImage(
                rgb.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888
            ).copy()
            self.frame_ready.emit(qt_image)

        with self._record_lock:
            self._release_writer()
        cap.release()
        self._camera_ready = False
        logger.info("Camera %d released.", self._camera_index)

    def stop(self) -> None:
        """Signal the capture loop to exit and wait for it to finish."""
        self._stop_requested = True
        self.quit()
        if not self.wait(1500):
            logger.warning("Camera worker thread did not stop gracefully; terminating.")
            self.terminate()
            self.wait()

    def start_recording(self, output_path: str) -> None:
        """Enable recording to the provided path.

        Writer initialization is deferred until the next valid frame so we can
        match the frame size of the actual capture stream.
        """
        with self._record_lock:
            self._record_target_path = output_path
            self._recording_enabled = True

    def stop_recording(self) -> None:
        """Disable recording and release any active writer."""
        with self._record_lock:
            self._recording_enabled = False
            self._release_writer()

    @property
    def is_recording(self) -> bool:
        """Return whether recording has been enabled."""
        with self._record_lock:
            return self._recording_enabled

    def _sync_recording_state(self, cap: cv2.VideoCapture, frame: np.ndarray) -> None:
        """Start/stop writer lazily based on recording toggle state."""
        with self._record_lock:
            if (
                self._camera_ready
                and self._recording_enabled
                and self._writer is None
                and self._record_target_path is not None
            ):
                height, width = frame.shape[:2]
                # Use a stable target FPS for encoding. Some camera backends
                # report unreliable CAP_PROP_FPS values that make playback run
                # too fast/slow when the file is decoded later.
                fps = VIDEO_RECORDING_FPS_FALLBACK

                Path(self._record_target_path).parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*VIDEO_RECORDING_FOURCC)
                self._writer = cv2.VideoWriter(self._record_target_path, fourcc, fps, (width, height))
                if not self._writer.isOpened():
                    logger.error("Cannot open video writer for %s", self._record_target_path)
                    self._writer = None
                    self._recording_enabled = False
                else:
                    logger.info("Recording to %s (%.1f fps, %dx%d)", self._record_target_path, fps, width, height)

            if not self._recording_enabled and self._writer is not None:
                self._release_writer()

    def _release_writer(self) -> None:
        """Release writer if active."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._next_record_deadline = None


class VideoFeed(QLabel):
    """A QLabel that displays a live OpenCV camera stream.

    The label scales the frame to fit its current size while preserving the
    aspect ratio.  When no camera is available the label shows a placeholder.

    In addition to rendering frames internally, ``frame_ready`` is emitted for
    each accepted frame so that secondary display widgets (e.g. a small preview
    in the biofeedback panel) can subscribe without opening a second camera.

    Args:
        camera_index: OpenCV camera index (default from ``config.py``).
        parent: Optional parent widget.

    Signals:
        frame_ready (QImage): Emitted for every accepted camera frame.
    """

    frame_ready = pyqtSignal(object)  # QImage — proxy for secondary displays

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._worker: _CameraWorker | None = None
        self._active = False

        self._showing_placeholder: bool = False

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self._show_placeholder("No video detected")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self, camera_index: int | None = None) -> None:
        """Start the capture thread and begin displaying frames.
        
        Args:
            camera_index: Optional override for the camera index.
        """
        if self._active:
            if camera_index is not None and camera_index != self._camera_index:
                self.stop()
            else:
                return
        
        if camera_index is not None:
            self._camera_index = camera_index

        self._active = True
        self._worker = _CameraWorker(self._camera_index)
        self._worker.setObjectName(f"CameraWorker-{self._camera_index}")
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()
        logger.info("VideoFeed started (camera %d).", self._camera_index)

    def stop(self) -> None:
        """Stop the capture thread."""
        if not self._active and self._worker is None:
            return

        self._active = False
        if self._worker is not None:
            # Disconnect signals immediately to avoid processing frames during shutdown.
            try:
                self._worker.frame_ready.disconnect(self._on_frame)
                self._worker.error_occurred.disconnect(self._on_error)
            except (TypeError, RuntimeError):
                pass
                
            logger.debug("Stopping camera worker thread...")
            self._worker.stop_recording()
            self._worker.stop()
            
            # Ensure it's dead before dropping the reference.
            if self._worker.isRunning():
                logger.warning("Camera worker still running after stop() - forcing wait.")
                self._worker.wait(2000)
                
            self._worker = None
            
        self._show_placeholder("No video detected")
        logger.info("VideoFeed stopped.")

    def start_recording(self, output_path: str) -> bool:
        """Start recording the live feed to ``output_path``.

        Returns:
            ``True`` if recording was requested, ``False`` if feed is inactive.
        """
        if not self._active or self._worker is None:
            return False
        self._worker.start_recording(output_path)
        return True

    def stop_recording(self) -> None:
        """Stop active recording, if any."""
        if self._worker is not None:
            self._worker.stop_recording()

    @property
    def is_recording(self) -> bool:
        """Return whether the feed is currently recording."""
        if self._worker is None:
            return False
        return self._worker.is_recording

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_frame(self, qt_image: QImage) -> None:
        """Display a pre-processed QImage.

        Args:
            qt_image: RGB QImage from worker.
        """
        if not self._active:
            return  # feed was stopped; discard any queued frames

        # Broadcast the raw image first so secondary display widgets can render
        # camera frames even while this primary label lives in a hidden stacked
        # page and therefore has no usable on-screen size yet.
        self.frame_ready.emit(qt_image)

        # Clear the placeholder stylesheet on the very first frame so the
        # background fill and border don't paint over the camera image.
        if self._showing_placeholder:
            self._showing_placeholder = False
            self.setStyleSheet("")
            self.setText("")

        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return  # widget not yet laid out — skip until it has a real size

        pixmap = QPixmap.fromImage(qt_image)
        # Fill the full widget while preserving aspect ratio (center-crop).
        # FastTransformation keeps rendering lightweight in the UI thread.
        scaled = pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(scaled)

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        """Display an error message when the camera cannot be opened.

        Args:
            message: Human-readable error description.
        """
        self._show_placeholder("No video detected")

    def _show_placeholder(self, text: str) -> None:
        """Display a styled placeholder text instead of a camera frame."""
        from app.ui.theme import COLOR_BACKGROUND, COLOR_BORDER, COLOR_FONT_MUTED, FONT_SUBTITLE
        self._showing_placeholder = True
        self.setPixmap(QPixmap())  # Clear any stale frame

        html = (
            f"<div align='center'>"
            f"<span style='font-size: 40px; color: {COLOR_FONT_MUTED};'>&#x25A6;</span><br><br>"
            f"<span style='font-size: {FONT_SUBTITLE}px; color: {COLOR_FONT_MUTED};'>"
            f"{text.upper()}"
            f"</span>"
            f"</div>"
        )
        self.setText(html)
        self.setStyleSheet(
            f"QLabel {{\n"
            f"   background-color: {COLOR_BACKGROUND};\n"
            f"   border: 1px solid {COLOR_BORDER};\n"
            f"   border-radius: 12px;\n"
            f"}}"
        )
