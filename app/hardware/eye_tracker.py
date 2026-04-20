"""Real-time eye tracker driver using the PuRe algorithm via USB camera.

This sensor captures frames from a dedicated USB camera, detects the pupil
using the PuRe (Santini et al., 2018) algorithm, and emits raw diameter
measurements in pixels.
"""

import sys
import time
import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from app.hardware.base_sensor import BaseSensor
from app.utils.config import EYE_TRACKER_CAMERA_INDEX
from app.utils.logger import get_logger

logger = get_logger(__name__)
_FALLBACK_NOTICE_LOGGED: bool = False


class _PupilWorker(QThread):
    """Background thread for camera capture and pupil detection.

    Uses the pypupilext (PuRe) detector with ROI optimization and
    center-position smoothing.
    """
    raw_pupil_received = pyqtSignal(float, float, float)
    connection_status_changed = pyqtSignal(bool, str)

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._running = True

    def run(self) -> None:
        """Main detection loop."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            if sys.platform == "darwin":
                msg = (
                    f"Eye tracker camera (index {self._camera_index}) could not be opened. "
                    "On macOS this is usually a camera permission issue. "
                    "Fix: System Settings → Privacy & Security → Camera → "
                    "enable access for Terminal (or VS Code), then restart the app."
                )
            else:
                msg = (
                    f"Eye tracker camera (index {self._camera_index}) not found or could not be opened. "
                    "Check that the USB eye tracker is connected and no other app is using it."
                )
            logger.error(msg)
            self.connection_status_changed.emit(False, msg)
            return

        self.connection_status_changed.emit(True, "Eye tracker connected")
        
        try:
            import pypupilext as pp  # type: ignore
            detector = pp.PuRe()
        except ImportError:
            global _FALLBACK_NOTICE_LOGGED
            if not _FALLBACK_NOTICE_LOGGED:
                logger.info("pypupilext not found. Using simple OpenCV fallback for pupil detection.")
                _FALLBACK_NOTICE_LOGGED = True
            class DummyPupil:
                def __init__(self, cx, cy, diameter, is_valid):
                    self.center = (cx, cy)
                    self._diameter = diameter
                    self._valid = is_valid
                def valid(self, conf):
                    return self._valid
                def diameter(self):
                    return self._diameter

            class FallbackPuRe:
                def runWithConfidence(self, gray_img):
                    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)
                    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)
                    M = cv2.moments(thresh)
                    if M["m00"] > 50:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        d = 2 * np.sqrt(M["m00"] / np.pi)
                        return DummyPupil(cx, cy, d, True)
                    return DummyPupil(float(gray_img.shape[1]/2), float(gray_img.shape[0]/2), 10.0, False)

            detector = FallbackPuRe()

        # Detection state
        last_good_center = None
        last_good_diameter = None
        last_good_time = 0.0
        smoothed_center = None

        # Hyperparameters (from test_pupil.py)
        roi_size = 160
        alpha = 0.3
        confidence_threshold = 0.35
        hold_time = 0.5  # seconds to hold position during blinks

        logger.info("Pupil detection worker started.")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Pupil worker: failed to capture frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            pupil = None
            cx, cy = 0.0, 0.0

            # 1. Try ROI optimization
            if last_good_center is not None:
                cx_roi, cy_roi = int(last_good_center[0]), int(last_good_center[1])
                x1 = max(0, cx_roi - roi_size)
                x2 = min(w, cx_roi + roi_size)
                y1 = max(0, cy_roi - roi_size)
                y2 = min(h, cy_roi + roi_size)

                gray_roi = gray[y1:y2, x1:x2]
                pupil_roi = detector.runWithConfidence(gray_roi)

                if pupil_roi.valid(confidence_threshold):
                    px, py = pupil_roi.center
                    cx, cy = px + x1, py + y1
                    pupil = pupil_roi

            # 2. Fallback to full frame
            if pupil is None:
                pupil_full = detector.runWithConfidence(gray)
                if pupil_full.valid(confidence_threshold):
                    cx, cy = pupil_full.center
                    pupil = pupil_full

            # 3. Process detection
            now = time.time()
            if pupil is not None:
                diameter = pupil.diameter()

                # Center position smoothing (EMA)
                if smoothed_center is None:
                    smoothed_center = (cx, cy)
                else:
                    sx = alpha * cx + (1 - alpha) * smoothed_center[0]
                    sy = alpha * cy + (1 - alpha) * smoothed_center[1]
                    smoothed_center = (sx, sy)

                cx, cy = smoothed_center
                last_good_center = (cx, cy)
                last_good_diameter = diameter
                last_good_time = now

                # Emit monocular diameter (left=diameter, right=0.0)
                self.raw_pupil_received.emit(float(diameter), 0.0, now)

            elif last_good_center is not None and (now - last_good_time) < hold_time:
                # Brief blink/loss: hold the last known position (but don't emit new diameter)
                pass
            else:
                # Total loss of signal
                last_good_center = None
                smoothed_center = None

            # Cap frame rate to ~30 FPS to avoid CPU saturation
            self.msleep(33)

        cap.release()
        logger.info("Pupil detection worker stopped.")

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._running = False
        self.wait()


class EyeTrackerSensor(BaseSensor):
    """Real eye tracker driver using the PuRe algorithm.

    Signals:
        raw_pupil_received (float, float, float):
            Emitted with ``(left_px, right_px, timestamp_s)``.
        connection_status_changed (bool, str):
            Emitted when the camera is opened or lost.
    """
    raw_pupil_received = pyqtSignal(float, float, float)
    connection_status_changed = pyqtSignal(bool, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: _PupilWorker | None = None

    def start(self) -> None:
        """Start the camera capture and detection thread."""
        if self._worker and self._worker.isRunning():
            return

        self._worker = _PupilWorker(camera_index=EYE_TRACKER_CAMERA_INDEX)
        self._worker.raw_pupil_received.connect(self.raw_pupil_received)
        self._worker.connection_status_changed.connect(self.connection_status_changed)
        self._worker.start()
        logger.info("EyeTrackerSensor started.")

    def stop(self) -> None:
        """Stop the background thread and release resources."""
        if self._worker:
            self._worker.stop()
            self._worker = None
        logger.info("EyeTrackerSensor stopped.")
