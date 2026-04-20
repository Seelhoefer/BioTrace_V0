"""Placeholder sensors when hardware is turned off in config.

These drivers never emit samples; they only report disconnected status so the
UI can show a clear message without inventing mock ECG or pupil data.
"""

from PyQt6.QtCore import pyqtSignal

from app.hardware.base_sensor import BaseSensor


class DisabledECGSensor(BaseSensor):
    """ECG path disabled via ``USE_PICO_ECG=False``."""

    raw_ecg_sample_received = pyqtSignal(float, float)
    connection_status_changed = pyqtSignal(bool, str)

    def start(self) -> None:
        self._running = True
        self.connection_status_changed.emit(
            False,
            "ECG disabled (USE_PICO_ECG=False). Enable Pico ECG in config to stream data.",
        )

    def stop(self) -> None:
        self._running = False


class DisabledEyeTracker(BaseSensor):
    """Eye tracker path disabled via ``USE_EYE_TRACKER=False``."""

    raw_pupil_received = pyqtSignal(float, float, float)
    connection_status_changed = pyqtSignal(bool, str)

    def start(self) -> None:
        self._running = True
        self.connection_status_changed.emit(
            False,
            "Eye tracker disabled (USE_EYE_TRACKER=False). Enable in config to stream data.",
        )

    def stop(self) -> None:
        self._running = False
