"""Hardware error counter driver stub.

This module provides the interface for the wire-touch sensor on the
laparoscopy box trainer. For Phase 6c, only the manual UI fallback is active;
this stub locks in the signal contract for Phase 6b hardware integration.
"""

from PyQt6.QtCore import pyqtSignal, QThread

from app.hardware.base_sensor import BaseSensor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorCounter(BaseSensor):
    """Driver for the laparoscopic box trainer error (wire-touch) sensor.

    Signals:
        error_detected: Emitted each time a wire-touch error occurs.
    """

    error_detected = pyqtSignal()

    def __init__(self) -> None:
        """Initialise the error counter sensor."""
        super().__init__()
        logger.debug("ErrorCounter stub initialised.")

    def start(self) -> None:
        """Begin monitoring hardware for wire-touch errors.

        Raises:
            NotImplementedError: Hardware error counter not yet implemented.
                                 Use ErrorInputWidget for manual counting.
        """
        raise NotImplementedError(
            "Hardware error counter not yet implemented. Use ErrorInputWidget for manual counting."
        )

    def stop(self) -> None:
        """Stop monitoring hardware."""
        # No-op in stub
        pass


class ErrorCounterWorker(QThread):
    """Background worker thread for polling GPIO/serial for wire-touch errors.

    TODO (Phase 6b): Implement GPIO polling or serial read loop here.
    """

    def __init__(self) -> None:
        """Initialise the worker thread."""
        super().__init__()

    def run(self) -> None:
        """Execute the read loop."""
        # Body is a pass for now; ready for Phase 6b hardware logic.
        pass
