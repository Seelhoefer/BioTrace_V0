"""Abstract base class for all BioTrace sensor drivers.

Every sensor driver (real or mock) must inherit ``BaseSensor`` and implement
``start()`` and ``stop()``.  Each subclass additionally declares its own
Qt signal for emitting raw data.

This contract ensures that swapping a mock sensor for a real hardware driver
requires zero changes in the processing or UI layers.

Example::

    class MyHRVSensor(BaseSensor):
        raw_ecg_sample_received = pyqtSignal(float, float)  # (ecg_value, timestamp)

        def start(self) -> None:
            self._running = True
            # open serial port, start read thread …

        def stop(self) -> None:
            self._running = False
            # close port, join thread …
"""

from abc import abstractmethod

from PyQt6.QtCore import QObject

from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseSensor(QObject):
    """Abstract base class for all sensor drivers.

    Inherits :class:`PyQt6.QtCore.QObject` to participate in the Qt
    signal/slot system.  Subclasses define their own typed signals.

    Subclasses **must** implement :meth:`start` and :meth:`stop`.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        """Whether the sensor is currently streaming data."""
        return self._running

    @abstractmethod
    def start(self) -> None:
        """Begin streaming data and emit signals.

        Implementations must set ``self._running = True`` and start any
        threads or timers required to produce data.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release all resources.

        Implementations must set ``self._running = False`` and join any
        background threads or stop timers cleanly before returning.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} running={self._running}>"
