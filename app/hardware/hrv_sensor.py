"""Stub hardware drivers for the real HRV sensor and eye tracker.

These files define the class interface that must be filled in once the physical
devices are available (Phase 6 of the roadmap). Production ECG uses
:class:`~app.hardware.pico_ecg_sensor.PicoECGSensor`; when ``USE_PICO_ECG`` is
false, :class:`~app.hardware.disabled_sensors.DisabledECGSensor` is used instead.
"""

from PyQt6.QtCore import QObject, pyqtSignal

from app.hardware.base_sensor import BaseSensor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HRVSensor(BaseSensor):
    """Real HRV sensor driver (serial / USB microcontroller).

    .. note::
        Not yet implemented. Use :class:`~app.hardware.pico_ecg_sensor.PicoECGSensor`
        for USB ECG.

    Signals:
        raw_ecg_sample_received (float, float):
            Emitted with ``(ecg_amplitude, timestamp_s)`` for each sample.
    """

    raw_ecg_sample_received = pyqtSignal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # TODO (Phase 6): initialise serial port from config.HRV_SENSOR_PORT

    def start(self) -> None:
        """Open serial connection and begin streaming raw ECG samples."""
        raise NotImplementedError(
            "HRVSensor is not implemented yet. Use PicoECGSensor for USB ECG."
        )

    def stop(self) -> None:
        """Close serial connection."""
        raise NotImplementedError(
            "HRVSensor is not implemented yet. Use PicoECGSensor for USB ECG."
        )
