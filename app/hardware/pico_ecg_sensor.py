"""Pi Pico ECG sensor driver for BioTrace.

Reads raw ECG samples from a Raspberry Pi Pico running YLab Zero
CircuitPython firmware over USB serial and forwards each sample to the
processing layer for windowed neurokit2 analysis (no host-side threshold
R-detector).

The Pi Pico runs ``sensory.print()`` in its Active state, producing one line
per sample interval in the format::

    Yeda0:(0.00234,) MOI0:(0.0,) MOI1:(1.0,)

The ``Yeda`` sensor is configured with ``reciprocal=True`` on the device,
meaning the transmitted value is ``1 / adc_voltage``.  This driver inverts it
back to the original ECG amplitude before emitting ``raw_ecg_sample_received``.

Usage::

    from app.hardware.pico_ecg_sensor import PicoECGSensor

    sensor = PicoECGSensor(port="/dev/tty.usbmodem101")
    sensor.raw_ecg_sample_received.connect(hrv_processor.on_ecg_sample)
    sensor.connection_status_changed.connect(handle_status)
    sensor.start()
    # … later …
    sensor.stop()
"""

import re
import time

import serial
import serial.tools.list_ports
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from app.hardware.base_sensor import BaseSensor
from app.utils.config import (
    PICO_ECG_BAUD,
    PICO_ECG_PORT,
    PICO_WALL_CONTACT_MIN_EVENT_INTERVAL_SECONDS,
    PICO_WALL_CONTACT_LOG_THROTTLE_SECONDS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_YEDA_RE = re.compile(r"Yeda\d+:\(([^,)]+),?\)")
_MOI_RE = re.compile(r"(MOI\d+):\(([^,)]+),?\)")

_PICO_VIDS: frozenset[int] = frozenset({
    0x2E8A,
    0x239A,
})
_PICO_DESC_KEYWORDS: tuple[str, ...] = ("circuitpython", "pico", "usbmodem")


def find_pico_port() -> str | None:
    """Scan connected serial ports and return the first Pi Pico device path."""
    for port_info in serial.tools.list_ports.comports():
        if port_info.vid in _PICO_VIDS:
            logger.info("Found Pico by VID 0x%04X on %s.", port_info.vid, port_info.device)
            return port_info.device
        desc = (port_info.description or "").lower()
        if any(kw in desc for kw in _PICO_DESC_KEYWORDS):
            logger.info(
                "Found Pico by description %r on %s.", port_info.description, port_info.device
            )
            return port_info.device
    return None


class _SerialWorker(QThread):
    """Background thread that reads lines from the Pi Pico serial port."""

    ecg_sample_ready = pyqtSignal(float, float)  # (value, timestamp_s)
    wall_contact_detected = pyqtSignal()
    connection_lost = pyqtSignal(str)

    def __init__(
        self,
        port: str,
        baud: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._port = port
        self._baud = baud
        self._stop_requested = False
        self._moi_states: dict[str, bool] = {}
        self._moi_seen = False
        self._last_wall_contact_event_ts: float = 0.0
        self._last_wall_contact_log_ts: float = 0.0

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        logger.debug("Serial worker thread %s started.", self.objectName())
        self._moi_states.clear()
        self._moi_seen = False
        self._last_wall_contact_event_ts = 0.0
        self._last_wall_contact_log_ts = 0.0

        import os

        if not os.path.exists(self._port):
            msg = f"Hardware not found at {self._port}."
            logger.warning(msg)
            self.connection_lost.emit(msg)
            return

        try:
            port = serial.Serial(
                self._port,
                baudrate=self._baud,
                timeout=1.0,
            )
        except serial.SerialException as exc:
            logger.error("Could not open serial port %s: %s", self._port, exc)
            self.connection_lost.emit(str(exc))
            return

        logger.info("Serial port %s opened at %d baud.", self._port, self._baud)

        _lines_received: int = 0
        _lines_matched: int = 0
        _samples_emitted: int = 0
        _WARN_INTERVAL: int = 200
        _MOI_WARN_INTERVAL: int = 500

        try:
            while not self._stop_requested:
                try:
                    raw_line = port.readline()
                except serial.SerialException as exc:
                    logger.error("Serial read error: %s", exc)
                    self.connection_lost.emit(str(exc))
                    break

                if not raw_line:
                    continue

                try:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue

                if self._detect_wall_contact(line):
                    self.wall_contact_detected.emit()

                _lines_received += 1

                if (
                    _lines_matched == 0
                    and _lines_received > 0
                    and _lines_received % _WARN_INTERVAL == 0
                ):
                    logger.warning(
                        "Pico serial: %d lines received, 0 matched Yeda pattern. "
                        "Check that the ECG channel name in the firmware matches "
                        "'Yeda<N>:(value,)'.",
                        _lines_received,
                    )

                if (
                    not self._moi_seen
                    and _lines_received > 0
                    and _lines_received % _MOI_WARN_INTERVAL == 0
                ):
                    logger.warning(
                        "Pico serial: %d lines received but no MOI wall-contact channel seen.",
                        _lines_received,
                    )

                ecg_value = _parse_yeda_value(line)
                if ecg_value is None:
                    continue

                _lines_matched += 1
                if _lines_matched == 1:
                    logger.info("Pico parser: first Yeda sample matched — ECG data flowing.")

                _samples_emitted += 1
                if _samples_emitted == 1:
                    logger.info("Pico ECG: first raw sample forwarded (value=%.6f).", ecg_value)
                self.ecg_sample_ready.emit(ecg_value, time.time())

        finally:
            try:
                port.close()
            except Exception:
                pass
            logger.info("Serial port %s closed.", self._port)

    def _detect_wall_contact(self, line: str) -> bool:
        moi_values = _parse_moi_values(line)
        if not moi_values:
            return False

        self._moi_seen = True

        rising_channels: list[str] = []
        for channel, value in moi_values.items():
            is_high = value >= 0.5
            was_high = self._moi_states.get(channel, False)
            self._moi_states[channel] = is_high
            if is_high and not was_high:
                rising_channels.append(channel)

        if rising_channels:
            now = time.time()
            if now - self._last_wall_contact_event_ts < PICO_WALL_CONTACT_MIN_EVENT_INTERVAL_SECONDS:
                return False
            self._last_wall_contact_event_ts = now

            if now - self._last_wall_contact_log_ts >= PICO_WALL_CONTACT_LOG_THROTTLE_SECONDS:
                logger.info(
                    "Pico wall contact detected on %s.",
                    ", ".join(sorted(rising_channels)),
                )
                self._last_wall_contact_log_ts = now
            else:
                logger.debug(
                    "Pico wall contact detected on %s (throttled).",
                    ", ".join(sorted(rising_channels)),
                )
            return True
        return False


def _parse_yeda_value(line: str) -> float | None:
    match = _YEDA_RE.search(line)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    # if value == 0.0:
    #     return None
    # return 1.0 / value          # <<< REMOVE OR COMMENT
    return value                  # <<< JUST RETURN RAW COUNT


def _parse_moi_values(line: str) -> dict[str, float]:
    matches = _MOI_RE.findall(line)
    if not matches:
        return {}
    values: dict[str, float] = {}
    for channel, raw_value in matches:
        try:
            values[channel] = float(raw_value)
        except ValueError:
            continue
    return values


class PicoECGSensor(BaseSensor):
    """Pi Pico ECG driver: streams raw ECG samples for neurokit2 processing."""

    raw_ecg_sample_received = pyqtSignal(float, float)  # (value, timestamp_s)
    wall_contact_detected = pyqtSignal()
    connection_status_changed = pyqtSignal(bool, str)

    def __init__(
        self,
        port: str | None = None,
        baud: int = PICO_ECG_BAUD,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if port is None:
            detected = find_pico_port()
            if detected is not None:
                self._port = detected
                logger.info("Auto-detected Pico port: %s", detected)
            else:
                self._port = PICO_ECG_PORT
                logger.warning(
                    "Pico not auto-detected; falling back to config port %s.", PICO_ECG_PORT
                )
        else:
            self._port = port
        self._baud = baud
        self._worker: _SerialWorker | None = None

    def start(self) -> None:
        if self._running:
            logger.warning("PicoECGSensor.start() called while already running.")
            return

        self._worker = _SerialWorker(self._port, self._baud, parent=None)
        self._worker.setObjectName("PicoECGSerialWorker")
        self._worker.ecg_sample_ready.connect(self._on_ecg_sample)
        self._worker.wall_contact_detected.connect(self.wall_contact_detected)
        self._worker.connection_lost.connect(self._on_connection_lost)
        self._worker.finished.connect(self._on_worker_finished)

        self._running = True
        self._worker.start()
        self.connection_status_changed.emit(True, f"Connected to {self._port}")
        logger.info("PicoECGSensor started on %s @ %d baud.", self._port, self._baud)

    def stop(self) -> None:
        if not self._running and self._worker is None:
            return

        self._running = False

        if self._worker is not None:
            try:
                self._worker.ecg_sample_ready.disconnect(self._on_ecg_sample)
                self._worker.wall_contact_detected.disconnect(self.wall_contact_detected)
                self._worker.connection_lost.disconnect(self._on_connection_lost)
            except (TypeError, RuntimeError):
                pass

            logger.debug("Stopping PicoECG worker thread...")
            self._worker.request_stop()
            if not self._worker.wait(3000):
                logger.warning("Serial worker did not exit within 3 s — terminating.")
                self._worker.terminate()
                self._worker.wait()
            self._worker = None

        self.connection_status_changed.emit(False, "Disconnected")
        logger.info("PicoECGSensor stopped.")

    @pyqtSlot(float, float)
    def _on_ecg_sample(self, value: float, timestamp: float) -> None:
        self.raw_ecg_sample_received.emit(value, timestamp)

    @pyqtSlot(str)
    def _on_connection_lost(self, error_message: str) -> None:
        self._running = False
        self.connection_status_changed.emit(False, f"Connection lost: {error_message}")
        logger.error("PicoECGSensor connection lost: %s", error_message)

    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        self._worker = None
        logger.debug("PicoECGSensor worker thread finished.")
