"""Unit tests for automatic Pi Pico USB serial port detection.

Tests cover the pure ``find_pico_port()`` function — no serial hardware
required.  All port enumeration is monkeypatched.

Run with:
    pytest tests/test_pico_port_detection.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.hardware.pico_ecg_sensor import find_pico_port


# ---------------------------------------------------------------------------
# Helpers — fake ListPortInfo objects
# ---------------------------------------------------------------------------


def _port(device: str, vid: int | None = None, pid: int | None = None, description: str = "") -> SimpleNamespace:
    """Build a minimal fake serial.tools.list_ports.ListPortInfo."""
    return SimpleNamespace(device=device, vid=vid, pid=pid, description=description)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindPicoPort:
    def test_returns_none_when_no_ports_available(self) -> None:
        with patch("serial.tools.list_ports.comports", return_value=[]):
            assert find_pico_port() is None

    def test_detects_official_pico_by_vid(self) -> None:
        """VID 0x2E8A is the Raspberry Pi Foundation vendor ID."""
        ports = [
            _port("/dev/ttyACM0", vid=0x2E8A, pid=0x0005),
        ]
        with patch("serial.tools.list_ports.comports", return_value=ports):
            assert find_pico_port() == "/dev/ttyACM0"

    def test_detects_adafruit_circuitpython_by_vid(self) -> None:
        """VID 0x239A is Adafruit's vendor ID (used by some CircuitPython builds)."""
        ports = [
            _port("/dev/cu.usbmodem1101", vid=0x239A, pid=0x8089),
        ]
        with patch("serial.tools.list_ports.comports", return_value=ports):
            assert find_pico_port() == "/dev/cu.usbmodem1101"

    def test_prefers_first_match_when_multiple_picos_present(self) -> None:
        ports = [
            _port("/dev/ttyACM0", vid=0x2E8A, pid=0x0005),
            _port("/dev/ttyACM1", vid=0x2E8A, pid=0x0005),
        ]
        with patch("serial.tools.list_ports.comports", return_value=ports):
            assert find_pico_port() == "/dev/ttyACM0"

    def test_ignores_unrecognised_devices(self) -> None:
        ports = [
            _port("/dev/ttyUSB0", vid=0x0403, pid=0x6001, description="FTDI"),
        ]
        with patch("serial.tools.list_ports.comports", return_value=ports):
            assert find_pico_port() is None

    def test_falls_back_to_description_match_when_no_vid(self) -> None:
        """Some systems report VID as None; match on description substring."""
        ports = [
            _port("/dev/cu.usbmodem101", vid=None, description="CircuitPython CDC Control"),
        ]
        with patch("serial.tools.list_ports.comports", return_value=ports):
            assert find_pico_port() == "/dev/cu.usbmodem101"

    def test_description_match_is_case_insensitive(self) -> None:
        ports = [
            _port("/dev/ttyACM0", vid=None, description="pico serial port"),
        ]
        with patch("serial.tools.list_ports.comports", return_value=ports):
            assert find_pico_port() == "/dev/ttyACM0"
