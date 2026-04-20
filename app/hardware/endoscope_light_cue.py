"""Optional serial commands for endoscope brightness during calibration.

The physical brightness wheel on many scopes is not computer-controlled.
When ``USE_ENDOSCOPE_LIGHT_SERIAL`` is True, this module sends short byte
sequences (configured as hex strings) to a USB-UART bridge that you wire
to whatever accepts commands (motor on the pot, DMX gateway, etc.).
"""

from __future__ import annotations

from app.utils.config import (
    ENDOSCOPE_LIGHT_BAUD,
    ENDOSCOPE_LIGHT_BRIGHT_HEX,
    ENDOSCOPE_LIGHT_PORT,
    ENDOSCOPE_LIGHT_RESTORE_HEX,
    USE_ENDOSCOPE_LIGHT_SERIAL,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_hex_command(s: str) -> bytes | None:
    """Parse a space- or comma-separated list of hex bytes into raw bytes."""
    if not s or not str(s).strip():
        return None
    parts = str(s).replace(",", " ").replace("0x", " ").split()
    out = bytearray()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(int(p, 16))
    return bytes(out) if out else None


def _write_port(port: str, baud: int, payload: bytes) -> None:
    try:
        import serial  # type: ignore
    except ImportError:
        logger.warning("pyserial not available — endoscope light serial cue skipped.")
        return
    try:
        with serial.Serial(port, baudrate=baud, timeout=0.5) as ser:
            ser.write(payload)
            ser.flush()
        logger.info("Endoscope light serial: wrote %d byte(s) to %s", len(payload), port)
    except OSError as exc:
        logger.error("Endoscope light serial write failed (%s): %s", port, exc)


def send_bright_command() -> None:
    """Send the configured “brightness maximum” command, if enabled."""
    if not USE_ENDOSCOPE_LIGHT_SERIAL:
        return
    if not ENDOSCOPE_LIGHT_PORT.strip():
        logger.debug("Endoscope light serial disabled: ENDOSCOPE_LIGHT_PORT is empty.")
        return
    payload = parse_hex_command(ENDOSCOPE_LIGHT_BRIGHT_HEX)
    if not payload:
        logger.debug("Endoscope light bright command empty (ENDOSCOPE_LIGHT_BRIGHT_HEX).")
        return
    _write_port(ENDOSCOPE_LIGHT_PORT.strip(), int(ENDOSCOPE_LIGHT_BAUD), payload)


def send_restore_command() -> None:
    """Send the configured restore / working-light command, if any."""
    if not USE_ENDOSCOPE_LIGHT_SERIAL:
        return
    if not ENDOSCOPE_LIGHT_PORT.strip():
        return
    payload = parse_hex_command(ENDOSCOPE_LIGHT_RESTORE_HEX)
    if not payload:
        return
    _write_port(ENDOSCOPE_LIGHT_PORT.strip(), int(ENDOSCOPE_LIGHT_BAUD), payload)
