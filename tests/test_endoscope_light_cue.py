"""Unit tests for endoscope light serial hex parsing."""

import pytest

from app.hardware.endoscope_light_cue import parse_hex_command


def test_parse_hex_command_empty() -> None:
    assert parse_hex_command("") is None
    assert parse_hex_command("   ") is None


def test_parse_hex_command_bytes() -> None:
    assert parse_hex_command("FF 01 64") == bytes([0xFF, 0x01, 0x64])
    assert parse_hex_command("0x0a,0x0b") == bytes([0x0A, 0x0B])


def test_parse_hex_command_single() -> None:
    assert parse_hex_command("00") == b"\x00"
