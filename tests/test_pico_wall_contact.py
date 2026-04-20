"""Unit tests for wall-contact parsing on the Pico serial stream."""

from app.hardware.pico_ecg_sensor import _SerialWorker, _parse_moi_values


def test_parse_moi_values_extracts_all_channels() -> None:
    line = "Yeda0:(0.00234,) MOI0:(0.0,) MOI1:(1.0,)"
    values = _parse_moi_values(line)
    assert values == {"MOI0": 0.0, "MOI1": 1.0}


def test_serial_worker_detects_rising_edge_only_once() -> None:
    worker = _SerialWorker("/dev/null", 115200)

    assert worker._detect_wall_contact("Yeda0:(20.0,) MOI0:(0.0,)") is False
    assert worker._detect_wall_contact("Yeda0:(20.0,) MOI0:(1.0,)") is True
    assert worker._detect_wall_contact("Yeda0:(20.0,) MOI0:(1.0,)") is False
    assert worker._detect_wall_contact("Yeda0:(20.0,) MOI0:(0.0,)") is False
    assert worker._detect_wall_contact("Yeda0:(20.0,) MOI0:(1.0,)") is True


def test_serial_worker_counts_any_moi_channel_rise() -> None:
    worker = _SerialWorker("/dev/null", 115200)

    assert worker._detect_wall_contact("MOI0:(0.0,) MOI1:(0.0,)") is False
    assert worker._detect_wall_contact("MOI0:(0.0,) MOI1:(1.0,)") is True
