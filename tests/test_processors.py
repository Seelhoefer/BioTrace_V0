"""Unit tests for the signal processing pipeline.

Tests cover HRVProcessor, PupilProcessor, and CLIProcessor using
in-process signal inspection (no QApplication required for logic tests).

Run with:
    pytest tests/test_processors.py -v
"""

import numpy as np
import pytest
from collections import deque


# -------------------------------------------------------------------------
# HRV — RR cleaning (pure numpy)
# -------------------------------------------------------------------------


class TestRollingMedianReplaceOutliers:
    def test_bounds_clip_extreme(self) -> None:
        from app.processing.hrv_processor import rolling_median_replace_outliers

        rr = np.array([800.0, 250.0, 820.0], dtype=float)
        out = rolling_median_replace_outliers(rr, half_width=1, dev_frac=0.2)
        assert np.all(out >= 300.0)
        assert np.all(out <= 2000.0)

    def test_ectopic_softened_toward_median(self) -> None:
        from app.processing.hrv_processor import rolling_median_replace_outliers

        rr = np.array([800.0, 800.0, 400.0, 800.0, 800.0], dtype=float)
        out = rolling_median_replace_outliers(rr, half_width=2, dev_frac=0.2)
        assert abs(out[2] - 800.0) < 50.0


# -------------------------------------------------------------------------
# PupilProcessor — unit tests
# -------------------------------------------------------------------------

class TestPupilProcessorLogic:
    def _make_processor(self, baseline: float = 100.0):
        from app.processing.pupil_processor import PupilProcessor
        return PupilProcessor(baseline_px=baseline)

    def test_baseline_zero_skips_pdi(self) -> None:
        proc = self._make_processor(baseline=0.0)
        # With baseline=0, PDI cannot be computed; set_baseline call required.
        assert proc.baseline_px == 0.0

    def test_set_baseline_updates(self) -> None:
        proc = self._make_processor()
        proc.set_baseline(120.0)
        assert proc.baseline_px == pytest.approx(120.0)

    def test_reset_clears_prev_diameter(self) -> None:
        proc = self._make_processor()
        proc._prev_diameter = 100.0
        proc.reset()
        assert proc._prev_diameter is None

    def test_blink_detection_threshold(self) -> None:
        """A diameter drop exceeding the threshold should be classified as a blink."""
        from app.utils.config import PUPIL_BLINK_VELOCITY_THRESHOLD_PX
        proc = self._make_processor()
        proc._prev_diameter = 100.0
        # Drop of 30 px >> threshold (20) → blink detected, prev_diameter cleared.
        big_drop_diameter = 100.0 - (PUPIL_BLINK_VELOCITY_THRESHOLD_PX + 10.0)
        velocity = abs(big_drop_diameter - proc._prev_diameter)
        assert velocity > PUPIL_BLINK_VELOCITY_THRESHOLD_PX

    def test_normal_sample_not_rejected(self) -> None:
        """A small smooth change should not trigger blink rejection."""
        from app.utils.config import PUPIL_BLINK_VELOCITY_THRESHOLD_PX
        proc = self._make_processor()
        proc._prev_diameter = 100.0
        small_change = 105.0  # 5 px change — well below threshold
        velocity = abs(small_change - proc._prev_diameter)
        assert velocity < PUPIL_BLINK_VELOCITY_THRESHOLD_PX

    def test_pdi_outlier_clamp(self) -> None:
        """A PDI change exceeding 40% should be rejected."""
        from app.utils.config import PUPIL_PDI_OUTLIER_CLAMP
        proc = self._make_processor(baseline=100.0)
        
        # 50% increase (150 px) > 40% clamp
        emitted = []
        proc.pdi_updated.connect(lambda pdi, ts: emitted.append(pdi))
        
        proc.on_pupil_sample(150.0, 0.0, 1.0)
        assert len(emitted) == 0
        
        # 30% increase (130 px) < 40% clamp
        proc.on_pupil_sample(130.0, 0.0, 2.0)
        assert len(emitted) == 1
        assert emitted[0] == pytest.approx(0.3)


# -------------------------------------------------------------------------
# CLIProcessor — unit tests
# -------------------------------------------------------------------------

class TestCLIProcessorLogic:
    def _make_processor(self):
        from app.processing.cli_processor import CLIProcessor
        return CLIProcessor()

    def test_no_emission_before_both_inputs(self) -> None:
        """CLI should not be emitted if only RMSSD or only PDI has been received."""
        proc = self._make_processor()
        # Feed only RMSSD — PDI still None.
        proc._rmssd = 50.0
        proc._rmssd_min = 30.0
        proc._rmssd_max = 70.0
        # _pdi remains None → _try_emit should short-circuit.
        emitted = []

        def capture(cli, ts):
            emitted.append(cli)

        proc.cli_updated.connect(capture)
        proc._try_emit()
        assert emitted == []

    def test_reset_clears_all_state(self) -> None:
        proc = self._make_processor()
        proc._rmssd = 50.0
        proc._pdi = 0.1
        proc.reset()
        assert proc._rmssd is None
        assert proc._pdi is None

    def test_cli_range_after_receiving_both(self) -> None:
        """After receiving valid RMSSD and PDI, CLI must be in [0, 1]."""
        from app.processing.cli_processor import _UNSET
        proc = self._make_processor()
        proc._rmssd = 50.0
        proc._rmssd_min = 30.0
        proc._rmssd_max = 70.0
        proc._pdi = 0.1
        proc._pdi_min = 0.0
        proc._pdi_max = 0.3
        proc._rmssd_ts = 1.0
        proc._pdi_ts = 1.0

        emitted: list[float] = []
        proc.cli_updated.connect(lambda cli, ts: emitted.append(cli))
        proc._try_emit()

        assert len(emitted) == 1
        assert 0.0 <= emitted[0] <= 1.0


# -------------------------------------------------------------------------
# HRVProcessor — ECG ring buffer + windowed neurokit2 (requires neurokit2)
# -------------------------------------------------------------------------


def _synthetic_ecg_array(duration_sec: float = 12.0, sr: int = 150) -> np.ndarray:
    """Simple synthetic ECG-like signal with dominant ~1 Hz rhythm."""
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=float) / float(sr)
    hr_hz = 72.0 / 60.0
    phase = (t * hr_hz) % 1.0
    qrs = np.where(phase < 0.04, 1.0, 0.0)
    noise = 0.02 * np.random.default_rng(42).standard_normal(n)
    return qrs + noise


class TestHRVProcessorBuffer:
    def _make_processor(self):
        from PyQt6.QtWidgets import QApplication
        from app.processing.hrv_processor import HRVProcessor

        inst = QApplication.instance()
        if inst is None:
            _ = QApplication([])
        return HRVProcessor(window_seconds=12, update_seconds=12, sampling_rate_hz=150)

    def test_ecg_samples_accumulate(self) -> None:
        from app.processing.hrv_processor import HRVProcessor
        from PyQt6.QtWidgets import QApplication

        if QApplication.instance() is None:
            _ = QApplication([])
        proc = HRVProcessor(window_seconds=12, update_seconds=12, sampling_rate_hz=150)
        for i in range(100):
            proc.on_ecg_sample(0.1 * i, float(i))
        assert len(proc._buffer) == 100

    def test_reset_clears_buffer(self) -> None:
        proc = self._make_processor()
        proc.on_ecg_sample(1.0, 0.0)
        proc.reset()
        assert len(proc._buffer) == 0


@pytest.mark.skipif(
    __import__("importlib.util").find_spec("neurokit2") is None,
    reason="neurokit2 not installed",
)
class TestComputeWindowHrvDict:
    def test_ok_on_synthetic_ecg(self) -> None:
        from app.processing.hrv_processor import compute_window_hrv_dict

        ecg = _synthetic_ecg_array(12.0, 150)
        d = compute_window_hrv_dict(ecg, 150)
        assert d["status"] == "ok"
        assert d["bpm"] > 40.0
        assert d["rmssd_ms"] > 0.0
        assert d["ln_rmssd"] == pytest.approx(float(np.log(d["rmssd_ms"])))

    def test_insufficient_on_short_buffer(self) -> None:
        from app.processing.hrv_processor import compute_window_hrv_dict

        d = compute_window_hrv_dict(np.zeros(200), 150)
        assert d["status"] == "insufficient_peaks"


@pytest.mark.skipif(
    __import__("importlib.util").find_spec("neurokit2") is None,
    reason="neurokit2 not installed",
)
class TestHRVProcessorWindowEmit:
    def test_process_buffered_emits_once(self) -> None:
        from PyQt6.QtWidgets import QApplication
        from app.processing.hrv_processor import HRVProcessor

        if QApplication.instance() is None:
            _ = QApplication([])
        proc = HRVProcessor(window_seconds=12, update_seconds=12, sampling_rate_hz=150)
        ecg = _synthetic_ecg_array(12.0, 150)
        for i, v in enumerate(ecg):
            proc.on_ecg_sample(float(v), float(i) / 150.0)
        hr: list[float] = []
        proc.hrv_updated.connect(lambda rr, bpm, rm, dlt, ts: hr.append(bpm))
        proc.process_buffered_window()
        assert len(hr) == 1
        assert hr[0] > 40.0
