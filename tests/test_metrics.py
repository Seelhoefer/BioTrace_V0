"""Unit tests for core metric formulas (RMSSD, PDI, CLI).

Run with:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from app.core.metrics import (
    compute_rmssd, 
    compute_pdi, 
    compute_cli, 
    normalize, 
    average_pupil_diameter
)


class TestRMSSD:
    def test_known_value(self) -> None:
        """RMSSD of [800, 810, 790, 820] should match manual calculation."""
        rr = np.array([800.0, 810.0, 790.0, 820.0])
        diffs = np.diff(rr)   # [10, -20, 30]
        expected = float(np.sqrt(np.mean(diffs ** 2)))
        assert compute_rmssd(rr) == pytest.approx(expected, rel=1e-6)

    def test_returns_zero_for_single_sample(self) -> None:
        assert compute_rmssd(np.array([800.0])) == 0.0

    def test_returns_zero_for_empty(self) -> None:
        assert compute_rmssd(np.array([])) == 0.0

    def test_always_non_negative(self) -> None:
        rr = np.random.uniform(600, 1000, 50)
        assert compute_rmssd(rr) >= 0.0


class TestPDI:
    def test_zero_when_equal_to_baseline(self) -> None:
        assert compute_pdi(100.0, 100.0) == pytest.approx(0.0)

    def test_positive_when_dilated(self) -> None:
        assert compute_pdi(120.0, 100.0) == pytest.approx(0.2)

    def test_negative_when_constricted(self) -> None:
        assert compute_pdi(80.0, 100.0) == pytest.approx(-0.2)

    def test_zero_baseline_guard(self) -> None:
        assert compute_pdi(100.0, 0.0) == 0.0

    def test_negative_baseline_guard(self) -> None:
        assert compute_pdi(100.0, -1.0) == 0.0


class TestAveragePupil:
    def test_binocular_mean(self) -> None:
        assert average_pupil_diameter(100.0, 120.0) == pytest.approx(110.0)

    def test_monocular_left_only(self) -> None:
        assert average_pupil_diameter(100.0, 0.0) == pytest.approx(100.0)
        assert average_pupil_diameter(100.0, None) == pytest.approx(100.0)

    def test_monocular_right_only(self) -> None:
        assert average_pupil_diameter(0.0, 120.0) == pytest.approx(120.0)
        assert average_pupil_diameter(None, 120.0) == pytest.approx(120.0)

    def test_all_unavailable_returns_none(self) -> None:
        assert average_pupil_diameter(0.0, 0.0) is None
        assert average_pupil_diameter(None, None) is None


class TestNormalize:
    def test_midpoint(self) -> None:
        assert normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)

    def test_min_maps_to_zero(self) -> None:
        assert normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_max_maps_to_one(self) -> None:
        assert normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_clamps_above_max(self) -> None:
        assert normalize(15.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_clamps_below_min(self) -> None:
        assert normalize(-5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_degenerate_range_returns_zero(self) -> None:
        assert normalize(5.0, 5.0, 5.0) == 0.0


class TestCLI:
    def test_equal_weights_midpoint(self) -> None:
        """With symmetric inputs, CLI should be around 0.5."""
        cli = compute_cli(
            rmssd=50.0, pdi=0.1,
            rmssd_min=30.0, rmssd_max=70.0,
            pdi_min=-0.1, pdi_max=0.3,
        )
        assert 0.0 <= cli <= 1.0

    def test_low_rmssd_high_pdi_gives_high_cli(self) -> None:
        """Low RMSSD (high stress) + high PDI (high load) → high CLI."""
        cli = compute_cli(
            rmssd=30.0, pdi=0.3,
            rmssd_min=30.0, rmssd_max=70.0,
            pdi_min=0.0, pdi_max=0.3,
        )
        assert cli == pytest.approx(1.0, abs=0.01)

    def test_zero_rmssd_guard(self) -> None:
        """RMSSD of 0 should not raise ZeroDivisionError."""
        cli = compute_cli(
            rmssd=0.0, pdi=0.0,
            rmssd_min=0.0, rmssd_max=50.0,
            pdi_min=0.0, pdi_max=0.3,
        )
        assert 0.0 <= cli <= 1.0

    def test_output_always_in_unit_interval(self) -> None:
        """CLI must always be in [0, 1]."""
        for _ in range(100):
            rmssd = float(np.random.uniform(10, 100))
            pdi = float(np.random.uniform(-0.5, 0.5))
            cli = compute_cli(
                rmssd=rmssd, pdi=pdi,
                rmssd_min=10.0, rmssd_max=100.0,
                pdi_min=-0.5, pdi_max=0.5,
            )
            assert 0.0 <= cli <= 1.0, f"CLI={cli} out of range"
