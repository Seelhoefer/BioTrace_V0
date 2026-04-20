"""Unit tests for metric extraction and normalization.

Verifies that raw simulator values are correctly transformed for the 
Schmettow 'error-domain' fitting engine.
"""

import numpy as np
import pytest
from app.analytics.lapsim_metrics import extract_metric_series
from app.analytics.lapsim_parser import TrialRecord

@pytest.fixture
def dummy_trials():
    return [
        TrialRecord(1, "2024-01-01 10:00", 60.0, 80.0, 2, "Pass"),
        TrialRecord(2, "2024-01-01 10:05", 50.0, 90.0, 1, "Pass"),
        TrialRecord(3, "2024-01-01 10:10", 45.0, 95.0, 0, "Pass"),
    ]

class TestLapsimMetrics:
    def test_extract_total_time(self, dummy_trials):
        t, v = extract_metric_series(dummy_trials, "Total Time (s)")
        assert np.array_equal(t, [1, 2, 3])
        # Time is already in error domain (lower is better)
        assert np.array_equal(v, [60.0, 50.0, 45.0])

    def test_extract_score_inverted(self, dummy_trials):
        t, v = extract_metric_series(dummy_trials, "Score")
        assert np.array_equal(t, [1, 2, 3])
        # max_score = 95.0
        # v1 = 95 - 80 = 15
        # v2 = 95 - 90 = 5
        # v3 = 95 - 95 = 0
        assert np.array_equal(v, [15.0, 5.0, 0.0])

    def test_extract_tissue_damage(self, dummy_trials):
        t, v = extract_metric_series(dummy_trials, "Tissue Damage (#)")
        assert np.array_equal(t, [1, 2, 3])
        assert np.array_equal(v, [2.0, 1.0, 0.0])

    def test_extract_drops_nones(self):
        trials = [
            TrialRecord(1, "...", 60.0, None, 2, "Pass"),
            TrialRecord(2, "...", None, 90.0, 1, "Pass"),
        ]
        
        t_time, v_time = extract_metric_series(trials, "Total Time (s)")
        assert len(t_time) == 1
        assert t_time[0] == 1
        
        t_score, v_score = extract_metric_series(trials, "Score")
        assert len(t_score) == 1
        assert t_score[0] == 2
