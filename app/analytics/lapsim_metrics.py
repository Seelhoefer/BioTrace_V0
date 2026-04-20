"""Metric normalization for LapSim learning curve analysis.

Converts various LapSim performance indicators into the 'error-direction'
format required by the Schmettow parametric model (where higher values 
represent worse performance, decreasing towards an asymptote).
"""

from typing import List, Tuple

import numpy as np
from app.analytics.lapsim_parser import TrialRecord
from app.utils.config import SCORE_MAX


def extract_metric_series(
    trials: List[TrialRecord],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (trial_numbers, error_values) parallel arrays for model fitting.

    Normalization logic:
    - Total Time: Used as-is (naturally decreases with skill).
    - Tissue Damage: Used as-is (naturally decreases with skill).
    - Score: Inverted. pseudo_errors = max_observed_score - current_score.

    Args:
        trials: List of trial data points from the parser.
        metric: The user-selected metric name.

    Returns:
        A tuple of (trial_number_array, error_value_array).
    """
    trial_nums = []
    raw_values = []

    # For score inversion, we find the dynamic ceiling of this specific dataset.
    score_max_local = 0.0
    if metric == "Score":
        valid_scores = [t.score for t in trials if t.score is not None]
        if valid_scores:
            score_max_local = max(valid_scores)

    for t in trials:
        val = None
        if metric == "Total Time (s)":
            val = t.total_time_s
        elif metric == "Score":
            if t.score is not None:
                val = score_max_local - t.score
        elif metric == "Tissue Damage (#)":
            val = float(t.tissue_damage) if t.tissue_damage is not None else None

        if val is not None:
            trial_nums.append(t.trial_number)
            raw_values.append(val)

    return np.array(trial_nums), np.array(raw_values)


def compute_performance_series(
    trials: List[TrialRecord],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute a composite performance error series for Schmettow model fitting.

    Combines speed (total time) and accuracy (tissue damage) into a single
    error metric scaled by SCORE_MAX. Higher values = worse performance 
    (error domain required by the Schmettow model).

    Formula:
        norm_speed  = (time  - min_time)  / (max_time  - min_time)   ∈ [0, 1]
        norm_acc    = (dmg   - min_dmg)   / (max_dmg   - min_dmg)    ∈ [0, 1]
        error       = (0.5 * norm_speed + 0.5 * norm_acc) * SCORE_MAX

    Degenerate cases (all values equal) collapse the corresponding component
    to 0 so the other dimension still drives the fit.

    Args:
        trials: Ordered list of TrialRecord objects for one participant/exercise.

    Returns:
        (trial_numbers, error_values, score_max).

    References:
        Schmettow, Chan, Groenier (2026). Parametric learning curve models
        for simulation-based surgery training.
    """
    valid: list[tuple[int, float, float]] = []
    for t in trials:
        if t.total_time_s is None:
            continue
        dmg = float(t.tissue_damage) if t.tissue_damage is not None else 0.0
        valid.append((t.trial_number, t.total_time_s, dmg))

    if not valid:
        return np.array([]), np.array([]), SCORE_MAX

    nums   = np.array([v[0] for v in valid], dtype=float)
    speeds = np.array([v[1] for v in valid], dtype=float)
    accs   = np.array([v[2] for v in valid], dtype=float)

    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    errors = (0.5 * _norm(speeds) + 0.5 * _norm(accs)) * SCORE_MAX
    return nums, errors, SCORE_MAX