"""Schmettow parametric learning curve fitting module.

This module implements the Schmettow model for fitting and predicting
learning trajectories based on error counts.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # sigmoid
from dataclasses import dataclass
from app.utils.config import SCORE_MAX, LC_MIN_SESSIONS

@dataclass
class SessionDataPoint:
    """Data point for a single session in the learning curve."""
    trial: int
    error_count: int
    performance_score: float

@dataclass
class SchmettowFit:
    """Results of a Schmettow model fit."""
    leff: float                # ∈ (0,1) - Learning efficiency. Higher = faster learning.
    maxp: float                # ≥ 0 - Asymptotic error floor (final plateau).
    scale: float               # > 0 - Magnitude of trainable component (initial amplitude).
    maxp_performance: float    # SCORE_MAX - maxp (display ceiling).
    r_squared: float           # Coefficient of determination (goodness of fit).
    predicted_errors: np.ndarray
    predicted_performance: np.ndarray

def get_mentor_message(mastery_pct: float) -> str:
    """Generate a context-aware assessment message based on mastery percentage."""
    if mastery_pct >= 80.0:
        return "You're approaching your performance ceiling. Excellent consistency."
    elif mastery_pct >= 40.0:
        remaining = 100.0 - mastery_pct
        return f"You're still {remaining:.0f}% from your potential. Keep grinding."
    else:
        return "Early stage. Each session matters most now — your curve is steepest here."

def fit_schmettow(
    trial_numbers: list[int] | np.ndarray,
    error_counts: list[float] | np.ndarray,
    score_max: float = SCORE_MAX,
) -> SchmettowFit | None:
    """Compute the Schmettow parametric learning curve fit using a 3-parameter model."""
    
    # Ensure inputs are numpy arrays and filter out NaN values
    t = np.array(trial_numbers, dtype=float)
    y = np.array(error_counts, dtype=float)
    
    valid_mask = ~np.isnan(y)
    t = t[valid_mask]
    y = y[valid_mask]

    if len(t) < LC_MIN_SESSIONS:
        return None

    # KISS: 3-parameter model. pexp removed for mathematical identifiability.
    def model(t_val, leff_raw, maxp_raw, scale_raw):
        leff = expit(leff_raw)
        maxp = np.exp(np.clip(maxp_raw, -20, 20))  # Safe clipping to prevent overflow
        scale = np.exp(np.clip(scale_raw, -20, 20))
        return scale * (1 - leff)**t_val + maxp

    try:
        min_errors = np.min(y)
        range_errors = np.max(y) - min_errors
        p0 = [0.0, np.log(max(0.1, min_errors)), np.log(max(1.0, range_errors))]

        popt, _ = curve_fit(model, t, y, p0=p0, maxfev=2000)

        leff = expit(popt[0])
        maxp = np.exp(np.clip(popt[1], -20, 20))
        scale = np.exp(np.clip(popt[2], -20, 20))

        # Return None if the data is entirely flat at zero
        if np.all(y == y[0]) and y[0] == 0:
            return None

        predicted_errors = model(t, *popt)
        predicted_performance = score_max - predicted_errors

        # Calculate R-squared
        ss_res = np.sum((y - predicted_errors)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return SchmettowFit(
            leff=leff,
            maxp=maxp,
            scale=scale,
            maxp_performance=score_max - maxp,
            r_squared=r_squared,
            predicted_errors=predicted_errors,
            predicted_performance=predicted_performance
        )
    except Exception:
        return None

def predict_at_trial(fit: SchmettowFit, trial: int, score_max: float = SCORE_MAX) -> float:
    """Compute predicted performance score at a given trial."""
    errors = fit.scale * (1 - fit.leff)**trial + fit.maxp
    return score_max - errors

def mastery_percent(fit: SchmettowFit, current_performance: float) -> float:
    """Calculate mastery percentage relative to the predicted ceiling."""
    if fit.maxp_performance <= 0:
        return 0.0
    percent = (current_performance / fit.maxp_performance) * 100
    return float(max(0.0, min(100.0, percent)))