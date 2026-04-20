"""Schmettow parametric learning curve fitting module.

Implements the 4-parameter Schmettow model for fitting and predicting
learning trajectories based on error counts:

    perf(t) = scale * (1 - leff)^(t + pexp) + maxp

Parameters:
    leff  ∈ (0,1)  Learning Efficiency — curvature / speed of improvement.
    maxp  ≥ 0      Maximum Performance — asymptotic error floor (plateau).
    pexp  ≥ 0      Previous Experience — shifts the curve along the trial axis.
    scale > 0      Magnitude of the trainable component (initial amplitude).

Reference:
    Schmettow, Chan, Groenier — Parametric learning curve models for
    simulation-based surgery training (github.com/schmettow/pub-learning-curves-surgery).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # sigmoid / inv_logit
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
    pexp: float                # ≥ 0 - Previous experience shift along trial axis.
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

def _schmettow_model(t_val, leff_raw, maxp_raw, scale_raw, pexp_raw):
    """4-parameter Schmettow model in raw (unconstrained) parameter space.

    Transformations ensure constraints:
        leff  = sigmoid(leff_raw)   → (0, 1)
        maxp  = exp(maxp_raw)       → (0, ∞)
        scale = exp(scale_raw)      → (0, ∞)
        pexp  = exp(pexp_raw)       → (0, ∞)
    """
    leff = expit(leff_raw)
    maxp = np.exp(np.clip(maxp_raw, -20, 20))
    scale = np.exp(np.clip(scale_raw, -20, 20))
    pexp = np.exp(np.clip(pexp_raw, -20, 20))
    return scale * (1 - leff)**(t_val + pexp) + maxp

def _schmettow_model_3p(t_val, leff_raw, maxp_raw, scale_raw):
    """3-parameter fallback (pexp fixed at 0)."""
    leff = expit(leff_raw)
    maxp = np.exp(np.clip(maxp_raw, -20, 20))
    scale = np.exp(np.clip(scale_raw, -20, 20))
    return scale * (1 - leff)**t_val + maxp

def fit_schmettow(
    trial_numbers: list[int] | np.ndarray,
    error_counts: list[float] | np.ndarray,
    score_max: float = SCORE_MAX,
) -> SchmettowFit | None:
    """Fit the Schmettow parametric learning curve.

    Tries the 4-parameter model first (with pexp). Falls back to the
    3-parameter model (pexp=0) when the extra parameter is not identifiable
    (e.g., too few data points or convergence failure).
    """
    t = np.array(trial_numbers, dtype=float)
    y = np.array(error_counts, dtype=float)

    valid_mask = ~np.isnan(y)
    t = t[valid_mask]
    y = y[valid_mask]

    if len(t) < LC_MIN_SESSIONS:
        return None

    # Return None if the data is entirely flat at zero
    if np.all(y == y[0]) and y[0] == 0:
        return None

    min_errors = np.min(y)
    range_errors = np.max(y) - min_errors

    # Initial guesses (raw parameter space)
    p0_3p = [0.0, np.log(max(0.1, min_errors)), np.log(max(1.0, range_errors))]
    p0_4p = p0_3p + [0.0]  # pexp_raw=0 → exp(0)=1 trial shift

    leff = maxp = scale = pexp = None
    predicted_errors = None
    used_4p = False

    # Try 4-parameter model first
    try:
        popt, _ = curve_fit(_schmettow_model, t, y, p0=p0_4p, maxfev=5000)
        leff = expit(popt[0])
        maxp = np.exp(np.clip(popt[1], -20, 20))
        scale = np.exp(np.clip(popt[2], -20, 20))
        pexp = np.exp(np.clip(popt[3], -20, 20))
        predicted_errors = _schmettow_model(t, *popt)
        used_4p = True
    except Exception:
        pass

    # Fall back to 3-parameter model
    if not used_4p:
        try:
            popt, _ = curve_fit(_schmettow_model_3p, t, y, p0=p0_3p, maxfev=2000)
            leff = expit(popt[0])
            maxp = np.exp(np.clip(popt[1], -20, 20))
            scale = np.exp(np.clip(popt[2], -20, 20))
            pexp = 0.0
            predicted_errors = _schmettow_model_3p(t, *popt)
        except Exception:
            return None

    predicted_performance = score_max - predicted_errors

    # R-squared
    ss_res = np.sum((y - predicted_errors)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return SchmettowFit(
        leff=leff,
        maxp=maxp,
        pexp=pexp,
        scale=scale,
        maxp_performance=score_max - maxp,
        r_squared=r_squared,
        predicted_errors=predicted_errors,
        predicted_performance=predicted_performance,
    )

def predict_at_trial(fit: SchmettowFit, trial: int, score_max: float = SCORE_MAX) -> float:
    """Compute predicted performance score at a given trial."""
    errors = fit.scale * (1 - fit.leff)**(trial + fit.pexp) + fit.maxp
    return score_max - errors

def mastery_percent(fit: SchmettowFit, current_performance: float) -> float:
    """Calculate mastery percentage relative to the predicted ceiling."""
    if fit.maxp_performance <= 0:
        return 0.0
    percent = (current_performance / fit.maxp_performance) * 100
    return float(max(0.0, min(100.0, percent)))
