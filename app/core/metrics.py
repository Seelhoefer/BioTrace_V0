"""Core metric formulas for BioTrace — RMSSD, PDI, and CLI.

All computations are pure functions with no side effects.
They have no dependency on Qt, the UI, or the database.

References:
    Task Force of ESC and NASPE (1996). Heart rate variability: standards
    of measurement, physiological interpretation, and clinical use.
    Circulation, 93(5), 1043–1065.
"""

import numpy as np

from app.utils.config import (
    CLI_WEIGHT_RMSSD,
    CLI_WEIGHT_PDI,
    RMSSD_MIN_SAMPLES,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RMSSD — Heart Rate Variability / Stress Proxy
# ---------------------------------------------------------------------------


def compute_rmssd(rr_intervals: np.ndarray) -> float:
    """Compute the Root Mean Square of Successive Differences (RMSSD).

    RMSSD is a time-domain HRV metric reflecting parasympathetic nervous
    system activity. Higher values indicate lower physiological stress.

    Formula:
        RMSSD = sqrt( mean( (RR[i+1] - RR[i])^2 ) )

    Args:
        rr_intervals: Array of successive RR intervals in milliseconds.
                      Minimum 2 values required.

    Returns:
        RMSSD value in milliseconds. Returns 0.0 if fewer than 2 intervals
        are provided.

    References:
        Task Force of ESC and NASPE (1996). Heart rate variability: standards
        of measurement. Circulation, 93(5), 1043-1065.
    """
    if len(rr_intervals) < RMSSD_MIN_SAMPLES:
        return 0.0

    successive_diffs = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
    return rmssd


# ---------------------------------------------------------------------------
# PDI — Pupil Dilation Index / Cognitive Load Proxy
# ---------------------------------------------------------------------------


def compute_pdi(current_diameter_px: float, baseline_diameter_px: float) -> float:
    """Compute the Pupil Dilation Index (PDI).

    PDI expresses the relative change in pupil diameter from the resting
    baseline established during calibration. Larger positive values indicate
    greater task-induced cognitive load.

    Formula:
        PDI = (current_diameter - baseline_diameter) / baseline_diameter

    Args:
        current_diameter_px: Current pupil diameter in pixels (camera units).
        baseline_diameter_px: Resting baseline diameter in pixels,
                              recorded during calibration.

    Returns:
        PDI as a dimensionless ratio. Returns 0.0 if the baseline is zero
        or negative (guard against division by zero).
    """
    if baseline_diameter_px <= 0.0:
        logger.warning("PDI computation skipped: baseline_diameter is %.3f", baseline_diameter_px)
        return 0.0

    pdi = (current_diameter_px - baseline_diameter_px) / baseline_diameter_px
    return float(pdi)


def average_pupil_diameter(left: float | None, right: float | None) -> float | None:
    """Return the mean of left and right pupil diameters, ignoring None or 0.0 values.

    This handles monocular input where one eye may be reported as 0.0 or None.

    Args:
        left: Left eye pupil diameter in pixels, or ``None``/``0.0`` if unavailable.
        right: Right eye pupil diameter in pixels, or ``None``/``0.0`` if unavailable.

    Returns:
        Mean diameter in pixels, or ``None`` if both values are unavailable.
    """
    values = [v for v in (left, right) if v is not None and v > 0.0]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def normalize(value: float, minimum: float, maximum: float) -> float:
    """Linearly normalize a value to the [0, 1] range.

    Args:
        value: The raw numeric value to normalize.
        minimum: The minimum of the observed range.
        maximum: The maximum of the observed range.

    Returns:
        Value scaled to [0.0, 1.0]. Returns 0.0 if ``minimum == maximum``
        (degenerate range guard).
    """
    if maximum == minimum:
        return 0.0
    normalized = (value - minimum) / (maximum - minimum)
    return float(np.clip(normalized, 0.0, 1.0))


# ---------------------------------------------------------------------------
# CLI — Cognitive Load Index
# ---------------------------------------------------------------------------


def compute_cli(
    rmssd: float,
    pdi: float,
    rmssd_min: float,
    rmssd_max: float,
    pdi_min: float,
    pdi_max: float,
    w1: float = CLI_WEIGHT_RMSSD,
    w2: float = CLI_WEIGHT_PDI,
) -> float:
    """Compute the Cognitive Load Index (CLI).

    CLI combines normalized stress (inverted RMSSD) and normalized cognitive
    load (PDI) into a single composite score in the [0, 1] range.

    Formula:
        CLI = w1 * norm(1 / RMSSD) + w2 * norm(PDI)

    Both inputs are normalized against the session-wide min/max so that the
    CLI adapts to the individual's physiological range.

    Args:
        rmssd: Current RMSSD value in milliseconds.
        pdi: Current Pupil Dilation Index (dimensionless ratio).
        rmssd_min: Session minimum RMSSD observed so far.
        rmssd_max: Session maximum RMSSD observed so far.
        pdi_min: Session minimum PDI observed so far.
        pdi_max: Session maximum PDI observed so far.
        w1: Weight for the RMSSD component. Defaults to ``CLI_WEIGHT_RMSSD``
            from ``config.py``.
        w2: Weight for the PDI component. Defaults to ``CLI_WEIGHT_PDI``
            from ``config.py``.

    Returns:
        CLI in the range [0.0, 1.0]. 0.0 = no cognitive load,
        1.0 = maximum cognitive load.

    Notes:
        Weights ``w1`` and ``w2`` should sum to 1.0. This is not enforced
        programmatically to allow experimental asymmetric weighting.
    """
    # High RMSSD → low stress → low CLI: we invert RMSSD before normalizing.
    inv_rmssd = (1.0 / rmssd) if rmssd > 0.0 else 0.0
    inv_rmssd_min = (1.0 / rmssd_max) if rmssd_max > 0.0 else 0.0
    inv_rmssd_max = (1.0 / rmssd_min) if rmssd_min > 0.0 else 0.0

    norm_stress = normalize(inv_rmssd, inv_rmssd_min, inv_rmssd_max)
    norm_load = normalize(pdi, pdi_min, pdi_max)

    cli = w1 * norm_stress + w2 * norm_load
    return float(np.clip(cli, 0.0, 1.0))
