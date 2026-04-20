"""NASA-TLX score calculation for BioTrace.

NASA Task Load Index (TLX) measures perceived workload across six dimensions.
This implementation supports both the weighted and unweighted (raw) variants.

References:
    Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX (Task Load
    Index): Results of empirical and theoretical research. Advances in
    Psychology, 52, 139–183.
"""

from dataclasses import dataclass, field

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NASATLXRatings:
    """Six raw NASA-TLX dimension ratings on a 0–100 scale.

    Attributes:
        mental_demand: How much mental and perceptual activity was required?
        physical_demand: How much physical activity was required?
        temporal_demand: How much time pressure was there?
        performance: How successful were you in accomplishing the task?
                     (Note: lower = better for performance in TLX scoring,
                     but the user rates on a 0–100 scale where 100 = perfect.)
        effort: How hard did you have to work to accomplish your level of
                performance?
        frustration: How insecure, discouraged, irritated, stressed, and
                     annoyed were you?
    """

    mental_demand: float = 50.0
    physical_demand: float = 50.0
    temporal_demand: float = 50.0
    performance: float = 50.0
    effort: float = 50.0
    frustration: float = 50.0

    @property
    def as_dict(self) -> dict[str, float]:
        """Return ratings as a plain dictionary."""
        return {
            "mental_demand": self.mental_demand,
            "physical_demand": self.physical_demand,
            "temporal_demand": self.temporal_demand,
            "performance": self.performance,
            "effort": self.effort,
            "frustration": self.frustration,
        }


@dataclass
class NASATLXWeights:
    """Pairwise-derived weights for each NASA-TLX dimension.

    Weights are non-negative integers that must sum to 15 (the total number
    of pairwise comparisons across 6 dimensions).  Default to equal weighting
    (each dimension wins 2.5 times on average), which approximates the
    *unweighted* Raw TLX.

    Attributes:
        mental_demand: Number of pairings won by mental demand.
        physical_demand: Number of pairings won by physical demand.
        temporal_demand: Number of pairings won by temporal demand.
        performance: Number of pairings won by performance.
        effort: Number of pairings won by effort.
        frustration: Number of pairings won by frustration.
    """

    mental_demand: int = 3
    physical_demand: int = 2
    temporal_demand: int = 3
    performance: int = 2
    effort: int = 3
    frustration: int = 2

    @property
    def as_dict(self) -> dict[str, int]:
        """Return weights as a plain dictionary."""
        return {
            "mental_demand": self.mental_demand,
            "physical_demand": self.physical_demand,
            "temporal_demand": self.temporal_demand,
            "performance": self.performance,
            "effort": self.effort,
            "frustration": self.frustration,
        }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_weighted_tlx(
    ratings: NASATLXRatings,
    weights: NASATLXWeights,
) -> float:
    """Compute the pairwise-weighted NASA-TLX overall workload score.

    Formula:
        TLX = sum(rating_i * weight_i) / 15

    The denominator is always 15 — the total number of pairwise comparisons
    for six dimensions (C(6,2) = 15).

    Args:
        ratings: Per-dimension ratings on a 0–100 scale.
        weights: Pairwise comparison win-counts per dimension (sum should = 15).

    Returns:
        Overall workload score in the range [0, 100].
        Higher scores indicate greater perceived workload.
    """
    total_weight = 15  # C(6,2) = 15 pairwise comparisons

    score = (
        ratings.mental_demand   * weights.mental_demand
        + ratings.physical_demand * weights.physical_demand
        + ratings.temporal_demand * weights.temporal_demand
        + ratings.performance     * weights.performance
        + ratings.effort          * weights.effort
        + ratings.frustration     * weights.frustration
    ) / total_weight

    logger.debug("NASA-TLX weighted score: %.2f", score)
    return float(score)


def compute_raw_tlx(ratings: NASATLXRatings) -> float:
    """Compute the unweighted Raw TLX score (simple mean of six dimensions).

    The Raw TLX is a faster, equally valid alternative to the weighted
    version when pairwise comparison data is unavailable.

    Args:
        ratings: Per-dimension ratings on a 0–100 scale.

    Returns:
        Mean of all six ratings in the range [0, 100].
    """
    values = list(ratings.as_dict.values())
    score = sum(values) / len(values)
    logger.debug("NASA-TLX raw score: %.2f", score)
    return float(score)
