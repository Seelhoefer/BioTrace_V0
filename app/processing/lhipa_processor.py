"""LHIPA (Low/High Index of Pupillary Activity) cognitive-load processor.

Implements the Duchowski et al. (2020) CHI algorithm with pupil preprocessing
from Kret & Sjak-Shie (2019), adapted to pixel-unit diameter streams from the
BioTrace eye tracker.

Algorithm summary
-----------------
1. **Preprocessing** — blink / dropout rejection (out-of-range and spike
   filter), followed by linear interpolation of gaps.
2. **LHIPA** — Symlet-16 wavelet decomposition; count thresholded coefficients
   at the lowest (LL) and highest (HH) frequency detail bands per second; take
   the log₂ ratio.  Higher cognitive load → more synchronised low-frequency
   pupil oscillation → *lower* LHIPA value.
3. **Calibration** — resting-period LHIPA distribution used to compute a
   personal z-score.
4. **Streaming** — 10-second sliding window, 1-second update cadence,
   consecutive-event hysteresis for level firing.

Public API
----------
- :func:`preprocess_pupil` — artefact rejection + interpolation.
- :func:`compute_lhipa` — wavelet-based LHIPA from a clean pupil window.
- :class:`PupilCalibration` — baseline statistics (pixel-unit).
- :func:`calibrate_from_baseline` — derive ``PupilCalibration`` from a resting
  pupil recording (no dark/bright phases needed).
- :class:`CognitiveLoadStreamer` — sliding-window streaming engine.
- :class:`LHIPAProcessor` — Qt ``QObject`` wrapper for the full pipeline.

Usage::

    # Wiring (done inside SessionManager):
    eye_tracker.raw_pupil_received.connect(lhipa_proc.on_pupil_sample)
    lhipa_proc.cognitive_load_updated.connect(session_manager.cognitive_load_updated)

    # Calibration phase:
    lhipa_proc.start_baseline_collection()
    # … eye tracker streams pupil data for ~60 s …
    lhipa_proc.finish_baseline_collection()

    # Session phase:
    lhipa_proc.reset()          # clears buffer, keeps calibration
    # … streaming fires automatically every 1 s …
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from scipy.interpolate import interp1d

from app.utils.config import (
    LHIPA_HIGH_CONSEC_COUNT,
    LHIPA_MOD_CONSEC_COUNT,
    LHIPA_UPDATE_SEC,
    LHIPA_WINDOW_SEC,
    PUPIL_MAX_DELTA_MM_PER_SEC,
    PUPIL_MAX_DELTA_PX,
    PUPIL_SAMPLING_RATE_HZ,
    PUPIL_VALID_MAX_MM,
    PUPIL_VALID_MAX_PX,
    PUPIL_VALID_MIN_MM,
    PUPIL_VALID_MIN_PX,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pywt  # type: ignore[import]
    _PWT_AVAILABLE = True
except Exception:  # noqa: BLE001
    pywt = None  # type: ignore[assignment]
    _PWT_AVAILABLE = False
    logger.warning(
        "PyWavelets not installed — LHIPAProcessor will be inactive. "
        "Install with: pip install PyWavelets"
    )

_WAVELET = "sym16"


# ---------------------------------------------------------------------------
# Building block 1: pupil preprocessing (Kret & Sjak-Shie, 2019)
# ---------------------------------------------------------------------------

def preprocess_pupil(
    pupil: np.ndarray,
    sampling_rate: float,
    *,
    in_mm: bool = False,
    valid_range_px: tuple[float, float] = (PUPIL_VALID_MIN_PX, PUPIL_VALID_MAX_PX),
    max_delta_px: float = PUPIL_MAX_DELTA_PX,
) -> np.ndarray:
    """Reject blink / dropout artefacts and interpolate gaps.

    Implements the Kret & Sjak-Shie (2019) preprocessing pipeline.
    When *in_mm* is ``True`` the array contains millimetre values and the
    research-specified valid range (1.5 – 9.0 mm) with a physiological spike
    filter (≤ 10 mm/s) is applied.  When *in_mm* is ``False`` (uncalibrated)
    the pixel-unit fallback thresholds are used instead.

    Steps:
    1. Values outside the valid range → NaN (blinks, sensor dropout).
    2. Zero values → NaN.
    3. Spike filter: physiologically impossible inter-sample jumps → NaN.
       Physiological limit: ≤ 10 mm/s ≈ ``10 / sampling_rate`` mm per sample.
    4. Linear interpolation of all NaN gaps.

    Args:
        pupil: Pupil diameter array (mm when *in_mm=True*, px otherwise).
        sampling_rate: Sensor sample rate in Hz.
        in_mm: ``True`` when values have been converted to millimetres via the
            iris px→mm scale.  Enables the research-specified valid range.
        valid_range_px: ``(min_px, max_px)`` fallback plausibility bounds
            (only used when *in_mm=False*).
        max_delta_px: Maximum allowed diameter change per sample in pixels
            (only used when *in_mm=False*).

    Returns:
        Cleaned diameter array of the same length.  Where fewer than two
        valid samples exist the array is returned as-is (all NaN).
    """
    p = pupil.astype(float).copy()
    p[p == 0.0] = np.nan

    if in_mm:
        # Research spec §2.7 (Kret & Sjak-Shie 2019): valid range 1.5 – 9.0 mm.
        p[(p < PUPIL_VALID_MIN_MM) | (p > PUPIL_VALID_MAX_MM)] = np.nan
        # Spike filter: physiological limit ≈ 10 mm/s.
        max_delta = PUPIL_MAX_DELTA_MM_PER_SEC / sampling_rate
    else:
        # Pixel-unit fallback when iris calibration is not available.
        p[(p < valid_range_px[0]) | (p > valid_range_px[1])] = np.nan
        max_delta = max_delta_px

    # Spike filter: physiologically impossible single-sample jumps.
    diffs = np.abs(np.diff(p, prepend=np.where(np.isnan(p[0]), 0.0, p[0])))
    p[diffs > max_delta] = np.nan

    # Linear interpolation of NaN gaps.
    valid = ~np.isnan(p)
    if valid.sum() < 2:
        return p
    idx = np.arange(len(p), dtype=float)
    f = interp1d(
        idx[valid],
        p[valid],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return f(idx)


# ---------------------------------------------------------------------------
# Building block 2: LHIPA (Duchowski et al., 2020)
# ---------------------------------------------------------------------------

def compute_lhipa(pupil: np.ndarray, sampling_rate: float) -> float:
    """Compute the Low/High Index of Pupillary Activity (LHIPA).

    Uses Symlet-16 wavelet decomposition.  Applies the universal threshold
    (Donoho & Johnstone) with hard thresholding to the lowest-frequency (LL)
    and highest-frequency (HH) detail bands, then returns::

        LHIPA = log₂(N_LL / N_HH)

    where N_LL and N_HH are the counts of surviving (above-threshold)
    coefficients per second.  **Higher cognitive load → lower LHIPA.**

    Args:
        pupil: Pre-processed pupil diameter array (pixels).
        sampling_rate: Sensor sample rate in Hz.

    Returns:
        LHIPA value, or ``float("nan")`` if the signal is too short, the
        decomposition yields fewer than two levels, or ``N_HH == 0``.
    """
    if not _PWT_AVAILABLE:
        return float("nan")

    if len(pupil) < 64:
        return float("nan")

    max_level = pywt.dwt_max_level(len(pupil), _WAVELET)
    if max_level < 2:
        logger.debug(
            "compute_lhipa: max_level=%d < 2 for %d samples; returning nan.",
            max_level,
            len(pupil),
        )
        return float("nan")

    coeffs = pywt.wavedec(pupil, _WAVELET, level=max_level)

    # coeffs = [cA_n, cD_n, cD_{n-1}, …, cD_1]
    # LL = lowest-frequency detail band (index 1 = cD_n).
    # HH = highest-frequency detail band (last element = cD_1).
    ll = coeffs[1]
    hh = coeffs[-1]

    # Universal threshold (Donoho & Johnstone), hard mode.
    def _threshold(band: np.ndarray) -> np.ndarray:
        sigma = np.std(band)
        thresh = sigma * np.sqrt(2.0 * np.log(max(len(band), 1)))
        return pywt.threshold(band, thresh, mode="hard")

    ll_thresh = _threshold(ll)
    hh_thresh = _threshold(hh)

    duration_sec = len(pupil) / sampling_rate
    count_ll = float(np.sum(np.abs(ll_thresh) > 0)) / duration_sec
    count_hh = float(np.sum(np.abs(hh_thresh) > 0)) / duration_sec

    if count_hh == 0.0:
        return float("nan")
    if count_ll == 0.0:
        return float("nan")

    return float(np.log2(count_ll / count_hh))


# ---------------------------------------------------------------------------
# Building block 3: calibration
# ---------------------------------------------------------------------------

@dataclass
class PupilCalibration:
    """Pupil calibration parameters (pixel-unit, personalised per session).

    Attributes:
        pupil_min_px: 5th-percentile resting diameter (proxy for constricted).
        pupil_max_px: 95th-percentile resting diameter (proxy for dilated).
        pupil_baseline_working_px: Median resting diameter under working light.
        lhipa_baseline_mean: Mean LHIPA across 10-s resting segments.
        lhipa_baseline_sd: Std dev of LHIPA across resting segments.
    """

    pupil_min_px: float
    pupil_max_px: float
    pupil_baseline_working_px: float
    lhipa_baseline_mean: float
    lhipa_baseline_sd: float

    def normalize(self, pupil_raw_px: float) -> float:
        """Normalise a raw pixel diameter to [0, 1].

        Args:
            pupil_raw_px: Raw diameter in pixels.

        Returns:
            Clamped normalised value in [0, 1].
        """
        span = self.pupil_max_px - self.pupil_min_px
        if span <= 0.0:
            return 0.0
        return float(np.clip(
            (pupil_raw_px - self.pupil_min_px) / span, 0.0, 1.0
        ))


def calibrate_from_baseline(
    baseline_pupil: np.ndarray,
    sampling_rate: float,
    window_sec: int = LHIPA_WINDOW_SEC,
    *,
    in_mm: bool = False,
) -> PupilCalibration | None:
    """Derive a ``PupilCalibration`` from a single resting-period recording.

    Simplified calibration: no separate dark/bright phases are needed.  The
    function computes robust percentile ranges from the cleaned signal and
    LHIPA statistics from overlapping 10-second windows (1-second stride).

    Args:
        baseline_pupil: Pupil diameter time series recorded during a quiet
            resting period (≥ 30 s recommended; 60 s is ideal).
        sampling_rate: Sensor rate in Hz.
        window_sec: Analysis window size in seconds (default 10 s).
        in_mm: ``True`` when values are in millimetres; applies the research-
            specified valid range (1.5 – 9.0 mm) and spike filter (≤ 10 mm/s).

    Returns:
        A :class:`PupilCalibration` instance, or ``None`` if the recording
        is too short or yields fewer than two valid LHIPA segments.
    """
    clean = preprocess_pupil(baseline_pupil, sampling_rate, in_mm=in_mm)

    valid = clean[~np.isnan(clean)]
    if len(valid) < 2:
        logger.warning("calibrate_from_baseline: insufficient valid pupil data.")
        return None

    pupil_min = float(np.percentile(valid, 5))
    pupil_max = float(np.percentile(valid, 95))
    pupil_baseline = float(np.median(valid))

    # Compute LHIPA in overlapping windows (10-s window, 1-s stride).
    window_samples = int(window_sec * sampling_rate)
    step_samples = max(1, int(1.0 * sampling_rate))

    lhipa_values: list[float] = []
    for start in range(0, len(clean) - window_samples, step_samples):
        val = compute_lhipa(clean[start : start + window_samples], sampling_rate)
        if not np.isnan(val):
            lhipa_values.append(val)

    if len(lhipa_values) < 2:
        logger.warning(
            "calibrate_from_baseline: only %d valid LHIPA segments (need ≥ 2). "
            "Extend calibration to ≥ 30 s for reliable z-scores.",
            len(lhipa_values),
        )
        return None

    logger.info(
        "LHIPA calibration: %d segments, mean=%.3f sd=%.3f (pupil %.1f–%.1f px, baseline=%.1f px).",
        len(lhipa_values),
        np.mean(lhipa_values),
        np.std(lhipa_values, ddof=1),
        pupil_min,
        pupil_max,
        pupil_baseline,
    )

    return PupilCalibration(
        pupil_min_px=pupil_min,
        pupil_max_px=pupil_max,
        pupil_baseline_working_px=pupil_baseline,
        lhipa_baseline_mean=float(np.mean(lhipa_values)),
        lhipa_baseline_sd=float(np.std(lhipa_values, ddof=1)),
    )


# ---------------------------------------------------------------------------
# Building block 4: real-time streaming
# ---------------------------------------------------------------------------

class CognitiveLoadStreamer:
    """10-second sliding window with 1-second cognitive-load updates.

    Applies :func:`preprocess_pupil` and :func:`compute_lhipa` on each
    update tick, then computes a personal z-score and fires a consecutive-
    event hysteresis to determine the current load level.

    Cognitive load interpretation:
        LHIPA *decreases* under higher cognitive load, so negative z-scores
        indicate elevated load:

        - ``"high"``     z < -1.5 for ≥ ``LHIPA_HIGH_CONSEC_COUNT`` consecutive ticks
        - ``"moderate"`` z < -1.0 for ≥ ``LHIPA_MOD_CONSEC_COUNT`` consecutive ticks
        - ``"low"``      otherwise

    Args:
        calibration: Personal baseline from :func:`calibrate_from_baseline`.
        sampling_rate: Sensor sample rate in Hz.
        window_sec: Sliding window duration.
        update_sec: Minimum seconds between output emissions.
        in_mm: Whether samples are in millimetres (True) or pixels (False).
    """

    def __init__(
        self,
        calibration: PupilCalibration,
        sampling_rate: float = PUPIL_SAMPLING_RATE_HZ,
        window_sec: int = LHIPA_WINDOW_SEC,
        update_sec: int = LHIPA_UPDATE_SEC,
        in_mm: bool = False,
    ) -> None:
        self.cal = calibration
        self.sr = sampling_rate
        self.in_mm = in_mm
        self.buffer: deque[float] = deque(maxlen=int(window_sec * sampling_rate))
        self.update_every: int = max(1, int(update_sec * sampling_rate))
        self.samples_since_update: int = 0
        self._consec_high: int = 0
        self._consec_moderate: int = 0

    def push(self, pupil_samples: np.ndarray) -> dict | None:
        """Add new raw pupil samples and return metrics when an update is due.

        No output is produced until the 10-second ring buffer is full **and**
        at least ``update_every`` new samples have arrived.

        Args:
            pupil_samples: Raw pupil diameters in pixels (any batch size).

        Returns:
            On success::

                {
                    "status": "ok",
                    "lhipa": float,
                    "z_score": float,
                    "cognitive_load_level": "low" | "moderate" | "high",
                    "pupil_normalized": float,   # [0, 1]
                }

            ``{"status": "invalid_window"}`` if LHIPA returns NaN.
            ``None`` if the update interval has not elapsed.
        """
        self.buffer.extend(pupil_samples)
        self.samples_since_update += len(pupil_samples)

        if (
            self.samples_since_update < self.update_every
            or len(self.buffer) < self.buffer.maxlen
        ):
            return None

        self.samples_since_update = 0
        window = preprocess_pupil(np.array(self.buffer, dtype=float), self.sr, in_mm=self.in_mm)
        lhipa = compute_lhipa(window, self.sr)

        if np.isnan(lhipa):
            return {"status": "invalid_window"}

        if self.cal.lhipa_baseline_sd <= 0.0:
            return {"status": "no_baseline_variance"}

        z = (lhipa - self.cal.lhipa_baseline_mean) / self.cal.lhipa_baseline_sd

        # Consecutive-event hysteresis (prevents spurious level flips).
        if z < -1.5:
            self._consec_high += 1
            self._consec_moderate = 0
        elif z < -1.0:
            self._consec_moderate += 1
            self._consec_high = 0
        else:
            self._consec_high = 0
            self._consec_moderate = 0

        if self._consec_high >= LHIPA_HIGH_CONSEC_COUNT:
            level = "high"
        elif self._consec_moderate >= LHIPA_MOD_CONSEC_COUNT:
            level = "moderate"
        else:
            level = "low"

        pupil_norm = self.cal.normalize(float(np.nanmedian(window)))

        return {
            "status": "ok",
            "lhipa": lhipa,
            "z_score": z,
            "cognitive_load_level": level,
            "pupil_normalized": pupil_norm,
        }

    def reset_buffer(self) -> None:
        """Clear the sample buffer without touching calibration."""
        self.buffer.clear()
        self.samples_since_update = 0
        self._consec_high = 0
        self._consec_moderate = 0


# ---------------------------------------------------------------------------
# Qt integration
# ---------------------------------------------------------------------------

class LHIPAProcessor(QObject):
    """Qt-integrated wrapper around :class:`CognitiveLoadStreamer`.

    Connects to ``EyeTrackerSensor.raw_pupil_received`` and emits
    ``cognitive_load_updated`` every second once the 10-second window fills
    and a calibration baseline has been established.

    Lifecycle::

        # Calibration phase:
        lhipa_proc.start_baseline_collection()
        # … eye tracker streams pupil data …
        lhipa_proc.finish_baseline_collection()  # computes calibration

        # Session phase:
        lhipa_proc.reset()                       # clears buffer, keeps calibration
        # … pupil data → on_pupil_sample → cognitive_load_updated …

    Signals:
        cognitive_load_updated (float, str, float):
            Emitted with ``(z_score, level, lhipa_value)`` on each 1-second
            tick.  ``level`` is ``"low"``, ``"moderate"``, or ``"high"``.
    """

    cognitive_load_updated = pyqtSignal(float, str, float)  # (z_score, level, lhipa)

    def __init__(
        self,
        sampling_rate: float = PUPIL_SAMPLING_RATE_HZ,
        window_sec: int = LHIPA_WINDOW_SEC,
        update_sec: int = LHIPA_UPDATE_SEC,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._sampling_rate = sampling_rate
        self._window_sec = window_sec
        self._update_sec = update_sec
        self._streamer: CognitiveLoadStreamer | None = None
        self._cal_buffer: list[float] = []
        self._collecting_baseline: bool = False
        # Set to True once iris px→mm scale is received from SessionManager.
        self._in_mm: bool = False

    def set_px_per_mm(self, px_per_mm: float) -> None:
        """Notify the processor that incoming values are now in millimetres.

        Called by SessionManager when iris calibration completes.  Switches
        ``preprocess_pupil`` to use the research-specified mm-based valid range
        (1.5 – 9.0 mm) and spike filter (≤ 10 mm/s).
        """
        if px_per_mm > 0.0:
            self._in_mm = True
            if self._streamer is not None:
                self._streamer.in_mm = True
            logger.info("LHIPAProcessor: switching to mm-unit preprocessing (px/mm=%.4f).", px_per_mm)

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def start_baseline_collection(self) -> None:
        """Begin accumulating raw pupil samples for calibration.

        Call alongside ``SessionManager.start_calibration()``.
        """
        self._cal_buffer = []
        self._collecting_baseline = True
        logger.info("LHIPAProcessor: baseline pupil collection started.")

    def finish_baseline_collection(self) -> None:
        """Stop accumulating and compute the LHIPA calibration baseline.

        Call alongside ``SessionManager.end_calibration()``.  If fewer than
        two valid LHIPA segments are found, the streamer will not be created
        and the processor remains inactive (existing PDI-based display continues).
        """
        self._collecting_baseline = False

        if not self._cal_buffer:
            logger.warning("LHIPAProcessor: no pupil data collected during calibration.")
            return

        baseline_arr = np.array(self._cal_buffer, dtype=float)
        logger.info(
            "LHIPAProcessor: computing calibration from %d samples (%.1f s) [in_mm=%s].",
            len(baseline_arr),
            len(baseline_arr) / self._sampling_rate,
            self._in_mm,
        )

        cal = calibrate_from_baseline(
            baseline_arr, self._sampling_rate, self._window_sec, in_mm=self._in_mm
        )
        if cal is None:
            logger.warning(
                "LHIPAProcessor: calibration failed — insufficient data. "
                "LHIPA will remain inactive; PDI-based display continues."
            )
            self._cal_buffer = []
            return

        self._streamer = CognitiveLoadStreamer(
            calibration=cal,
            sampling_rate=self._sampling_rate,
            window_sec=self._window_sec,
            update_sec=self._update_sec,
            in_mm=self._in_mm,
        )
        self._cal_buffer = []
        logger.info("LHIPAProcessor: streamer created (calibration OK).")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the sample buffer without discarding the calibration.

        Call at ``SessionManager.start_session()`` to prevent calibration
        data from leaking into the first analysis window.
        """
        if self._streamer is not None:
            self._streamer.reset_buffer()
        self._collecting_baseline = False
        logger.info("LHIPAProcessor reset (calibration preserved).")

    # ------------------------------------------------------------------
    # Pupil sample slot
    # ------------------------------------------------------------------

    @pyqtSlot(float, float, float)
    def on_pupil_sample(
        self, left_px: float, right_px: float, _timestamp_s: float
    ) -> None:
        """Process one raw pupil measurement from the eye tracker.

        Averages left and right diameters (ignoring zeros / missing eyes),
        appends to the calibration buffer when active, and pushes to the
        sliding-window streamer.

        Args:
            left_px: Left pupil diameter in pixels (0 if unavailable).
            right_px: Right pupil diameter in pixels (0 if unavailable).
            _timestamp_s: Unix wall-clock timestamp (unused internally).
        """
        # Average non-zero diameters.
        valid = [d for d in (left_px, right_px) if d > 0.0]
        if not valid:
            return
        diameter = float(np.mean(valid))

        if self._collecting_baseline:
            self._cal_buffer.append(diameter)

        if self._streamer is None:
            return

        result = self._streamer.push(np.array([diameter], dtype=float))
        if result is None or result.get("status") != "ok":
            return

        self.cognitive_load_updated.emit(
            float(result["z_score"]),
            str(result["cognitive_load_level"]),
            float(result["lhipa"]),
        )
        logger.debug(
            "LHIPA update: z=%.2f level=%s LHIPA=%.3f pupil_norm=%.2f",
            result["z_score"],
            result["cognitive_load_level"],
            result["lhipa"],
            result["pupil_normalized"],
        )
