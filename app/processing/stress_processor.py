"""Stress processor using the neurokit2 ECG pipeline for BioTrace.

- ``nk_ecg_rmssd_ms``: neurokit2 ECG → RMSSD (ms) for one segment.
- ``compute_stress_metrics_nk``: RMSSD vs mean calibration RMSSD, classified by
  **percent change** bands (not z-scores).
- ``HRVStreamer``: 30-second sliding window with 5-second update cadence.
- ``StressProcessor``: Qt ``QObject`` wrapper emitting ``stress_updated``.

Baseline workflow::

    stress_proc.start_baseline_collection()
    # … sensor streams ECG …
    stress_proc.finish_baseline_collection()   # mean RMSSD per 30 s segment

    stress_proc.reset()                        # clears buffer, keeps baseline

Graceful degradation:
    If neurokit2 is not installed the processor logs a warning but remains
    silent — the RMSSD-based stress display in LiveView continues as before.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.utils.config import (
    PICO_ECG_SAMPLE_RATE_HZ,
    STRESS_NK_UPDATE_SEC,
    STRESS_NK_WINDOW_SEC,
    STRESS_RMSSD_PCT_MILD_THRESHOLD,
    STRESS_RMSSD_PCT_STABLE_THRESHOLD,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import neurokit2 as nk  # type: ignore[import]
    _NK_AVAILABLE = True
except Exception:  # noqa: BLE001 — catches ImportError, AttributeError from binary deps
    nk = None  # type: ignore[assignment]
    _NK_AVAILABLE = False
    logger.warning(
        "neurokit2 could not be imported — StressProcessor will be inactive. "
        "Ensure neurokit2 and its dependencies (pyarrow, pandas) are compatible. "
        "Install with: pip install neurokit2"
    )


def nk_ecg_rmssd_ms(
    ecg_segment: np.ndarray,
    sampling_rate: int = PICO_ECG_SAMPLE_RATE_HZ,
) -> dict:
    """Run neurokit2 ECG processing and return RMSSD for one segment.

    Returns:
        ``{"status": "ok", "rmssd_ms": float, "ln_rmssd": float}`` or a failure dict.
    """
    if not _NK_AVAILABLE:
        return {"status": "neurokit2_unavailable"}

    try:
        _signals, info = nk.ecg_process(
            ecg_segment,
            sampling_rate=sampling_rate,
            method="pantompkins1985",
        )
        r_peaks = info["ECG_R_Peaks"]

        if len(r_peaks) < 4:
            return {"status": "insufficient_peaks"}

        hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
        rmssd_ms = float(hrv_time["HRV_RMSSD"].iloc[0])

        if rmssd_ms <= 0:
            return {"status": "invalid_rmssd"}

        ln_rmssd = float(np.log(rmssd_ms))
        return {"status": "ok", "rmssd_ms": rmssd_ms, "ln_rmssd": ln_rmssd}
    except Exception as exc:  # noqa: BLE001
        logger.warning("nk_ecg_rmssd_ms failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def _stress_level_from_pct_change(pct_change: float) -> str:
    """Map RMSSD % change vs baseline to stable / mild / high."""
    if pct_change >= STRESS_RMSSD_PCT_STABLE_THRESHOLD:
        return "stable"
    if pct_change >= STRESS_RMSSD_PCT_MILD_THRESHOLD:
        return "mild"
    return "high"


def compute_stress_metrics_nk(
    ecg_segment: np.ndarray,
    baseline_rmssd_ms: float,
    sampling_rate: int = PICO_ECG_SAMPLE_RATE_HZ,
) -> dict:
    """Run neurokit2 on ``ecg_segment`` and classify stress vs baseline RMSSD (%).

    Args:
        ecg_segment: Raw ECG amplitude array (30 s window recommended).
        baseline_rmssd_ms: Mean RMSSD (ms) from resting calibration segments.
        sampling_rate: ECG sample rate in Hz.

    Returns:
        On success::

            {
                "status": "ok",
                "rmssd_ms": float,
                "ln_rmssd": float,
                "pct_change_rmssd": float,
                "stress_level": "stable" | "mild" | "high",
            }

        On failure: ``{"status": "<reason>"}``.
    """
    if not _NK_AVAILABLE:
        return {"status": "neurokit2_unavailable"}

    if baseline_rmssd_ms <= 0:
        return {"status": "no_baseline"}

    result = nk_ecg_rmssd_ms(ecg_segment, sampling_rate=sampling_rate)
    if result.get("status") != "ok":
        return result

    rmssd_ms = float(result["rmssd_ms"])
    ln_rmssd = float(result["ln_rmssd"])
    pct_change = ((rmssd_ms - baseline_rmssd_ms) / baseline_rmssd_ms) * 100.0
    level = _stress_level_from_pct_change(pct_change)

    return {
        "status": "ok",
        "rmssd_ms": rmssd_ms,
        "ln_rmssd": ln_rmssd,
        "pct_change_rmssd": pct_change,
        "stress_level": level,
    }


class HRVStreamer:
    """30-second sliding ECG window with 5-second update cadence."""

    def __init__(
        self,
        sampling_rate: int = PICO_ECG_SAMPLE_RATE_HZ,
        window_sec: int = STRESS_NK_WINDOW_SEC,
        update_sec: int = STRESS_NK_UPDATE_SEC,
    ) -> None:
        self.sr: int = sampling_rate
        self.buffer: deque[float] = deque(maxlen=window_sec * sampling_rate)
        self.update_every: int = update_sec * sampling_rate
        self.samples_since_update: int = 0
        self.baseline_rmssd_ms: float | None = None

    def set_baseline(self, baseline_ecg: np.ndarray) -> None:
        """Set mean calibration RMSSD (ms) from overlapping 30 s segments (5 s stride)."""
        if not _NK_AVAILABLE:
            logger.warning("HRVStreamer.set_baseline skipped: neurokit2 not installed.")
            return

        segment_len = 30 * self.sr
        if len(baseline_ecg) < segment_len:
            logger.warning(
                "Baseline ECG too short (%d samples, need %d for a 30-s segment). "
                "Stress % scores will be unavailable.",
                len(baseline_ecg),
                segment_len,
            )
            return

        rmssds: list[float] = []
        stride = self.sr * 5
        for start in range(0, len(baseline_ecg) - segment_len, stride):
            seg = baseline_ecg[start : start + segment_len]
            seg_result = nk_ecg_rmssd_ms(seg, sampling_rate=self.sr)
            if seg_result.get("status") == "ok":
                rmssds.append(float(seg_result["rmssd_ms"]))

        if len(rmssds) < 2:
            logger.warning(
                "Only %d valid baseline segments found (need ≥ 2). "
                "Extend calibration duration (≥ 60 s) for reliable stress scoring.",
                len(rmssds),
            )
            return

        self.baseline_rmssd_ms = float(np.mean(rmssds))
        logger.info(
            "HRVStreamer baseline set: mean RMSSD=%.2f ms (%d segments).",
            self.baseline_rmssd_ms,
            len(rmssds),
        )

    def push(self, samples: np.ndarray) -> dict | None:
        self.buffer.extend(samples)
        self.samples_since_update += len(samples)

        if self.baseline_rmssd_ms is None:
            return None

        if (
            self.samples_since_update >= self.update_every
            and len(self.buffer) == self.buffer.maxlen
        ):
            self.samples_since_update = 0
            return compute_stress_metrics_nk(
                np.array(self.buffer, dtype=float),
                self.baseline_rmssd_ms,
                sampling_rate=self.sr,
            )
        return None

    def reset_buffer(self) -> None:
        self.buffer.clear()
        self.samples_since_update = 0


class StressProcessor(QObject):
    """Qt wrapper: ``raw_ecg_sample_received`` → ``stress_updated``."""

    stress_updated = pyqtSignal(float, str, float)  # (pct_change_rmssd, stress_level, rmssd_ms)

    def __init__(
        self,
        sampling_rate: int = PICO_ECG_SAMPLE_RATE_HZ,
        window_sec: int = STRESS_NK_WINDOW_SEC,
        update_sec: int = STRESS_NK_UPDATE_SEC,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._streamer = HRVStreamer(
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            update_sec=update_sec,
        )
        self._cal_ecg_buffer: list[float] = []
        self._collecting_baseline: bool = False

    def start_baseline_collection(self) -> None:
        self._cal_ecg_buffer = []
        self._collecting_baseline = True
        logger.info("StressProcessor: baseline ECG collection started.")

    def finish_baseline_collection(self) -> None:
        """Compute mean RMSSD baseline from calibration ECG."""
        self._collecting_baseline = False

        if not self._cal_ecg_buffer:
            logger.warning("StressProcessor: no ECG data collected during calibration.")
            return

        baseline_ecg = np.array(self._cal_ecg_buffer, dtype=float)
        logger.info(
            "StressProcessor: computing baseline from %d samples (%.1f s).",
            len(baseline_ecg),
            len(baseline_ecg) / self._streamer.sr,
        )
        self._streamer.set_baseline(baseline_ecg)
        self._cal_ecg_buffer = []

    def reset(self) -> None:
        self._streamer.reset_buffer()
        self._collecting_baseline = False
        logger.info("StressProcessor reset (baseline preserved).")

    @pyqtSlot(float, float)
    def on_ecg_sample(self, ecg_value: float, _timestamp_s: float) -> None:
        if self._collecting_baseline:
            self._cal_ecg_buffer.append(ecg_value)

        result = self._streamer.push(np.array([ecg_value], dtype=float))
        if result is not None and result.get("status") == "ok":
            self.stress_updated.emit(
                float(result["pct_change_rmssd"]),
                str(result["stress_level"]),
                float(result["rmssd_ms"]),
            )
            logger.debug(
                "Stress update: pct=%.1f%% level=%s RMSSD=%.1f ms",
                result["pct_change_rmssd"],
                result["stress_level"],
                result["rmssd_ms"],
            )
