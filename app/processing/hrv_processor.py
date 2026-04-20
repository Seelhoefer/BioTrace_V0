"""Windowed ECG → HRV / BPM processing for BioTrace (neurokit2).

Buffers raw ECG samples from ``raw_ecg_sample_received(value, timestamp_s)``,
runs Pan–Tompkins via ``neurokit2.ecg_process`` on a sliding window (default
30 s, refreshed every 5 s), applies RR artefact rules, and emits smoothed BPM
plus RMSSD (linear ms and ln for stress analytics).

Usage::

    sensor.raw_ecg_sample_received.connect(hrv_processor.on_ecg_sample)
    hrv_processor.rmssd_updated.connect(live_view.on_rmssd_updated)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from app.core.metrics import compute_rmssd
from app.utils.config import (
    HRV_ECG_MIN_WINDOW_SEC,
    HRV_ECG_UPDATE_SEC,
    HRV_ECG_WINDOW_SEC,
    HRV_RR_MEDIAN_DEV_FRAC,
    HRV_RR_PHYS_MAX_MS,
    HRV_RR_PHYS_MIN_MS,
    HRV_RR_ROLLING_HALF_WIDTH,
    PICO_ECG_SAMPLE_RATE_HZ,
    RMSSD_MIN_SAMPLES,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import neurokit2 as nk  # type: ignore[import-untyped]

    _NK_AVAILABLE = True
except Exception:  # noqa: BLE001
    nk = None  # type: ignore[assignment]
    _NK_AVAILABLE = False
    logger.warning(
        "neurokit2 could not be imported — HRVProcessor will emit "
        "neurokit2_unavailable status only. Install with: pip install neurokit2"
    )


def rolling_median_replace_outliers(
    rr_ms: np.ndarray,
    half_width: int = HRV_RR_ROLLING_HALF_WIDTH,
    dev_frac: float = HRV_RR_MEDIAN_DEV_FRAC,
) -> np.ndarray:
    """Apply local median + ectopic rule: replace RR deviating > dev_frac from local median.

    Intervals outside ``[HRV_RR_PHYS_MIN_MS, HRV_RR_PHYS_MAX_MS]`` are first
    replaced with NaN, then each finite sample is compared to the median of its
    symmetric neighbourhood; if it deviates by more than ``dev_frac × median``,
    it is replaced by that median (Lipponen-style soft correction).
    """
    rr = np.asarray(rr_ms, dtype=float).copy()
    n = rr.size
    if n == 0:
        return rr

    phys_ok = (rr >= HRV_RR_PHYS_MIN_MS) & (rr <= HRV_RR_PHYS_MAX_MS)
    rr[~phys_ok] = np.nan

    out = rr.copy()
    for i in range(n):
        if not np.isfinite(out[i]):
            continue
        lo = max(0, i - half_width)
        hi = min(n, i + half_width + 1)
        neigh = rr[lo:hi]
        neigh = neigh[np.isfinite(neigh)]
        if neigh.size == 0:
            continue
        med = float(np.median(neigh))
        if med <= 0:
            continue
        if abs(out[i] - med) > dev_frac * med:
            out[i] = med

    # Fill remaining NaNs with global median of physically valid original samples.
    base = rr[np.isfinite(rr) & phys_ok]
    fill = float(np.median(base)) if base.size else np.nan
    for i in range(n):
        if not np.isfinite(out[i]) and np.isfinite(fill):
            out[i] = fill
    return out


def compute_window_hrv_dict(
    ecg: np.ndarray,
    sampling_rate: int = PICO_ECG_SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    """Run neurokit2 on one ECG window and return HRV metrics dict.

    Returns:
        ``status`` one of ``ok``, ``neurokit2_unavailable``, ``insufficient_peaks``,
        ``invalid_rmssd``, ``error``.
        On ``ok``: ``bpm``, ``rmssd_ms``, ``ln_rmssd``, ``rr_median_ms``, ``n_peaks``.
    """
    if not _NK_AVAILABLE:
        return {"status": "neurokit2_unavailable"}

    ecg = np.asarray(ecg, dtype=float).ravel()
    min_samples = int(max(5, HRV_ECG_MIN_WINDOW_SEC) * sampling_rate)
    if ecg.size < min_samples:
        return {"status": "insufficient_peaks"}

    try:
        _signals, info = nk.ecg_process(
            ecg,
            sampling_rate=sampling_rate,
            method="pantompkins1985",
        )
        r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_window_hrv_dict: ecg_process failed: %s", exc)
        return {"status": "error", "message": str(exc)}

    if r_peaks.size < 4:
        return {"status": "insufficient_peaks"}

    rr_raw = np.diff(r_peaks) / float(sampling_rate) * 1000.0
    rr_clean = rolling_median_replace_outliers(rr_raw)
    rr_clean = rr_clean[np.isfinite(rr_clean)]
    if rr_clean.size < RMSSD_MIN_SAMPLES:
        return {"status": "insufficient_peaks"}

    rmssd = compute_rmssd(rr_clean)
    if rmssd <= 0.0:
        return {"status": "invalid_rmssd"}

    rr_med = float(np.median(rr_clean))
    bpm = 60_000.0 / rr_med if rr_med > 0 else 0.0
    ln_rmssd = float(np.log(rmssd))

    return {
        "status": "ok",
        "bpm": float(bpm),
        "rmssd_ms": float(rmssd),
        "ln_rmssd": ln_rmssd,
        "rr_median_ms": rr_med,
        "n_peaks": int(r_peaks.size),
    }


class HRVProcessor(QObject):
    """Sliding-window ECG buffer with periodic neurokit2 HRV extraction.

    Signals:
        rmssd_updated (float, float):
            ``(rmssd_ms, timestamp_s)`` — linear RMSSD for RMSSD card / charts.
        ln_rmssd_updated (float, float):
            ``(ln_rmssd, timestamp_s)`` — ln transform for stress analytics.
        hrv_updated (float, float, float, float, float):
            ``(rr_median_ms, bpm_smoothed, rmssd_ms, delta_rmssd_ms, timestamp_s)``
            for persistence (``rr_median_ms`` is window median NN after cleaning).
        hrv_window_metrics (dict):
            Full latest ``compute_window_hrv_dict`` result (including ``status``).
    """

    rmssd_updated = pyqtSignal(float, float)
    ln_rmssd_updated = pyqtSignal(float, float)
    hrv_updated = pyqtSignal(float, float, float, float, float)
    hrv_window_metrics = pyqtSignal(dict)

    def __init__(
        self,
        window_seconds: int = HRV_ECG_WINDOW_SEC,
        update_seconds: int = HRV_ECG_UPDATE_SEC,
        sampling_rate_hz: int = PICO_ECG_SAMPLE_RATE_HZ,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._sr = sampling_rate_hz
        self._window_samples = int(window_seconds * sampling_rate_hz)
        self._min_samples = int(HRV_ECG_MIN_WINDOW_SEC * sampling_rate_hz)
        self._buffer: deque[float] = deque(maxlen=self._window_samples)

        self._prev_rmssd: float = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(int(update_seconds * 1000))
        self._timer.timeout.connect(self._on_timer_tick)

    @pyqtSlot(float, float)
    def on_ecg_sample(self, ecg_value: float, _timestamp_s: float) -> None:
        """Append one raw ECG sample (after firmware inversion) to the ring buffer."""
        self._buffer.append(float(ecg_value))
        if not self._timer.isActive():
            self._timer.start()

    def _on_timer_tick(self) -> None:
        self.process_buffered_window()

    def process_buffered_window(self) -> dict[str, Any] | None:
        """Run neurokit2 on the current buffer if long enough; emit Qt signals.

        Returns:
            The metrics dict, or ``None`` if skipped.
        """
        if len(self._buffer) < self._min_samples:
            return None

        ecg = np.array(self._buffer, dtype=float)
        result = compute_window_hrv_dict(ecg, self._sr)
        self.hrv_window_metrics.emit(dict(result))

        ts = time.time()
        status = result.get("status", "error")
        if status != "ok":
            logger.debug("HRV window skipped: %s", result)
            return result

        rmssd = float(result["rmssd_ms"])
        ln_rmssd = float(result["ln_rmssd"])
        bpm = float(result["bpm"])
        rr_med = float(result["rr_median_ms"])
        delta_rmssd = rmssd - self._prev_rmssd
        self._prev_rmssd = rmssd

        self.rmssd_updated.emit(rmssd, ts)
        self.ln_rmssd_updated.emit(ln_rmssd, ts)
        self.hrv_updated.emit(rr_med, bpm, rmssd, delta_rmssd, ts)
        return result

    def reset(self) -> None:
        """Clear buffer and RMSSD history (session boundaries)."""
        self._buffer.clear()
        self._prev_rmssd = 0.0
        self._timer.stop()
        logger.info("HRVProcessor reset.")
