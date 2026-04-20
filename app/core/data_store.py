"""In-session ring-buffer data store for BioTrace.

During a live session all raw and processed samples are accumulated in memory
using fixed-size deques (ring buffers).  At session end, the caller is
responsible for persisting the data to SQLite via the repositories.

The DataStore has no Qt or UI dependency — it is a plain Python object that
can be unit-tested without a running QApplication.

Usage::

    store = DataStore()
    store.add_rr_interval(timestamp=12.3, rr_ms=850.0)
    rr_array = store.get_rr_timestamps_and_intervals()
"""

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

from app.utils.config import RMSSD_WINDOW_SECONDS
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Default maximum points kept in memory per signal type.
# At 1 Hz HRV and 10 Hz pupil, 3 600 s of data comfortably fits in memory.
_MAX_HRV_SAMPLES: int = 3600
_MAX_PUPIL_SAMPLES: int = 36000
_MAX_CLI_SAMPLES: int = 36000


# ---------------------------------------------------------------------------
# Named tuple types (lightweight, no Qt dependency)
# ---------------------------------------------------------------------------


class HRVSample(NamedTuple):
    """A single HRV data point."""
    timestamp:    float        # seconds since session start
    rr_interval:  float        # inter-beat interval in milliseconds
    rmssd:        float | None = None  # rolling RMSSD (ms), None until computed
    bpm:          float | None = None  # instantaneous BPM = 60 000 / rr_interval
    delta_rmssd:  float | None = None  # rmssd − rmssd_previous (stress trend)


class PupilSample(NamedTuple):
    """A single pupil measurement."""
    timestamp: float
    left_diameter: float | None   # pixels (camera units)
    right_diameter: float | None  # pixels (camera units)
    pdi: float | None             # computed pupil dilation index


class CLISample(NamedTuple):
    """A single CLI computation result."""
    timestamp: float
    cli: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------


class DataStore:
    """In-memory ring-buffer store for all live-session biometric data.

    Attributes:
        session_id: The database ID of the currently active session, set
                    externally once the session row is created.
        baseline_rmssd: Resting RMSSD from calibration (milliseconds).
        baseline_rmssd_std: Standard deviation of RMSSD in calibration (milliseconds).
        baseline_pupil_px: Resting pupil diameter from calibration (pixels).
        baseline_pupil_px_std: Standard deviation of pupil diameter in calibration (pixels).
    """

    def __init__(self) -> None:
        self.session_id: int | None = None
        self.baseline_rmssd: float = 0.0
        self.baseline_rmssd_std: float = 0.0
        self.baseline_pupil_px: float = 0.0
        self.baseline_pupil_px_std: float = 0.0

        self._hrv: deque[HRVSample] = deque(maxlen=_MAX_HRV_SAMPLES)
        self._pupil: deque[PupilSample] = deque(maxlen=_MAX_PUPIL_SAMPLES)
        self._cli: deque[CLISample] = deque(maxlen=_MAX_CLI_SAMPLES)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_hrv_sample(
        self,
        timestamp: float,
        rr_interval: float,
        rmssd: float | None = None,
        bpm: float | None = None,
        delta_rmssd: float | None = None,
    ) -> None:
        """Append a new HRV data point.

        Args:
            timestamp: Seconds since session start.
            rr_interval: RR interval in milliseconds.
            rmssd: Rolling RMSSD in milliseconds, or ``None`` if not yet computed.
            bpm: Instantaneous heart rate (60 000 / rr_interval), or ``None``.
            delta_rmssd: Change in RMSSD since the previous window, or ``None``.
        """
        self._hrv.append(HRVSample(timestamp, rr_interval, rmssd, bpm, delta_rmssd))

    def add_pupil_sample(
        self,
        timestamp: float,
        left_diameter: float | None,
        right_diameter: float | None,
        pdi: float | None = None,
    ) -> None:
        """Append a new pupil measurement.

        Args:
            timestamp: Seconds since session start.
            left_diameter: Left eye diameter in pixels, or ``None``.
            right_diameter: Right eye diameter in pixels, or ``None``.
            pdi: Pupil Dilation Index if computed, otherwise ``None``.
        """
        self._pupil.append(PupilSample(timestamp, left_diameter, right_diameter, pdi))

    def add_cli_sample(self, timestamp: float, cli: float) -> None:
        """Append a CLI computation result.

        Args:
            timestamp: Seconds since session start.
            cli: Cognitive Load Index in [0.0, 1.0].
        """
        self._cli.append(CLISample(timestamp, cli))

    # ------------------------------------------------------------------
    # Read — windowed slice for RMSSD computation
    # ------------------------------------------------------------------

    def get_recent_rr_intervals(self, window_seconds: int = RMSSD_WINDOW_SECONDS):
        """Return RR intervals from the last ``window_seconds`` of data.

        Args:
            window_seconds: Duration of the trailing window in seconds.

        Returns:
            List of RR interval values (float, milliseconds) within the window.
        """
        if not self._hrv:
            return []
        latest_ts = self._hrv[-1].timestamp
        cutoff = latest_ts - window_seconds
        return [s.rr_interval for s in self._hrv if s.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Read — full snapshots (for post-session persistence / export)
    # ------------------------------------------------------------------

    @property
    def hrv_samples(self) -> list[HRVSample]:
        """Snapshot of all HRV samples in chronological order."""
        return list(self._hrv)

    @property
    def pupil_samples(self) -> list[PupilSample]:
        """Snapshot of all pupil samples in chronological order."""
        return list(self._pupil)

    @property
    def cli_samples(self) -> list[CLISample]:
        """Snapshot of all CLI samples in chronological order."""
        return list(self._cli)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all in-memory samples (call between sessions)."""
        self._hrv.clear()
        self._pupil.clear()
        self._cli.clear()
        self.session_id = None
        logger.info("DataStore cleared.")
