"""Cognitive Load Index (CLI) processor for BioTrace.

CLI is defined as the normalized Pupil Dilation Index (PDI).  The eye tracker
is the sole input — RMSSD / HRV drives the separate Physical Stress gauge and
is not used here.

Running session min/max normalization ensures the gauge adapts to each
individual's physiological range instead of hard-coded limits.

Usage::

    pupil_proc = PupilProcessor(baseline_px=100.0)
    cli_proc = CLIProcessor()
    pupil_proc.pdi_updated.connect(cli_proc.on_pdi_updated)
    cli_proc.cli_updated.connect(live_view.on_cli_updated)
"""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.core.metrics import normalize
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel: not yet seen any data.
_UNSET = float("inf")


class CLIProcessor(QObject):
    """Maps raw PDI to a normalized Cognitive Load Index in [0, 1].

    Uses session-wide running min/max so the gauge fills the full 0–100 %
    range relative to what was observed in this session.

    Signals:
        cli_updated (float, float):
            Emitted with ``(cli, timestamp_s)`` for each accepted PDI sample.
            CLI is in the range [0.0, 1.0].
    """

    cli_updated = pyqtSignal(float, float)  # (cli, timestamp_s)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pdi: float | None = None
        self._pdi_ts: float = 0.0
        self._pdi_min: float = _UNSET
        self._pdi_max: float = -_UNSET

    # on_rmssd_updated is kept so existing signal wiring in session.py does not
    # need to change.  RMSSD is intentionally unused here — it drives the
    # Physical Stress gauge directly in LiveView instead.
    @pyqtSlot(float, float)
    def on_rmssd_updated(self, _rmssd: float, _timestamp_s: float) -> None:
        """Accept RMSSD signal (unused — stress is shown separately)."""

    @pyqtSlot(float, float)
    def on_pdi_updated(self, pdi: float, timestamp_s: float) -> None:
        """Receive a PDI sample and emit a normalized CLI value.

        Args:
            pdi: Pupil Dilation Index (or raw diameter in px when no baseline).
            timestamp_s: Unix timestamp of the sample.
        """
        self._pdi = pdi
        self._pdi_ts = timestamp_s
        self._pdi_min = min(self._pdi_min, pdi)
        self._pdi_max = max(self._pdi_max, pdi)

        if self._pdi_min == _UNSET or self._pdi_max == -_UNSET:
            return

        cli = normalize(pdi, self._pdi_min, self._pdi_max)
        self.cli_updated.emit(cli, timestamp_s)

    def reset(self) -> None:
        """Clear all state between sessions."""
        self._pdi = None
        self._pdi_min = _UNSET
        self._pdi_max = -_UNSET
        logger.info("CLIProcessor reset.")
