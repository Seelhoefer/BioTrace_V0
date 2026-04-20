"""Session lifecycle manager for BioTrace — Phase 4 update.

Phase 4 additions:
- ``start_calibration()`` / ``end_calibration()`` — drives the calibration
  collection mode where sensors run but data is accumulated only for
  baseline computation (not stored as session samples).
- DataStore writes now connected to all three processors so every metric
  sample is captured in memory during live sessions.
- Calibration baselines persisted to the database via CalibrationRepository.
- Session samples bulk-persisted to the database on session end.

Session state machine:
    IDLE → CALIBRATING → READY → RUNNING → IDLE
"""

import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.core.data_store import DataStore
from app.core.metrics import average_pupil_diameter
from app.hardware.mock_sensors import MockHRVSensor, MockEyeTracker
from app.hardware.error_counter import ErrorCounter
from app.processing.hrv_processor import HRVProcessor, compute_window_hrv_dict
from app.utils.config import (
    USE_PICO_ECG,
    USE_EYE_TRACKER,
    PICO_ECG_PORT,
    PICO_ECG_BAUD,
    PICO_ECG_SAMPLE_RATE_HZ,
    SESSIONS_DIR,
    CALIBRATION_DURATION_SECONDS,
    CALIBRATION_MIN_ECG_SECONDS,
    ERROR_EVENT_DEBOUNCE_SECONDS,
)
from app.processing.pupil_processor import PupilProcessor
from app.processing.cli_processor import CLIProcessor
from app.storage.database import DatabaseManager
from app.storage.session_repository import SessionRepository
from app.storage.calibration_repository import CalibrationRepository
from app.storage.export import SessionExporter
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionState(Enum):
    """Finite-state machine states for the session lifecycle."""
    IDLE        = auto()
    CALIBRATING = auto()
    READY       = auto()   # calibration done, session not yet started
    RUNNING     = auto()
    PAUSED      = auto()


class SessionManager(QObject):
    """Orchestrates the full calibration → session lifecycle.

    Signals:
        calibration_complete (float, float):
            Emitted when baseline recording finishes, carrying
            ``(baseline_rmssd_ms, baseline_pupil_px)``.
        session_started (int):
            Emitted with the new session's database ID.
        session_ended (int):
            Emitted with the session ID when a session ends.
        session_paused (int):
            Emitted when a session is paused.
        session_resumed (int):
            Emitted when a session is resumed.
        rmssd_updated (float, float):
            Forwarded from HRVProcessor — ``(rmssd_ms, timestamp_s)``.
        pdi_updated (float, float):
            Forwarded from PupilProcessor — ``(pdi, timestamp_s)``.
        cli_updated (float, float):
            Forwarded from CLIProcessor — ``(cli, timestamp_s)``.
    """

    calibration_complete  = pyqtSignal(float, float)  # (baseline_rmssd, baseline_pupil_mm)
    session_started       = pyqtSignal(int)
    session_ended         = pyqtSignal(int)
    session_paused        = pyqtSignal(int)
    session_resumed       = pyqtSignal(int)
    rmssd_updated         = pyqtSignal(float, float)
    pdi_updated           = pyqtSignal(float, float)
    cli_updated           = pyqtSignal(float, float)
    bpm_updated           = pyqtSignal(float, float)  # (bpm, timestamp_s)
    hrv_connection_changed = pyqtSignal(bool, str)    # (connected, message)
    eye_connection_changed = pyqtSignal(bool, str)    # (connected, message)
    camera_connection_changed = pyqtSignal(bool, str) # (connected, message)
    error_count_updated   = pyqtSignal(int)

    def __init__(self, db: DatabaseManager, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._state = SessionState.IDLE
        self._session_id: int | None = None
        self._session_dir: Path | None = None
        self._session_start_time: float = 0.0
        self._error_count: int = 0
        self._last_error_event_ts: float = 0.0
        self._recording_path: str | None = None

        self._db = db
        # ── Repositories ───────────────────────────────────────────────
        self._session_repo = SessionRepository(db)
        self._cal_repo     = CalibrationRepository(db)
        self._exporter     = SessionExporter(db)

        # ── Data store ─────────────────────────────────────────────────
        self._data_store = DataStore()

        # ── Calibration accumulators (raw, pre-baseline) ───────────────
        self._cal_ecg_samples: list[float] = []
        self._cal_rmssd_values: deque[float] = deque()
        self._cal_pupils: deque[float] = deque()       # avg diameters in pixels
        self._cal_duration: int = 0

        # ── Baselines (set after calibration) ─────────────────────────
        self._baseline_rmssd:     float = 0.0
        self._baseline_rmssd_std: float = 0.0
        self._baseline_pupil_px:  float = 0.0
        self._baseline_pupil_std: float = 0.0

        # ── Sensors ────────────────────────────────────────────────────
        if USE_PICO_ECG:
            from app.hardware.pico_ecg_sensor import PicoECGSensor
            self._hrv_sensor = PicoECGSensor(port=None, baud=PICO_ECG_BAUD, parent=self)
        else:
            self._hrv_sensor = MockHRVSensor(self)

        if USE_EYE_TRACKER:
            from app.hardware.eye_tracker import EyeTrackerSensor
            self._eye_tracker = EyeTrackerSensor(parent=self)
        else:
            self._eye_tracker = MockEyeTracker(self)

        self._error_counter = ErrorCounter()  # stub for Phase 6b hardware

        # ── Processors ─────────────────────────────────────────────────
        self._hrv_proc   = HRVProcessor(parent=self)
        self._pupil_proc = PupilProcessor(parent=self)
        self._cli_proc   = CLIProcessor(parent=self)

        self._wire_signals()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        """Connect all internal signals and slots."""
        # Raw sensor → processors
        self._hrv_sensor.raw_ecg_sample_received.connect(self._hrv_proc.on_ecg_sample)
        self._eye_tracker.raw_pupil_received.connect(self._pupil_proc.on_pupil_sample)

        # Raw sensor → calibration accumulators
        self._hrv_sensor.raw_ecg_sample_received.connect(self._on_cal_ecg)
        self._eye_tracker.raw_pupil_received.connect(self._on_cal_pupil)
        self._hrv_proc.rmssd_updated.connect(self._on_cal_rmssd)

        # Processors → CLI combiner
        self._hrv_proc.rmssd_updated.connect(self._cli_proc.on_rmssd_updated)
        self._pupil_proc.pdi_updated.connect(self._cli_proc.on_pdi_updated)

        # Processors → public signals (for LiveView)
        self._hrv_proc.rmssd_updated.connect(self.rmssd_updated)
        self._pupil_proc.pdi_updated.connect(self.pdi_updated)
        self._cli_proc.cli_updated.connect(self.cli_updated)

        # Processors → DataStore (in-session sample recording)
        self._hrv_proc.hrv_updated.connect(self._store_hrv)
        self._pupil_proc.pdi_updated.connect(self._store_pdi)
        self._cli_proc.cli_updated.connect(self._store_cli)

        # HRV processor → BPM forwarding (always, regardless of session state)
        self._hrv_proc.hrv_updated.connect(self._forward_bpm)

        # Sensor connection status → public signals
        self._hrv_sensor.connection_status_changed.connect(self.hrv_connection_changed)
        self._eye_tracker.connection_status_changed.connect(self.eye_connection_changed)
        if hasattr(self._hrv_sensor, "wall_contact_detected"):
            self._hrv_sensor.wall_contact_detected.connect(self._on_hardware_error)

        # Mock camera connection since it's managed externally by the UI widgets
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.camera_connection_changed.emit(True, "Camera Mock"))
        
        # Error counter (hardware box wire-touch) → internal slot
        self._error_counter.error_detected.connect(self._on_hardware_error)

    # ------------------------------------------------------------------
    # Calibration accumulator slots
    # ------------------------------------------------------------------

    @pyqtSlot(float, float)
    def _on_cal_ecg(self, ecg_value: float, _timestamp: float) -> None:
        """Accumulate raw ECG samples during calibration (neurokit2 baseline)."""
        if self._state == SessionState.CALIBRATING:
            self._cal_ecg_samples.append(float(ecg_value))

    @pyqtSlot(float, float, float)
    def _on_cal_pupil(self, left_px: float, right_px: float, _timestamp: float) -> None:
        """Accumulate raw pupil diameters during calibration."""
        if self._state == SessionState.CALIBRATING:
            avg = average_pupil_diameter(left_px, right_px)
            if avg is not None:
                self._cal_pupils.append(avg)

    @pyqtSlot(float, float)
    def _on_cal_rmssd(self, rmssd: float, _timestamp: float) -> None:
        """Accumulate rolling RMSSD values during calibration."""
        if self._state == SessionState.CALIBRATING and rmssd > 0.0:
            self._cal_rmssd_values.append(rmssd)

    # ------------------------------------------------------------------
    # DataStore write slots (active during RUNNING state only)
    # ------------------------------------------------------------------

    @pyqtSlot(float, float, float, float, float)
    def _store_hrv(
        self,
        rr_ms: float,
        bpm: float,
        rmssd: float,
        delta_rmssd: float,
        timestamp_s: float,
    ) -> None:
        """Store one windowed HRV snapshot in the DataStore.

        Receives ``hrv_updated`` from :class:`HRVProcessor` (neurokit2 window).

        Args:
            rr_ms: Median NN interval (ms) after cleaning in that window.
            bpm: Smoothed BPM from the same window.
            rmssd: RMSSD (ms) from cleaned RR in the window.
            delta_rmssd: Change in RMSSD since the previous successful window.
            timestamp_s: Unix timestamp of the analysis.
        """
        if self._state == SessionState.RUNNING:
            elapsed = timestamp_s - self._session_start_time
            self._data_store.add_hrv_sample(elapsed, rr_ms, rmssd, bpm, delta_rmssd)

    @pyqtSlot(float, float, float, float, float)
    def _forward_bpm(
        self,
        _rr_ms: float,
        bpm: float,
        _rmssd: float,
        _delta_rmssd: float,
        timestamp_s: float,
    ) -> None:
        """Forward smoothed window BPM from HRVProcessor to the UI layer."""
        self.bpm_updated.emit(bpm, timestamp_s)

    @pyqtSlot(float, float)
    def _store_pdi(self, pdi: float, timestamp: float) -> None:
        if self._state == SessionState.RUNNING:
            elapsed = timestamp - self._session_start_time
            self._data_store.add_pupil_sample(elapsed, None, None, pdi)

    @pyqtSlot(float, float)
    def _store_cli(self, cli: float, timestamp: float) -> None:
        if self._state == SessionState.RUNNING:
            elapsed = timestamp - self._session_start_time
            self._data_store.add_cli_sample(elapsed, cli)

    # ------------------------------------------------------------------
    # Error count management
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_hardware_error(self) -> None:
        """Increment error count when hardware sensor detects a touch."""
        if self._state == SessionState.RUNNING:
            now = time.time()
            if now - self._last_error_event_ts < ERROR_EVENT_DEBOUNCE_SECONDS:
                return
            self._last_error_event_ts = now
            self._error_count += 1
            self.error_count_updated.emit(self._error_count)

    def increment_error_count(self) -> None:
        """Manually increment the error count (from UI)."""
        if self._state == SessionState.RUNNING:
            self._error_count += 1
            self.error_count_updated.emit(self._error_count)

    def decrement_error_count(self) -> None:
        """Manually decrement the error count (from UI), floored at 0."""
        if self._state == SessionState.RUNNING and self._error_count > 0:
            self._error_count -= 1
            self.error_count_updated.emit(self._error_count)

    # ------------------------------------------------------------------
    # Public API — calibration
    # ------------------------------------------------------------------

    def start_calibration(self) -> None:
        """Start sensor streaming in calibration (baseline) mode.

        The sensors will emit data, but processor output is not displayed
        in the UI until the full baseline recording is complete.  Raw
        samples are accumulated for baseline computation.
        """
        if self._state not in (SessionState.IDLE, SessionState.READY):
            logger.warning("start_calibration() called in state %s — ignored.", self._state)
            return

        self._cal_ecg_samples.clear()
        self._cal_rmssd_values.clear()
        self._cal_pupils.clear()
        self._state = SessionState.CALIBRATING

        self._hrv_sensor.start()
        if USE_EYE_TRACKER:
            self._eye_tracker.start()
        logger.info("Calibration baseline recording started.")

    def end_calibration(self, duration_seconds: int) -> tuple[float, float]:
        """Stop sensors, compute baselines from accumulated data, and persist.

        This method is called by the CalibrationView countdown timer after
        the full baseline window has elapsed.

        Args:
            duration_seconds: Actual recording duration (may differ from the
                              configured target if stopped early).

        Returns:
            ``(baseline_rmssd_ms, baseline_pupil_px)`` — the computed baselines.
        """
        if self._state != SessionState.CALIBRATING:
            logger.warning("end_calibration() called in state %s — ignored.", self._state)
            return (0.0, 0.0)

        self._hrv_sensor.stop()
        if USE_EYE_TRACKER:
            self._eye_tracker.stop()

        # Compute baselines (RMSSD from full calibration ECG via neurokit2).
        ecg_arr = np.array(self._cal_ecg_samples, dtype=float)
        rmssd_array = np.array(list(self._cal_rmssd_values), dtype=float)
        pupil_array = np.array(list(self._cal_pupils), dtype=float)
        min_ecg_samples = int(CALIBRATION_MIN_ECG_SECONDS * PICO_ECG_SAMPLE_RATE_HZ)
        ecg_ok = len(ecg_arr) >= min_ecg_samples
        if duration_seconds >= CALIBRATION_DURATION_SECONDS and ecg_ok:
            nk_result = compute_window_hrv_dict(ecg_arr, PICO_ECG_SAMPLE_RATE_HZ)
            if nk_result.get("status") == "ok":
                self._baseline_rmssd = float(nk_result["rmssd_ms"])
                if rmssd_array.size > 1:
                    self._baseline_rmssd_std = float(np.std(rmssd_array, ddof=1))
                elif rmssd_array.size == 1:
                    self._baseline_rmssd_std = 0.0
                else:
                    self._baseline_rmssd_std = 0.0
            elif rmssd_array.size > 0:
                self._baseline_rmssd = float(np.mean(rmssd_array))
                self._baseline_rmssd_std = (
                    float(np.std(rmssd_array, ddof=1)) if rmssd_array.size > 1 else 0.0
                )
                logger.warning(
                    "Calibration RMSSD from rolling updates only (neurokit2 full-ECG status=%s).",
                    nk_result.get("status"),
                )
            else:
                self._baseline_rmssd = 0.0
                self._baseline_rmssd_std = 0.0
                logger.warning(
                    "Calibration RMSSD baseline rejected: neurokit2 status=%s on %d ECG samples.",
                    nk_result.get("status"),
                    len(ecg_arr),
                )
        else:
            self._baseline_rmssd = 0.0
            self._baseline_rmssd_std = 0.0
            logger.warning(
                "Calibration RMSSD baseline rejected (duration=%ds, n_ecg=%d; requires >=%ds and >=%d samples).",
                duration_seconds,
                len(ecg_arr),
                CALIBRATION_DURATION_SECONDS,
                min_ecg_samples,
            )

        if len(pupil_array) > 0:
            self._baseline_pupil_px = float(np.mean(pupil_array))
            self._baseline_pupil_std = float(np.std(pupil_array))
        else:
            self._baseline_pupil_px = 0.0
            self._baseline_pupil_std = 0.0

        self._cal_duration = duration_seconds

        # Pass baseline to pupil processor.
        self._pupil_proc.set_baseline(self._baseline_pupil_px)
        self._data_store.baseline_rmssd = self._baseline_rmssd
        self._data_store.baseline_rmssd_std = self._baseline_rmssd_std
        self._data_store.baseline_pupil_px = self._baseline_pupil_px
        self._data_store.baseline_pupil_px_std = self._baseline_pupil_std

        self._state = SessionState.READY
        logger.info(
            "Calibration complete: RMSSD=%.2f±%.2f ms, pupil=%.3f±%.3f px (n_ecg=%d, n_rmssd_win=%d, n_pupils=%d)",
            self._baseline_rmssd,
            self._baseline_rmssd_std,
            self._baseline_pupil_px,
            self._baseline_pupil_std,
            len(self._cal_ecg_samples),
            len(self._cal_rmssd_values),
            len(self._cal_pupils),
        )
        self.calibration_complete.emit(self._baseline_rmssd, self._baseline_pupil_px)
        return (self._baseline_rmssd, self._baseline_pupil_px)

    # ------------------------------------------------------------------
    # Public API — session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> int:
        """Create a session record, persist calibration baseline, begin streaming.

        Returns:
            The new session's database ID.
        """
        if self._state == SessionState.RUNNING:
            logger.warning("start_session() called while already running (id=%d).", self._session_id)
            return self._session_id

        # Reset processors and data store.
        self._hrv_proc.reset()
        self._pupil_proc.reset()
        self._cli_proc.reset()
        self._data_store.clear()
        self._error_count = 0
        self._last_error_event_ts = 0.0
        self.error_count_updated.emit(0)

        # Restore baseline in pupil processor (cleared by reset).
        if self._baseline_pupil_px > 0.0:
            self._pupil_proc.set_baseline(self._baseline_pupil_px)

        # Create DB session record.
        started_at = datetime.now(tz=timezone.utc)
        self._session_id = self._session_repo.create_session(started_at)
        self._data_store.session_id = self._session_id

        # Create session directory: sessions/session_ID_YYYYMMDD_HHMMSS
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path(SESSIONS_DIR) / f"session_{self._session_id}_{ts}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Session directory created: %s", self._session_dir)

        # Persist calibration baseline if we have one.
        if self._baseline_rmssd > 0.0 or self._baseline_pupil_px > 0.0:
            self._cal_repo.save_calibration(
                session_id=self._session_id,
                baseline_rmssd=self._baseline_rmssd,
                baseline_rmssd_std=self._baseline_rmssd_std,
                baseline_pupil_px=self._baseline_pupil_px,
                baseline_pupil_std=self._baseline_pupil_std,
                duration_seconds=self._cal_duration,
            )

        self._session_start_time = time.time()
        self._state = SessionState.RUNNING

        self._hrv_sensor.start()
        if USE_EYE_TRACKER:
            self._eye_tracker.start()

        self.session_started.emit(self._session_id)
        logger.info("Session %d started (state=RUNNING).", self._session_id)
        return self._session_id

    def pause_session(self) -> None:
        """Suspend data recording and clock (state: PAUSED)."""
        if self._state == SessionState.RUNNING:
            self._state = SessionState.PAUSED
            self.session_paused.emit(self._session_id)
            logger.info("Session %d paused.", self._session_id)

    def resume_session(self) -> None:
        """Continue data recording and clock (state: RUNNING)."""
        if self._state == SessionState.PAUSED:
            self._state = SessionState.RUNNING
            self.session_resumed.emit(self._session_id)
            logger.info("Session %d resumed.", self._session_id)

    def set_recording_path(self, path: str | None) -> None:
        """Store the filesystem path of the video recording for this session.

        Called by the UI (LiveView) when recording stops.

        Args:
            path: Absolute filesystem path to the MP4 file.
        """
        self._recording_path = path
        logger.debug("Recording path set: %s", path)

    def end_session(self, notes: str = "") -> None:
        """Stop sensors, bulk-persist all session samples, finalize DB record.

        Args:
            notes: Optional free-text notes to store with the session.
        """
        if self._state != SessionState.RUNNING:
            logger.warning("end_session() called in state %s — ignored.", self._state)
            return

        self._hrv_sensor.stop()
        if USE_EYE_TRACKER:
            self._eye_tracker.stop()

        # Finalize the database record.
        ended_at = datetime.now(tz=timezone.utc)
        self._session_repo.end_session(
            self._session_id, ended_at, notes, error_count=self._error_count
        )

        # Store recording path if one was set.
        if self._recording_path:
            self._session_repo.set_video_path(self._session_id, self._recording_path)

        # Bulk-persist all in-memory samples.
        self._persist_samples()

        # ── AUTOMATIC EXCEL EXPORT ───────────────────────────────────
        if self._session_id and self._session_dir:
            excel_path = self._session_dir / f"session_{self._session_id}.xlsx"
            try:
                self._exporter.export_excel(self._session_id, excel_path)
                logger.info("Automatic session export completed: %s", excel_path)
            except Exception as e:
                logger.error("Failed to automatically export session %d: %s", self._session_id, e)

        finished_id = self._session_id
        self._session_id = None
        self._recording_path = None
        self._state = SessionState.IDLE

        self.session_ended.emit(finished_id)
        logger.info("Session %d ended and persisted.", finished_id)

    def _persist_samples(self) -> None:
        """Bulk-write all DataStore samples to the database."""
        if self._session_id is None and self._data_store.session_id is None:
            return
        sid = self._data_store.session_id or self._session_id

        hrv = [
            (s.timestamp, s.rr_interval, s.rmssd, s.bpm, s.delta_rmssd)
            for s in self._data_store.hrv_samples
        ]
        pupil = [(s.timestamp, s.left_diameter, s.right_diameter, s.pdi)
                 for s in self._data_store.pupil_samples]
        cli = [(s.timestamp, s.cli) for s in self._data_store.cli_samples]

        if hrv:
            self._cal_repo.save_hrv_samples_bulk(sid, hrv)
        if pupil:
            self._cal_repo.save_pupil_samples_bulk(sid, pupil)
        if cli:
            self._cal_repo.save_cli_samples_bulk(sid, cli)

        logger.info(
            "Persisted sample bulk: %d HRV, %d pupil, %d CLI for session %d.",
            len(hrv), len(pupil), len(cli), sid,
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_session_dir(self) -> Path | None:
        """The subfolder for the current session's data (if active)."""
        return self._session_dir

    @property
    def state(self) -> SessionState:
        """Current state of the session lifecycle."""
        return self._state

    @property
    def baseline_rmssd(self) -> float:
        """Resting RMSSD baseline in milliseconds (0.0 if not calibrated)."""
        return self._baseline_rmssd

    @property
    def baseline_rmssd_std(self) -> float:
        """Calibration RMSSD baseline standard deviation in milliseconds."""
        return self._baseline_rmssd_std

    @property
    def baseline_pupil_px(self) -> float:
        """Resting pupil diameter baseline in pixels (0.0 if not calibrated)."""
        return self._baseline_pupil_px

    @property
    def baseline_pupil_std(self) -> float:
        """Calibration pupil baseline standard deviation in pixels."""
        return self._baseline_pupil_std

    @property
    def data_store(self) -> DataStore:
        """Read-only access to the current session's data buffer."""
        return self._data_store

    @property
    def current_session_id(self) -> int | None:
        """Database ID of the active session, or None."""
        return self._session_id

    def set_pupil_baseline(self, baseline_px: float) -> None:
        """Manually override the pupil baseline (used by CalibrationView)."""
        self._baseline_pupil_px = baseline_px
        self._pupil_proc.set_baseline(baseline_px)
