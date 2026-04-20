"""Runtime configuration constants for BioTrace.

All thresholds, hardware settings, and algorithm weights live here.
No magic numbers anywhere else in the codebase.

To change behaviour, edit these values — no other file should need touching.
"""

# ---------------------------------------------------------------------------
# Hardware — Serial / USB
# ---------------------------------------------------------------------------

HRV_SENSOR_PORT: str = "/dev/tty.usbserial-0001"  # macOS default; override as needed
HRV_SENSOR_BAUD: int = 115200

EYE_TRACKER_PORT: str = "/dev/tty.usbserial-0002"
EYE_TRACKER_BAUD: int = 115200

# Eye tracker — USB camera (separate from endoscopy camera)
EYE_TRACKER_CAMERA_INDEX: int = 0

CAMERA_INDEX: int = 1  # OpenCV camera index (usually 1 for external USB on laptops)

# Number of frames to attempt during camera warmup before declaring failure.
# Covers the macOS/AVFoundation delay where isOpened()=True before streaming starts.
CAMERA_WARMUP_FRAMES: int = 60  # ~2 seconds at 30 fps

# Video recording output settings.
SESSIONS_DIR: str = "sessions"
VIDEO_RECORDINGS_DIR: str = "recordings"  # legacy, replaced by session-specific subfolders
VIDEO_RECORDING_FPS_FALLBACK: float = 30.0
VIDEO_RECORDING_FOURCC: str = "mp4v"

# ---------------------------------------------------------------------------
# Hardware — Raspberry Pi Pico ECG (YLab Zero firmware)
# ---------------------------------------------------------------------------

# Set to True to use the real Pico over USB serial; False uses MockHRVSensor.
USE_PICO_ECG: bool = True

# Set to True when a real eye tracker is physically connected; False disables
# MockEyeTracker so no fake pupil/PDI/CLI data appears in the live session.
USE_EYE_TRACKER: bool = True

# USB serial port for the Pi Pico (macOS: /dev/tty.usbmodem*, Linux: /dev/ttyACM*).
PICO_ECG_PORT: str = "/dev/cu.usbmodem1101"
PICO_ECG_BAUD: int = 115200

# Yeda analog ECG sensor sample rate (set in firmware: 1.0/150 interval).
PICO_ECG_SAMPLE_RATE_HZ: int = 150

# ---------------------------------------------------------------------------
# Signal Processing — ECG / HRV (neurokit2 sliding window)
# ---------------------------------------------------------------------------

# Raw ECG ring buffer length and neurokit2 re-analysis cadence.
HRV_ECG_WINDOW_SEC: int = 30
HRV_ECG_UPDATE_SEC: int = 5

# Shared window/update for StressProcessor (same physical buffer contract).
STRESS_NK_WINDOW_SEC: int = HRV_ECG_WINDOW_SEC
STRESS_NK_UPDATE_SEC: int = HRV_ECG_UPDATE_SEC

# RMSSD % change vs mean calibration RMSSD: pct = (RMSSD_now − baseline) / baseline × 100.
# Stable: pct ≥ STRESS_RMSSD_PCT_STABLE_THRESHOLD (0 % down to −10 %).
# Mild: STRESS_RMSSD_PCT_MILD_THRESHOLD ≤ pct < stable threshold (−40 % to −10 %).
# High: pct < STRESS_RMSSD_PCT_MILD_THRESHOLD (< −40 %).
# Used by StressProcessor and NeedleGauge arc widths (same bands on 0–100 % “drop” scale).
STRESS_RMSSD_PCT_STABLE_THRESHOLD: float = -10.0
STRESS_RMSSD_PCT_MILD_THRESHOLD: float = -40.0

# Minimum buffered ECG duration (seconds) before the first neurokit2 analysis.
HRV_ECG_MIN_WINDOW_SEC: float = 10.0

# RR artefact handling after R-peak detection (milliseconds / fractions).
HRV_RR_PHYS_MIN_MS: float = 300.0   # > 200 BPM rejected
HRV_RR_PHYS_MAX_MS: float = 2000.0  # < 30 BPM rejected
HRV_RR_MEDIAN_DEV_FRAC: float = 0.20  # ectopic: replace if |RR − med| > 20% of local median
HRV_RR_ROLLING_HALF_WIDTH: int = 3    # median window = 2 * half_width + 1 beats

# RMSSD sliding window duration (seconds) — legacy name; matches HRV_ECG_WINDOW_SEC.
RMSSD_WINDOW_SECONDS: int = 30

# Minimum number of RR intervals required before RMSSD is meaningful.
RMSSD_MIN_SAMPLES: int = 2

# Deprecated: per-beat filter (replaced by window RR cleaning). Kept for DB migrations / docs.
HRV_MIN_RR_MS: float = 500.0

# Minimum continuous ECG at calibration end for neurokit2 baseline (seconds).
CALIBRATION_MIN_ECG_SECONDS: float = 10.0

# Pupil blink rejection: max drop in pixels per sample.
# Tune after testing on real hardware. Start at 20.
PUPIL_BLINK_VELOCITY_THRESHOLD_PX: float = 20.0

# Pupil outlier clamp: discard if |PDI| > this fraction.
# 0.40 = 40 % change from baseline — physiological upper bound.
PUPIL_PDI_OUTLIER_CLAMP: float = 0.40

# Calibration baseline recording duration (seconds).
CALIBRATION_DURATION_SECONDS: int = 20

# Minimum RR intervals required for accepting an RMSSD calibration baseline.
# For a 20 s window this should stay lenient enough for normal resting HR.
CALIBRATION_MIN_RR_INTERVALS: int = 15

# ---------------------------------------------------------------------------
# Cognitive Load Index (CLI)
# ---------------------------------------------------------------------------

# Weights must sum to 1.0.
CLI_WEIGHT_RMSSD: float = 0.5
CLI_WEIGHT_PDI: float = 0.5

# Alert zone thresholds (CLI range 0.0 – 1.0).
CLI_THRESHOLD_LOW: float = 0.33     # green  → yellow boundary
CLI_THRESHOLD_HIGH: float = 0.66    # yellow → red boundary

# Adaptive pupil-workload thresholding (live dashboard).
WORKLOAD_BASELINE_SECONDS: float = 5.0
WORKLOAD_PUPIL_SMOOTHING_SECONDS: float = 1.0
WORKLOAD_PUPIL_ROLLING_SECONDS: float = 30.0
WORKLOAD_THRESHOLD_FACTOR: float = 0.997
WORKLOAD_STATE_PERSIST_SECONDS: float = 0.2

# Minimum time between two accepted hardware error events (wire contacts).
ERROR_EVENT_DEBOUNCE_SECONDS: float = 0.5

# Pi Pico wall-contact flood protection.
PICO_WALL_CONTACT_MIN_EVENT_INTERVAL_SECONDS: float = 0.5
PICO_WALL_CONTACT_LOG_THROTTLE_SECONDS: float = 5.0

# ---------------------------------------------------------------------------
# Mock Sensor — development / testing
# ---------------------------------------------------------------------------

# Simulated RR interval range (milliseconds).
MOCK_RR_MIN_MS: float = 700.0
MOCK_RR_MAX_MS: float = 1000.0

# Simulated pupil diameter range (pixels/camera units).
MOCK_PUPIL_MIN_PX: float = 80.0
MOCK_PUPIL_MAX_PX: float = 160.0

# How often mock sensors emit new data (milliseconds between signals).
MOCK_EMIT_INTERVAL_MS: int = 1000   # 1 Hz for HRV
MOCK_PUPIL_INTERVAL_MS: int = 100   # 10 Hz for pupil

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH: str = "biotrace.db"

# ---------------------------------------------------------------------------
# Learning Curves
# ---------------------------------------------------------------------------

SCORE_MAX: int = 100.00          # Maximum possible performance score per session
LC_MIN_SESSIONS: int = 5     # Minimum sessions with error data before curve is fitted
