# BioTrace Methods Summary (Hardware + Signal Logic)

## 1. System Objective
BioTrace estimates **physiological stress** and **cognitive workload** during a simulated task by combining:
- ECG-derived heart-rate variability (RMSSD)
- Video-based pupil dilation dynamics
- Event-level task errors (wire contacts)

The implementation is designed for **real-time feedback**, **session-level interpretability**, and **post-session reproducibility**.

## 2. Hardware and Acquisition Stack

### ECG (physical stress channel)
- Device: Raspberry Pi Pico streaming YLab Zero ECG samples over USB serial.
- Sampling rate: `150 Hz`.
- Port auto-detection supports Pico-compatible USB VID/device descriptions.
- Core module: `app/hardware/pico_ecg_sensor.py`.

### Eye tracker (cognitive workload channel)
- Device: dedicated USB camera.
- Pupil detection: PuRe (`pypupilext`) when available; OpenCV fallback otherwise.
- Output: monocular pupil diameter in pixels (timestamped).
- Core module: `app/hardware/eye_tracker.py`.

### Task error channel
- Hardware wall-contact events are debounced before counting.
- Debounce: `0.5 s` minimum between accepted events.
- Core flow: `app/core/session.py`.

## 3. Calibration Protocol
A baseline phase is run before each session:
- Duration: `20 s` (`CALIBRATION_DURATION_SECONDS`)
- RMSSD baseline accepted only if at least `15` RR intervals are collected (`CALIBRATION_MIN_RR_INTERVALS`)
- Pupil baseline: mean pupil diameter over calibration samples

This baseline anchors all percent-change computations and keeps interpretation participant-specific.

## 4. Signal Processing Pipeline

### 4.1 ECG -> RR intervals -> RMSSD
1. Raw ECG samples are baseline-corrected with a slow EMA.
2. Adaptive threshold + refractory-period peak detector identifies R-peaks.
3. RR interval is computed from peak spacing.
4. RMSSD is computed over a 30-second sliding window.

RMSSD equation:

`RMSSD = sqrt(mean((RR[i+1] - RR[i])^2))`

### 4.2 Pupil diameter -> PDI
1. Left/right pupil diameters are averaged (when valid).
2. Blink artifacts are rejected using velocity threshold.
3. Extreme outliers are clamped.
4. Pupil Dilation Index (PDI) is computed relative to calibration baseline.

PDI equation:

`PDI = (current_diameter - baseline_diameter) / baseline_diameter`

## 5. Live Metrics and Equations

### 5.1 Percent change from baseline (timeline axis)
For stress (RMSSD), the live timeline uses:

`percent_delta = ((value - baseline_mean) / baseline_mean) * 100`

For pupil and threshold traces, PDI ratio values are shown as percent by multiplying by 100.

### 5.2 Stress % (gauge)
From RMSSD percent change:

`stress_pct = clamp(-percent_delta, 0, 100)`

Interpretation:
- RMSSD below baseline -> positive stress%
- RMSSD above baseline -> lower stress%

### 5.3 High vs Low Cognitive Load (adaptive)
Workload is not based on a fixed absolute pupil cutoff. It uses adaptive thresholding:
1. Smooth PDI over `1 s`
2. Compute rolling mean over `30 s`
3. Threshold = `rolling_mean * 0.997`
4. State = `HIGH` if `smoothed_pdi > threshold`, else `LOW`
5. A persistence rule (`0.2 s`) prevents rapid label flicker

Important clarification for presentation:
- The HIGH/LOW workload threshold is **continuously changing** during the session.
- The current implementation does **not** use a fixed `20%` cutoff for workload state.
- The `40%` value is used as an **outlier rejection clamp** (`|PDI| > 0.40`), not as the HIGH/LOW boundary.

## 6. Post-Session Event Definitions

### Stress events
Using RMSSD percent change from baseline:
- Stress event: crossing below `-10%`
- Severe stress event: crossing below `-40%`
- Counted on **crossings** (state transitions), not per-sample occupancy

### High workload events
- Counted when smoothed pupil signal crosses from `<= threshold` to `> threshold`
- Uses same smoothing/rolling/threshold logic as live view

## 7. Why These Code/Algorithm Choices

1. **Deterministic, low-latency processing**
- Threshold and windowed methods are transparent and stable for real-time UI.
- Easier to validate clinically/experimentally than black-box estimators.

2. **Participant-normalized metrics**
- Baseline-referenced formulas reduce inter-subject variability.
- Session-adaptive workload thresholding avoids brittle fixed pupil cutoffs.

3. **Artifact robustness**
- ECG refractory logic and RR plausibility filters reduce double-counting/noise.
- Blink-velocity rejection and PDI outlier clamp improve pupil signal quality.

4. **Interpretability for training context**
- % change from baseline, threshold crossings, and explicit HIGH/LOW states are straightforward for trainees and faculty to interpret during debrief.

## 8. Threshold Selection Rationale

- `PICO_RPEAK_THRESHOLD_FACTOR = 0.65`: chosen to separate R-peaks from lower-amplitude wave components (e.g., T-wave contamination risk).
- `PICO_RPEAK_REFRACTORY_SAMPLES = 90` at 150 Hz (600 ms): suppresses double detection in rapid succession.
- `PUPIL_BLINK_VELOCITY_THRESHOLD_PX = 20`: practical blink/noise gate for per-frame pixel jumps.
- `PUPIL_PDI_OUTLIER_CLAMP = 0.40`: removes physiologically implausible >40% instantaneous relative changes.
- `WORKLOAD_THRESHOLD_FACTOR = 0.997` with rolling windows: intentionally near local baseline to detect subtle sustained upward shifts while persistence filtering controls false positives.
- Stress event cutoffs (`-10%`, `-40%`): two-tier event stratification for moderate vs severe baseline-relative RMSSD suppression.

## 9. Reproducibility Notes
- All constants are centralized in `app/utils/config.py`.
- Processing code is separated by modality (`hrv_processor.py`, `pupil_processor.py`, `cli_processor.py`).
- Session lifecycle, calibration gating, and persistence are centralized in `app/core/session.py`.

This architecture supports clear parameter auditing and straightforward methods reporting for presentations and manuscripts.

## 10. External Resource Used
- `/Users/razhanr/Downloads/EBSCO-FullText-04_12_2026.pdf`

How it is used in this methods narrative:
- As supporting literature context for pupil-dilation-driven workload interpretation.
- The implemented thresholds and equations above remain the **source-of-truth from code** for this project.
