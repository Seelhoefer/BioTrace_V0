#!/usr/bin/env python3
"""Live pupil-detection debug viewer.

Run from the project root **without** the full Qt app:

    python tools/pupil_debug.py

Press keys while the window is focused:
    q / Esc  — quit
    d        — toggle darkness-map overlay (shows which regions pass the gate)
    t        — toggle threshold mask overlay
    +/-      — increase / decrease DARKNESS_RATIO by 0.05
    [/]      — increase / decrease MIN_CIRCULARITY by 0.05

The window title always shows the current tuning values and the last measured
diameter so you can adjust and see the effect in real time.
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

# Allow running from project root without pip-installing the package.
sys.path.insert(0, ".")

from app.utils.config import (  # noqa: E402
    EYE_TRACKER_CAMERA_INDEX,
    EYE_PUPIL_DETECTION_ZOOM,
    PUPIL_DARKNESS_RATIO,
    PUPIL_MIN_CIRCULARITY,
    PUPIL_MIN_ASPECT,
    PUPIL_MIN_AREA_FRAC,
    PUPIL_MAX_AREA_FRAC,
    PUPIL_MIN_AREA_PX,
    PUPIL_SCORE_W_CIRCULARITY,
    PUPIL_SCORE_W_ASPECT,
    PUPIL_SCORE_W_DARKNESS,
    PUPIL_SCORE_W_CENTER,
    PUPIL_SCORE_W_CONSISTENCY,
)

# ---------------------------------------------------------------------------
# Tunable parameters (modified interactively via key presses)
# ---------------------------------------------------------------------------
_params: dict[str, float] = {
    "darkness_ratio":   float(PUPIL_DARKNESS_RATIO),
    "min_circularity":  float(PUPIL_MIN_CIRCULARITY),
    "min_aspect":       float(PUPIL_MIN_ASPECT),
    "min_area_frac":    float(PUPIL_MIN_AREA_FRAC),
    "max_area_frac":    float(PUPIL_MAX_AREA_FRAC),
    "min_area_px":      float(PUPIL_MIN_AREA_PX),
}

_show_darkness = False
_show_thresh   = False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect(
    gray_roi: np.ndarray,
    params: dict[str, float],
    frame_w: int,
    frame_h: int,
    roi_x1: int,
    roi_y1: int,
) -> tuple[dict | None, np.ndarray, float, np.ndarray]:
    """Run detection and return (best_candidate | None, thresh_mask, frame_median, blurred_roi)."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    frame_median = float(np.median(blurred))
    darkness_cutoff = frame_median * params["darkness_ratio"]

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_h, roi_w = gray_roi.shape
    img_area = float(roi_h * roi_w)
    min_area = max(params["min_area_px"], img_area * params["min_area_frac"])
    max_area = img_area * params["max_area_frac"]
    frame_center = (float(frame_w) / 2.0, float(frame_h) / 2.0)

    best: dict | None = None
    best_score = -1.0
    rejected: list[dict] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        perim = float(cv2.arcLength(cnt, True))
        if perim <= 0:
            continue
        circularity = float((4.0 * np.pi * area) / (perim * perim))
        if circularity < params["min_circularity"]:
            rejected.append({"cnt": cnt, "reason": f"circ={circularity:.2f}"})
            continue

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx_l, cy_l), (axis_a, axis_b), angle = ellipse
            major = float(max(axis_a, axis_b))
            minor = float(min(axis_a, axis_b))
            if major <= 0 or minor <= 0:
                continue
            aspect = minor / major
            diameter = (major + minor) / 2.0
            cx = float(cx_l + roi_x1)
            cy = float(cy_l + roi_y1)
            ell = ((cx, cy), (float(ellipse[1][0]), float(ellipse[1][1])), float(angle))
        else:
            (cx_l, cy_l), radius = cv2.minEnclosingCircle(cnt)
            diameter = float(radius) * 2.0
            aspect = 1.0
            cx = float(cx_l + roi_x1)
            cy = float(cy_l + roi_y1)
            ell = ((cx, cy), (diameter, diameter), 0.0)

        if aspect < params["min_aspect"]:
            rejected.append({"cnt": cnt, "reason": f"asp={aspect:.2f}"})
            continue

        cnt_mask = np.zeros(blurred.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        mean_int = float(cv2.mean(blurred, mask=cnt_mask)[0])
        if mean_int >= darkness_cutoff:
            rejected.append({"cnt": cnt, "reason": f"bright={mean_int:.0f}>={darkness_cutoff:.0f}"})
            continue

        darkness = max(0.0, 1.0 - mean_int / max(1.0, frame_median))
        dist = float(np.hypot(cx - frame_center[0], cy - frame_center[1]))
        center_score = 1.0 - min(1.0, dist / max(1.0, np.hypot(frame_w, frame_h)))

        score = (
            float(PUPIL_SCORE_W_CIRCULARITY) * circularity
            + float(PUPIL_SCORE_W_ASPECT) * aspect
            + float(PUPIL_SCORE_W_DARKNESS) * darkness
            + float(PUPIL_SCORE_W_CENTER) * center_score
        )

        candidate = {
            "cx": cx, "cy": cy, "diameter": diameter,
            "circularity": circularity, "aspect": aspect,
            "darkness": darkness, "mean_int": mean_int,
            "frame_median": frame_median, "score": score,
            "ellipse": ell, "cnt": cnt,
            "area": area,
        }
        if score > best_score:
            best_score = score
            best = candidate

    return best, mask, frame_median, blurred, rejected  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

CYAN   = (255, 200,  50)
GREEN  = ( 50, 220,  80)
RED    = ( 50,  50, 255)
YELLOW = ( 40, 200, 220)
GRAY   = (160, 160, 160)


def _put(frame: np.ndarray, text: str, row: int, color=GRAY) -> None:
    cv2.putText(frame, text, (8, 18 + row * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def _annotate(
    frame: np.ndarray,
    best: dict | None,
    rejected: list[dict],
    mask: np.ndarray,
    blurred_roi: np.ndarray,
    roi_x1: int, roi_y1: int,
    frame_median: float,
    params: dict[str, float],
    show_darkness: bool,
    show_thresh: bool,
    fps: float,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Darkness overlay (red channel)
    if show_darkness:
        brightness_frac = (blurred_roi.astype(np.float32) / max(1.0, frame_median))
        dark_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        roi_h, roi_w = blurred_roi.shape
        dark_map = np.clip(255 - (brightness_frac * 128).astype(np.uint8), 0, 255)
        dark_map_rgb = cv2.applyColorMap(dark_map, cv2.COLORMAP_JET)
        dark_overlay[roi_y1:roi_y1 + roi_h, roi_x1:roi_x1 + roi_w] = dark_map_rgb
        frame = cv2.addWeighted(frame, 0.6, dark_overlay, 0.4, 0)

    # Threshold overlay
    if show_thresh:
        thresh_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        roi_h, roi_w = mask.shape
        thresh_rgb[roi_y1:roi_y1 + roi_h, roi_x1:roi_x1 + roi_w, 2] = mask  # red channel
        frame = cv2.addWeighted(frame, 0.7, thresh_rgb, 0.3, 0)

    # Rejected blobs (thin red)
    for r in rejected:
        cnt = r["cnt"].copy()
        cnt[:, :, 0] += roi_x1
        cnt[:, :, 1] += roi_y1
        cv2.drawContours(frame, [cnt], -1, RED, 1, cv2.LINE_AA)

    # Best detection
    if best is not None:
        cx, cy = int(best["cx"]), int(best["cy"])
        d = best["diameter"]
        ell = best["ellipse"]

        # Ellipse
        e_center = (int(ell[0][0]), int(ell[0][1]))
        e_axes = (max(1, int(ell[1][0] / 2)), max(1, int(ell[1][1] / 2)))
        cv2.ellipse(frame, e_center, e_axes, float(ell[2]), 0, 360, GREEN, 2, cv2.LINE_AA)

        # Centre dot
        cv2.circle(frame, (cx, cy), 3, GREEN, -1, cv2.LINE_AA)

        # Diameter circle
        cv2.circle(frame, (cx, cy), max(1, int(d / 2)), CYAN, 1, cv2.LINE_AA)

        _put(frame, f"d={d:.1f}px  circ={best['circularity']:.2f}  asp={best['aspect']:.2f}", 0, GREEN)
        _put(frame, f"dark={best['darkness']:.2f}  blob={best['mean_int']:.0f}  median={frame_median:.0f}", 1, GREEN)
        _put(frame, f"score={best['score']:.2f}  area={best['area']:.0f}", 2, GREEN)
    else:
        _put(frame, "NO PUPIL DETECTED", 0, RED)
        _put(frame, f"frame median={frame_median:.0f}", 1, GRAY)

    # Param legend (bottom)
    _put(frame, f"dark_ratio={params['darkness_ratio']:.2f} (+/-)   circ>={params['min_circularity']:.2f} ([ ])   asp>={params['min_aspect']:.2f}", h // 20 * 19 - 40, YELLOW)
    _put(frame, f"FPS={fps:.0f}  D=darkness  T=threshold  Q=quit", h // 20 * 19 - 20, GRAY)

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(EYE_TRACKER_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {EYE_TRACKER_CAMERA_INDEX}")
        sys.exit(1)

    print(f"Camera {EYE_TRACKER_CAMERA_INDEX} opened. Press Q or Esc to quit.")
    print("Keys: +/- darkness_ratio  [/] min_circularity  D=darkness map  T=threshold mask")

    global _show_darkness, _show_thresh  # noqa: PLW0603

    zoom = max(1.0, float(EYE_PUPIL_DETECTION_ZOOM))
    frame_times: list[float] = []
    prev_d: float | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        t0 = time.monotonic()
        gray_native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_nat, w_nat = gray_native.shape

        # Apply zoom (centre-crop + resize)
        zoom_x1, zoom_y1 = 0, 0
        if zoom > 1.0:
            crop_w = max(16, int(round(w_nat / zoom)))
            crop_h = max(16, int(round(h_nat / zoom)))
            zoom_x1 = max(0, (w_nat - crop_w) // 2)
            zoom_y1 = max(0, (h_nat - crop_h) // 2)
            zoom_x2 = min(w_nat, zoom_x1 + crop_w)
            zoom_y2 = min(h_nat, zoom_y1 + crop_h)
            gray = cv2.resize(gray_native[zoom_y1:zoom_y2, zoom_x1:zoom_x2],
                              (w_nat, h_nat), interpolation=cv2.INTER_LINEAR)
        else:
            gray = gray_native

        # ROI (80% of frame, centred)
        roi_frac = 0.80
        roi_w = max(32, int(w_nat * roi_frac))
        roi_h = max(32, int(h_nat * roi_frac))
        rx1 = max(0, (w_nat - roi_w) // 2)
        ry1 = max(0, (h_nat - roi_h) // 2)
        rx2 = min(w_nat, rx1 + roi_w)
        ry2 = min(h_nat, ry1 + roi_h)
        gray_roi = gray[ry1:ry2, rx1:rx2]

        best, mask, frame_median, blurred_roi, rejected = _detect(
            gray_roi, _params, w_nat, h_nat, rx1, ry1
        )

        # Convert zoomed coords back to native
        if best is not None and zoom > 1.0:
            best["cx"]       = zoom_x1 + best["cx"] / zoom
            best["cy"]       = zoom_y1 + best["cy"] / zoom
            best["diameter"] = best["diameter"] / zoom
            ell = best["ellipse"]
            best["ellipse"] = (
                (zoom_x1 + ell[0][0] / zoom, zoom_y1 + ell[0][1] / zoom),
                (ell[1][0] / zoom, ell[1][1] / zoom),
                ell[2],
            )
        if rejected and zoom > 1.0:
            for r in rejected:
                c = r["cnt"].astype(np.float32)
                c[:, :, 0] = zoom_x1 + c[:, :, 0] / zoom
                c[:, :, 1] = zoom_y1 + c[:, :, 1] / zoom
                r["cnt"] = c.astype(np.int32)

        frame_times.append(time.monotonic() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / max(1e-6, sum(frame_times) / len(frame_times))

        display = frame.copy()
        display = _annotate(
            display, best, rejected, mask, blurred_roi,
            rx1, ry1, frame_median, _params,
            _show_darkness, _show_thresh, fps,
        )

        d_str = f"{best['diameter']:.1f}px" if best else "none"
        cv2.setWindowTitle(
            "pupil_debug",
            f"Pupil debug  d={d_str}  dark={_params['darkness_ratio']:.2f}  "
            f"circ={_params['min_circularity']:.2f}",
        )
        cv2.imshow("pupil_debug", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            _show_darkness = not _show_darkness
        elif key == ord("t"):
            _show_thresh = not _show_thresh
        elif key == ord("+"):
            _params["darkness_ratio"] = min(0.95, _params["darkness_ratio"] + 0.05)
        elif key == ord("-"):
            _params["darkness_ratio"] = max(0.10, _params["darkness_ratio"] - 0.05)
        elif key == ord("]"):
            _params["min_circularity"] = min(0.95, _params["min_circularity"] + 0.05)
        elif key == ord("["):
            _params["min_circularity"] = max(0.10, _params["min_circularity"] - 0.05)

    cap.release()
    cv2.destroyAllWindows()

    # Print final params so they can be pasted back into config.py
    print("\n--- Final tuning values ---")
    for k, v in _params.items():
        print(f"  {k.upper()}: {v:.4f}")


if __name__ == "__main__":
    main()
