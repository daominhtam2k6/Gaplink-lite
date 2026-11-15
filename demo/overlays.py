"""OpenCV drawing utilities for the side-by-side GapLink-Lite vs C-3 demo videos."""
import math
import cv2
import numpy as np

# Palette (BGR for OpenCV)
PALETTE = [
    (237, 179, 99),   # Sky Blue
    (85, 173, 246),   # Warm Orange
    (145, 211, 104),  # Mint Green
    (129, 129, 252),  # Coral Red
    (244, 148, 183),  # Soft Purple
    (179, 135, 246),  # Pink
    (209, 206, 0),    # Cyan
    (94, 219, 250),   # Gold
]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return PALETTE[abs(int(track_id)) % len(PALETTE)]


def draw_occlusion_zone(img: np.ndarray, box_cxcywh: np.ndarray, label: str = "OCCLUDED (AN)"):
    """Draw a semi-transparent dark-red occlusion overlay over the hidden ground-truth box."""
    h_img, w_img = img.shape[:2]
    cx, cy, w, h = [float(v) for v in box_cxcywh]
    x1 = int(max(0, round(cx - 0.5 * w)))
    y1 = int(max(0, round(cy - 0.5 * h)))
    x2 = int(min(w_img - 1, round(cx + 0.5 * w)))
    y2 = int(min(h_img - 1, round(cy + 0.5 * h)))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    tint = np.zeros_like(roi)
    tint[:, :] = (40, 40, 180)
    cv2.addWeighted(roi, 0.45, tint, 0.55, 0.0, dst=roi)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 240), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (x1 + 3, max(14, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 255), 1, cv2.LINE_AA)


def draw_track_box(img: np.ndarray, box_cxcywh: np.ndarray, track_id: int, is_recovered: bool = False, is_wrong: bool = False):
    cx, cy, w, h = [float(v) for v in box_cxcywh]
    x1 = int(round(cx - 0.5 * w))
    y1 = int(round(cy - 0.5 * h))
    x2 = int(round(cx + 0.5 * w))
    y2 = int(round(cy + 0.5 * h))
    col = color_for_id(track_id)
    thick = 2
    tag = f"ID {track_id}"
    if is_recovered:
        col = (80, 245, 120)
        thick = 3
        tag = f"ID {track_id} [RECOVERED]"
    elif is_wrong:
        col = (60, 60, 255)
        thick = 3
        tag = f"ID {track_id} [ID-SWITCH]"

    cv2.rectangle(img, (x1, y1), (x2, y2), col, thick, lineType=cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    by1 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, by1), (x1 + tw + 6, by1 + th + 5), col, -1)
    cv2.putText(img, tag, (x1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (15, 15, 20), 1, cv2.LINE_AA)


def draw_ghost_and_ellipse(img: np.ndarray, pred_box: np.ndarray, sigma_xy: np.ndarray, track_id: int, gap: int):
    """Draw dashed ghost box and 90% confidence ellipse (2.146 * sigma_x, 2.146 * sigma_y)."""
    cx, cy, w, h = [float(v) for v in pred_box]
    sx = max(2.0, float(sigma_xy[0]))
    sy = max(2.0, float(sigma_xy[1]))
    rx = int(round(2.146 * sx))
    ry = int(round(2.146 * sy))
    center = (int(round(cx)), int(round(cy)))
    col = color_for_id(track_id)

    # Semi-transparent ellipse fill
    overlay = img.copy()
    cv2.ellipse(overlay, center, (rx, ry), 0.0, 0.0, 360.0, col, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.20, img, 0.80, 0.0, dst=img)
    cv2.ellipse(img, center, (rx, ry), 0.0, 0.0, 360.0, col, 1, lineType=cv2.LINE_AA)

    # Ghost center crosshair
    cv2.drawMarker(img, center, col, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
    cv2.putText(img, f"G:{track_id}(g={gap})", (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


def draw_hud_banner(img: np.ndarray, title: str, frame_id: int, stats: dict):
    """Draw top HUD bar with method name and live cumulative counters."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0.0, dst=img)
    txt = (
        f"{title} | Frame {frame_id:04d} | "
        f"Recovered: {stats.get('rec', 0)}/{stats.get('tot', 0)} | "
        f"WrongLink: {stats.get('wrong', 0)} | NewID(Frag): {stats.get('frag', 0)}"
    )
    cv2.putText(img, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 245, 255), 1, cv2.LINE_AA)
