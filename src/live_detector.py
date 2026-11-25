"""Real-time vehicle detection (YOLOv8) and online 128-bin HSV histogram extraction for raw surveillance video."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch

from src.events import box_iou_cxcywh
from src.io_mot import center_in_ignored

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the repo is unpacked


# COCO vehicle class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_FOUR_WHEEL_CLASSES = (2, 5, 7)
COCO_ALL_VEHICLE_CLASSES = (2, 3, 5, 7)


def extract_online_hsv_hists(frame_bgr: np.ndarray, boxes_cxcywh: np.ndarray) -> np.ndarray:
    """Compute L1-normalized 128-bin (8x4x4) HSV histograms from the inner 0.6w x 0.6h crop of each box.
    Matches the exact histogram quantization used to train GapLink-Lite (`scripts/extract_hist.py`).
    """
    K = len(boxes_cxcywh)
    if K == 0 or frame_bgr is None or frame_bgr.size == 0:
        return np.zeros((K, 128), dtype=np.float32)

    H_im, W_im = frame_bgr.shape[:2]
    hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out = np.zeros((K, 128), dtype=np.float32)

    for i in range(K):
        cx, cy, w, h = [float(v) for v in boxes_cxcywh[i]]
        iw = max(2.0, 0.6 * w)
        ih = max(2.0, 0.6 * h)
        x1 = int(np.clip(round(cx - 0.5 * iw), 0, W_im - 1))
        y1 = int(np.clip(round(cy - 0.5 * ih), 0, H_im - 1))
        x2 = int(np.clip(round(cx + 0.5 * iw), x1 + 1, W_im))
        y2 = int(np.clip(round(cy + 0.5 * ih), y1 + 1, H_im))
        crop = hsv_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
        s = float(hist.sum())
        if s > 1e-6:
            q = np.clip(np.round((hist / s) * 255.0), 0, 255).astype(np.float32)
            qs = float(q.sum())
            if qs > 1e-6:
                out[i] = q / qs
    return out


def class_agnostic_nms_cxcywh(
    boxes_cxcywh: np.ndarray,
    scores: np.ndarray,
    iou_thr: float = 0.55,
) -> np.ndarray:
    """Return kept indices after greedy class-agnostic NMS (prevents double car+truck boxes on vans/buses)."""
    if len(boxes_cxcywh) == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou_cxcywh(boxes_cxcywh[i : i + 1], boxes_cxcywh[rest])[0]
        order = rest[ious < iou_thr]
    return np.array(keep, dtype=np.int32)


class LiveVehicleDetector:
    """Wrapper around Ultralytics YOLOv8 for real-time surveillance vehicle detection + online HSV extraction."""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | None = None,
        conf_thr: float = 0.28,
        nms_iou: float = 0.52,
        imgsz: int = 640,
        min_box_h: float = 14.0,
        include_motorcycles: bool = False,
    ):
        from ultralytics import YOLO

        if weights is None:
            default_rel = ROOT / "release" / "yolov8n.pt"
            weights = default_rel if default_rel.exists() else "yolov8n.pt"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(weights))
        self.conf_thr = float(conf_thr)
        self.nms_iou = float(nms_iou)
        self.imgsz = int(imgsz)
        self.min_box_h = float(min_box_h)
        self.classes = list(COCO_ALL_VEHICLE_CLASSES if include_motorcycles else COCO_FOUR_WHEEL_CLASSES)

    def _postprocess_one(
        self,
        res,
        frame_bgr: np.ndarray,
        ignore_xywh: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if res.boxes is None or len(res.boxes) == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "cls_ids": np.zeros((0,), dtype=np.int32),
                "hists": np.zeros((0, 128), dtype=np.float32),
            }

        xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls_ids = res.boxes.cls.detach().cpu().numpy().astype(np.int32)

        w = np.maximum(2.0, xyxy[:, 2] - xyxy[:, 0])
        h = np.maximum(2.0, xyxy[:, 3] - xyxy[:, 1])
        cx = xyxy[:, 0] + 0.5 * w
        cy = xyxy[:, 1] + 0.5 * h
        boxes = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

        valid = (h >= self.min_box_h) & (w >= 10.0)
        if ignore_xywh is not None and len(ignore_xywh) > 0:
            xywh = np.stack([xyxy[:, 0], xyxy[:, 1], w, h], axis=1)
            in_ig = center_in_ignored(xywh, ignore_xywh)
            valid = valid & (~in_ig)

        boxes = boxes[valid]
        scores = scores[valid]
        cls_ids = cls_ids[valid]

        if len(boxes) > 1:
            keep = class_agnostic_nms_cxcywh(boxes, scores, iou_thr=self.nms_iou)
            boxes = boxes[keep]
            scores = scores[keep]
            cls_ids = cls_ids[keep]

        hists = extract_online_hsv_hists(frame_bgr, boxes)
        return {
            "boxes": boxes,
            "scores": scores,
            "cls_ids": cls_ids,
            "hists": hists,
        }

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        ignore_xywh: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run YOLOv8 vehicle detection and online HSV extraction on a single BGR frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "cls_ids": np.zeros((0,), dtype=np.int32),
                "hists": np.zeros((0, 128), dtype=np.float32),
            }

        res = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thr,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            classes=self.classes,
            agnostic_nms=True,
            device=self.device,
            verbose=False,
        )[0]
        return self._postprocess_one(res, frame_bgr, ignore_xywh=ignore_xywh)

    def detect_batch(
        self,
        frames_bgr: list[np.ndarray],
        ignore_xywh: np.ndarray | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Batch GPU inference producing identical per-frame outputs as detect_frame for fast benchmark runs."""
        if not frames_bgr:
            return []
        results = self.model.predict(
            source=frames_bgr,
            conf=self.conf_thr,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            classes=self.classes,
            agnostic_nms=True,
            device=self.device,
            verbose=False,
        )
        return [
            self._postprocess_one(res, frm, ignore_xywh=ignore_xywh)
            for res, frm in zip(results, frames_bgr)
        ]
