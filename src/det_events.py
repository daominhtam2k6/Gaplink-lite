"""Detector-aware training events (v1.3): log the exact events the live tracker builds on YOLOv8n boxes.

On TRAIN (and val) sequences only, YOLO boxes are matched post hoc to ground truth (IoU >= 0.45) to obtain
identity labels. The tracker is driven by an oracle associator (teacher forcing) so tracks stay clean, and
every re-association event it builds is logged with labels, so GapLink-Lite sees at training time the same
input distribution as in deployment: detector jitter, 2..5-frame flicker gaps, false-positive candidates.
The test split is never used here.
"""
from __future__ import annotations

import numpy as np

from src.events import event_is_hard
from src.tracker import OnlineTracker


def label_from_gt(trk_gt: np.ndarray, cand_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """label[i] = candidate index with the same GT id as lost track i (or -1); cand_label is the inverse map."""
    N, M = len(trk_gt), len(cand_gt)
    label = np.full((N,), -1, dtype=np.int32)
    cand_label = np.full((M,), -1, dtype=np.int32)
    for i, g in enumerate(trk_gt):
        if g < 0:
            continue
        for j in range(M):
            if cand_gt[j] == g and cand_label[j] < 0:
                label[i], cand_label[j] = j, i
                break
    return label, cand_label


class OracleAssociator:
    """Links exactly the GT-consistent pairs (training-data generation only)."""
    name = "oracle"

    def decide(self, ev: dict) -> np.ndarray:
        return np.asarray(ev.get("_oracle_label", np.full((len(ev["t_lost"]),), -1)), dtype=np.int32).copy()


class LoggingTracker(OnlineTracker):
    def __init__(self, *a, hard_factor: float = 1.15, **kw):
        super().__init__(*a, **kw)
        self.logged: list[dict] = []
        self.hard_factor = hard_factor
        self._cur_boxes = np.zeros((0, 4), dtype=np.float32)
        self._cur_gt = np.zeros((0,), dtype=np.int32)

    def step(self, frame_id, det_boxes, det_hists, det_gt_tids=None):
        self._cur_boxes = np.asarray(det_boxes, dtype=np.float32)
        self._cur_gt = (np.asarray(det_gt_tids, dtype=np.int32) if det_gt_tids is not None
                        else np.full((len(det_boxes),), -1, dtype=np.int32))
        return super().step(frame_id, det_boxes, det_hists, det_gt_tids)

    def _cand_gt(self, cand_boxes: np.ndarray) -> np.ndarray:
        out = np.full((len(cand_boxes),), -1, dtype=np.int32)
        for j, b in enumerate(cand_boxes):
            hit = np.nonzero(np.all(self._cur_boxes == b[None, :], axis=1))[0]
            if len(hit):
                out[j] = self._cur_gt[hit[0]]
        return out

    def build_event_from_subpools(self, frame_id, sub_lost, sub_cand_boxes, sub_cand_hists, vis_boxes_arr):
        ev = super().build_event_from_subpools(frame_id, sub_lost, sub_cand_boxes, sub_cand_hists, vis_boxes_arr)
        trk_gt = np.array([t.last_gt_tid for t in sub_lost], dtype=np.int32)
        cand_gt = self._cand_gt(np.asarray(sub_cand_boxes, dtype=np.float32))
        label, cand_label = label_from_gt(trk_gt, cand_gt)
        ev["_oracle_label"] = label
        if np.all(trk_gt >= 0):  # only fully labelled events are kept for training
            N = len(label)
            true_box = np.zeros((N, 4), dtype=np.float32)
            has_t = label >= 0
            true_box[has_t] = ev["cand"][label[has_t]]  # target = the YOLO box of the same vehicle
            lev = {k: v for k, v in ev.items() if k != "_oracle_label"}
            lev.update({"label": label.copy(), "cand_label": cand_label, "true_box": true_box,
                        "has_true_target": has_t, "is_hard": bool(event_is_hard(ev["cand"], label, self.hard_factor)),
                        "side": np.zeros((N,), dtype=np.int8), "mode": "det", "source": "yolo"})
            self.logged.append(lev)
        return ev
