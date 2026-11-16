"""Minimal 2-stage online multi-object tracker for evaluating C-1..C-4 and GapLink-Lite."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.events import EventCfg, assemble_event, build_neighbor_context, box_iou_cxcywh


class TrackState:
    def __init__(self, track_id: int, frame_id: int, box: np.ndarray, hist_vec: np.ndarray, gt_tid: int = -1, k_len: int = 20):
        self.track_id = int(track_id)
        self.k_len = k_len
        self.boxes = [box.astype(np.float32).copy()]
        self.frames = [int(frame_id)]
        self.color_ema = hist_vec.astype(np.float32).copy()
        self.last_frame = int(frame_id)
        self.age = 0
        self.ctx_lost = np.zeros((8, 6), dtype=np.float32)
        self.ctx_mask_lost = np.zeros((8,), dtype=bool)
        self.initial_gt_tid = int(gt_tid)
        self.last_gt_tid = int(gt_tid)

    def update(self, frame_id: int, box: np.ndarray, hist_vec: np.ndarray, gt_tid: int = -1, alpha: float = 0.8):
        self.boxes.append(box.astype(np.float32).copy())
        self.frames.append(int(frame_id))
        if len(self.boxes) > self.k_len:
            self.boxes = self.boxes[-self.k_len :]
            self.frames = self.frames[-self.k_len :]
        if hist_vec.sum() > 1e-6:
            if self.color_ema.sum() > 1e-6:
                self.color_ema = alpha * self.color_ema + (1.0 - alpha) * hist_vec.astype(np.float32)
            else:
                self.color_ema = hist_vec.astype(np.float32).copy()
            s = float(self.color_ema.sum())
            if s > 1e-6:
                self.color_ema /= s
        self.last_frame = int(frame_id)
        self.age = 0
        if gt_tid != -1:
            self.last_gt_tid = int(gt_tid)

    def get_padded_hist(self) -> tuple[np.ndarray, int]:
        L = min(len(self.boxes), self.k_len)
        out = np.zeros((self.k_len, 4), dtype=np.float32)
        out[self.k_len - L : self.k_len] = np.stack(self.boxes[-L:], axis=0)
        return out, L


class OnlineTracker:
    """2-stage online tracker:
    - Stage 1: IoU + Hungarian on active tracks (age == 0) with threshold `iou_thr` (default 0.30).
    - Stage 2: If associator is not None (C-2, C-3, C-4, or GapLink-Lite), match lost tracks (2 <= age <= a_max)
      to unmatched detections via `assemble_event` + `associator.decide(ev)`.
      For C-1 (`associator=None`), Stage 2 is skipped.
    """
    def __init__(
        self,
        associator=None,
        iou_thr: float = 0.30,
        a_max: int = 95,
        W: int = 960,
        H: int = 540,
        seq_name: str = "seq",
    ):
        self.associator = associator
        self.iou_thr = float(iou_thr)
        self.a_max = int(a_max)
        self.W = int(W)
        self.H = int(H)
        self.seq_name = seq_name
        self.cfg = EventCfg()

        self.next_id = 1
        self.active_tracks: list[TrackState] = []
        self.lost_tracks: list[TrackState] = []
        self.reassoc_log: list[dict] = []
        self.frame_outputs: dict[int, list[dict]] = {}

    def build_event_from_subpools(
        self,
        frame_id: int,
        sub_lost: list[TrackState],
        sub_cand_boxes: np.ndarray,
        sub_cand_hists: np.ndarray,
        vis_boxes_arr: np.ndarray,
    ) -> dict:
        N = len(sub_lost)
        K = self.cfg.hist_len
        t_lost = np.zeros((N,), dtype=np.int32)
        hist = np.zeros((N, K, 4), dtype=np.float32)
        hist_len = np.zeros((N,), dtype=np.int32)
        track_color = np.zeros((N, 128), dtype=np.float32)
        ctx_lost = np.zeros((N, self.cfg.n_ctx, 6), dtype=np.float32)
        ctx_conf = np.zeros((N, self.cfg.n_ctx, 6), dtype=np.float32)
        ctx_mask_lost = np.zeros((N, self.cfg.n_ctx), dtype=bool)
        ctx_mask_conf = np.zeros((N, self.cfg.n_ctx), dtype=bool)

        for i, trk in enumerate(sub_lost):
            t_lost[i] = trk.last_frame
            h_pad, h_l = trk.get_padded_hist()
            hist[i] = h_pad
            hist_len[i] = h_l
            track_color[i] = trk.color_ema
            ctx_lost[i] = trk.ctx_lost
            ctx_mask_lost[i] = trk.ctx_mask_lost
            ctx_conf[i], ctx_mask_conf[i] = build_neighbor_context(
                trk.boxes[-1], vis_boxes_arr, 1.0, self.cfg.n_ctx
            )

        return assemble_event(
            seq_name=self.seq_name,
            t_lost=t_lost,
            t_conf=frame_id,
            hist=hist,
            hist_len=hist_len,
            track_color=track_color,
            cand=sub_cand_boxes,
            cand_hist=sub_cand_hists,
            ctx_lost=ctx_lost,
            ctx_conf=ctx_conf,
            ctx_mask_lost=ctx_mask_lost,
            ctx_mask_conf=ctx_mask_conf,
            W=self.W,
            H=self.H,
        )

    def step(self, frame_id: int, det_boxes: np.ndarray, det_hists: np.ndarray, det_gt_tids: np.ndarray | None = None):
        K = len(det_boxes)
        if det_gt_tids is None:
            det_gt_tids = np.full((K,), -1, dtype=np.int32)

        # Stage 1: IoU matching between active_tracks and current detections
        matched_act = {}
        matched_det = set()
        if len(self.active_tracks) > 0 and K > 0:
            act_boxes = np.stack([t.boxes[-1] for t in self.active_tracks], axis=0)
            ious = box_iou_cxcywh(act_boxes, det_boxes)
            r_ind, c_ind = linear_sum_assignment(-ious)
            for r, c in zip(r_ind, c_ind):
                if ious[r, c] >= self.iou_thr:
                    matched_act[int(r)] = int(c)
                    matched_det.add(int(c))

        # Snapshot active neighbor boxes at previous frame for newly lost tracks
        prev_vis_boxes = np.stack([t.boxes[-1] for t in self.active_tracks], axis=0) if self.active_tracks else np.zeros((0, 4), dtype=np.float32)

        new_active = []
        newly_lost = []
        for r_idx, trk in enumerate(self.active_tracks):
            if r_idx in matched_act:
                c = matched_act[r_idx]
                trk.update(frame_id, det_boxes[c], det_hists[c], int(det_gt_tids[c]))
                new_active.append(trk)
            else:
                trk.age = frame_id - trk.last_frame
                trk.ctx_lost, trk.ctx_mask_lost = build_neighbor_context(
                    trk.boxes[-1], prev_vis_boxes, 0.0, self.cfg.n_ctx
                )
                newly_lost.append(trk)

        vis_boxes_now = np.stack([t.boxes[-1] for t in new_active], axis=0) if new_active else np.zeros((0, 4), dtype=np.float32)

        eligible_lost = []
        for trk in self.lost_tracks:
            trk.age = frame_id - trk.last_frame
            if 2 <= trk.age <= self.a_max:
                eligible_lost.append(trk)

        unmatched_det_indices = [c for c in range(K) if c not in matched_det]

        reassoc_matched_lost = set()
        reassoc_matched_det = set()

        if self.associator is not None and len(eligible_lost) > 0 and len(unmatched_det_indices) > 0:
            eligible_lost.sort(key=lambda t: t.age)
            n_max = self.cfg.n_trk_max
            m_max = self.cfg.n_cand_max
            cand_indices = unmatched_det_indices[:m_max]

            for start_l in range(0, len(eligible_lost), n_max):
                sub_lost = eligible_lost[start_l : start_l + n_max]
                avail_cand_pos = [j for j in range(len(cand_indices)) if cand_indices[j] not in reassoc_matched_det]
                if not avail_cand_pos:
                    break
                sub_cand_idx = [cand_indices[j] for j in avail_cand_pos]
                sub_cand_boxes = det_boxes[sub_cand_idx]
                sub_cand_hists = det_hists[sub_cand_idx]

                ev = self.build_event_from_subpools(
                    frame_id, sub_lost, sub_cand_boxes, sub_cand_hists, vis_boxes_now
                )
                dec = self.associator.decide(ev)
                for i_local, j_local in enumerate(dec):
                    if j_local >= 0 and j_local < len(sub_cand_idx):
                        det_c = sub_cand_idx[int(j_local)]
                        if det_c in reassoc_matched_det:
                            continue
                        trk = sub_lost[i_local]
                        gap_val = frame_id - trk.last_frame
                        gt_before = trk.initial_gt_tid
                        gt_now = int(det_gt_tids[det_c])
                        self.reassoc_log.append({
                            "frame": int(frame_id),
                            "track_id": int(trk.track_id),
                            "gap": int(gap_val),
                            "gt_tid_before": int(gt_before),
                            "gt_tid_after": int(gt_now),
                            "correct": bool(gt_before != -1 and gt_before == gt_now),
                        })
                        trk.update(frame_id, det_boxes[det_c], det_hists[det_c], gt_now)
                        new_active.append(trk)
                        reassoc_matched_lost.add(trk.track_id)
                        reassoc_matched_det.add(det_c)

        next_lost = []
        for trk in eligible_lost:
            if trk.track_id not in reassoc_matched_lost:
                next_lost.append(trk)
        for trk in newly_lost:
            if trk.age <= self.a_max:
                next_lost.append(trk)
        self.lost_tracks = next_lost

        for c in unmatched_det_indices:
            if c not in reassoc_matched_det:
                nt = TrackState(
                    track_id=self.next_id,
                    frame_id=frame_id,
                    box=det_boxes[c],
                    hist_vec=det_hists[c],
                    gt_tid=int(det_gt_tids[c]),
                    k_len=self.cfg.hist_len,
                )
                self.next_id += 1
                new_active.append(nt)

        self.active_tracks = new_active
        self.frame_outputs[frame_id] = [
            {
                "track_id": t.track_id,
                "box": t.boxes[-1].copy(),
                "gt_tid": t.last_gt_tid,
                "initial_gt_tid": t.initial_gt_tid,
            }
            for t in self.active_tracks
        ]
        return self.frame_outputs[frame_id]
