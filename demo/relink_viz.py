"""Identity-recovery visualiser: what happens between "vehicle lost" and "vehicle re-linked".

For each tracker (left = baseline, right = GapLink) it records
  * the Stage-2 decisions of the associator, with its prediction (mu, sigma) and score matrix,
  * the outcome of every re-link, judged against ground truth when the clip has it:
      ok     - the lost ID was re-linked to the same real vehicle
      wrong  - the lost ID was re-linked to a different real vehicle
      missed - the real vehicle came back under a NEW ID while its old ID was still waiting
and draws ghost boxes, the predicted position / uncertainty during the gap, and the verdicts.

Nothing here changes tracking: the associator wrapper returns exactly the inner decision.
"""
from __future__ import annotations

import numpy as np
import cv2

VERDICT_TTL = 60      # frames a verdict label stays on screen
MIN_GAP = 3           # re-links after >= MIN_GAP missing frames are judged (same lower bound as the report's gap bins)
GHOST_MIN_AGE = 8     # draw the dashed "lost" box only for longer gaps, to keep the picture readable
EDGE = 10             # px: a track lost at the image border has most likely left the scene -> no ghost
PRED_TTL = 3          # frames a prediction stays valid without a fresh decision
MAX_EVENTS = 80

COL = {
    "ghost": (170, 170, 170),
    "ok": (70, 200, 70),
    "wrong": (40, 40, 235),
    "missed": (200, 60, 200),
    "unk": (200, 200, 200),
    "pred_gl": (255, 200, 40),    # cyan-ish (BGR)
    "pred_bl": (0, 150, 255),     # orange
    "cand": (0, 220, 255),
}


class ExplainedAssociator:
    """Wraps an associator; keeps the explanation of its last decision in `.last`."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.kind = "gaplink" if hasattr(inner, "infer_event") else ("kalman" if hasattr(inner, "predict_gaussian") else "other")
        self.last: dict | None = None

    def __getattr__(self, item):
        # delegate everything else (infer_event, tau, theta, ...) so callers that probe the
        # associator (e.g. TargetLockManager) keep using the real model
        return getattr(self.inner, item)

    def decide(self, ev: dict) -> np.ndarray:
        N, M = len(ev["t_lost"]), len(ev["cand"])
        if self.kind == "gaplink" and N > 0 and M > 0:
            from scipy.optimize import linear_sum_assignment
            res = self.inner.infer_event(ev)
            P = np.asarray(res["P"], dtype=np.float32)
            dec = np.full((N,), -1, dtype=np.int32)
            P_real, P_dust = P[:N, :M], P[:N, M]
            r_ind, c_ind = linear_sum_assignment(-P_real)
            for r, c in zip(r_ind, c_ind):  # identical rule to GapLinkAssociator.decide
                if P_real[r, c] >= self.inner.tau and P_real[r, c] > P_dust[r]:
                    dec[r] = int(c)
            sig = np.asarray(res.get("sigma_cal", res["sigma_px"]), dtype=np.float32)
            self.last = {"kind": "gaplink", "mu": np.asarray(res["pred_box"], np.float32)[:, :2], "sigma": sig[:, :2],
                         "score": P, "dec": dec, "thr": float(self.inner.tau)}
            return dec
        dec = np.asarray(self.inner.decide(ev), dtype=np.int32)
        if self.kind == "kalman" and N > 0:
            mus, vars_ = self.inner.predict_gaussian(ev)
            C = self.inner.cost(ev) if M > 0 else np.zeros((N, 0), np.float32)
            self.last = {"kind": "kalman", "mu": mus[:, :2].astype(np.float32),
                         "sigma": np.sqrt(np.maximum(vars_[:, :2], 1.0)).astype(np.float32),
                         "score": np.asarray(C, np.float32), "dec": dec, "thr": float(self.inner.theta)}
        else:
            self.last = {"kind": self.kind, "mu": None, "sigma": None, "score": None, "dec": dec, "thr": None}
        return dec


class RelinkRecorder:
    def __init__(self, assoc: ExplainedAssociator | None, min_hits: int = 3):
        self.assoc = assoc
        self.min_hits = int(min_hits)
        self.reset()

    def reset(self):
        self.pred: dict[int, tuple] = {}          # tid -> (frame, mu(2,), sigma(2,))
        self.events: list[dict] = []
        self.lost_info: dict[int, dict] = {}      # tid -> {"box", "since", "gt"}
        self.flash: dict[int, dict] = {}          # tid -> {"verdict", "frame", "from", "gap", "was"}
        self.gt_owner: dict[int, int] = {}
        self.seen: set[int] = set()
        self.counts = {"ok": 0, "wrong": 0, "missed": 0, "unk": 0}
        self.ghosts: list[dict] = []
        self._n_log = 0
        self.last_f = -1
        self.crops: dict[int, np.ndarray] = {}    # tid -> last visible crop (BGR)
        self.history: list[dict] = []             # wrong / missed events with before-after crops

    # called by OnlineTracker after every Stage-2 decision
    def on_event(self, frame_id, sub_lost, sub_cand_idx, ev, dec):
        exp = self.assoc.last if self.assoc is not None else None
        tids = [int(t.track_id) for t in sub_lost]
        if exp is not None and exp.get("mu") is not None:
            for i, tid in enumerate(tids):
                self.pred[tid] = (int(frame_id), exp["mu"][i].copy(), exp["sigma"][i].copy())
        snap = {"frame": int(frame_id), "kind": exp["kind"] if exp else "other", "tids": tids,
                "gaps": [int(frame_id - t.last_frame) for t in sub_lost],
                "cand": np.asarray(ev["cand"], np.float32).round(1).tolist(),
                "dec": [int(d) for d in np.asarray(dec).tolist()],
                "score": None if not exp or exp.get("score") is None else np.round(exp["score"], 3).tolist(),
                "thr": None if not exp else exp.get("thr")}
        self.events.append(snap)
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]

    def update(self, tracker, step_out: dict, cur_f: int, frame_bgr: np.ndarray | None = None):
        """Once per frame, after tracker.step(). With frame_bgr, keeps before/after crops of wrong / missed events."""
        if cur_f <= self.last_f:
            if cur_f < self.last_f:
                self.reset()
            else:
                return
        self.last_f = cur_f
        active = {int(o["track_id"]) for o in step_out.get("tracks", [])}
        boxes_now = {int(o["track_id"]): o["box"] for o in step_out.get("tracks", [])}

        # 1) re-links made this frame
        log = getattr(tracker, "reassoc_log", [])
        for e in log[self._n_log:]:
            tid = int(e["track_id"])
            info = self.lost_info.get(tid, {})
            g0 = int(info.get("gt", e.get("gt_tid_before", -1)))
            g1 = int(e.get("gt_tid_after", -1))
            v = "unk" if (g0 < 0 or g1 < 0) else ("ok" if g0 == g1 else "wrong")
            gap = int(e.get("gap", 0))
            if gap >= MIN_GAP:
                self.counts[v] += 1
                if v != "unk":
                    self.flash[tid] = {"verdict": v, "frame": cur_f, "from": info.get("box"), "gap": gap}
                if v == "wrong":
                    self._add_history("wrong", cur_f, tid, tid, gap, self.crops.get(tid),
                                      _crop(frame_bgr, boxes_now.get(tid)))
            self.lost_info.pop(tid, None)
            self.pred.pop(tid, None)
        self._n_log = len(log)

        # 2) currently lost (ghost) tracks
        self.ghosts = []
        for t in getattr(tracker, "lost_tracks", []):
            if t.hits < self.min_hits:
                continue
            tid = int(t.track_id)
            self.lost_info[tid] = {"box": t.boxes[-1].copy(), "since": int(t.last_frame), "gt": int(t.last_gt_tid)}
            b = t.boxes[-1]
            at_edge = (b[0] - b[2] / 2 < EDGE or b[1] - b[3] / 2 < EDGE or
                       b[0] + b[2] / 2 > tracker.W - EDGE or b[1] + b[3] / 2 > tracker.H - EDGE)
            self.ghosts.append({"tid": tid, "box": b.copy(), "age": int(cur_f - t.last_frame), "edge": bool(at_edge)})
        live_lost = {g["tid"] for g in self.ghosts}
        for tid in list(self.lost_info):
            if tid not in live_lost and tid not in active:
                self.lost_info.pop(tid, None)      # expired

        # 3) vehicles that came back under a new ID while the old one was still waiting
        for o in step_out.get("tracks", []):
            tid = int(o["track_id"])
            g = int(o.get("gt_tid", -1))
            if tid not in self.seen:
                self.seen.add(tid)
                owner = self.gt_owner.get(g) if g >= 0 else None
                gap_o = int(cur_f - self.lost_info[owner]["since"]) if owner in self.lost_info else 0
                if owner is not None and owner != tid and owner in live_lost and gap_o >= MIN_GAP:
                    self.counts["missed"] += 1
                    self.flash[tid] = {"verdict": "missed", "frame": cur_f, "from": self.lost_info[owner]["box"],
                                       "gap": int(cur_f - self.lost_info[owner]["since"]), "was": owner}
                    self._add_history("missed", cur_f, tid, owner, gap_o, self.crops.get(owner),
                                      _crop(frame_bgr, boxes_now.get(tid)))
            if g >= 0:
                self.gt_owner[g] = tid

        # 4) remember what every visible vehicle looks like (for the next before/after pair)
        if frame_bgr is not None:
            for tid, b in boxes_now.items():
                c = _crop(frame_bgr, b)
                if c is not None:
                    self.crops[tid] = c
            keep = active | live_lost
            for tid in [t for t in self.crops if t not in keep]:
                self.crops.pop(tid, None)

    def _add_history(self, kind, frame, tid, was, gap, before, after):
        self.history.append({"kind": kind, "frame": int(frame), "tid": int(tid), "was": int(was), "gap": int(gap),
                             "before": _b64(before), "after": _b64(after)})

    def event_at(self, frame: int) -> dict | None:
        for e in reversed(self.events):
            if e["frame"] == frame:
                return e
        return None

    def latest_event(self) -> dict | None:
        return self.events[-1] if self.events else None


def _crop(frame_bgr, box, pad: float = 0.15):
    if frame_bgr is None or box is None:
        return None
    H, W = frame_bgr.shape[:2]
    cx, cy, w, h = [float(v) for v in box]
    x1, y1 = int(max(0, cx - w * (0.5 + pad))), int(max(0, cy - h * (0.5 + pad)))
    x2, y2 = int(min(W, cx + w * (0.5 + pad))), int(min(H, cy + h * (0.5 + pad)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


def _b64(img, max_side: int = 120) -> str:
    if img is None or img.size == 0:
        return ""
    import base64
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")) if ok else ""


def attach_relink(vms, min_hits: int = 3) -> RelinkRecorder:
    """Wrap the engine's associator and hook the recorder into its tracker."""
    trk = vms.tracker
    assoc = trk.associator
    if assoc is not None and not isinstance(assoc, ExplainedAssociator):
        assoc = ExplainedAssociator(assoc)
        trk.associator = assoc
    rec = RelinkRecorder(assoc, min_hits=min_hits)
    trk.event_hook = rec.on_event
    vms.relink = rec
    return rec


# ----------------------------------------------------------------------------------------------- drawing
def _dashed_rect(img, p1, p2, col, dash=7, th=1):
    x1, y1 = p1
    x2, y2 = p2
    for xa in range(x1, x2, dash * 2):
        cv2.line(img, (xa, y1), (min(xa + dash, x2), y1), col, th, cv2.LINE_AA)
        cv2.line(img, (xa, y2), (min(xa + dash, x2), y2), col, th, cv2.LINE_AA)
    for ya in range(y1, y2, dash * 2):
        cv2.line(img, (x1, ya), (x1, min(ya + dash, y2)), col, th, cv2.LINE_AA)
        cv2.line(img, (x2, ya), (x2, min(ya + dash, y2)), col, th, cv2.LINE_AA)


def _tag(img, txt, x, y, bg, fg=(255, 255, 255), fs=0.5, th=1):
    H, W = img.shape[:2]
    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x = int(np.clip(x, 2, W - tw - 8))
    y = int(np.clip(y, tht + 60, H - 4))
    cv2.rectangle(img, (x, y - tht - 5), (x + tw + 8, y + 4), bg, -1)
    cv2.putText(img, txt, (x + 4, y - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, fg, th, cv2.LINE_AA)


def _xyxy(b):
    cx, cy, w, h = [float(v) for v in b]
    return int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))


def draw_relink_layer(canvas: np.ndarray, rec: RelinkRecorder, step_out: dict, cur_f: int,
                      is_gaplink: bool, show_cands: bool = False) -> None:
    H, W = canvas.shape[:2]
    pcol = COL["pred_gl"] if is_gaplink else COL["pred_bl"]

    # ghosts + prediction during the gap (all uncertainty fills blended in ONE pass)
    fills = []
    for g in rec.ghosts:
        if g["edge"] or g["age"] < GHOST_MIN_AGE:
            continue
        x1, y1, x2, y2 = _xyxy(g["box"])
        _dashed_rect(canvas, (x1, y1), (x2, y2), COL["ghost"], th=1)
        _tag(canvas, f"#{g['tid']} {g['age']}f", x1, y1 - 3, (45, 45, 45), (230, 230, 230), fs=0.38)
        p = rec.pred.get(g["tid"])
        if p is not None and cur_f - p[0] < PRED_TTL:
            mu, sg = p[1], p[2]
            c0 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            c1 = (int(np.clip(mu[0], 0, W - 1)), int(np.clip(mu[1], 0, H - 1)))
            cap_ = 2.0 * max(12.0, float(y2 - y1))   # keep the 2-sigma ellipse readable
            ax = (int(np.clip(2.0 * sg[0], 4, cap_)), int(np.clip(2.0 * sg[1], 4, cap_)))
            fills.append((c1, ax))
            cv2.ellipse(canvas, c1, ax, 0, 0, 360, pcol, 1, cv2.LINE_AA)
            cv2.line(canvas, c0, c1, pcol, 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, c1, pcol, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    if fills:
        ov = canvas.copy()
        for c1, ax in fills:
            cv2.ellipse(ov, c1, ax, 0, 0, 360, pcol, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.12, canvas, 0.88, 0, dst=canvas)

    # verdicts on the vehicle that was re-linked / re-born
    boxes = {int(o["track_id"]): o["box"] for o in step_out.get("tracks", [])}
    for tid, fl in list(rec.flash.items()):
        if cur_f - fl["frame"] > VERDICT_TTL:
            rec.flash.pop(tid, None)
            continue
        b = boxes.get(tid)
        if b is None:
            continue
        x1, y1, x2, y2 = _xyxy(b)
        v = fl["verdict"]
        col = COL[v]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        if fl.get("from") is not None:
            fx1, fy1, fx2, fy2 = _xyxy(fl["from"])
            cv2.arrowedLine(canvas, ((fx1 + fx2) // 2, (fy1 + fy2) // 2), ((x1 + x2) // 2, (y1 + y2) // 2), col, 2,
                            cv2.LINE_AA, tipLength=0.08)
        txt = {"ok": f"RE-LINKED #{tid}  (gap {fl['gap']}f)",
               "wrong": f"WRONG LINK #{tid} -> other car",
               "missed": f"MISSED: new #{tid} (was #{fl.get('was')})",
               "unk": f"re-linked #{tid} (no GT)"}[v]
        _tag(canvas, txt, x1, y1 - 6, col, (255, 255, 255) if v != "unk" else (20, 20, 20), fs=0.52, th=2 if v != "unk" else 1)

    # candidates of the decision taken this frame (to read the score matrix in the dashboard)
    if show_cands:
        ev = rec.event_at(cur_f)
        if ev is not None:
            for j, cb in enumerate(ev["cand"]):
                x1, y1, x2, y2 = _xyxy(cb)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COL["cand"], 1, cv2.LINE_AA)
                _tag(canvas, f"c{j + 1}", x2 - 10, y2 + 14, COL["cand"], (15, 15, 15), fs=0.42)


def relink_summary(rec: RelinkRecorder | None) -> dict:
    if rec is None:
        return {"ok": 0, "wrong": 0, "missed": 0, "unk": 0, "lost_now": 0}
    d = dict(rec.counts)
    d["lost_now"] = len(rec.ghosts)
    return d


def event_json(rec: RelinkRecorder | None, frame: int | None = None) -> dict | None:
    if rec is None:
        return None
    return rec.event_at(frame) if frame is not None else rec.latest_event()
