"""Tracking core shared by the GapLink live console (demo/gaplink_live*.py).

Pane      one tracker (YOLOv8n detections -> Stage 1 IoU -> Stage 2 GapLink-Lite) with the surveillance rules
          (forbidden route Z_A -> Z_B, target lock), its persistent target path (seen / hidden) and drawing helpers.
ClipRunner two panes on one clip with optional UA-DETRAC ground truth (used for offline checks; the live console
          uses a single Pane through demo/gaplink_live_core.LiveRunner).

Ground truth, when present, is only used to judge events post hoc; it never enters the trackers.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from demo.clip_meta import load_clip_meta
from demo.relink_viz import EDGE, MIN_GAP, VERDICT_TTL, RelinkRecorder, _b64, _crop
from demo.render_security_demo import load_clip_dets_cache, load_clip_gt_meta, match_frame_dets_to_gt
from src.security_analytics import PolygonZone, SecurityVMSEngine
from src.tracker import OnlineTracker

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("relink", "route", "lock")

# BGR colours
C_GRAY = (175, 175, 175)
C_OK = (70, 200, 70)
C_WRONG = (40, 40, 235)
C_NEW = (200, 60, 200)
C_ROUTE_OK = (30, 30, 230)
C_ROUTE_WID = (0, 190, 255)
C_ROUTE_FALSE = (200, 60, 200)
C_PRED = (255, 200, 40)
C_UNV = (0, 200, 255)
HIDDEN_STATES = ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD")
MIN_TRAIL_GAP = 8                 # hidden runs shorter than this are detector drop-outs, not occlusions
C_TRAIL = (90, 220, 110)          # observed path of the target
C_TRAIL_DARK = (20, 40, 20)
CHI2_90_2D = 2.146               # sqrt of the 90% quantile of a chi-square with 2 dof: 90% ellipse = 2.146 sigma            # amber: re-link that cannot be verified (no ground truth)
ZONE_COL = {"Z_A": (0, 200, 255), "Z_B": (60, 60, 240)}


def zones_for_clip(meta: dict | None, W: int, H: int) -> list[PolygonZone]:
    """Only Z_A (forbidden origin) and Z_B (restricted destination) are used."""
    zs = []
    for z in (meta or {}).get("zones", []):
        if z["zone_id"] in ("Z_A", "Z_B"):
            role = "entry" if z["zone_id"] == "Z_A" else "restricted"
            zs.append(PolygonZone(z["zone_id"], z["zone_id"], [tuple(p) for p in z["points"]], role=role,
                                  normalized=True, color_bgr=ZONE_COL[z["zone_id"]]))
    return zs


# ------------------------------------------------------------------------------------------------ judges
class RouteJudge:
    """ok       : same real vehicle at entry (Z_A) and at the alert (Z_B)
       wrong_id : the vehicle at Z_B really came from Z_A, but the alert's ID started on another vehicle
       false    : the vehicle at Z_B never was in Z_A (no real violation)"""

    def __init__(self, gt_meta: dict | None, W: int, H: int):
        self.W, self.H = W, H
        self.traj: dict[int, list[tuple[int, np.ndarray]]] = {}
        if gt_meta:
            for f, (bx, ti) in gt_meta["by_frame"].items():
                for b, t in zip(bx, ti):
                    self.traj.setdefault(int(t), []).append((int(f), b))

    def visited(self, gt_tid: int, zone: PolygonZone | None, until: int) -> bool:
        if zone is None:
            return False
        return any(f <= until and zone.contains_box(b, self.W, self.H) for f, b in self.traj.get(gt_tid, []))

    def judge(self, alert: dict, zone_a: PolygonZone | None) -> str:
        g0, g1 = int(alert.get("origin_gt_tid", -1)), int(alert.get("dest_gt_tid", -1))
        if g1 < 0 or not self.traj:
            return "unk"
        if g0 == g1:
            return "ok"
        return "wrong_id" if self.visited(g1, zone_a, int(alert["frame"])) else "false"


class Pane:
    """One tracker + its judges, history and drawing state."""

    def __init__(self, label: str, assoc, W: int, H: int, fps: float, zones: list[PolygonZone], gt_meta, is_gl: bool):
        self.label, self.is_gl, self.W, self.H = label, is_gl, W, H
        self.has_gt = gt_meta is not None
        # without ground truth: every re-link and every new ID that appears while a track is lost is shown,
        # but none of them is called right or wrong
        self.unv = {"relink": 0, "new": 0}
        self.unv_flash: dict[int, tuple[str, int, int]] = {}    # tid -> (kind, frame, gap or lost tid)
        self._n_log_unv = 0
        self._seen_unv: set[int] = set()
        self.zones = zones
        self.vms = SecurityVMSEngine(OnlineTracker(associator=assoc, W=W, H=H, min_hits=3), zones=zones,
                                     forbidden_routes=[("Z_A", "Z_B")], max_dwell_sec=1e9, fps=fps)
        self.rec = RelinkRecorder(None, min_hits=3)
        self.judge = RouteJudge(gt_meta, W, H)
        self.route = {"ok": 0, "wrong_id": 0, "false": 0, "unk": 0}
        self.route_flash: dict[int, tuple[str, int]] = {}
        self.history: list[dict] = []
        self._n_rec_hist = 0
        self._reset_path()
        self._lock_bad_logged = False
        self._prev_lock_state = "IDLE"
        self.last = None

    # ---------------------------------------------------------------- step
    def step(self, f: int, det: dict, gt_tids, frame, mode: str) -> dict:
        res = self.vms.step(f, det["boxes"], det["hists"], det_gt_tids=gt_tids, frame_bgr=frame)
        self._prev_crops = dict(self.rec.crops)       # last crops before this frame (for "before" images)
        self.rec.update(self.vms.tracker, res, f, frame if mode == "relink" else None)
        if mode == "relink" and not self.has_gt:
            self._relink_unverified(f, res, frame)
        elif mode == "relink":
            for h in self.rec.history[self._n_rec_hist:]:
                self.history.append(dict(h))
            self._n_rec_hist = len(self.rec.history)
        elif mode == "route":
            za = next((z for z in self.zones if z.zone_id == "Z_A"), None)
            for a in res["route_alerts"]:
                v = self.judge.judge(a, za)
                self.route[v] += 1
                self.route_flash[int(a["track_id"])] = (v, f)
                self.history.append({"kind": "route_" + v, "frame": f, "tid": int(a["track_id"]), "was": int(a["track_id"]),
                                     "gap": int(a.get("occlusions_bridged", 0)),
                                     "before": a.get("entry_crop", ""), "after": a.get("alert_crop", "")})
        self._lock_step(f, res, frame)        # the target lock works in every mode
        self.last = res
        return res

    def _relink_unverified(self, f, res, frame):
        tr = self.vms.tracker
        log = getattr(tr, "reassoc_log", [])
        boxes = {int(o["track_id"]): o["box"] for o in res.get("tracks", [])}
        for e in log[self._n_log_unv:]:
            gap = int(e.get("gap", 0))
            if gap < MIN_GAP:
                continue
            tid = int(e["track_id"])
            self.unv["relink"] += 1
            self.unv_flash[tid] = ("relink", f, gap)
            self.history.append({"kind": "relink_unv", "frame": f, "tid": tid, "was": tid, "gap": gap,
                                 "before": _b64(self._prev_crops.get(tid)), "after": _b64(_crop(frame, boxes.get(tid)))})
        self._n_log_unv = len(log)
        # confirmed tracks hidden inside the image (a track lost at the border has most likely left the scene)
        lost = {int(t.track_id): t.boxes[-1] for t in getattr(tr, "lost_tracks", [])
                if t.hits >= 3 and f - int(t.last_frame) >= MIN_GAP and not self._at_edge(t.boxes[-1])}
        for o in res.get("tracks", []):
            tid = int(o["track_id"])
            if tid in self._seen_unv:
                continue
            self._seen_unv.add(tid)
            # a new ID that appears inside the image while a confirmed vehicle is still hidden
            # (vehicles entering at the border are simply new vehicles)
            if lost and not self._at_edge(o["box"]):
                near = min(lost, key=lambda k: float(np.hypot(lost[k][0] - o["box"][0], lost[k][1] - o["box"][1])))
                self.unv["new"] += 1
                self.unv_flash[tid] = ("new", f, near)
                self.history.append({"kind": "new_unv", "frame": f, "tid": tid, "was": near, "gap": 0,
                                     "before": _b64(self._prev_crops.get(near)),
                                     "after": _b64(_crop(frame, o["box"]))})

    def _at_edge(self, b) -> bool:
        cx, cy, w, h = [float(v) for v in b]
        return (cx - w / 2 < EDGE or cy - h / 2 < EDGE or cx + w / 2 > self.W - EDGE or cy + h / 2 > self.H - EDGE)

    def _lock_step(self, f, res, frame):
        """Follow the locked target: record its observed and predicted positions and log incidents."""
        lk = res.get("lock") or {}
        st = self._lock_state(lk)
        if st == "IDLE" or lk.get("track_id") is None:
            self._prev_lock_state = st
            return
        hidden = st in HIDDEN_STATES
        was_hidden = self._prev_lock_state in HIDDEN_STATES
        if self._path_open:
            if st in ("LOCKED", "REACQUIRED") and lk.get("box") is not None:
                b = lk["box"]
                if was_hidden:
                    gap = int(f - self._hidden_since) if self._hidden_since else 0
                    if gap >= MIN_TRAIL_GAP:          # short detector drop-outs are not marked
                        if self._pending_hide is not None:
                            self.tevents.append(self._pending_hide)
                        self.tevents.append({"kind": "reacq", "frame": f, "x": float(b[0]), "y": float(b[1]), "gap": gap})
                        self.lock_flash = ("ok", f)
                    self._pending_hide = None
                self.tpath.append((f, float(b[0]), float(b[1]), "obs"))
            elif hidden and lk.get("pred_box") is not None:
                if not was_hidden:
                    self._hidden_since = f
                    if self.tpath:
                        _, x, y, _k = self.tpath[-1]
                        self._pending_hide = {"kind": "hide", "frame": f, "x": x, "y": y, "gap": 0}
                pb = lk["pred_box"]
                self.tpath.append((f, float(pb[0]), float(pb[1]), "pred"))
        if st in ("HIJACKED_WRONG", "LOST_EXPIRED") and not self._lock_bad_logged:
            self._lock_bad_logged = True
            self._path_open = False
            if self._pending_hide is not None:
                self.tevents.append(self._pending_hide)
                self._pending_hide = None
            if st == "LOST_EXPIRED":
                kind = "lost"
            else:
                kind = "newid" if lk.get("reason") == "new_id" else "wrong"
            x, y = (self.tpath[-1][1], self.tpath[-1][2]) if self.tpath else (0.0, 0.0)
            if kind != "lost" and lk.get("box") is not None:
                x, y = float(lk["box"][0]), float(lk["box"][1])
            self.tevents.append({"kind": kind, "frame": f, "x": x, "y": y, "gap": int(lk.get("gap_frames", 0) or 0)})
            self.lock_flash = (kind, f)
            tl = self.vms.target_lock
            was = int(lk.get("track_id") or -1)
            self.history.append({"kind": "lock_" + kind, "frame": f,
                                 "tid": int(lk.get("new_track_id") or was), "was": was,
                                 "gap": int(lk.get("gap_frames", 0)), "before": _b64(tl.lock_crop_img),
                                 "after": _b64(_crop(frame, lk.get("box")))})
        self._prev_lock_state = st

    def lock_at(self, x: float, y: float, frame_id: int, frame) -> int | None:
        if self.last is None:
            return None
        self._reset_path()
        self._lock_bad_logged, self._prev_lock_state = False, "LOCKED"
        tid = self.vms.target_lock.lock_by_click(x, y, self.last["tracks"], frame_id=frame_id, frame_bgr=frame,
                                                 clear_totals=True)
        if tid is not None:
            o = next((o for o in self.last["tracks"] if int(o["track_id"]) == int(tid)), None)
            if o is not None:
                self.tpath.append((int(frame_id), float(o["box"][0]), float(o["box"][1]), "obs"))
            self.tevents.append({"kind": "lock", "frame": int(frame_id), "x": self.tpath[-1][1] if self.tpath else x,
                                 "y": self.tpath[-1][2] if self.tpath else y, "gap": 0})
            # show the lock at once, even while the video is paused (the engine reports it from the next frame)
            self.last = dict(self.last, lock={"state": "LOCKED", "track_id": int(tid), "gap_frames": 0,
                                              "box": o["box"] if o is not None else None})
        return tid

    def unlock(self):
        self.vms.target_lock.reset(clear_totals=True)
        self._reset_path()
        self._lock_bad_logged, self._prev_lock_state = False, "IDLE"
        if self.last is not None:
            self.last = dict(self.last, lock={"state": "IDLE"})

    def _lock_state(self, lk: dict) -> str:
        """Lock state as shown. Without ground truth a 'wrong vehicle' verdict would only come from a colour
        heuristic (a partly hidden car changes colour statistics), so it is not shown: the viewer judges by eye."""
        st = lk.get("state", "IDLE")
        if st == "HIJACKED_WRONG" and not self.has_gt and lk.get("reason") != "new_id":
            return "LOCKED"
        return st

    def _reset_path(self):
        self.tpath: list[tuple[int, float, float, str]] = []     # (frame, x, y, "obs" | "pred")
        self.tevents: list[dict] = []
        self._path_open = True
        self._hidden_since = 0
        self._pending_hide: dict | None = None
        self.lock_flash = None

    def lock_metrics(self) -> dict:
        tl = self.vms.target_lock
        lk = (self.last or {}).get("lock") or {}
        return {"state": self._lock_state(lk), "reason": lk.get("reason"), "tid": lk.get("track_id"),
                "new_tid": lk.get("new_track_id"), "gap": int(lk.get("gap_frames", 0) or 0),
                "reacquired": sum(1 for e in self.tevents if e["kind"] == "reacq"),
                "wrong": tl.hijack_count if self.has_gt else 0, "nogt": not self.has_gt,
                "longest": max([e["gap"] for e in self.tevents if e["kind"] == "reacq"] + [int(lk.get("gap_frames", 0) or 0)]),
                "hidden_frames": sum(1 for p in self.tpath if p[3] == "pred")}

    def metrics(self, mode: str) -> dict:
        if mode == "relink" and not self.has_gt:
            return {"relink": self.unv["relink"], "new": self.unv["new"], "nogt": True}
        if mode == "relink":
            c = self.rec.counts
            tot = c["ok"] + c["wrong"] + c["missed"]
            return {"ok": c["ok"], "total": tot, "wrong": c["wrong"], "missed": c["missed"]}
        if mode == "route":
            return dict(self.route, nogt=not self.has_gt)
        return self.lock_metrics()

    # ---------------------------------------------------------------- draw
    def draw(self, frame, f: int, mode: str, show_ids: bool = True, show_ellipse: bool = False,
             show_trail: bool = True) -> np.ndarray:
        img = frame.copy()
        res = self.last or {"tracks": []}
        tracks = res["tracks"]
        if mode == "route":
            ov = img.copy()
            for z in self.zones:
                pts = z.to_pixel_array(self.W, self.H).astype(np.int32)
                cv2.fillPoly(ov, [pts], z.color_bgr)
            cv2.addWeighted(ov, 0.18, img, 0.82, 0, dst=img)
            for z in self.zones:
                pts = z.to_pixel_array(self.W, self.H).astype(np.int32)
                cv2.polylines(img, [pts], True, z.color_bgr, 2, cv2.LINE_AA)
                c = pts.mean(axis=0).astype(int)
                _tag(img, z.zone_id, int(c[0]) - 14, int(c[1]), z.color_bgr, (15, 15, 15), 0.6, 2)

        lk_res = res.get("lock") or {}                  # the target lock is drawn in every mode
        locked = lk_res.get("track_id")
        new_tid = lk_res.get("new_track_id") if lk_res.get("reason") == "new_id" else None
        late = []
        for o in tracks:
            tid = int(o["track_id"])
            x1, y1, x2, y2 = _xyxy(o["box"])
            col, thick, lbl = C_GRAY, 1, (f"#{tid}" if show_ids else "")
            if new_tid is not None and tid == int(new_tid):
                col, thick = C_NEW, 3
                lbl = f"#{tid} NEW ID (target was #{locked})"
            elif locked is not None and tid == int(locked):
                st = self._lock_state(lk_res)
                bad = st == "HIJACKED_WRONG"
                col, thick = (C_WRONG if bad else C_OK), 3
                re_flash = self.lock_flash and self.lock_flash[0] == "ok" and f - self.lock_flash[1] <= VERDICT_TTL
                lbl = (f"#{tid} WRONG CAR" if bad else
                       (f"#{tid} RE-ACQUIRED" if re_flash else f"#{tid} TARGET"))
            elif mode == "relink" and not self.has_gt:
                fu = self.unv_flash.get(tid)
                if fu and f - fu[1] <= VERDICT_TTL:
                    if fu[0] == "relink":
                        col, thick, lbl = C_UNV, 3, f"#{tid} RE-LINKED after {fu[2]}f ?"
                    else:
                        col, thick, lbl = C_NEW, 3, f"#{tid} NEW ID (#{fu[2]} hidden)"
            elif mode == "relink":
                fl = self.rec.flash.get(tid)
                if fl and f - fl["frame"] <= VERDICT_TTL and fl["verdict"] in ("ok", "wrong", "missed"):
                    v = fl["verdict"]
                    col, thick = {"ok": C_OK, "wrong": C_WRONG, "missed": C_NEW}[v], 3
                    lbl = {"ok": f"#{tid} RE-LINKED", "wrong": f"#{tid} WRONG LINK",
                           "missed": f"#{tid} NEW ID (was #{fl.get('was')})"}[v]
            elif mode == "route":
                fl = self.route_flash.get(tid)
                if fl and f - fl[1] <= VERDICT_TTL:
                    v = fl[0]
                    col = {"ok": C_ROUTE_OK, "wrong_id": C_ROUTE_WID, "false": C_ROUTE_FALSE}.get(v, C_GRAY)
                    thick = 3
                    lbl = {"ok": f"#{tid} VIOLATION", "wrong_id": f"#{tid} VIOLATION, WRONG ID",
                           "false": f"#{tid} FALSE VIOLATION"}.get(v, f"#{tid} violation (no GT)")
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
            if lbl and locked is not None and tid == int(locked):
                late.append((lbl, x1, y2 + 22, col))    # target label under the box: the trail arrives from above
            elif lbl:
                big = thick > 1
                _tag(img, lbl, x1, y1 - 4, col if big else (28, 30, 36), (255, 255, 255) if big else (235, 235, 235),
                     0.55 if big else 0.42, 2 if big else 1)

        self._draw_lock(img, f, show_ellipse, show_trail)
        for lbl, x, y, col in late:                     # drawn last so trail markers never cover it
            _tag(img, lbl, x, y, col, (255, 255, 255), 0.55, 2)
        self._header(img, mode)
        return img

    def _draw_lock(self, img, f, show_ellipse: bool = False, show_trail: bool = True):
        """Whole path of the locked target since the click: solid where it was seen, dashed where it was hidden
        (the positions predicted by this pane's method), with markers where it was hidden and re-acquired."""
        lk = (self.last or {}).get("lock") or {}
        if lk.get("track_id") is None and not self.tpath:
            return
        if show_trail and len(self.tpath) >= 2:
            runs, cur = [], [self.tpath[0]]
            for p in self.tpath[1:]:
                if p[3] == cur[-1][3]:
                    cur.append(p)
                else:
                    runs.append(cur)
                    cur = [cur[-1], p] if p[3] == "pred" else [p]    # a hidden run starts where the car was last seen
            runs.append(cur)
            for run in runs:
                pts = np.array([[p[1], p[2]] for p in run], np.float32)
                if run[-1][3] == "obs" and len(pts) >= 5:            # smooth the detector jitter of observed centres
                    k = 5
                    pad = np.vstack([np.repeat(pts[:1], k // 2, 0), pts, np.repeat(pts[-1:], k // 2, 0)])
                    pts = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in (0, 1)], axis=1)
                ip = np.round(pts).astype(np.int32)
                if len(ip) < 2:
                    continue
                if run[-1][3] == "obs":
                    cv2.polylines(img, [ip], False, C_TRAIL_DARK, 6, cv2.LINE_AA)
                    cv2.polylines(img, [ip], False, C_TRAIL, 3, cv2.LINE_AA)
                else:
                    _dashed_polyline(img, ip, (25, 25, 25), 5)
                    _dashed_polyline(img, ip, C_PRED, 3)
            for ev in self.tevents:
                c = (int(round(ev["x"])), int(round(ev["y"])))
                if ev["kind"] == "lock":
                    cv2.circle(img, c, 7, (25, 25, 25), -1, cv2.LINE_AA)
                    cv2.circle(img, c, 5, C_TRAIL, -1, cv2.LINE_AA)
                elif ev["kind"] == "hide":
                    cv2.circle(img, c, 8, (25, 25, 25), 3, cv2.LINE_AA)
                    cv2.circle(img, c, 8, C_PRED, 2, cv2.LINE_AA)
                elif ev["kind"] == "reacq":
                    cv2.circle(img, c, 9, (25, 25, 25), -1, cv2.LINE_AA)
                    cv2.circle(img, c, 7, C_OK, -1, cv2.LINE_AA)
                    t = f"+{ev['gap']}f"                    # label left of the dot, away from the box ahead
                    (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    _tag(img, t, c[0] - tw - 16, c[1] + 5, C_OK, (15, 15, 15), 0.45, 1)
                else:                                                  # lost / wrong / new id
                    col = {"lost": (150, 150, 150), "newid": C_NEW}.get(ev["kind"], C_WRONG)
                    for dx in (-7, 7):
                        cv2.line(img, (c[0] - 7, c[1] - dx), (c[0] + 7, c[1] + dx), (25, 25, 25), 5, cv2.LINE_AA)
                        cv2.line(img, (c[0] - 7, c[1] - dx), (c[0] + 7, c[1] + dx), col, 3, cv2.LINE_AA)
                    txt = {"lost": "target lost", "newid": "target got a new ID", "wrong": "wrong vehicle"}[ev["kind"]]
                    _tag(img, txt, c[0] + 10, c[1] + 22, col, (255, 255, 255), 0.45, 1)
        st = self._lock_state(lk)
        if st in HIDDEN_STATES and lk.get("pred_box") is not None:
            if show_ellipse and lk.get("sigma_xy") is not None:
                self._draw_ellipse(img, lk["pred_box"], lk["sigma_xy"])
            x1, y1, x2, y2 = _xyxy(lk["pred_box"])
            _dashed(img, (x1, y1), (x2, y2), C_PRED, 2)
            _tag(img, f"#{lk['track_id']} hidden {lk.get('gap_frames', 0)}f - predicted", x1, y1 - 4, C_PRED, (15, 15, 15), 0.5, 1)

    def _draw_ellipse(self, img, pred_box, sigma_xy):
        """90% ellipse of the predicted position: learned sigma (GapLink-Lite, not calibrated for the tuned model)
        or Kalman covariance (C-3)."""
        cx, cy = float(pred_box[0]), float(pred_box[1])
        ax = [max(2.0, CHI2_90_2D * float(v)) for v in sigma_xy[:2]]
        if not all(np.isfinite([cx, cy] + ax)):
            return
        H, W = img.shape[:2]
        ax = [min(a, 0.5 * max(W, H)) for a in ax]   # keep a runaway Kalman covariance on screen
        centre, axes = (int(round(cx)), int(round(cy))), (int(round(ax[0])), int(round(ax[1])))
        ov = img.copy()
        cv2.ellipse(ov, centre, axes, 0, 0, 360, C_PRED, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.18, img, 0.82, 0, dst=img)
        cv2.ellipse(img, centre, axes, 0, 0, 360, C_PRED, 2, cv2.LINE_AA)
        txt = "90% ellipse (model, uncalibrated)" if self.is_gl else "90% ellipse (Kalman)"
        _tag(img, txt, int(cx - ax[0]), int(cy + ax[1]) + 18, C_PRED, (15, 15, 15), 0.45, 1)

    def _header(self, img, mode):
        W = img.shape[1]
        cv2.rectangle(img, (0, 0), (W, 34), (16, 18, 24), -1)
        cv2.putText(img, self.label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (120, 230, 120) if self.is_gl else (130, 150, 255), 2, cv2.LINE_AA)
        m = self.metrics(mode)
        if mode == "relink" and m.get("nogt"):
            items = [(f"Re-links {m['relink']}", C_UNV), (f"New IDs {m['new']}", C_NEW), ("NO GT - check by eye", (90, 90, 90))]
        elif mode == "relink":
            items = [(f"Re-linked OK {m['ok']}/{m['total']}", C_OK), (f"Wrong link {m['wrong']}", C_WRONG),
                     (f"New ID {m['missed']}", C_NEW)]
        elif mode == "route" and m.get("nogt"):
            items = [(f"Violations {m['unk']}", C_ROUTE_OK), ("NO GT - check by eye", (90, 90, 90))]
        elif mode == "route":
            items = [(f"Violation, correct ID {m['ok']}", C_ROUTE_OK), (f"Wrong ID {m['wrong_id']}", C_ROUTE_WID),
                     (f"False {m['false']}", C_ROUTE_FALSE)]
        elif m.get("nogt"):
            items = [(f"Re-acquired {m['reacquired']}", C_OK), ("NO GT - check by eye", (90, 90, 90))]
        else:
            items = [(f"Re-acquired {m['reacquired']}", C_OK), (f"Target lost {m['wrong']}", C_WRONG)]
        x = W - 8
        for txt, col in reversed(items):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x -= tw + 14
            cv2.rectangle(img, (x, 7), (x + tw + 10, 28), col, -1)
            cv2.putText(img, txt, (x + 5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            x -= 6


def _xyxy(b):
    cx, cy, w, h = [float(v) for v in b]
    return int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))


def _tag(img, txt, x, y, bg, fg, fs=0.45, th=1):
    H, W = img.shape[:2]
    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x = int(np.clip(x, 2, W - tw - 8))
    y = int(np.clip(y, tht + 40, H - 4))
    cv2.rectangle(img, (x, y - tht - 5), (x + tw + 8, y + 4), bg, -1)
    cv2.putText(img, txt, (x + 4, y - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, fg, th, cv2.LINE_AA)


def _dashed_polyline(img, pts: np.ndarray, col, th=2, dash=10.0, gap=7.0):
    """Dashed line along a polyline, with dashes measured along the curve."""
    pts = pts.astype(np.float32)
    seg = np.diff(pts, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    tot = float(ln.sum())
    if tot < 1:
        return
    cum = np.concatenate([[0.0], np.cumsum(ln)])

    def at(d):
        i = int(np.clip(np.searchsorted(cum, d) - 1, 0, len(ln) - 1))
        t = 0.0 if ln[i] == 0 else (d - cum[i]) / ln[i]
        return tuple(np.round(pts[i] + t * seg[i]).astype(int))

    d = 0.0
    while d < tot:
        e = min(tot, d + dash)
        cv2.line(img, at(d), at(e), col, th, cv2.LINE_AA)
        d = e + gap


def _dashed(img, p1, p2, col, th=1, dash=8):
    (x1, y1), (x2, y2) = p1, p2
    for xa in range(x1, x2, dash * 2):
        cv2.line(img, (xa, y1), (min(xa + dash, x2), y1), col, th, cv2.LINE_AA)
        cv2.line(img, (xa, y2), (min(xa + dash, x2), y2), col, th, cv2.LINE_AA)
    for ya in range(y1, y2, dash * 2):
        cv2.line(img, (x1, ya), (x1, min(ya + dash, y2)), col, th, cv2.LINE_AA)
        cv2.line(img, (x2, ya), (x2, min(ya + dash, y2)), col, th, cv2.LINE_AA)


# ------------------------------------------------------------------------------------------------ clip runner
class ClipRunner:
    """Frame-by-frame driver for one clip with two panes (used by the server, the renderer and the checker)."""

    def __init__(self, clip_path: Path, scenario: str, bl_assoc, gl_assoc, bl_label: str, gl_label: str,
                 zones_override: list[dict] | None = None):
        self.path, self.mode = Path(clip_path), scenario
        self.meta = load_clip_meta(self.path) or {}
        if zones_override is not None:
            self.meta = dict(self.meta, zones=zones_override)
        self.dets = load_clip_dets_cache(self.path) or {}
        self.gt = load_clip_gt_meta(self.path)
        self.has_gt = self.gt is not None
        if self.path.exists():
            cap = cv2.VideoCapture(str(self.path))
            self.W, self.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        else:                                   # analysis without pixels: size from meta
            self.W, self.H = (int(v) for v in self.meta["size"])
            fps, self.n = float(self.meta.get("fps", 25.0)), int(self.meta["n_frames"])
        self.fps = fps if 5 <= fps <= 60 else 25.0
        zl, zr = zones_for_clip(self.meta, self.W, self.H), zones_for_clip(self.meta, self.W, self.H)
        self.left = Pane(bl_label, bl_assoc, self.W, self.H, self.fps, zl, self.gt, False)
        self.right = Pane(gl_label, gl_assoc, self.W, self.H, self.fps, zr, self.gt, True)
        self.lock_sched = [tuple(x) for x in self.meta.get("lock", [])] if scenario == "lock" else []
        self.manual_lock = False

    def step(self, f: int, frame) -> None:
        d = self.dets.get(f)
        if d is None:
            d = {"boxes": np.zeros((0, 4), np.float32), "hists": np.zeros((0, 128), np.float32)}
        g = match_frame_dets_to_gt(d["boxes"], self.gt, f)
        self.left.step(f, d, g, frame, self.mode)
        self.right.step(f, d, g, frame, self.mode)
        if self.mode == "lock" and not self.manual_lock:
            for lf, lx, ly in self.lock_sched:
                if int(lf) == f:
                    self.left.lock_at(lx, ly, f, frame)
                    self.right.lock_at(lx, ly, f, frame)

    def lock_click(self, x: float, y: float, f: int, frame) -> dict:
        self.manual_lock = True
        return {"bl": self.left.lock_at(x, y, f, frame), "gl": self.right.lock_at(x, y, f, frame)}

    def draw(self, frame, f: int, show_ids: bool = True, show_ellipse: bool = False, show_trail: bool = True) -> np.ndarray:
        return np.concatenate([self.left.draw(frame, f, self.mode, show_ids, show_ellipse, show_trail),
                               self.right.draw(frame, f, self.mode, show_ids, show_ellipse, show_trail)], axis=1)

    def unlock(self) -> None:
        self.manual_lock = True
        self.left.unlock()
        self.right.unlock()

    def history(self) -> list[dict]:
        out = [dict(h, side="bl") for h in self.left.history] + [dict(h, side="gl") for h in self.right.history]
        return sorted(out, key=lambda h: -h["frame"])

    def events(self) -> list[dict]:
        """Timeline marks of the locked target and of the incidents: {side, kind, frame, gap}."""
        out = []
        for side, p in (("bl", self.left), ("gl", self.right)):
            out += [{"side": side, "kind": e["kind"], "frame": int(e["frame"]), "gap": int(e.get("gap", 0))}
                    for e in p.tevents if e["kind"] != "hide"]
            out += [{"side": side, "kind": h["kind"], "frame": int(h["frame"]), "gap": int(h.get("gap", 0))}
                    for h in p.history if not h["kind"].startswith("lock_")]
        return sorted(out, key=lambda e: e["frame"])

    def metrics(self) -> dict:
        return {"bl": self.left.metrics(self.mode), "gl": self.right.metrics(self.mode),
                "lock": {"bl": self.left.lock_metrics(), "gl": self.right.lock_metrics()}}


def save_zones_to_meta(clip_path: Path, zones: list[dict]) -> None:
    mp = Path(clip_path).with_name(Path(clip_path).stem + "_meta.json")
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    keep = [z for z in meta.get("zones", []) if z["zone_id"] not in {x["zone_id"] for x in zones}]
    meta["zones"] = keep + zones
    meta["zones_edited"] = True
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
