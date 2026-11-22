"""Core of the lean GapLink-VMS demo: three clips, two trackers side by side (baseline vs GapLink).

  1. relink : identity recovery after occlusion   (re-linked OK / total, wrong link, new ID)
  2. route  : forbidden route Z_A -> Z_B          (violation with correct ID, violation with wrong ID, false violation)
  3. lock   : click a vehicle, follow it behind a large vehicle, predicted path while hidden, re-acquire

All verdicts are judged post hoc against the clip's UA-DETRAC ground truth (<clip>_gt.npz); GT never enters
the trackers. Only wrong IDs, new IDs and route violations go to the before/after history.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from demo.clip_meta import load_clip_meta
from demo.relink_viz import MIN_GAP, VERDICT_TTL, RelinkRecorder, _b64, _crop
from demo.render_security_demo import load_clip_dets_cache, load_clip_gt_meta, match_frame_dets_to_gt
from src.security_analytics import PolygonZone, SecurityVMSEngine
from src.tracker import OnlineTracker

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "demo" / "sample_videos"
CLIP_SLOTS = [(1, "relink", ["demo1_relink"]),
              (2, "route", ["demo2_route"]),
              (3, "lock", ["demo3_lock"])]

# BGR colours
C_GRAY = (175, 175, 175)
C_OK = (70, 200, 70)
C_WRONG = (40, 40, 235)
C_NEW = (200, 60, 200)
C_ROUTE_OK = (30, 30, 230)
C_ROUTE_WID = (0, 190, 255)
C_ROUTE_FALSE = (200, 60, 200)
C_PRED = (255, 200, 40)
ZONE_COL = {"Z_A": (0, 200, 255), "Z_B": (60, 60, 240)}


def find_clips() -> list[dict]:
    out = []
    for n, scen, names in CLIP_SLOTS:
        for nm in names:
            p = SAMPLE_DIR / f"{nm}.mp4"
            if p.exists():
                out.append({"slot": n, "scenario": scen, "id": p.name, "path": p})
                break
    return out


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
        self.zones = zones
        self.vms = SecurityVMSEngine(OnlineTracker(associator=assoc, W=W, H=H, min_hits=3), zones=zones,
                                     forbidden_routes=[("Z_A", "Z_B")], max_dwell_sec=1e9, fps=fps)
        self.rec = RelinkRecorder(None, min_hits=3)
        self.judge = RouteJudge(gt_meta, W, H)
        self.route = {"ok": 0, "wrong_id": 0, "false": 0, "unk": 0}
        self.route_flash: dict[int, tuple[str, int]] = {}
        self.history: list[dict] = []
        self._n_rec_hist = 0
        self.pred_path: list[tuple[float, float]] = []
        self.path_until = -1
        self.lock_flash: tuple[str, int] | None = None
        self._lock_bad_logged = False
        self._prev_lock_state = "IDLE"
        self.last = None

    # ---------------------------------------------------------------- step
    def step(self, f: int, det: dict, gt_tids, frame, mode: str) -> dict:
        res = self.vms.step(f, det["boxes"], det["hists"], det_gt_tids=gt_tids, frame_bgr=frame)
        self.rec.update(self.vms.tracker, res, f, frame if mode == "relink" else None)
        if mode == "relink":
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
        elif mode == "lock":
            self._lock_step(f, res, frame)
        self.last = res
        return res

    def _lock_step(self, f, res, frame):
        lk = res.get("lock") or {}
        st = lk.get("state", "IDLE")
        hidden = st in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD")
        if hidden and lk.get("pred_box") is not None:
            if self._prev_lock_state not in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD"):
                tr = lk.get("trail") or []
                self.pred_path = [tr[-1]] if tr else []
            pb = lk["pred_box"]
            self.pred_path.append((float(pb[0]), float(pb[1])))
            self.path_until = f + 10 ** 6
        elif self._prev_lock_state in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD"):
            self.path_until = f + 50          # keep the predicted path visible for 2 s after the gap
            if st == "REACQUIRED":
                self.lock_flash = ("ok", f)
        if st in ("HIJACKED_WRONG", "LOST_EXPIRED") and not self._lock_bad_logged:
            self._lock_bad_logged = True
            if st == "LOST_EXPIRED":
                kind = "lost"
            else:
                kind = "newid" if lk.get("reason") == "new_id" else "wrong"
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
        self.pred_path, self.path_until, self.lock_flash = [], -1, None
        self._lock_bad_logged, self._prev_lock_state = False, "LOCKED"
        return self.vms.target_lock.lock_by_click(x, y, self.last["tracks"], frame_id=frame_id, frame_bgr=frame,
                                                  clear_totals=True)

    def metrics(self, mode: str) -> dict:
        if mode == "relink":
            c = self.rec.counts
            tot = c["ok"] + c["wrong"] + c["missed"]
            return {"ok": c["ok"], "total": tot, "wrong": c["wrong"], "missed": c["missed"]}
        if mode == "route":
            return dict(self.route)
        tl = self.vms.target_lock
        lk = (self.last or {}).get("lock") or {}
        return {"state": lk.get("state", "IDLE"), "reason": lk.get("reason"), "gap": int(lk.get("gap_frames", 0) or 0),
                "reacquired": tl.reacquire_count, "wrong": tl.hijack_count}

    # ---------------------------------------------------------------- draw
    def draw(self, frame, f: int, mode: str, show_ids: bool = True) -> np.ndarray:
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

        lk_res = (res.get("lock") or {}) if mode == "lock" else {}
        locked = lk_res.get("track_id") if mode == "lock" else None
        new_tid = lk_res.get("new_track_id") if lk_res.get("reason") == "new_id" else None
        for o in tracks:
            tid = int(o["track_id"])
            x1, y1, x2, y2 = _xyxy(o["box"])
            col, thick, lbl = C_GRAY, 1, (f"#{tid}" if show_ids else "")
            if mode == "relink":
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
            elif mode == "lock" and new_tid is not None and tid == int(new_tid):
                col, thick = C_NEW, 3
                lbl = f"#{tid} NEW ID (target was #{locked})"
            elif mode == "lock" and locked is not None and tid == int(locked):
                st = (res.get("lock") or {}).get("state", "LOCKED")
                bad = st == "HIJACKED_WRONG"
                col, thick = (C_WRONG if bad else C_OK), 3
                re_flash = self.lock_flash and self.lock_flash[0] == "ok" and f - self.lock_flash[1] <= VERDICT_TTL
                lbl = f"#{tid} WRONG CAR" if bad else (f"#{tid} RE-ACQUIRED" if re_flash else f"#{tid} TARGET")
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
            if lbl:
                big = thick > 1
                _tag(img, lbl, x1, y1 - 4, col if big else (28, 30, 36), (255, 255, 255) if big else (235, 235, 235),
                     0.55 if big else 0.42, 2 if big else 1)

        if mode == "lock":
            self._draw_lock(img, f)
        self._header(img, mode)
        return img

    def _draw_lock(self, img, f):
        lk = (self.last or {}).get("lock") or {}
        if lk.get("track_id") is None:
            return
        tr = np.array(lk.get("trail") or [], np.float32)
        if len(tr) >= 2:
            # observed path = detector box centres: smooth the box jitter, and draw nothing for a vehicle
            # that has not really moved (displacement < half its height)
            k = min(7, len(tr))
            sm = np.stack([np.convolve(tr[:, i], np.ones(k) / k, mode="valid") for i in (0, 1)], axis=1)
            ref = lk.get("box") if lk.get("box") is not None else lk.get("pred_box")
            hh = float(ref[3]) if ref is not None else 40.0
            if len(sm) >= 2 and float(np.hypot(*(sm[-1] - sm[0]))) >= 0.5 * hh:
                cv2.polylines(img, [np.round(sm).astype(np.int32)], False, C_OK, 2, cv2.LINE_AA)
        if self.pred_path and f <= self.path_until and len(self.pred_path) >= 2:
            pts = np.round(np.array(self.pred_path, np.float32)).astype(np.int32)
            for i in range(len(pts) - 1):
                if i % 2 == 0:
                    cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), C_PRED, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[-1]), 4, C_PRED, -1, cv2.LINE_AA)
        st = lk.get("state")
        if st in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD") and lk.get("pred_box") is not None:
            x1, y1, x2, y2 = _xyxy(lk["pred_box"])
            _dashed(img, (x1, y1), (x2, y2), C_PRED, 2)
            _tag(img, f"#{lk['track_id']} hidden {lk.get('gap_frames', 0)}f - predicted", x1, y1 - 4, C_PRED, (15, 15, 15), 0.5, 1)

    def _header(self, img, mode):
        W = img.shape[1]
        cv2.rectangle(img, (0, 0), (W, 34), (16, 18, 24), -1)
        cv2.putText(img, self.label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (120, 230, 120) if self.is_gl else (130, 150, 255), 2, cv2.LINE_AA)
        m = self.metrics(mode)
        if mode == "relink":
            items = [(f"Re-linked OK {m['ok']}/{m['total']}", C_OK), (f"Wrong link {m['wrong']}", C_WRONG),
                     (f"New ID {m['missed']}", C_NEW)]
        elif mode == "route":
            items = [(f"Violation, correct ID {m['ok']}", C_ROUTE_OK), (f"Wrong ID {m['wrong_id']}", C_ROUTE_WID),
                     (f"False {m['false']}", C_ROUTE_FALSE)]
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
        cap = cv2.VideoCapture(str(self.path))
        self.W, self.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.fps = fps if 5 <= fps <= 60 else 25.0
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
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

    def draw(self, frame, f: int, show_ids: bool = True) -> np.ndarray:
        return np.concatenate([self.left.draw(frame, f, self.mode, show_ids),
                               self.right.draw(frame, f, self.mode, show_ids)], axis=1)

    def history(self) -> list[dict]:
        out = [dict(h, side="bl") for h in self.left.history] + [dict(h, side="gl") for h in self.right.history]
        return sorted(out, key=lambda h: -h["frame"])

    def metrics(self) -> dict:
        return {"bl": self.left.metrics(self.mode), "gl": self.right.metrics(self.mode)}


def save_zones_to_meta(clip_path: Path, zones: list[dict]) -> None:
    mp = Path(clip_path).with_name(Path(clip_path).stem + "_meta.json")
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    keep = [z for z in meta.get("zones", []) if z["zone_id"] not in {x["zone_id"] for x in zones}]
    meta["zones"] = keep + zones
    meta["zones_edited"] = True
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
