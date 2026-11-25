"""GapLink-only live console: one tracker (YOLOv8n -> Stage 1 IoU -> Stage 2 GapLink-Lite), drawn for the eye.

No baseline and no ground truth: nothing is scored. What is drawn lets a viewer check the identities by eye:
  - every vehicle keeps one colour per ID, with a short trail;
  - a confirmed vehicle that disappears inside the image becomes a dashed "hidden" box at the place it was last seen;
  - when Stage 2 gives the ID back, the box flashes "ID #k kept after N frames hidden" and a dashed bridge joins the place
    it was last seen to the place it came back (same colour as before the occlusion: a new ID would get a new colour);
  - the locked target (click) keeps its full path: solid where seen, dashed where hidden (GapLink's predicted positions),
    with an optional 90% ellipse (uncalibrated v1.3 model, illustration);
  - forbidden route Z_A -> Z_B alerts with the zones drawn in the console.
"""
from __future__ import annotations

import colorsys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from demo.clip_meta import load_clip_meta
from demo.relink_viz import EDGE, _b64, _crop
from demo.render_security_demo import load_clip_dets_cache
from demo.vms_lite_core import (C_ROUTE_OK, Pane, _dashed_polyline, _tag, _xyxy, zones_for_clip)

MODES = ("relink", "route", "lock")
LINK_MIN_GAP = 12         # shorter gaps are detector flicker: the ID is kept, but no "kept after" flash is drawn
FLASH = 50                # frames a re-link stays highlighted (2 s at 25 fps)
TRAIL_LEN = 125           # frames of trail per vehicle (5 s)
GHOST_MIN = 3             # frames before a missing vehicle is drawn as hidden
GHOST_MIN_H = 45          # px: small far vehicles are not drawn as hidden (clutter in dense scenes)
GHOST_MIN_HITS = 15       # only vehicles followed for a while (not one-off or parked flicker) are drawn as hidden
LINK_MIN_H = 30           # px: re-links of tiny far vehicles are counted but not flashed
LABEL_MIN_H = 28          # px: smaller boxes get no ID label (still a coloured box)
TRAIL_MIN_MOVE = 25.0     # px: parked vehicles get no trail
C_LINK = (80, 220, 255)   # BGR amber-yellow for "ID kept" flashes (distinct from the per-ID colours' saturation)


def id_colour(tid: int) -> tuple[int, int, int]:
    """Stable, well separated colour per ID (golden-ratio hue walk), BGR."""
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class LiveRunner:
    def __init__(self, clip_path: Path, mode: str, gl_assoc, label: str, zones_override: list[dict] | None = None):
        self.path, self.mode = Path(clip_path), mode if mode in MODES else "relink"
        self.meta = load_clip_meta(self.path) or {}
        if zones_override is not None:
            self.meta = dict(self.meta, zones=zones_override)
        self.dets = load_clip_dets_cache(self.path) or {}
        if self.path.exists():
            cap = cv2.VideoCapture(str(self.path))
            self.W, self.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        else:                       # analysis without pixels: size and length from the meta file
            self.W, self.H = (int(v) for v in self.meta["size"])
            fps, self.n = float(self.meta.get("fps", 25.0)), int(self.meta["n_frames"])
        self.fps = fps if 5 <= fps <= 60 else 25.0
        self.zones = zones_for_clip(self.meta, self.W, self.H)
        self.pane = Pane(label, gl_assoc, self.W, self.H, self.fps, self.zones, None, True)
        self.trails: dict[int, deque] = {}         # tid -> (frame, x, y)
        self.last_seen: dict[int, tuple[int, list]] = {}   # tid -> (frame, box)
        self.crops: dict[int, np.ndarray] = {}
        self.links: list[dict] = []                # every re-link with gap >= LINK_MIN_GAP
        self.n_links_all = 0
        self._n_log = 0
        self.hidden: list[dict] = []
        self.alerts: list[dict] = []

    # ---------------------------------------------------------------- step
    def step(self, f: int, frame) -> None:
        d = self.dets.get(f) or {"boxes": np.zeros((0, 4), np.float32), "hists": np.zeros((0, 128), np.float32)}
        res = self.pane.step(f, d, None, frame, self.mode if self.mode != "relink" else "live")
        boxes = {int(o["track_id"]): [float(v) for v in o["box"][:4]] for o in res["tracks"]}
        log = getattr(self.pane.vms.tracker, "reassoc_log", [])
        for e in log[self._n_log:]:
            tid, gap = int(e["track_id"]), int(e.get("gap", 0))
            self.n_links_all += 1
            if gap < LINK_MIN_GAP or tid not in boxes:
                continue
            ls = self.last_seen.get(tid)
            frm_box = ls[1] if ls else boxes[tid]
            self.links.append({"tid": tid, "frame": f, "gap": gap, "from": frm_box[:2], "to": boxes[tid][:2],
                               "big": boxes[tid][3] >= LINK_MIN_H,
                               "box": boxes[tid], "before": _b64(self.crops.get(tid)), "after": _b64(_crop(frame, boxes[tid]))})
            tr = self.trails.setdefault(tid, deque(maxlen=TRAIL_LEN))
            tr.append((f - 1, frm_box[0], frm_box[1], "bridge"))     # bridge: last seen -> back again (drawn dashed)
        self._n_log = len(log)
        for tid, b in boxes.items():
            self.trails.setdefault(tid, deque(maxlen=TRAIL_LEN)).append((f, b[0], b[1], "seen"))
            self.last_seen[tid] = (f, b)
            if frame is not None and f % 3 == 0:
                c = _crop(frame, b)
                if c is not None and c.size:
                    self.crops[tid] = c
        self.hidden = []
        for t in getattr(self.pane.vms.tracker, "lost_tracks", []):
            tid, gap = int(t.track_id), f - int(t.last_frame)
            if (t.hits >= GHOST_MIN_HITS and gap >= GHOST_MIN and tid in self.last_seen
                    and self.last_seen[tid][1][3] >= GHOST_MIN_H
                    and not self._at_edge(self.last_seen[tid][1])):
                self.hidden.append({"tid": tid, "gap": gap, "box": self.last_seen[tid][1]})
        for a in (res.get("route_alerts", []) if self.mode == "route" else []):
            self.alerts.append({"tid": int(a["track_id"]), "frame": f, "before": a.get("entry_crop", ""),
                                "after": a.get("alert_crop", "")})

    def _at_edge(self, b) -> bool:
        cx, cy, w, h = b[:4]
        return cx - w / 2 < EDGE or cy - h / 2 < EDGE or cx + w / 2 > self.W - EDGE or cy + h / 2 > self.H - EDGE

    def lock_click(self, x: float, y: float, f: int, frame) -> int | None:
        return self.pane.lock_at(x, y, f, frame)

    def unlock(self) -> None:
        self.pane.unlock()

    # ---------------------------------------------------------------- read-outs
    def stats(self) -> dict:
        res = self.pane.last or {"tracks": []}
        gaps = [l["gap"] for l in self.links]
        return {"tracked": len(res["tracks"]), "hidden": len(self.hidden), "links": len(self.links),
                "links_long": sum(1 for g in gaps if g >= 25), "longest": max(gaps) if gaps else 0,
                "alerts": len(self.alerts), "lock": self.pane.lock_metrics()}

    def events(self) -> list[dict]:
        out = [{"kind": "link", "frame": l["frame"], "gap": l["gap"], "tid": l["tid"]} for l in self.links]
        out += [{"kind": "alert", "frame": a["frame"], "gap": 0, "tid": a["tid"]} for a in self.alerts]
        out += [{"kind": e["kind"], "frame": int(e["frame"]), "gap": int(e.get("gap", 0)), "tid": None}
                for e in self.pane.tevents if e["kind"] != "hide"]
        return sorted(out, key=lambda e: e["frame"])

    def history(self) -> list[dict]:
        h = [{"kind": "link", "frame": l["frame"], "tid": l["tid"], "gap": l["gap"], "before": l["before"],
              "after": l["after"]} for l in self.links]
        h += [{"kind": "alert", "frame": a["frame"], "tid": a["tid"], "gap": 0, "before": a["before"], "after": a["after"]}
              for a in self.alerts]
        h += [{"kind": x["kind"], "frame": x["frame"], "tid": x["tid"], "gap": x.get("gap", 0), "before": x.get("before", ""),
               "after": x.get("after", "")} for x in self.pane.history if x["kind"].startswith("lock_")]
        return sorted(h, key=lambda e: -e["frame"])

    # ---------------------------------------------------------------- draw
    def draw(self, frame, f: int, show_ids=True, show_trails=True, show_ellipse=False, overlay=True) -> np.ndarray:
        img = frame.copy()
        if not overlay:
            return img
        res = self.pane.last or {"tracks": []}
        if self.mode == "route":
            ov = img.copy()
            for z in self.zones:
                cv2.fillPoly(ov, [z.to_pixel_array(self.W, self.H).astype(np.int32)], z.color_bgr)
            cv2.addWeighted(ov, 0.16, img, 0.84, 0, dst=img)
            for z in self.zones:
                pts = z.to_pixel_array(self.W, self.H).astype(np.int32)
                cv2.polylines(img, [pts], True, z.color_bgr, 2, cv2.LINE_AA)
                c = pts.mean(axis=0).astype(int)
                _tag(img, z.zone_id, int(c[0]) - 14, int(c[1]), z.color_bgr, (15, 15, 15), 0.6, 2)
        active = {int(o["track_id"]): o["box"] for o in res["tracks"]}
        locked = (res.get("lock") or {}).get("track_id")
        # trails of every vehicle (the locked one has its own full path)
        if show_trails:
            for tid, tr in self.trails.items():
                if tid not in active or (locked is not None and tid == int(locked)) or len(tr) < 2:
                    continue
                col = id_colour(tid)
                pts = [p for p in tr if f - p[0] <= TRAIL_LEN]
                if (len(pts) < 2 or active[tid][3] < LABEL_MIN_H
                        or np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) < TRAIL_MIN_MOVE):
                    continue
                for a, b in zip(pts[:-1], pts[1:]):
                    p0, p1 = (int(a[1]), int(a[2])), (int(b[1]), int(b[2]))
                    if a[3] == "bridge":
                        _dashed_polyline(img, np.array([p0, p1]), (20, 20, 20), 5, 9, 6)
                        _dashed_polyline(img, np.array([p0, p1]), col, 2, 9, 6)
                    else:
                        cv2.line(img, p0, p1, col, 2, cv2.LINE_AA)
        # hidden vehicles: dashed box where last seen
        for h in self.hidden:
            if locked is not None and h["tid"] == int(locked):
                continue
            x1, y1, x2, y2 = _xyxy(h["box"])
            col = id_colour(h["tid"])
            _dashed_polyline(img, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]), col, 2, 8, 6)
            if show_ids:
                _tag(img, f"#{h['tid']} hidden", x1, y1 - 4, (30, 30, 30), col, 0.42, 1)
        # live vehicles
        flash = {l["tid"]: l for l in self.links if 0 <= f - l["frame"] <= FLASH and l["big"]}
        alert = {a["tid"]: a for a in self.alerts if 0 <= f - a["frame"] <= FLASH}
        late = []
        for tid, b in active.items():
            x1, y1, x2, y2 = _xyxy(b)
            col = id_colour(tid)
            if locked is not None and tid == int(locked):
                continue                       # drawn by the lock layer below
            if tid in alert and self.mode == "route":
                cv2.rectangle(img, (x1, y1), (x2, y2), C_ROUTE_OK, 3, cv2.LINE_AA)
                late.append((f"#{tid} FORBIDDEN ROUTE", x1, y1 - 4, C_ROUTE_OK, (255, 255, 255)))
            elif tid in flash:
                cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (20, 20, 20), 5, cv2.LINE_AA)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
                late.append((f"#{tid} ID KEPT after {flash[tid]['gap']}f hidden", x1, y1 - 4, col, (15, 15, 15)))
            else:
                small = b[3] < LABEL_MIN_H
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 1 if small else 2, cv2.LINE_AA)
                if show_ids and not small:
                    _tag(img, f"#{tid}", x1, y1 - 3, col, (15, 15, 15), 0.45, 1)
        # locked target: full path + state
        lk = res.get("lock") or {}
        self.pane._draw_lock(img, f, show_ellipse, True)
        if locked is not None and int(locked) in active:
            x1, y1, x2, y2 = _xyxy(active[int(locked)])
            fl = self.pane.lock_flash
            again = fl and fl[0] == "ok" and f - fl[1] <= FLASH
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (20, 20, 20), 6, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (70, 210, 70), 3, cv2.LINE_AA)
            late.append((f"#{locked} TARGET" + (" RE-ACQUIRED" if again else ""), x1, y2 + 22, (70, 210, 70),
                         (255, 255, 255)))
        elif lk.get("reason") == "new_id" and lk.get("new_track_id") is not None and int(lk["new_track_id"]) in active:
            x1, y1, x2, y2 = _xyxy(active[int(lk["new_track_id"])])
            late.append((f"target got new ID #{lk['new_track_id']}", x1, y2 + 22, (200, 60, 200), (255, 255, 255)))
        for txt, x, y, bg, fg in late:
            _tag(img, txt, x, y, bg, fg, 0.55, 2)
        return img
