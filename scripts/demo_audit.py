"""Strict audit of what a viewer sees in the GapLink live console, against UA-DETRAC GT (offline only).

Used by scripts/check_showcase.py (one showcase clip, fresh detections, same path as an upload).

Per frame the tracks are matched ONE-TO-ONE to GT boxes (Hungarian on IoU, IoU >= MATCH_IOU). Then:
  phantom     a track box lying mostly inside another track box (cab/trailer split, part of a bus...): it is
              not a vehicle, so an "ID KEPT" flash on it is a visible error;
  id switch   a GT vehicle matched to track a, later to track b != a: ANY gap, Stage 1 or Stage 2 (what a careful viewer
              sees as a colour/number change); 'visible' if the box is >= VISIBLE_H px high and not at the image edge;
  re-link     each "ID KEPT" event of the console (gap >= LINK_MIN_GAP): same vehicle / WRONG (another vehicle) /
              PHANTOM / unverified (no GT at one end, e.g. a parked car in an ignored area).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCH_IOU = 0.5
PHANTOM_INSIDE = 0.75     # share of the box area inside another (larger) track box / unmatched inside a GT vehicle
PARTIAL_AREA = 0.55       # a matched track smaller than this share of its GT box is only a part of the vehicle
VISIBLE_H = 30.0
EDGE = 12


def _xyxy(b):
    return b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


def inside(a, b) -> float:
    """Share of box a's area inside box b."""
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih / max(1e-6, a[2] * a[3])


def match(tracks: dict[int, list], gts: list[tuple[int, list]]) -> dict[int, int]:
    """track id -> gt id, one-to-one, IoU >= MATCH_IOU."""
    if not tracks or not gts:
        return {}
    tids = list(tracks)
    m = np.array([[iou(tracks[t], g[1]) for g in gts] for t in tids])
    r, c = linear_sum_assignment(-m)
    return {tids[i]: gts[j][0] for i, j in zip(r, c) if m[i, j] >= MATCH_IOU}


def phantoms(tracks: dict[int, list]) -> set[int]:
    out = set()
    for t, b in tracks.items():
        for u, o in tracks.items():
            if u != t and o[2] * o[3] > b[2] * b[3] and inside(b, o) >= PHANTOM_INSIDE:
                out.add(t)
                break
    return out


class Auditor:
    """Feed it one frame at a time (tracks + GT), then call report() with the console's re-link events."""

    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.m: dict[int, dict[int, int]] = {}       # frame -> {tid: gt}
        self.ph: dict[int, set[int]] = {}             # frame -> phantom tids
        self.box: dict[int, dict[int, list]] = {}     # frame -> {tid: box}

    def add(self, f: int, tracks: dict[int, list], gts: list[tuple[int, list]]) -> None:
        self.box[f] = tracks
        ph = phantoms(tracks)
        m = match({t: b for t, b in tracks.items() if t not in ph}, gts)
        gbox = dict(gts)
        # split vehicle: an unmatched track lying mostly inside a GT vehicle (cab/trailer, front/back of a bus), and a
        # matched track covering too little of its vehicle; both are "part of a vehicle" boxes
        for t, b in tracks.items():
            if t in m:
                g = gbox[m[t]]
                if b[2] * b[3] < PARTIAL_AREA * g[2] * g[3]:
                    ph.add(t)
            elif any(inside(b, g) >= PHANTOM_INSIDE for g in gbox.values()):
                ph.add(t)
        self.ph[f] = ph
        self.m[f] = m

    def _visible(self, b) -> bool:
        x1, y1, x2, y2 = _xyxy(b)
        return b[3] >= VISIBLE_H and x1 > EDGE and y1 > EDGE and x2 < self.W - EDGE and y2 < self.H - EDGE

    def report(self, links: list[dict], strong_gap: int = 25) -> dict:
        # id switches: per GT vehicle, consecutive matched tracks
        last: dict[int, tuple[int, int]] = {}
        switches = []
        for f in sorted(self.m):
            for tid, g in self.m[f].items():
                if g in last and last[g][0] != tid:
                    b = self.box[f][tid]
                    switches.append({"frame": f, "gt": g, "old": last[g][0], "new": tid, "gap": f - last[g][1],
                                     "visible": self._visible(b), "h": round(float(b[3]), 1)})
                last[g] = (tid, f)
        out_links = []
        for l in links:
            f, tid, gap = l["frame"], l["tid"], l["gap"]
            after = self.m.get(f, {}).get(tid)
            before = None
            for k in range(f - gap, max(0, f - gap - 30), -1):        # last frames before it went missing
                if tid in self.m.get(k, {}):
                    before = self.m[k][tid]
                    break
            ph = tid in self.ph.get(f, set())
            v = ("PHANTOM" if ph else "same vehicle" if before is not None and before == after
                 else "WRONG" if before is not None and after is not None else "unverified")
            b = self.box.get(f, {}).get(tid, [0, 0, 0, 0])
            out_links.append({"frame": f, "tid": tid, "gap": gap, "gt_before": before, "gt_after": after, "verdict": v,
                              "flashed": bool(b[3] >= 30)})
        vis_sw = [s for s in switches if s["visible"]]
        flashed = [x for x in out_links if x["flashed"]]
        n_ph = sum(len(v) for v in self.ph.values())
        return {
            "links": out_links,
            "links_same": sum(1 for x in out_links if x["verdict"] == "same vehicle"),
            "links_strong_same": sum(1 for x in out_links if x["verdict"] == "same vehicle" and x["gap"] >= strong_gap),
            "links_wrong": sum(1 for x in out_links if x["verdict"] == "WRONG"),
            "links_phantom_flashed": sum(1 for x in flashed if x["verdict"] == "PHANTOM"),
            "links_unverified_flashed": sum(1 for x in flashed if x["verdict"] == "unverified"),
            "id_switches": switches, "id_switches_visible": len(vis_sw),
            "phantom_track_frames": n_ph,
            "visible_errors": len(vis_sw) + sum(1 for x in flashed if x["verdict"] in ("WRONG", "PHANTOM")),
        }
