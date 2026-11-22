"""Sidecar metadata for showcase clips: <video_stem>_meta.json next to the .mp4.

{
  "scenario": "route" | "dwell" | "lock",
  "title": "...",
  "zones": [{"zone_id": "Z_A", "name": "...", "role": "entry", "points": [[x, y], ...]}],   # normalised [0,1]
  "forbidden_routes": [["Z_A", "Z_B"]],
  "max_dwell_sec": 5.0,
  "key_moments": [{"frame": 212, "cx": 0.41, "cy": 0.37, "label": "..."}],                 # clip frame index, normalised centre
  "lock": [[frame, x_px, y_px]],
  "source": {"seq": "MVI_40701", "f0": 1201, "f1": 1576},
  "expected": {...}                                                                        # counts from scripts/mine_demo3.py
}
"""
from __future__ import annotations

import json
from pathlib import Path

ZONE_COLORS = {"Z_A": (0, 165, 255), "Z_C": (220, 195, 40), "Z_B": (55, 55, 240), "Z_DWELL": (215, 85, 195)}


def meta_path(video_path: Path) -> Path:
    return Path(video_path).with_name(f"{Path(video_path).stem}_meta.json")


def load_clip_meta(video_path: Path | None) -> dict | None:
    if video_path is None:
        return None
    p = meta_path(video_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def zones_from_meta(meta: dict):
    from src.security_analytics import PolygonZone
    zones = [PolygonZone(z["zone_id"], z.get("name", z["zone_id"]), [tuple(map(float, p)) for p in z["points"]],
                         role=z.get("role", "both"), normalized=True, color_bgr=ZONE_COLORS.get(z["zone_id"], (0, 200, 255)))
             for z in meta.get("zones", [])]
    forb = [tuple(r) for r in meta.get("forbidden_routes", [["Z_A", "Z_B"]])]
    return zones, forb, float(meta.get("max_dwell_sec", 5.0))


def slow_windows(meta: dict | None, pre: int = 30, post: int = 40) -> list[tuple[int, int, float, float]]:
    """[(start, end, cx_norm, cy_norm)] around each key moment."""
    if not meta:
        return []
    return [(max(1, int(k["frame"]) - pre), int(k["frame"]) + post, float(k.get("cx", 0.5)), float(k.get("cy", 0.5)))
            for k in meta.get("key_moments", [])]
