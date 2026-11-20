"""Real-world Vehicle Security Analytics Engine (GapLink-VMS).

Provides three concrete, OCR-free security camera modules directly on top of OnlineTracker:
  1. ZoneRouteMonitor: Zone-to-zone forbidden maneuver alerts (Z_A -> Z_B) & Origin-Destination (OD) flow matrix.
  2. DwellTimeMonitor: Restricted-zone stopping / anti-loitering dwell-time alerts resilient to passing occluders.
  3. TargetLockManager: Click-to-lock suspect pursuit with 2-sigma uncertainty zone and Dustbin hold protection.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import cv2
import numpy as np

from src.baselines import kalman_predict_1track


def crop_vehicle_b64(frame_bgr: np.ndarray | None, box_cxcywh: np.ndarray, pad: float = 0.22, max_side: int = 140) -> str:
    """Extract a padded vehicle crop from a real camera frame and encode as base64 JPEG for alert evidence."""
    if frame_bgr is None or frame_bgr.size == 0:
        return ""
    H, W = frame_bgr.shape[:2]
    cx, cy, w, h = [float(v) for v in box_cxcywh]
    pw, ph = w * (1.0 + 2.0 * pad), h * (1.0 + 2.0 * pad)
    x1 = int(np.clip(round(cx - 0.5 * pw), 0, W - 1))
    y1 = int(np.clip(round(cy - 0.5 * ph), 0, H - 1))
    x2 = int(np.clip(round(cx + 0.5 * pw), x1 + 1, W))
    y2 = int(np.clip(round(cy + 0.5 * ph), y1 + 1, H))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    ch, cw = crop.shape[:2]
    if max(ch, cw) > max_side:
        scale = max_side / float(max(ch, cw))
        crop = cv2.resize(crop, (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))))
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class PolygonZone:
    """2D polygonal region on the camera view (supports either pixel or [0,1]-normalized coordinates)."""
    zone_id: str
    name: str
    points: list[tuple[float, float]]
    role: str = "both"  # "entry", "exit", "restricted", "dwell", "both"
    normalized: bool = False
    color_bgr: tuple[int, int, int] = (0, 200, 255)

    def to_pixel_array(self, W: int = 960, H: int = 540) -> np.ndarray:
        pts = np.asarray(self.points, dtype=np.float32)
        if self.normalized:
            pts = pts.copy()
            pts[:, 0] *= float(W)
            pts[:, 1] *= float(H)
        return pts

    def contains_point(self, x: float, y: float, W: int = 960, H: int = 540) -> bool:
        poly = self.to_pixel_array(W, H)
        if len(poly) < 3:
            return False
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0.0

    def contains_box(self, box_cxcywh: np.ndarray, W: int = 960, H: int = 540) -> bool:
        cx, cy, _, h = [float(v) for v in box_cxcywh]
        # Check either ground-contact anchor (cx, cy + 0.25*h) or geometric center (cx, cy)
        return self.contains_point(cx, cy + 0.25 * h, W, H) or self.contains_point(cx, cy, W, H)


class ZoneRouteMonitor:
    """Tracks Origin-Destination (OD) zone flows and detects forbidden zone-to-zone routes (e.g. Z_A -> Z_B)."""

    def __init__(
        self,
        zones: list[PolygonZone],
        forbidden_routes: list[tuple[str, str]] | None = None,
        fps: float = 25.0,
        W: int = 960,
        H: int = 540,
    ):
        self.zones = [z for z in zones if z.role != "dwell"]
        self.zone_map = {z.zone_id: z for z in self.zones}
        self.forbidden_routes = set(forbidden_routes or [])
        self.fps = max(1.0, float(fps))
        self.W = int(W)
        self.H = int(H)

        self.track_origin: dict[int, dict] = {}
        self.track_zone_seq: dict[int, list[str]] = {}
        self.completed_od: set[tuple[int, str, str]] = set()
        self.od_counts: dict[tuple[str, str], int] = {}
        self.unverified_entries: dict[str, set[int]] = {}
        self.alerts: list[dict] = []
        self.true_alerts: list[dict] = []
        self.false_alarms: list[dict] = []
        self._next_alert_id = 1

    def step(
        self,
        frame_id: int,
        active_outputs: list[dict],
        reassoc_by_track: dict[int, int] | None = None,
        wrong_reassoc_tids: set[int] | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> list[dict]:
        reassoc_by_track = reassoc_by_track or {}
        wrong_reassoc_tids = wrong_reassoc_tids or set()
        new_alerts = []

        for out in active_outputs:
            tid = int(out["track_id"])
            box = out["box"]
            matched_zones = [z for z in self.zones if z.contains_box(box, self.W, self.H)]
            if not matched_zones:
                continue

            for z in matched_zones:
                zid = z.zone_id
                seq_list = self.track_zone_seq.setdefault(tid, [])
                if not seq_list or seq_list[-1] != zid:
                    seq_list.append(zid)

                if tid not in self.track_origin:
                    if z.role in ("entry", "both"):
                        self.track_origin[tid] = {
                            "zone_id": zid,
                            "frame": int(frame_id),
                            "box": box.copy(),
                            "gt_tid": int(out.get("initial_gt_tid", out.get("gt_tid", -1))),
                            "reassoc_cnt": int(reassoc_by_track.get(tid, 0)),
                            "crop_b64": crop_vehicle_b64(frame_bgr, box),
                        }
                    elif z.role in ("exit", "restricted"):
                        # Track spawned mid-scene (broken ID or abstained dustbin) without known entry zone
                        self.unverified_entries.setdefault(zid, set()).add(tid)
                else:
                    orig = self.track_origin[tid]
                    src_z = orig["zone_id"]
                    if zid != src_z and z.role in ("exit", "restricted", "both"):
                        key = (tid, src_z, zid)
                        if key not in self.completed_od:
                            self.completed_od.add(key)
                            self.od_counts[(src_z, zid)] = self.od_counts.get((src_z, zid), 0) + 1

                            if (src_z, zid) in self.forbidden_routes:
                                n_reassoc = int(reassoc_by_track.get(tid, 0)) - int(orig["reassoc_cnt"])
                                g_orig = int(orig["gt_tid"])
                                g_dest = int(out.get("gt_tid", -1))
                                is_fp = bool(
                                    (tid in wrong_reassoc_tids)
                                    or (n_reassoc > 0 and g_orig != -1 and g_dest != -1 and g_orig != g_dest)
                                )
                                alert = {
                                    "alert_id": f"RT-{self._next_alert_id:03d}",
                                    "type": "FORBIDDEN_ROUTE",
                                    "frame": int(frame_id),
                                    "time_sec": round(frame_id / self.fps, 2),
                                    "track_id": tid,
                                    "origin_zone": src_z,
                                    "dest_zone": zid,
                                    "entry_frame": int(orig["frame"]),
                                    "transit_sec": round((frame_id - orig["frame"]) / self.fps, 2),
                                    "occlusions_bridged": max(0, n_reassoc),
                                    "origin_gt_tid": g_orig,
                                    "dest_gt_tid": g_dest,
                                    "is_false_alarm": is_fp,
                                    "entry_crop": orig["crop_b64"],
                                    "alert_crop": crop_vehicle_b64(frame_bgr, box),
                                    "box": box.copy(),
                                }
                                self._next_alert_id += 1
                                self.alerts.append(alert)
                                if is_fp:
                                    self.false_alarms.append(alert)
                                else:
                                    self.true_alerts.append(alert)
                                new_alerts.append(alert)
        return new_alerts


class DwellTimeMonitor:
    """Monitors vehicle dwell time inside restricted / no-stopping security zones across occlusions."""

    def __init__(
        self,
        dwell_zones: list[PolygonZone],
        max_dwell_sec: float = 6.0,
        fps: float = 25.0,
        exit_grace_frames: int = 6,
        W: int = 960,
        H: int = 540,
    ):
        self.zones = [z for z in dwell_zones if z.role in ("dwell", "restricted", "both")]
        self.max_dwell_sec = float(max_dwell_sec)
        self.fps = max(1.0, float(fps))
        self.exit_grace_frames = int(exit_grace_frames)
        self.W = int(W)
        self.H = int(H)

        # Keyed by (zone_id, track_id)
        self.states: dict[tuple[str, int], dict] = {}
        self.alerts: list[dict] = []
        self.true_alerts: list[dict] = []
        self.false_alarms: list[dict] = []
        self._next_alert_id = 1

    def step(
        self,
        frame_id: int,
        active_outputs: list[dict],
        wrong_reassoc_tids: set[int] | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> tuple[list[dict], dict[int, float]]:
        """Update dwell timers and return (new_alerts, active_dwell_seconds_by_track_id)."""
        wrong_reassoc_tids = wrong_reassoc_tids or set()
        new_alerts = []
        active_timers: dict[int, float] = {}

        for out in active_outputs:
            tid = int(out["track_id"])
            box = out["box"]
            for z in self.zones:
                key = (z.zone_id, tid)
                in_z = z.contains_box(box, self.W, self.H)
                if in_z:
                    st = self.states.get(key)
                    if st is None:
                        st = {
                            "first_frame": int(frame_id),
                            "last_in_frame": int(frame_id),
                            "outside_streak": 0,
                            "occlusions_bridged": 0,
                            "alerted": False,
                            "anchor_box": box.copy(),
                            "origin_gt_tid": int(out.get("initial_gt_tid", out.get("gt_tid", -1))),
                            "entry_crop": crop_vehicle_b64(frame_bgr, box),
                        }
                        self.states[key] = st
                    else:
                        gap = int(frame_id) - int(st["last_in_frame"])
                        if gap >= 2 and st["outside_streak"] == 0:
                            st["occlusions_bridged"] += 1
                        st["last_in_frame"] = int(frame_id)
                        st["outside_streak"] = 0

                        # Stationary velocity check: moving vehicles passing through the zone must NOT accumulate dwell time!
                        anc = st["anchor_box"]
                        h_ref = max(16.0, float(anc[3]))
                        drift = float(np.hypot(float(box[0] - anc[0]), float(box[1] - anc[1]))) / h_ref
                        is_hijacked_id = (tid in wrong_reassoc_tids) or (
                            st["occlusions_bridged"] > 0
                            and st["origin_gt_tid"] != -1
                            and int(out.get("gt_tid", -1)) != -1
                            and st["origin_gt_tid"] != int(out.get("gt_tid", -1))
                        )
                        if drift > 0.45 and not st["alerted"] and not is_hijacked_id:
                            # Vehicle is actively moving along the road -> reset stationary timer anchor
                            st["first_frame"] = int(frame_id)
                            st["anchor_box"] = box.copy()
                            st["occlusions_bridged"] = 0

                    dwell_s = (int(frame_id) - int(st["first_frame"])) / self.fps
                    active_timers[tid] = max(active_timers.get(tid, 0.0), dwell_s)

                    if dwell_s >= self.max_dwell_sec and not st["alerted"]:
                        st["alerted"] = True
                        g_orig = int(st["origin_gt_tid"])
                        g_now = int(out.get("gt_tid", -1))
                        n_occ = int(st["occlusions_bridged"])
                        is_fp = bool(
                            (tid in wrong_reassoc_tids)
                            or (n_occ > 0 and g_orig != -1 and g_now != -1 and g_orig != g_now)
                        )
                        alert = {
                            "alert_id": f"DW-{self._next_alert_id:03d}",
                            "type": "DWELL_TIME_ALERT",
                            "frame": int(frame_id),
                            "time_sec": round(frame_id / self.fps, 2),
                            "zone_id": z.zone_id,
                            "track_id": tid,
                            "dwell_sec": round(dwell_s, 2),
                            "max_dwell_sec": self.max_dwell_sec,
                            "occlusions_bridged": int(st["occlusions_bridged"]),
                            "origin_gt_tid": g_orig,
                            "now_gt_tid": g_now,
                            "is_false_alarm": is_fp,
                            "entry_crop": st["entry_crop"],
                            "alert_crop": crop_vehicle_b64(frame_bgr, box),
                            "box": box.copy(),
                        }
                        self._next_alert_id += 1
                        self.alerts.append(alert)
                        if is_fp:
                            self.false_alarms.append(alert)
                        else:
                            self.true_alerts.append(alert)
                        new_alerts.append(alert)
                else:
                    st = self.states.get(key)
                    if st is not None:
                        st["outside_streak"] += 1
                        if st["outside_streak"] >= self.exit_grace_frames:
                            del self.states[key]

        return new_alerts, active_timers


class TargetLockManager:
    """Click-to-Lock suspect target pursuit across occlusions with 2-sigma search zone & Dustbin protection."""

    def __init__(self, fps: float = 25.0, reacquire_flash_frames: int = 25):
        self.fps = max(1.0, float(fps))
        self.reacquire_flash_frames = int(reacquire_flash_frames)
        self.reacquire_count: int = 0
        self.hijack_count: int = 0
        self.dustbin_holds: int = 0
        self.events: list[dict] = []
        self.reset(clear_totals=True)

    def reset(self, clear_totals: bool = True):
        self.locked_track_id: int | None = None
        self.lock_start_frame: int = 0
        self.locked_initial_gt_tid: int = -1
        self.locked_initial_hist: np.ndarray | None = None
        self.locked_initial_box: np.ndarray | None = None
        self.lock_crop_b64: str = ""
        self.lock_crop_img: np.ndarray | None = None
        self.state: str = "IDLE"  # IDLE, LOCKED, OCCLUDED_SEARCHING, AMBIGUOUS_HOLD, REACQUIRED, HIJACKED_WRONG, LOST_EXPIRED
        self.trail: list[tuple[float, float]] = []
        self.reacquire_until_frame: int = 0
        self.last_gap_frames: int = 0
        self._counted_hijack_for_lock: bool = False
        if clear_totals:
            self.reacquire_count = 0
            self.hijack_count = 0
            self.dustbin_holds = 0
            self.events = []

    def lock_by_track_id(
        self,
        track_id: int,
        active_outputs: list[dict],
        frame_id: int = 0,
        frame_bgr: np.ndarray | None = None,
        clear_totals: bool = False,
    ) -> bool:
        for o in active_outputs:
            if int(o["track_id"]) == int(track_id):
                self.reset(clear_totals=clear_totals)
                self.locked_track_id = int(track_id)
                self.lock_start_frame = int(frame_id)
                self.locked_initial_gt_tid = int(o.get("gt_tid", -1))
                self.locked_initial_box = o["box"].copy()
                if "hist" in o and o["hist"] is not None:
                    self.locked_initial_hist = o["hist"].copy()
                self.lock_crop_b64 = crop_vehicle_b64(frame_bgr, o["box"])
                if frame_bgr is not None and frame_bgr.size > 0:
                    H, W = frame_bgr.shape[:2]
                    cx, cy, w, h = [float(v) for v in o["box"]]
                    x1 = int(np.clip(round(cx - 0.6 * w), 0, W - 1))
                    y1 = int(np.clip(round(cy - 0.6 * h), 0, H - 1))
                    x2 = int(np.clip(round(cx + 0.6 * w), x1 + 1, W))
                    y2 = int(np.clip(round(cy + 0.6 * h), y1 + 1, H))
                    self.lock_crop_img = frame_bgr[y1:y2, x1:x2].copy()
                self.state = "LOCKED"
                self.trail.append((float(o["box"][0]), float(o["box"][1])))
                return True
        return False

    def lock_by_click(
        self,
        x: float,
        y: float,
        active_outputs: list[dict],
        frame_id: int = 0,
        frame_bgr: np.ndarray | None = None,
        max_dist: float = 75.0,
        clear_totals: bool = False,
    ) -> int | None:
        if not active_outputs:
            return None
        best_tid = None
        best_score = 1e9
        for o in active_outputs:
            cx, cy, w, h = [float(v) for v in o["box"]]
            in_box = (abs(x - cx) <= 0.6 * w) and (abs(y - cy) <= 0.6 * h)
            dist = float(np.hypot(x - cx, y - cy))
            score = dist * (0.25 if in_box else 1.0)
            if (in_box or dist <= max_dist) and score < best_score:
                best_score = score
                best_tid = int(o["track_id"])
        if best_tid is not None:
            self.lock_by_track_id(best_tid, active_outputs, frame_id=frame_id, frame_bgr=frame_bgr, clear_totals=clear_totals)
        return best_tid

    def step(
        self,
        frame_id: int,
        active_outputs: list[dict],
        tracker,
        associator=None,
        wrong_reassoc_tids: set[int] | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> dict:
        if self.locked_track_id is None:
            return {"state": "IDLE", "track_id": None}
        wrong_reassoc_tids = wrong_reassoc_tids or set()

        # 1. Check if locked target is currently active/visible
        for o in active_outputs:
            if int(o["track_id"]) == self.locked_track_id:
                box = o["box"]
                now_gt = int(o.get("gt_tid", -1))
                if self.locked_initial_gt_tid != -1 and now_gt != -1:
                    is_hijacked = bool(now_gt != self.locked_initial_gt_tid)
                else:
                    # GT-free hijack detection for user-uploaded custom videos
                    is_hijacked = bool(self.locked_track_id in wrong_reassoc_tids)
                    if (
                        not is_hijacked
                        and self.state in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD")
                        and self.locked_initial_hist is not None
                        and o.get("hist") is not None
                    ):
                        h0 = self.locked_initial_hist
                        h1 = o["hist"]
                        n0, n1 = float(np.linalg.norm(h0)), float(np.linalg.norm(h1))
                        if n0 > 1e-6 and n1 > 1e-6:
                            cs = float(np.dot(h0, h1) / (n0 * n1 + 1e-8))
                            if cs < 0.56:
                                is_hijacked = True
                if is_hijacked:
                    if not self._counted_hijack_for_lock:
                        self.hijack_count += 1
                        self._counted_hijack_for_lock = True
                    self.state = "HIJACKED_WRONG"
                else:
                    if self.state in ("OCCLUDED_SEARCHING", "AMBIGUOUS_HOLD"):
                        self.reacquire_count += 1
                        self.reacquire_until_frame = int(frame_id) + self.reacquire_flash_frames
                        ev = {
                            "type": "TARGET_REACQUIRED",
                            "frame": int(frame_id),
                            "track_id": self.locked_track_id,
                            "gap_frames": int(self.last_gap_frames),
                            "gap_sec": round(self.last_gap_frames / self.fps, 2),
                            "now_gt_tid": now_gt,
                            "origin_gt_tid": int(self.locked_initial_gt_tid),
                            "crop_b64": crop_vehicle_b64(frame_bgr, box),
                        }
                        self.events.append(ev)

                    self.state = "REACQUIRED" if frame_id <= self.reacquire_until_frame else "LOCKED"
                self.trail.append((float(box[0]), float(box[1])))
                if len(self.trail) > 150:
                    self.trail = self.trail[-150:]
                return {
                    "state": self.state,
                    "track_id": self.locked_track_id,
                    "box": box.copy(),
                    "pred_box": None,
                    "sigma_xy": None,
                    "gap_frames": 0,
                    "reacquire_count": self.reacquire_count,
                    "hijack_count": self.hijack_count,
                    "dustbin_holds": self.dustbin_holds,
                    "trail": list(self.trail),
                    "now_gt_tid": now_gt,
                    "origin_gt_tid": int(self.locked_initial_gt_tid),
                }

        # 2. Check if locked target is in tracker.lost_tracks (occluded)
        # First check if the true target vehicle (locked_initial_gt_tid) has already re-emerged under a DIFFERENT track_id
        # (meaning this tracker failed to re-associate the locked target and fragmented/hijacked it!)
        if self.locked_initial_gt_tid != -1:
            for o in active_outputs:
                if int(o.get("gt_tid", -1)) == self.locked_initial_gt_tid and int(o["track_id"]) != self.locked_track_id:
                    if not self._counted_hijack_for_lock:
                        self.hijack_count += 1
                        self._counted_hijack_for_lock = True
                    self.state = "HIJACKED_WRONG"
                    return {
                        "state": "HIJACKED_WRONG",
                        "track_id": self.locked_track_id,
                        "box": o["box"].copy(),
                        "pred_box": None,
                        "sigma_xy": None,
                        "gap_frames": self.last_gap_frames,
                        "reacquire_count": self.reacquire_count,
                        "hijack_count": self.hijack_count,
                        "dustbin_holds": self.dustbin_holds,
                        "trail": list(self.trail),
                        "now_gt_tid": int(o["track_id"]),
                        "origin_gt_tid": int(self.locked_initial_gt_tid),
                    }

        lost_obj = None
        for lt in tracker.lost_tracks:
            if int(lt.track_id) == self.locked_track_id:
                lost_obj = lt
                break

        if lost_obj is not None and 1 <= lost_obj.age <= tracker.a_max:
            self.last_gap_frames = int(frame_id - lost_obj.last_frame)
            lost_obj.age = self.last_gap_frames
            if associator is not None and hasattr(associator, "infer_event"):
                vis_now = (
                    np.stack([o["box"] for o in active_outputs], axis=0)
                    if active_outputs
                    else np.zeros((0, 4), dtype=np.float32)
                )
                ev_tmp = tracker.build_event_from_subpools(
                    frame_id,
                    [lost_obj],
                    np.zeros((1, 4), dtype=np.float32),
                    np.zeros((1, 128), dtype=np.float32),
                    vis_now,
                )
                inf = associator.infer_event(ev_tmp)
                pred_box = inf["pred_box"][0].copy()
                sigma_xy = inf["sigma_cal"][0, :2].copy()
            else:
                h_pad, h_len = lost_obj.get_padded_hist()
                mu, var = kalman_predict_1track(h_pad, h_len, self.last_gap_frames)
                pred_box = mu.copy()
                sigma_xy = np.sqrt(np.maximum(1.0, var[:2])).astype(np.float32)

            # Check if newly spawned tracks appeared nearby during this frame while associator held off (Dustbin protection)
            spawned_nearby = False
            for o in active_outputs:
                if int(o.get("hits", 1)) == tracker.min_hits:
                    dist = float(np.hypot(o["box"][0] - pred_box[0], o["box"][1] - pred_box[1]))
                    if dist <= 3.0 * max(20.0, float(max(sigma_xy[0], sigma_xy[1]))):
                        spawned_nearby = True
                        break
            if spawned_nearby:
                self.dustbin_holds += 1
                self.state = "AMBIGUOUS_HOLD"
            elif self.state != "AMBIGUOUS_HOLD":
                self.state = "OCCLUDED_SEARCHING"

            return {
                "state": self.state,
                "track_id": self.locked_track_id,
                "box": None,
                "pred_box": pred_box,
                "sigma_xy": sigma_xy,
                "gap_frames": self.last_gap_frames,
                "reacquire_count": self.reacquire_count,
                "hijack_count": self.hijack_count,
                "dustbin_holds": self.dustbin_holds,
                "trail": list(self.trail),
                "now_gt_tid": -1,
                "origin_gt_tid": int(self.locked_initial_gt_tid),
            }

        # 3. Target exceeded max occlusion age without re-association
        if not self._counted_hijack_for_lock:
            self.hijack_count += 1
            self._counted_hijack_for_lock = True
        self.state = "LOST_EXPIRED"
        return {
            "state": "LOST_EXPIRED",
            "track_id": self.locked_track_id,
            "box": None,
            "pred_box": None,
            "sigma_xy": None,
            "gap_frames": self.last_gap_frames,
            "reacquire_count": self.reacquire_count,
            "hijack_count": self.hijack_count,
            "dustbin_holds": self.dustbin_holds,
            "trail": list(self.trail),
            "now_gt_tid": -1,
            "origin_gt_tid": int(self.locked_initial_gt_tid),
        }


class SecurityVMSEngine:
    """Unified 3-in-1 security analytics wrapper around OnlineTracker."""

    def __init__(
        self,
        tracker,
        zones: list[PolygonZone] | None = None,
        forbidden_routes: list[tuple[str, str]] | None = None,
        max_dwell_sec: float = 6.0,
        fps: float = 25.0,
    ):
        self.tracker = tracker
        self.fps = float(fps)
        zones = zones or []
        self.zones = zones
        self.route_mon = ZoneRouteMonitor(
            zones=zones,
            forbidden_routes=forbidden_routes,
            fps=fps,
            W=tracker.W,
            H=tracker.H,
        )
        self.dwell_mon = DwellTimeMonitor(
            dwell_zones=zones,
            max_dwell_sec=max_dwell_sec,
            fps=fps,
            W=tracker.W,
            H=tracker.H,
        )
        self.target_lock = TargetLockManager(fps=fps)
        self._reassoc_counter: dict[int, int] = {}
        self._wrong_reassoc_tids: set[int] = set()
        self._last_log_idx: int = 0

    def step(
        self,
        frame_id: int,
        det_boxes: np.ndarray,
        det_hists: np.ndarray,
        det_gt_tids: np.ndarray | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> dict:
        outputs = self.tracker.step(frame_id, det_boxes, det_hists, det_gt_tids)
        while self._last_log_idx < len(self.tracker.reassoc_log):
            rec = self.tracker.reassoc_log[self._last_log_idx]
            tid = int(rec["track_id"])
            self._reassoc_counter[tid] = self._reassoc_counter.get(tid, 0) + 1
            gb = int(rec.get("gt_tid_before", -1))
            ga = int(rec.get("gt_tid_after", -1))
            if gb != -1 and ga != -1:
                if gb != ga:
                    self._wrong_reassoc_tids.add(tid)
            else:
                # GT-free appearance & kinematic mismatch check for custom uploaded videos
                cs = float(rec.get("color_cos_sim", 1.0))
                sz_d = float(rec.get("size_log_diff", 0.0))
                pj = float(rec.get("pos_jump_norm", 0.0))
                if cs < 0.56 or sz_d > 0.62 or (cs < 0.68 and pj > 3.6):
                    self._wrong_reassoc_tids.add(tid)
            self._last_log_idx += 1

        route_alerts = self.route_mon.step(
            frame_id,
            outputs,
            reassoc_by_track=self._reassoc_counter,
            wrong_reassoc_tids=self._wrong_reassoc_tids,
            frame_bgr=frame_bgr,
        )
        dwell_alerts, dwell_timers = self.dwell_mon.step(
            frame_id,
            outputs,
            wrong_reassoc_tids=self._wrong_reassoc_tids,
            frame_bgr=frame_bgr,
        )
        lock_info = self.target_lock.step(
            frame_id,
            outputs,
            self.tracker,
            associator=self.tracker.associator,
            wrong_reassoc_tids=self._wrong_reassoc_tids,
            frame_bgr=frame_bgr,
        )

        return {
            "frame": int(frame_id),
            "tracks": outputs,
            "route_alerts": route_alerts,
            "dwell_alerts": dwell_alerts,
            "dwell_timers": dwell_timers,
            "lock": lock_info,
        }


def auto_calibrate_zones_from_detections(
    dets_by_frame: dict[int, dict],
    W: int = 960,
    H: int = 540,
    zone_colors: dict[str, tuple[int, int, int]] | None = None,
) -> list[PolygonZone]:
    """Automatically fit Z_A (Restricted Origin), Z_C (Permitted Origin), Z_B (Destination), and Z_DWELL
    to the dominant traffic flow of ANY user-uploaded camera video.
    """
    from src.tracker import OnlineTracker

    colors = zone_colors or {
        "Z_A": (0, 165, 255),
        "Z_C": (220, 195, 40),
        "Z_B": (55, 55, 240),
        "Z_DWELL": (215, 85, 195),
    }
    trk = OnlineTracker(associator=None, W=W, H=H, min_hits=2)
    track_pts: dict[int, list[np.ndarray]] = {}
    for f in sorted(dets_by_frame.keys())[:180]:
        d = dets_by_frame[f]
        outs = trk.step(f, d["boxes"], d["hists"])
        for o in outs:
            track_pts.setdefault(int(o["track_id"]), []).append(o["box"][:2].copy())

    disps = []
    all_pts = []
    for pts in track_pts.values():
        if len(pts) >= 4:
            all_pts.extend(pts)
        if len(pts) >= 8:
            dx = float(pts[-1][0] - pts[0][0])
            dy = float(pts[-1][1] - pts[0][1])
            if np.hypot(dx, dy) >= 35.0:
                disps.append([dx, dy])

    if len(disps) >= 3:
        med_v = np.median(np.array(disps, dtype=np.float32), axis=0)
        norm = float(np.hypot(med_v[0], med_v[1]))
        u_flow = (med_v / norm) if norm > 1e-3 else np.array([0.0, 1.0], dtype=np.float32)
    else:
        u_flow = np.array([0.0, 1.0], dtype=np.float32)
    u_perp = np.array([-u_flow[1], u_flow[0]], dtype=np.float32)

    pts_arr = np.array(all_pts, dtype=np.float32) if len(all_pts) >= 6 else np.array([[0.5 * W, 0.5 * H]], dtype=np.float32)
    center = np.median(pts_arr, axis=0)
    rel = pts_arr - center[None, :]
    s_proj = rel @ u_flow
    p_proj = rel @ u_perp

    if len(s_proj) >= 6:
        s_10, s_40, s_60, s_94 = np.percentile(s_proj, [8, 38, 62, 94])
        p_02, p_med, p_98 = np.percentile(p_proj, [3, 50, 97])
    else:
        s_10, s_40, s_60, s_94 = -0.32 * H, -0.06 * H, 0.08 * H, 0.36 * H
        p_02, p_med, p_98 = -0.35 * W, 0.0, 0.35 * W
    pad_p = 35.0

    def norm_poly(s_lo, s_hi, p_lo, p_hi) -> list[tuple[float, float]]:
        corners = [
            center + s_lo * u_flow + p_lo * u_perp,
            center + s_hi * u_flow + p_lo * u_perp,
            center + s_hi * u_flow + p_hi * u_perp,
            center + s_lo * u_flow + p_hi * u_perp,
        ]
        return [
            (
                round(float(np.clip(c[0] / float(W), 0.01, 0.99)), 3),
                round(float(np.clip(c[1] / float(H), 0.01, 0.99)), 3),
            )
            for c in corners
        ]

    return [
        PolygonZone("Z_A", "Z_A: Nhanh Cam", norm_poly(s_10 - 20, s_40, p_02 - pad_p, p_med), role="entry", normalized=True, color_bgr=colors["Z_A"]),
        PolygonZone("Z_C", "Z_C: Nhanh Hop Le", norm_poly(s_10 - 20, s_40, p_med, p_98 + pad_p), role="entry", normalized=True, color_bgr=colors["Z_C"]),
        PolygonZone("Z_B", "Z_B: Vung Cam Xam Nhap", norm_poly(s_60, s_94 + 30, p_02 - pad_p, p_98 + pad_p), role="restricted", normalized=True, color_bgr=colors["Z_B"]),
        PolygonZone("Z_DWELL", "Z_DWELL: Vung Cam Dung Do (>5.0s)", [(0.10, 0.20), (0.90, 0.20), (0.90, 0.88), (0.10, 0.88)], role="dwell", normalized=True, color_bgr=colors["Z_DWELL"]),
    ]


def auto_pick_best_lock_schedule(
    dets_by_frame: dict[int, dict],
    gl_assoc,
    W: int = 960,
    H: int = 540,
) -> list[tuple[int, float, float]]:
    """Find prominent vehicles in any uploaded video that undergo a clean occlusion recovery under GapLink-Lite."""
    from src.tracker import OnlineTracker

    trk = OnlineTracker(associator=gl_assoc, W=W, H=H, min_hits=3)
    first_seen: dict[int, tuple[int, float, float, float]] = {}
    for f in sorted(dets_by_frame.keys())[:240]:
        d = dets_by_frame[f]
        outs = trk.step(f, d["boxes"], d["hists"])
        for o in outs:
            tid = int(o["track_id"])
            if tid not in first_seen and f >= 5:
                area = float(o["box"][2] * o["box"][3])
                first_seen[tid] = (f, float(o["box"][0]), float(o["box"][1]), area)

    best = None
    best_score = -1.0
    for rec in trk.reassoc_log:
        tid = int(rec["track_id"])
        gap = int(rec["gap"])
        cs = float(rec.get("color_cos_sim", 0.8))
        if tid in first_seen and gap >= 4 and cs >= 0.62:
            f0, cx0, cy0, area = first_seen[tid]
            score = gap * np.sqrt(max(100.0, area)) * cs
            if score > best_score and rec["frame"] > f0 + 4:
                best_score = score
                best = (f0 + 1, cx0, cy0)

    if best is not None:
        return [best]
    # Fallback: pick largest vehicle visible in frames 8..30
    for f in sorted(dets_by_frame.keys())[5:30]:
        boxes = dets_by_frame[f]["boxes"]
        if len(boxes) > 0:
            areas = boxes[:, 2] * boxes[:, 3]
            idx = int(np.argmax(areas))
            return [(f, float(boxes[idx, 0]), float(boxes[idx, 1]))]
    return [(10, 0.5 * W, 0.5 * H)]
