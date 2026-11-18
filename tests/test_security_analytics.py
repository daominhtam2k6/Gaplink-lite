"""Unit tests for GapLink-VMS real-world security analytics modules and OnlineTracker min_hits hygiene."""
import numpy as np
from src.tracker import OnlineTracker
from src.baselines import KalmanMahalanobisBaseline
from src.security_analytics import (
    PolygonZone,
    SecurityVMSEngine,
)


def _hist(bin_idx: int) -> np.ndarray:
    v = np.zeros((128,), dtype=np.float32)
    v[bin_idx % 128] = 1.0
    return v


def test_min_hits_filters_1frame_false_positives():
    """Verify min_hits=3 suppresses 1-2 frame detector flickers from entering lost_tracks while keeping real vehicles."""
    trk = OnlineTracker(associator=KalmanMahalanobisBaseline(theta=25.0), min_hits=3)
    # Frame 1: real car at (100,200) + ghost flicker at (700,400)
    out1 = trk.step(1, np.array([[100, 200, 40, 30], [700, 400, 35, 25]], dtype=np.float32), np.stack([_hist(1), _hist(9)]))
    assert len(out1) == 0  # tentative (hits=1 < 3)

    # Frame 2: real car continues, ghost disappears
    out2 = trk.step(2, np.array([[104, 200, 40, 30]], dtype=np.float32), np.stack([_hist(1)]))
    assert len(out2) == 0
    assert len(trk.lost_tracks) == 0  # ghost flicker did NOT pollute lost_tracks!

    # Frame 3: real car hits=3 -> confirmed!
    out3 = trk.step(3, np.array([[108, 200, 40, 30]], dtype=np.float32), np.stack([_hist(1)]))
    assert len(out3) == 1
    assert out3[0]["hits"] == 3


def test_forbidden_route_and_od_across_occlusion():
    """Verify Z_A -> Z_B triggers FORBIDDEN_ROUTE when ID is preserved across occlusion, while Z_C -> Z_B is legal."""
    z_a = PolygonZone("Z_A", "Gate Guest", [(50, 150), (200, 150), (200, 300), (50, 300)], role="entry")
    z_c = PolygonZone("Z_C", "Gate Staff", [(50, 350), (200, 350), (200, 500), (50, 500)], role="entry")
    z_b = PolygonZone("Z_B", "Restricted Vault", [(450, 150), (650, 150), (650, 500), (450, 500)], role="restricted")

    # Tracker WITH re-association vs WITHOUT re-association (C-1)
    vms_relink = SecurityVMSEngine(
        OnlineTracker(associator=KalmanMahalanobisBaseline(theta=25.0)),
        zones=[z_a, z_c, z_b],
        forbidden_routes=[("Z_A", "Z_B")],
        fps=10.0,
    )
    vms_norelink = SecurityVMSEngine(
        OnlineTracker(associator=None),
        zones=[z_a, z_c, z_b],
        forbidden_routes=[("Z_A", "Z_B")],
        fps=10.0,
    )

    for f in range(1, 26):
        # Car 1: goes Z_A (x=100..500, y=220), occluded at frames 10..15 (bus crossing)
        # Car 2: goes Z_C -> Z_B legally (x=100..500, y=420), unoccluded
        boxes, hists = [], []
        x1 = 100.0 + f * 16.0
        if not (10 <= f <= 15):
            boxes.append([x1, 220.0, 44.0, 30.0])
            hists.append(_hist(10))
        boxes.append([x1, 420.0, 44.0, 30.0])
        hists.append(_hist(50))

        b_arr = np.array(boxes, dtype=np.float32)
        h_arr = np.stack(hists, axis=0)
        vms_relink.step(f, b_arr, h_arr)
        vms_norelink.step(f, b_arr, h_arr)

    # Relink tracker catches exact Z_A -> Z_B violation AND counts legal Z_C -> Z_B without false alarm
    assert len(vms_relink.route_mon.alerts) == 1
    al = vms_relink.route_mon.alerts[0]
    assert al["origin_zone"] == "Z_A" and al["dest_zone"] == "Z_B"
    assert al["occlusions_bridged"] == 1
    assert vms_relink.route_mon.od_counts.get(("Z_A", "Z_B"), 0) == 1
    assert vms_relink.route_mon.od_counts.get(("Z_C", "Z_B"), 0) == 1

    # C-1 (no relink) misses Z_A -> Z_B violation completely and marks Z_B entry as unverified!
    assert len(vms_norelink.route_mon.alerts) == 0
    assert len(vms_norelink.route_mon.unverified_entries.get("Z_B", set())) == 1


def test_dwell_time_survives_passing_occluder_and_target_lock():
    """Verify stopped car inside restricted zone accumulates continuous dwell timer across occlusion without resetting."""
    z_dwell = PolygonZone("Z_NO_STOP", "Bank Gate Curb", [(200, 150), (400, 150), (400, 300), (200, 300)], role="dwell")
    vms_relink = SecurityVMSEngine(
        OnlineTracker(associator=KalmanMahalanobisBaseline(theta=25.0)),
        zones=[z_dwell],
        max_dwell_sec=2.0,
        fps=10.0,
    )
    vms_norelink = SecurityVMSEngine(
        OnlineTracker(associator=None),
        zones=[z_dwell],
        max_dwell_sec=2.0,
        fps=10.0,
    )

    for f in range(1, 28):
        # Stopped car at (300, 220), occluded twice at frames 8..11 and 17..19 (< 2.0s visible chunks!)
        if f in (8, 9, 10, 11, 17, 18, 19):
            b_arr = np.zeros((0, 4), dtype=np.float32)
            h_arr = np.zeros((0, 128), dtype=np.float32)
        else:
            b_arr = np.array([[300.0, 220.0, 46.0, 32.0]], dtype=np.float32)
            h_arr = np.stack([_hist(3)])

        res_r = vms_relink.step(f, b_arr, h_arr)
        vms_norelink.step(f, b_arr, h_arr)
        if f == 2:
            locked_tid = vms_relink.target_lock.lock_by_click(302.0, 221.0, res_r["tracks"], frame_id=f)
            assert locked_tid is not None

    # Relink bridges both occlusions -> triggers DWELL_TIME_ALERT at f=21 (2.0s) and re-acquires target lock twice!
    assert len(vms_relink.dwell_mon.alerts) == 1
    assert vms_relink.dwell_mon.alerts[0]["occlusions_bridged"] >= 1
    assert vms_relink.target_lock.reacquire_count == 2

    # Without relink, timer resets to 0.0s on every passing bus -> 0 alerts!
    assert len(vms_norelink.dwell_mon.alerts) == 0
