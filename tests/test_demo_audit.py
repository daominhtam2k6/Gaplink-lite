"""scripts/demo_audit.py: one-to-one matching, phantom (part-of-vehicle) boxes, every ID change."""
from scripts.demo_audit import Auditor, match, phantoms

CAR = [100, 100, 60, 40]


def test_match_is_one_to_one():
    m = match({1: CAR, 2: [102, 100, 60, 40]}, [(7, CAR)])
    assert list(m.values()) == [7] and len(m) == 1


def test_phantom_inside_another_track():
    assert phantoms({1: [200, 100, 120, 60], 2: [230, 100, 50, 50]}) == {2}


def _feed(aud, frames):
    for f, tracks, gts in frames:
        aud.add(f, tracks, gts)


def test_relink_same_vehicle_and_switch_counted():
    a = Auditor(960, 540)
    fr = [(f, {1: CAR}, [(7, CAR)]) for f in range(1, 11)]
    fr += [(f, {}, [(7, CAR)]) for f in range(11, 31)]           # hidden, GT still annotated
    fr += [(f, {1: CAR}, [(7, CAR)]) for f in range(31, 41)]       # same ID back
    fr += [(f, {5: CAR}, [(7, CAR)]) for f in range(41, 51)]       # then a new ID: visible switch
    _feed(a, fr)
    r = a.report([{"frame": 31, "tid": 1, "gap": 21}])
    assert r["links"][0]["verdict"] == "same vehicle" and r["links_strong_same"] == 0 and r["links_same"] == 1
    assert len(r["id_switches"]) == 1 and r["id_switches"][0]["new"] == 5 and r["visible_errors"] == 1


def test_part_of_vehicle_link_is_phantom():
    truck = [300, 200, 200, 60]
    bed = [350, 200, 90, 55]                                       # trailer box inside the truck GT
    a = Auditor(960, 540)
    _feed(a, [(f, {1: truck, 2: bed}, [(9, truck)]) for f in range(1, 21)])
    r = a.report([{"frame": 20, "tid": 2, "gap": 15}])
    assert r["links"][0]["verdict"] == "PHANTOM" and r["links_phantom_flashed"] == 1 and r["visible_errors"] == 1


def test_wrong_link():
    other = [400, 300, 60, 40]
    a = Auditor(960, 540)
    _feed(a, [(f, {1: CAR}, [(7, CAR), (8, other)]) for f in range(1, 11)]
          + [(f, {}, [(7, CAR), (8, other)]) for f in range(11, 26)]
          + [(f, {1: other}, [(7, CAR), (8, other)]) for f in range(26, 30)])
    r = a.report([{"frame": 26, "tid": 1, "gap": 15}])
    assert r["links"][0]["verdict"] == "WRONG" and r["links_wrong"] == 1
