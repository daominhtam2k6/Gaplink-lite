"""Rehearsal check of a showcase clip through the SAME path as an upload (prepare: 25 fps, 960 px, YOLOv8n) and
GapLink-Lite, audited against the UA-DETRAC GT of its source frames (offline only: nothing is shown in the console).

    python -m scripts.check_showcase demo/showcase/demo1.mp4 [--follow 17,21]

Strict audit (scripts/demo_audit.py): one-to-one matching to GT, phantom boxes (part of another vehicle), every ID
change of a GT vehicle. Writes runs/check_showcase/<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gt_frame(seq, f):
    sl = seq.by_frame.get(f)
    rows = seq.rows[sl] if sl is not None else np.zeros((0, 9))
    return [(int(x[1]), [x[2] + x[4] / 2, x[3] + x[5] / 2, x[4], x[5]]) for x in rows]


def run_audit(runner, n: int, gt_of, frames=None) -> dict:
    """Step a LiveRunner over n frames (frames: optional iterator of images) and audit it."""
    from scripts.demo_audit import Auditor
    aud = Auditor(runner.W, runner.H)
    for f in range(1, n + 1):
        frm = next(frames) if frames is not None else None
        runner.step(f, frm)
        tracks = {int(o["track_id"]): [float(v) for v in o["box"][:4]] for o in runner.pane.last["tracks"]}
        aud.add(f, tracks, gt_of(f))
    return aud.report(runner.links)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    a = ap.parse_args()
    from demo.gaplink_live import make_gaplink
    from demo.gaplink_live_core import LiveRunner
    from scripts.prepare_video import prepare
    from src.io_mot import load_seq
    src = json.loads((ROOT / "demo" / "showcase" / "SOURCES.json").read_text(encoding="utf-8"))
    rec = next(r for r in src if r["file"] == a.clip.name)
    seq = load_seq(ROOT / "data" / "mot", rec["seq"])
    f0 = rec["frames"][0]
    out = Path(tempfile.mkdtemp())
    prepare(a.clip, a.clip.stem, out_dir=out)
    gl, label = make_gaplink()
    r = LiveRunner(out / f"{a.clip.stem}.mp4", "relink", gl, label)
    cap = cv2.VideoCapture(str(out / f"{a.clip.stem}.mp4"))
    def frames_gen():
        while True:
            ok, im = cap.read()
            yield im if ok else None
    frames = frames_gen()
    res = run_audit(r, r.n, lambda f: gt_frame(seq, f0 + f - 1), frames)
    res = {"clip": a.clip.name, "source": rec, "detections": "fresh", **res}
    od = ROOT / "runs" / "check_showcase"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{a.clip.stem}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{a.clip.name}: ID KEPT same vehicle {res['links_same']} (>=25f: {res['links_strong_same']}), "
          f"WRONG {res['links_wrong']}, PHANTOM shown {res['links_phantom_flashed']}, unverified shown "
          f"{res['links_unverified_flashed']} | ID changes {len(res['id_switches'])} (visible {res['id_switches_visible']}) "
          f"| VISIBLE ERRORS {res['visible_errors']}")
    for x in res["links"]:
        print(f"   kept   f{x['frame']:4d} #{x['tid']:<3d} gap {x['gap']:3d}  {x['verdict']}{'' if x['flashed'] else ' (small)'}")
    for x in res["id_switches"]:
        print(f"   IDCHG  f{x['frame']:4d} gt {x['gt']} #{x['old']} -> #{x['new']} after {x['gap']} f"
              f"{'' if x['visible'] else ' (small/edge)'}")


if __name__ == "__main__":
    main()
