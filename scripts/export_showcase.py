"""Write raw showcase clips (no overlay, no cached detections) from UA-DETRAC test frames for the GapLink live demo.

    python -m scripts.export_showcase --seq MVI_40905 --f0 51 --f1 350 --name 01_tow_truck

writes demo/showcase/<name>.mp4 (H.264, 25 fps, original 960x540 frames, CRF 17) and appends its source and SHA-256 to
demo/showcase/SOURCES.json. These files are what is opened and uploaded in front of the audience: the console
detects and tracks them from scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "demo" / "showcase"


def export(seq_name: str, f0: int, f1: int, name: str, note: str = "") -> dict:
    import imageio.v2 as imageio
    from src.io_mot import load_seq
    seq = load_seq(ROOT / "data" / "mot", seq_name)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{name}.mp4"
    tmp = OUT / f"{name}_tmp.mp4"
    w = imageio.get_writer(str(tmp), fps=25, codec="libx264", quality=None, macro_block_size=None, pixelformat="yuv420p",
                           ffmpeg_params=["-crf", "17", "-preset", "slow"])
    n = 0
    for f in range(f0, f1 + 1):
        im = None
        for nm in (f"img{f:05d}.jpg", f"image{f:06d}.jpg"):
            if (seq.img_dir / nm).exists():
                im = cv2.imread(str(seq.img_dir / nm))
                break
        if im is None:
            raise SystemExit(f"missing frame {f} in {seq.img_dir}")
        w.append_data(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        n += 1
    w.close()
    tmp.replace(dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    rec = {"file": dst.name, "sha256": sha, "dataset": "UA-DETRAC", "split": "test", "seq": seq_name, "frames": [f0, f1],
           "n_frames": n, "fps": 25, "note": note or "raw frames, no overlay; selected for the GapLink-only demo "
                                                  "(strict audit, scripts/demo_audit.py), not a reported result"}
    sp = OUT / "SOURCES.json"
    recs = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else []
    recs = [r for r in recs if r["file"] != dst.name] + [rec]
    sp.write_text(json.dumps(recs, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--f0", type=int, required=True)
    ap.add_argument("--f1", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    r = export(a.seq, a.f0, a.f1, a.name, a.note)
    print(f"wrote demo/showcase/{r['file']}  {r['n_frames']} frames  sha256 {r['sha256'][:16]}...")


if __name__ == "__main__":
    main()
