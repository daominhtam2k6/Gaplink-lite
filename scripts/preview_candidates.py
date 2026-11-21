"""Contact sheet of the top mined demo candidates (for choosing clips with the right scene context).

    python -m scripts.preview_candidates --top 6
writes runs/demo_mining/preview_<scenario>.jpg : for each candidate, frames at clip start / key moment / end,
with the security zones and the key-moment location drawn.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def frame(seq, f):
    p = seq.img_dir / f"img{f:05d}.jpg"
    if not p.exists():
        p = seq.img_dir / f"image{f:06d}.jpg"
    im = cv2.imread(str(p))
    return im if im is not None else np.zeros((seq.H, seq.W, 3), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="runs/demo_mining/candidates_C-3.json")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    import scripts.eval_live_security as els
    from src.io_mot import load_seq
    cand = json.loads((ROOT / a.candidates).read_text(encoding="utf-8"))
    col = {"Z_A": (0, 165, 255), "Z_C": (220, 195, 40), "Z_B": (55, 55, 240), "Z_DWELL": (215, 85, 195)}
    for sc in ("route", "dwell", "lock"):
        rows = []
        for r, c in enumerate(cand[sc][:a.top]):
            seq = load_seq(ROOT / "data" / "mot", c["seq"])
            zones = els.build_adaptive_security_zones(seq)[0]
            k = c["key_frames"][0] if c.get("key_frames") else (c["f0"] + c["f1"]) // 2
            tiles = []
            for f in (c["f0"], k, c["f1"]):
                im = frame(seq, f)
                ov = im.copy()
                for z in zones:
                    if sc != "dwell" and z.zone_id == "Z_DWELL":
                        continue
                    if sc == "dwell" and z.zone_id != "Z_DWELL":
                        continue
                    cv2.polylines(ov, [z.to_pixel_array(seq.W, seq.H).astype(np.int32)], True, col[z.zone_id], 3)
                im = cv2.addWeighted(ov, 0.8, im, 0.2, 0)
                if sc == "lock":
                    tr = seq.tracks.get(int(c["gt_tid"]))
                    if tr is not None and f in list(tr["frames"]):
                        b = tr["boxes"][list(tr["frames"]).index(f)]
                        cv2.rectangle(im, (int(b[0] - b[2] / 2), int(b[1] - b[3] / 2)), (int(b[0] + b[2] / 2), int(b[1] + b[3] / 2)), (80, 255, 80), 3)
                else:
                    for x in c["alerts_baseline"] + c["alerts_gaplink"]:
                        if x.get("box") and abs(x["frame"] - f) < 3:
                            b = x["box"]
                            cv2.rectangle(im, (int(b[0] - b[2] / 2), int(b[1] - b[3] / 2)), (int(b[0] + b[2] / 2), int(b[1] + b[3] / 2)),
                                          (0, 0, 255) if x["fp"] else (80, 255, 80), 3)
                cv2.putText(im, f"f{f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                tiles.append(cv2.resize(im, (480, 270)))
            row = np.concatenate(tiles, axis=1)
            info = f"#{r} {c['seq']} [{c['f0']}-{c['f1']}] score {c['score']} GL {c.get('gaplink')} BL {c.get('baseline')}"
            if sc == "lock":
                info += f" gap {c['gap']} disp {c['displacement_px']}"
            bar = np.full((34, row.shape[1], 3), 30, np.uint8)
            cv2.putText(bar, info[:150], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            rows.append(np.concatenate([bar, row], axis=0))
        if rows:
            out = ROOT / "runs" / "demo_mining" / f"preview_{sc}.jpg"
            cv2.imwrite(str(out), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])
            print("wrote", out)


if __name__ == "__main__":
    main()
