"""Select the 4 most illustrative test clips where GapLink-Lite outperforms C-3 and render side-by-side MP4 videos + stitched demo_full.mp4."""
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import torch
import yaml

from src.model.gaplink import GapLinkLite, GapLinkAssociator
from demo.render import render_side_by_side_clip

ROOT = Path(r"D:\gaplink")


def main():
    test_res = json.loads((ROOT / "runs" / "test_final" / "test_results.json").read_text(encoding="utf-8"))
    per_seq = test_res["per_seq_tracker"]["full"]

    # Rank test sequences by (IRR_GapLink - IRR_C3) * n_intervals
    scored = []
    for s_name, m_dict in per_seq.items():
        n_int = m_dict["GapLink-Lite"]["n_intervals"]
        if n_int < 5:
            continue
        diff = m_dict["GapLink-Lite"]["IRR"] - m_dict["C-3"]["IRR"]
        scored.append((diff * n_int, diff, n_int, s_name))
    scored.sort(reverse=True)
    chosen_seqs = [x[3] for x in scored[:4]]
    print("Top 4 test sequences selected for demo clips:", chosen_seqs)

    rel_dir = ROOT / "release"
    tracker_cfg = yaml.safe_load((rel_dir / "tracker.yaml").read_text(encoding="utf-8"))
    calib_params = json.loads((rel_dir / "calib.json").read_text(encoding="utf-8"))
    tau = float(tracker_cfg["gaplink_tau"])
    th_c3 = float(tracker_cfg["c3_theta"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(rel_dir / "final.pt", map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = GapLinkLite(
        use_ctx=bool(cfg.get("use_ctx", True)),
        use_color=bool(cfg.get("use_color", True)),
        gap_aware=bool(cfg.get("gap_aware", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gl_assoc = GapLinkAssociator(model, device=str(device), tau=tau, calib=calib_params)

    out_dir = ROOT / "demo" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    clip_infos = []

    for idx_c, s_name in enumerate(chosen_seqs, start=1):
        out_p = out_dir / f"clip_{idx_c:02d}_{s_name}.mp4"
        print(f"Rendering {out_p.name}...")
        info = render_side_by_side_clip(
            seq_name=s_name,
            out_mp4=out_p,
            start_frame=1,
            end_frame=240,
            associator_right=gl_assoc,
            theta_c3=th_c3,
            seed=7000 + idx_c,
            fps=20,
        )
        clip_paths.append(out_p)
        clip_infos.append(info)

    # Concatenate into demo_full.mp4 and extract representative frames for the report
    full_mp4 = out_dir / "demo_full.mp4"
    writer = imageio.get_writer(str(full_mp4), fps=20, codec="libx264", quality=7, macro_block_size=None)
    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for idx_c, cp in enumerate(clip_paths, start=1):
        cap = cv2.VideoCapture(str(cp))
        f_cnt = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            f_cnt += 1
            if f_cnt == 90:
                cv2.imwrite(str(fig_dir / f"fig_demo_clip_{idx_c:02d}.png"), frame_bgr)
            writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()
    writer.close()

    (out_dir / "demo_manifest.json").write_text(json.dumps(clip_infos, indent=2), encoding="utf-8")
    print("All 4 demo clips and demo_full.mp4 rendered successfully!")


if __name__ == "__main__":
    main()
