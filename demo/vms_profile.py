"""Single source of truth for the GapLink-VMS demo configuration.

Everything comes from frozen release files, never from constants in the demo code:
  - GapLink side: release/live_v1.3.yaml -> checkpoint + tau selected on validation (falls back to v1.1).
  - Baselines: thresholds selected on validation for the live setting (same file), C-4 lambdas from tracker.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

BASELINE_INFO = {
    "C-2": "C-2 Vận tốc không đổi",
    "C-3": "C-3 Kalman + Mahalanobis (kiểu SORT/DeepSORT, tự cài đặt)",
    "C-4": "C-4 Kalman + kích thước + màu",
}
# Demo default: industry-style Kalman tracker family (SORT/DeepSORT/ByteTrack all use a Kalman motion model).
# C-2 (strongest validation-selected live baseline, report Table live v1.3) stays one click away in the UI.
DEFAULT_BASELINE = "C-3"


def resolve_root() -> Path:
    cfg = ROOT / "configs" / "paths.yaml"
    if cfg.exists():
        r = ROOT / yaml.safe_load(cfg.read_text(encoding="utf-8"))["root_dir"]
        if r.exists():
            return r
    return ROOT


@dataclass
class LiveProfile:
    version: str
    ckpt: Path
    tau: float
    thetas: dict
    tracker: dict

    @property
    def gaplink_label(self) -> str:
        return f"GapLink-Lite {self.version} (τ={self.tau:g})"

    def baseline_label(self, name: str) -> str:
        return f"{BASELINE_INFO.get(name, name)} (θ={self.thetas[name]:g})"


def load_profile(root: Path | None = None, version: str = "v1.3") -> LiveProfile:
    root = root or resolve_root()
    rel = root / "release"
    tcfg = yaml.safe_load((rel / "tracker.yaml").read_text(encoding="utf-8"))
    y = rel / "live_v1.3.yaml"
    if version == "v1.3" and y.exists():
        op = yaml.safe_load(y.read_text(encoding="utf-8"))
        ck = op["v13_ckpt"].replace("\\", "/")
        ckpt = (root / ck) if op.get("gaplink_kind_live") == "GapLink-v13" else rel / "final.pt"
        return LiveProfile("v1.3", ckpt, float(op["gaplink_tau_live"]),
                           {"C-2": float(op["c2_theta_live"]), "C-3": float(op["c3_theta_live"]),
                            "C-4": float(op["c4_theta_live"])}, tcfg)
    return LiveProfile("v1.1", rel / "final.pt", float(tcfg["gaplink_tau"]),
                       {"C-2": float(tcfg["c2_theta"]), "C-3": float(tcfg["c3_theta"]), "C-4": float(tcfg["c4_theta"])}, tcfg)


def make_baseline(name: str, prof: LiveProfile):
    from src.baselines import ConstantVelocityBaseline, KalmanMahalanobisBaseline
    th = prof.thetas[name]
    if name == "C-2":
        return ConstantVelocityBaseline(theta=th)
    if name == "C-3":
        return KalmanMahalanobisBaseline(theta=th, name="C-3")
    if name == "C-4":
        return KalmanMahalanobisBaseline(theta=th, lambda_s=float(prof.tracker["c4_lambda_s"]),
                                         lambda_c=float(prof.tracker["c4_lambda_c"]), name="C-4")
    raise ValueError(name)


def make_gaplink(prof: LiveProfile, device: str):
    import torch
    from src.model.gaplink import GapLinkAssociator, GapLinkLite
    m = GapLinkLite(use_ctx=True, use_color=True, gap_aware=True).to(device)
    m.load_state_dict(torch.load(prof.ckpt, map_location=device, weights_only=False)["model"])
    m.eval()
    return GapLinkAssociator(m, device=device, tau=prof.tau, calib=None)
