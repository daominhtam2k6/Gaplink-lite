from pathlib import Path

import pytest

from demo.vms_profile import BASELINE_INFO, DEFAULT_BASELINE, load_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "release" / "live_v1.3.yaml").exists(), reason="no live v1.3 release")
def test_profile_reads_frozen_v13_operating_points():
    import yaml
    op = yaml.safe_load((ROOT / "release" / "live_v1.3.yaml").read_text(encoding="utf-8"))
    p = load_profile(ROOT)
    assert p.version == "v1.3"
    assert p.tau == pytest.approx(op["gaplink_tau_live"])
    assert p.thetas["C-2"] == pytest.approx(op["c2_theta_live"])
    assert p.ckpt.name == "live_v1.3.pt" and p.ckpt.exists()
    assert DEFAULT_BASELINE in BASELINE_INFO and "θ=" in p.baseline_label(DEFAULT_BASELINE)


@pytest.mark.skipif(not (ROOT / "release" / "tracker.yaml").exists(), reason="no release")
def test_profile_v11_fallback():
    p = load_profile(ROOT, version="v1.1")
    assert p.version == "v1.1" and p.ckpt.name == "final.pt"
