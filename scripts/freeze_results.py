"""Freeze GapLink-Lite v1.1 results: hash every artifact the reported numbers depend on.

Usage (Windows):
    D:\\gaplink\\.venv\\Scripts\\python.exe D:\\gaplink\\scripts\\freeze_results.py           # verify
    D:\\gaplink\\.venv\\Scripts\\python.exe D:\\gaplink\\scripts\\freeze_results.py --write   # (re)create manifest

The manifest stores paths relative to root_dir (configs/paths.yaml) so it verifies on any machine.
--write refuses to overwrite an existing manifest unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
MANIFEST_REL = "release/FREEZE_MANIFEST.json"
RESULTS_COPY_DIR = "release/results_v1.1"

# Artifacts the frozen numbers depend on (relative to root).
ARTIFACTS = [
    # model + operating point
    "release/final.pt",
    "release/final.sha256",
    "release/calib.json",
    "release/tracker.yaml",
    "release/main.yaml",
    "release/yolov8n.pt",
    # data definition
    "data/splits.csv",
    "data/cache/val_events.pkl",
    "data/cache/test_events.pkl",
    # selection on val (must precede test)
    "runs/d4_val/val_summary.json",
    "runs/baselines_val.json",
    # test outputs (single locked run)
    "runs/test_final/test_results.json",
    "runs/test_final/live_security_results.json",
    "runs/test_final/evidence_chain_results.json",
    "runs/test_final/audit_diagnostics.json",
    # protocol
    "docs/PREREG.md",  # DEVIATIONS.md is a living log, intentionally not hashed
]
# Small result files copied under release/ so they are versioned in git (runs/ is gitignored).
COPY_TO_RELEASE = [a for a in ARTIFACTS if a.startswith("runs/") and a.endswith(".json")]


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def resolve_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root)
    cfg = ROOT_DEFAULT / "configs" / "paths.yaml"
    if cfg.exists():
        r = Path(yaml.safe_load(cfg.read_text(encoding="utf-8"))["root_dir"])
        if r.exists():
            return r
    return ROOT_DEFAULT


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_manifest(root: Path) -> dict:
    files = {}
    for rel in ARTIFACTS:
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"missing artifact: {rel}")
        files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    # internal consistency checks
    ckpt = files["release/final.pt"]["sha256"]
    declared = (root / "release/final.sha256").read_text().split()[0]
    tr = yaml.safe_load((root / "release/tracker.yaml").read_text())
    tres = json.loads((root / "runs/test_final/test_results.json").read_text())
    ev = (root / "data/cache/test_events.sha256").read_text().split()[0]
    checks = {
        "final.pt == final.sha256": ckpt == declared,
        "final.pt == tracker.yaml:final_sha256": ckpt == tr.get("final_sha256"),
        "final.pt == test_results:final_sha256": ckpt == tres.get("final_sha256"),
        "test_events.pkl == test_results:test_events_sha256": files["data/cache/test_events.pkl"]["sha256"] == tres.get("test_events_sha256") == ev,
        "tau == tracker.yaml:gaplink_tau": tres.get("tau") == tr.get("gaplink_tau"),
    }
    return {
        "freeze_name": "gaplink-lite-v1.1-results",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit_at_freeze": git_commit(root),
        "final_sha256": ckpt,
        "consistency_checks": checks,
        "files": files,
    }


def verify(root: Path) -> int:
    mpath = root / MANIFEST_REL
    if not mpath.exists():
        print(f"[FAIL] no manifest at {mpath}")
        return 2
    m = json.loads(mpath.read_text(encoding="utf-8"))
    bad = 0
    for rel, meta in m["files"].items():
        p = root / rel
        if not p.exists():
            print(f"[MISSING] {rel}"); bad += 1; continue
        got = sha256_file(p)
        if got != meta["sha256"]:
            print(f"[CHANGED] {rel}: {meta['sha256'][:12]} -> {got[:12]}"); bad += 1
    for rel in COPY_TO_RELEASE:
        c = root / RESULTS_COPY_DIR / Path(rel).name
        if not c.exists() or sha256_file(c) != m["files"][rel]["sha256"]:
            print(f"[COPY MISMATCH] {c}"); bad += 1
    for k, ok in m["consistency_checks"].items():
        if not ok:
            print(f"[CHECK FAILED AT FREEZE] {k}"); bad += 1
    print("[OK] frozen artifacts verified" if bad == 0 else f"[FAIL] {bad} problem(s)")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = resolve_root(a.root)
    if not a.write:
        return verify(root)
    mpath = root / MANIFEST_REL
    if mpath.exists() and not a.force:
        print(f"manifest exists ({mpath}); use --force only with a documented deviation")
        return 3
    m = build_manifest(root)
    out = root / RESULTS_COPY_DIR
    out.mkdir(parents=True, exist_ok=True)
    for rel in COPY_TO_RELEASE:
        dst = out / Path(rel).name
        if dst.exists() and sha256_file(dst) == m["files"][rel]["sha256"]:
            continue  # identical (and possibly read-only) copy already in place
        shutil.copy2(root / rel, dst)
    mpath.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m["consistency_checks"], indent=2))
    print(f"wrote {mpath}")
    return 0 if all(m["consistency_checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
