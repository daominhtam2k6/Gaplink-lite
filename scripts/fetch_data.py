"""Resume UA-DETRAC download if incomplete, optionally fetch XMLs, and extract archives."""
import os
import json
import hashlib
import zipfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the repo is unpacked

RAW_DIR = ROOT / "data" / "raw"
IMG_ROOT = RAW_DIR / "images"

EXPECTED_SIZES = {
    "ua_detrac_training_set.zip": 5630247258,
    "ua_detrac_test_set.zip": 4252838102,
}

HF_URLS = {
    "ua_detrac_training_set.zip": "https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main/ua_detrac_training_set.zip",
    "ua_detrac_test_set.zip": "https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main/ua_detrac_test_set.zip",
}

ALBANY_XML_URLS = {
    "DETRAC-Train-Annotations-XML.zip": "http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML.zip",
    "DETRAC-Test-Annotations-XML.zip": "http://detrac-db.rit.albany.edu/Data/DETRAC-Test-Annotations-XML.zip",
}

def resume_curl(url: str, dest: Path, expected_size: int | None = None):
    for attempt in range(5):
        if dest.exists() and expected_size and dest.stat().st_size >= expected_size:
            print(f"[OK] {dest.name} complete ({dest.stat().st_size:,} bytes)")
            return True
        cur = dest.stat().st_size if dest.exists() else 0
        print(f"[DL] {dest.name} attempt {attempt+1}: current {cur:,} / {expected_size or '?'} bytes")
        cmd = [
            "curl.exe", "-L", "-C", "-", "--retry", "5", "--retry-delay", "2",
            "-o", str(dest), url
        ]
        subprocess.run(cmd, cwd=str(RAW_DIR))
    return dest.exists() and (expected_size is None or dest.stat().st_size >= expected_size)

def try_optional_xmls():
    xml_dir = RAW_DIR / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in ALBANY_XML_URLS.items():
        dest = RAW_DIR / fname
        if not dest.exists() or dest.stat().st_size < 100_000:
            cmd = ["curl.exe", "-L", "--connect-timeout", "10", "-m", "120", "-o", str(dest), url]
            subprocess.run(cmd, cwd=str(RAW_DIR))
        if dest.exists() and zipfile.is_zipfile(dest):
            print(f"[EXTRACT XML] {fname}")
            with zipfile.ZipFile(dest, "r") as z:
                z.extractall(xml_dir)

def extract_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        top_dirs = {m.split("/")[0] for m in members if "/" in m}
        already = all((out_dir / td / "annotations.csv").exists() for td in top_dirs if td.startswith("detrac_"))
        if already and len(top_dirs) >= 35:
            print(f"[SKIP EXTRACT] {zip_path.name} already extracted ({len(top_dirs)} sequences)")
            return len(top_dirs)
        print(f"[EXTRACT] {zip_path.name} -> {out_dir} ({len(members)} files)...")
        z.extractall(out_dir)
        return len(top_dirs)

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, exp_sz in EXPECTED_SIZES.items():
        resume_curl(HF_URLS[fname], RAW_DIR / fname, exp_sz)

    try_optional_xmls()

    n_train = extract_zip(RAW_DIR / "ua_detrac_training_set.zip", IMG_ROOT / "train")
    n_test = extract_zip(RAW_DIR / "ua_detrac_test_set.zip", IMG_ROOT / "test")

    manifest = {
        "train_zip_size": (RAW_DIR / "ua_detrac_training_set.zip").stat().st_size,
        "test_zip_size": (RAW_DIR / "ua_detrac_test_set.zip").stat().st_size,
        "n_train_sequences": n_train,
        "n_test_sequences": n_test,
    }
    (RAW_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("MANIFEST:", json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
