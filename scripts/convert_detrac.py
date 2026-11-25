"""Convert extracted UA-DETRAC sequences (CSV + optional XML) into internal MOTChallenge format.
Properly separates `class == 'ignored'` (target_id == -1) rows into `{seq}_ignore.txt` and keeps only real vehicle tracks (`target_id > 0`).
"""
import csv
import json
from pathlib import Path
import numpy as np
from lxml import etree
from src.io_mot import center_in_ignored

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the repo is unpacked

CLASS_MAP = {
    "car": 1,
    "van": 2,
    "bus": 3,
    "others": 4,
    "truck": 4,
    "vehicle": 1,
}
FORBIDDEN_HUMAN = {"person", "pedestrian", "human", "people", "man", "woman"}


def load_optional_sequence_xml(xml_dir: Path, seq_name: str):
    """Return (ignore_boxes (K,4), class_by_frame_id dict, vis_by_frame_id dict) if official XML exists."""
    xml_path = xml_dir / f"{seq_name}.xml"
    if not xml_path.exists():
        matches = list(xml_dir.rglob(f"*{seq_name}.xml")) if xml_dir.exists() else []
        if not matches:
            return np.zeros((0, 4), dtype=np.float32), {}, {}
        xml_path = matches[0]

    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:
        return np.zeros((0, 4), dtype=np.float32), {}, {}

    ig_boxes = []
    ig_el = root.find("ignored_region")
    if ig_el is not None:
        for b in ig_el.findall("box"):
            ig_boxes.append([
                float(b.attrib["left"]),
                float(b.attrib["top"]),
                float(b.attrib["width"]),
                float(b.attrib["height"]),
            ])
    ignore = np.array(ig_boxes, dtype=np.float32) if ig_boxes else np.zeros((0, 4), dtype=np.float32)

    cls_map = {}
    vis_map = {}
    for fr in root.findall("frame"):
        fnum = int(fr.attrib["num"])
        tlist = fr.find("target_list")
        if tlist is None:
            continue
        for tgt in tlist.findall("target"):
            tid = int(tgt.attrib["id"])
            attr_el = tgt.find("attribute")
            if attr_el is not None:
                vtype = attr_el.attrib.get("vehicle_type", "car").lower().strip()
                cls_map[(fnum, tid)] = CLASS_MAP.get(vtype, 4)
                trunc = float(attr_el.attrib.get("truncation_ratio", 0.0))
                vis_map[(fnum, tid)] = max(0.0, min(1.0, 1.0 - trunc))
    return ignore, cls_map, vis_map


def convert_seq_folder(seq_dir: Path, source_split: str, xml_dir: Path, mot_dir: Path) -> dict | None:
    csv_path = seq_dir / "annotations.csv"
    if not csv_path.exists():
        return None

    folder_name = seq_dir.name
    seq_name = folder_name[folder_name.find("MVI_"):] if "MVI_" in folder_name else folder_name

    ignore_xml, cls_map, vis_map = load_optional_sequence_xml(xml_dir, seq_name)

    rows = []
    ignore_set = set()
    for b in ignore_xml:
        ignore_set.add((round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2]), 2), round(float(b[3]), 2)))

    max_frame = 0
    W, H = 960, 540

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cls_str = r.get("class", "vehicle").lower().strip()
            if cls_str in FORBIDDEN_HUMAN:
                continue
            fname = r["filename"]
            digits = "".join(ch for ch in fname if ch.isdigit())
            if not digits:
                continue
            fnum = int(digits)
            max_frame = max(max_frame, fnum)
            W = int(float(r.get("width", 960)))
            H = int(float(r.get("height", 540)))
            xmin = float(r["xmin"])
            ymin = float(r["ymin"])
            xmax = float(r["xmax"])
            ymax = float(r["ymax"])
            w = max(1.0, xmax - xmin)
            h = max(1.0, ymax - ymin)
            tid = int(float(r["target_id"]))
            conf = float(r.get("confidence", 1.0))

            # Separate static ignore regions (`class == 'ignored'` or `tid < 0`) from vehicle trajectories!
            if cls_str == "ignored" or tid < 0:
                ignore_set.add((round(xmin, 2), round(ymin, 2), round(w, 2), round(h, 2)))
                continue

            if (fnum, tid) in cls_map:
                cid = cls_map[(fnum, tid)]
            else:
                cid = CLASS_MAP.get(cls_str, 1)
            vis = vis_map.get((fnum, tid), 1.0)
            rows.append([fnum, tid, xmin, ymin, w, h, conf, cid, vis])

    if not rows:
        return None

    ignore = np.array(sorted(ignore_set), dtype=np.float32) if ignore_set else np.zeros((0, 4), dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    if len(ignore) > 0:
        inside = center_in_ignored(arr[:, 2:6], ignore)
        arr = arr[~inside]

    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]

    mot_file = mot_dir / f"{seq_name}.txt"
    ig_file = mot_dir / f"{seq_name}_ignore.txt"
    meta_file = mot_dir / f"{seq_name}_meta.json"

    with open(mot_file, "w", encoding="utf-8") as f:
        for r in arr:
            f.write(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.2f},{int(r[7])},{r[8]:.3f}\n")

    with open(ig_file, "w", encoding="utf-8") as f:
        for b in ignore:
            f.write(f"{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}\n")

    meta = {
        "name": seq_name,
        "folder_name": folder_name,
        "W": W,
        "H": H,
        "n_frames": max_frame,
        "img_dir": str(seq_dir),
        "source_split": source_split,
        "n_rows": int(len(arr)),
        "n_tracks": int(len(np.unique(arr[:, 1]))),
        "n_ignored_regions": int(len(ignore)),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    raw_dir = ROOT / "data" / "raw"
    img_root = raw_dir / "images"
    xml_dir = raw_dir / "xml"
    mot_dir = ROOT / "data" / "mot"
    mot_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for sp in ("train", "test"):
        sp_dir = img_root / sp
        if not sp_dir.exists():
            continue
        for seq_dir in sorted(sp_dir.iterdir()):
            if seq_dir.is_dir():
                meta = convert_seq_folder(seq_dir, sp, xml_dir, mot_dir)
                if meta is not None:
                    converted.append(meta)

    print(f"Converted {len(converted)} sequences into {mot_dir}")


if __name__ == "__main__":
    main()
