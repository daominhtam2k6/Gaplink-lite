import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the repo is unpacked

def test_splits_disjoint_if_exists():
    splits_path = ROOT / "data" / "splits.csv"
    if not splits_path.exists():
        return
    with open(splits_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    seqs_by_split = {"train": set(), "val": set(), "test": set()}
    locs_by_split = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        sp = r["split"]
        assert sp in seqs_by_split
        seqs_by_split[sp].add(r["seq"])
        locs_by_split[sp].add(r["location"])
    # No sequence in two splits
    assert len(seqs_by_split["train"] & seqs_by_split["val"]) == 0
    assert len(seqs_by_split["train"] & seqs_by_split["test"]) == 0
    assert len(seqs_by_split["val"] & seqs_by_split["test"]) == 0
    # No location in two splits
    assert len(locs_by_split["train"] & locs_by_split["val"]) == 0
    assert len(locs_by_split["train"] & locs_by_split["test"]) == 0
    assert len(locs_by_split["val"] & locs_by_split["test"]) == 0
