"""Padding/collation and pool dataset for GapLink-Lite."""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

def pad_event_to_fixed(ev: dict, n_max: int = 6, m_max: int = 10, k_len: int = 20, n_ctx: int = 8) -> dict:
    N = min(len(ev["t_lost"]), n_max)
    M = min(len(ev["cand"]), m_max)

    hist = np.zeros((n_max, k_len, 4), dtype=np.float32)
    hist_len = np.zeros((n_max,), dtype=np.int32)
    track_color = np.zeros((n_max, 128), dtype=np.float32)
    gap = np.ones((n_max,), dtype=np.int32)
    gap_bin = np.zeros((n_max,), dtype=np.int8)
    ctx_lost = np.zeros((n_max, n_ctx, 6), dtype=np.float32)
    ctx_conf = np.zeros((n_max, n_ctx, 6), dtype=np.float32)
    ctx_mask_lost = np.zeros((n_max, n_ctx), dtype=bool)
    ctx_mask_conf = np.zeros((n_max, n_ctx), dtype=bool)
    trk_mask = np.zeros((n_max,), dtype=bool)
    label = np.full((n_max,), -1, dtype=np.int64)
    true_box = np.zeros((n_max, 4), dtype=np.float32)
    has_true_target = np.zeros((n_max,), dtype=bool)
    side = np.zeros((n_max,), dtype=np.int8)

    cand = np.zeros((m_max, 4), dtype=np.float32)
    cand_hist = np.zeros((m_max, 128), dtype=np.float32)
    cand_mask = np.zeros((m_max,), dtype=bool)
    cand_label = np.full((m_max,), -1, dtype=np.int64)

    hist[:N] = ev["hist"][:N]
    hist_len[:N] = ev["hist_len"][:N]
    track_color[:N] = ev["track_color"][:N]
    gap[:N] = ev["gap"][:N]
    gap_bin[:N] = ev["gap_bin"][:N]
    ctx_lost[:N] = ev["ctx_lost"][:N]
    ctx_conf[:N] = ev["ctx_conf"][:N]
    ctx_mask_lost[:N] = ev["ctx_mask_lost"][:N]
    ctx_mask_conf[:N] = ev["ctx_mask_conf"][:N]
    trk_mask[:N] = True
    if "label" in ev:
        lbl = ev["label"][:N].copy()
        lbl[lbl >= M] = -1
        label[:N] = lbl
    if "true_box" in ev:
        true_box[:N] = ev["true_box"][:N]
    if "has_true_target" in ev:
        has_true_target[:N] = ev["has_true_target"][:N]
    if "side" in ev:
        side[:N] = ev["side"][:N]

    cand[:M] = ev["cand"][:M]
    cand_hist[:M] = ev["cand_hist"][:M]
    cand_mask[:M] = True
    if "cand_label" in ev:
        clbl = ev["cand_label"][:M].copy()
        clbl[clbl >= N] = -1
        cand_label[:M] = clbl

    return {
        "W": float(ev.get("W", 960)),
        "H": float(ev.get("H", 540)),
        "hist": hist,
        "hist_len": hist_len,
        "track_color": track_color,
        "gap": gap,
        "gap_bin": gap_bin,
        "ctx_lost": ctx_lost,
        "ctx_conf": ctx_conf,
        "ctx_mask_lost": ctx_mask_lost,
        "ctx_mask_conf": ctx_mask_conf,
        "trk_mask": trk_mask,
        "label": label,
        "true_box": true_box,
        "has_true_target": has_true_target,
        "side": side,
        "cand": cand,
        "cand_hist": cand_hist,
        "cand_mask": cand_mask,
        "cand_label": cand_label,
        "is_hard": bool(ev.get("is_hard", False)),
    }

class EventListDataset(Dataset):
    def __init__(self, events: list[dict]):
        self.items = [pad_event_to_fixed(e) for e in events]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class CurriculumPoolDataset(Dataset):
    """Pre-generated pool dataset with easy/hard split for curriculum sampling."""
    def __init__(self, easy_events: list[dict], hard_events: list[dict]):
        self.easy = [pad_event_to_fixed(e) for e in easy_events]
        self.hard = [pad_event_to_fixed(e) for e in hard_events]

    def sample_batch(self, batch_size: int, hard_prob: float, rng: np.random.Generator) -> dict:
        n_hard = int(round(batch_size * float(np.clip(hard_prob, 0.0, 1.0))))
        n_easy = batch_size - n_hard
        idx_h = rng.integers(0, len(self.hard), size=n_hard) if len(self.hard) > 0 else []
        idx_e = rng.integers(0, len(self.easy), size=n_easy) if len(self.easy) > 0 else []
        batch_list = [self.hard[i] for i in idx_h] + [self.easy[i] for i in idx_e]
        return collate_padded_events(batch_list)

def collate_padded_events(batch: list[dict]) -> dict:
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], np.ndarray):
            arr = np.stack(vals, axis=0)
            out[k] = torch.from_numpy(arr)
        elif isinstance(vals[0], bool):
            out[k] = torch.tensor(vals, dtype=torch.bool)
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals, dtype=torch.float32)
    return out
