import os
import json
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset

ROOT = os.path.join("..", "data", "traffic_labelled", "2C_preprocessed_neuralode")
SEQS_DIR = os.path.join(ROOT, "sequences")
META_DIR = os.path.join(ROOT, "sequences_meta")
STATS_PATH = os.path.join(ROOT, "train_stats.npz")

PCTL = 90

meta_len = pd.read_parquet(META_DIR, columns=["seq_len", "split"]).drop_duplicates()
train_lens = meta_len[meta_len["split"] == "train"]["seq_len"].values
MAX_LEN = int(np.percentile(train_lens, PCTL)) if len(train_lens) else 128

print(f"[INFO] Dynamic MAX_LEN={MAX_LEN} (train P{PCTL})")

seqs_ds = ds.dataset(SEQS_DIR, format="parquet", partitioning="hive")

EVENT_IGNORE_INDEX = -1

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_one_sequence(split: str, day: str, seq_id: str, benign_only: bool, columns=None) -> pd.DataFrame:
    filt = (
        (ds.field("split") == split) &
        (ds.field("day") == day) &
        (ds.field("seq_id") == seq_id)
    )
    if benign_only:
        filt = filt & (ds.field("seq_y") == 0)

    tbl = seqs_ds.to_table(filter=filt, columns=list(columns) if columns is not None else None)
    return tbl.to_pandas()

def compute_train_stats(max_rows: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    cols = ["x", "split"]
    tbl = seqs_ds.to_table(
        filter=(ds.field("split") == "train"),
        columns=cols,
    )
    df = tbl.to_pandas()

    xs = []
    total = 0

    for arr in df["x"].values:
        if arr is None:
            continue

        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                x = arr
            elif arr.ndim == 1:
                x = np.stack(list(arr), axis=0)
            else:
                continue
        else:
            if len(arr) == 0:
                continue
            if isinstance(arr[0], np.ndarray):
                x = np.stack(arr, axis=0)
            else:
                x = np.asarray(arr, dtype=np.float32)
                if x.ndim == 1:
                    x = np.stack(list(arr), axis=0)

        x = x.astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        xs.append(x)
        total += x.shape[0]

        if total >= max_rows:
            break

    if not xs:
        raise ValueError("No train sequences found to compute stats")

    X = np.concatenate(xs, axis=0)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)

    np.savez(STATS_PATH, mean=mean, std=std)
    return mean, std

def load_or_create_train_stats() -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(STATS_PATH):
        d = np.load(STATS_PATH)
        return d["mean"].astype(np.float32), d["std"].astype(np.float32)

    return compute_train_stats()

TRAIN_MEAN, TRAIN_STD = load_or_create_train_stats()

def _tail_crop_list(x: List, max_len: int) -> List:
    if len(x) <= max_len:
        return x
    return x[-max_len:]

def encode_ode_sequence(
    t_list: List[int],
    x_list: List[List[float]],
    y_list: Optional[List[int]],
    max_len: int,
    dt_clip: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    t_list = _tail_crop_list(t_list, max_len)
    x_list = _tail_crop_list(x_list, max_len)
    if y_list is not None:
        y_list = _tail_crop_list(y_list, max_len)

    seq_len = len(t_list)
    if seq_len == 0:
        raise ValueError("Empty sequence after crop")

    t = np.asarray(t_list, dtype=np.float32)
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0

    if dt_clip is not None:
        dt = np.clip(dt, 0.0, float(dt_clip))
        dt = dt / float(dt_clip)

    def _to_2d_float32(x_list) -> np.ndarray:
        if isinstance(x_list, np.ndarray):
            if x_list.ndim == 2:
                x = x_list
            elif x_list.ndim == 1:
                x = np.stack(list(x_list), axis=0)
            else:
                raise ValueError(f"Unexpected x_list ndim={x_list.ndim}")
        else:
            if len(x_list) == 0:
                raise ValueError("Empty x_list")
            if isinstance(x_list[0], np.ndarray):
                x = np.stack(x_list, axis=0)
            else:
                x = np.asarray(x_list, dtype=np.float32)
                if x.ndim == 1:
                    x = np.stack(list(x_list), axis=0)

        x = x.astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    x = _to_2d_float32(x_list)
    feat_dim = int(x.shape[1])
    x = (x - TRAIN_MEAN) / TRAIN_STD
    x = np.clip(x, -5.0, 5.0)

    mask = np.ones((seq_len,), dtype=np.int64)

    if y_list is not None:
        y_seq = np.asarray(y_list, dtype=np.int64)
    else:
        y_seq = None

    if seq_len < max_len:
        pad_n = max_len - seq_len
        t_pad = np.zeros((pad_n,), dtype=np.float32)
        dt_pad = np.zeros((pad_n,), dtype=np.float32)
        x_pad = np.zeros((pad_n, feat_dim), dtype=np.float32)
        mask_pad = np.zeros((pad_n,), dtype=np.int64)

        t = np.concatenate([t, t_pad], axis=0)
        dt = np.concatenate([dt, dt_pad], axis=0)
        x = np.concatenate([x, x_pad], axis=0)
        mask = np.concatenate([mask, mask_pad], axis=0)

        if y_seq is not None:
            y_pad = np.full((pad_n,), EVENT_IGNORE_INDEX, dtype=np.int64)
            y_seq = np.concatenate([y_seq, y_pad], axis=0)

    out = {
        "t": t,
        "dt": dt,
        "x": x,
        "mask": mask,
    }
    if y_seq is not None:
        out["y_seq"] = y_seq
    return out


class CICNeuralODESequenceDataset(Dataset):
    def __init__(
        self,
        split: str,
        meta_df: pd.DataFrame,
        max_len: int,
        benign_only: bool,
        dt_clip: Optional[float] = None,
        return_event_labels: bool = True,
    ):
        self.split = split
        self.max_len = int(max_len)
        self.benign_only = bool(benign_only)
        self.dt_clip = dt_clip
        self.return_event_labels = bool(return_event_labels)

        df = meta_df[meta_df["split"] == split].copy()
        self.rows = df[["seq_id", "day", "entity", "seq_y"]].drop_duplicates().reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows.iloc[idx]
        seq_id = row["seq_id"]
        day = row["day"]
        entity = row["entity"]
        seq_y = int(row["seq_y"])

        cols = ("seq_id", "day", "entity", "seq_y", "t", "x", "y_seq")
        hit = load_one_sequence(self.split, day, seq_id, self.benign_only, columns=cols)
        if hit.empty:
            raise KeyError(f"seq_id not found: {seq_id}")

        r = hit.iloc[0]
        t_list = r["t"]
        x_list = r["x"]
        y_list = r["y_seq"] if self.return_event_labels else None

        enc = encode_ode_sequence(
            t_list=t_list,
            x_list=x_list,
            y_list=y_list,
            max_len=self.max_len,
            dt_clip=self.dt_clip,
        )

        out = {
            "x": torch.from_numpy(enc["x"]),
            "dt": torch.from_numpy(enc["dt"]),
            "mask": torch.from_numpy(enc["mask"]),
            "seq_y": torch.tensor(seq_y, dtype=torch.long),
            "seq_id": seq_id,
            "day": day,
            "entity": entity,
        }

        if self.return_event_labels:
            out["y_seq"] = torch.from_numpy(enc["y_seq"])

        return out