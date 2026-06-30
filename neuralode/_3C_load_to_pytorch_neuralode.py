from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "traffic_labelled" / "2C_preprocessed_neuralode"

EVENT_IGNORE_INDEX = -100
DEFAULT_STATS_VERSION = 3


def resolve_root(root: Optional[str] = None) -> Path:
    if root is None:
        return DEFAULT_ROOT
    return Path(root).resolve()


def save_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_sequence_build_config(root: Optional[str] = None) -> Dict[str, object]:
    resolved_root = resolve_root(root)
    config_path = resolved_root / "sequence_build_config.json"
    if not config_path.exists():
        return {}
    return load_json(config_path)


def load_split_summary(root: Optional[str] = None) -> pd.DataFrame:
    resolved_root = resolve_root(root)
    summary_path = resolved_root / "split_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


def _stats_paths(root: Path) -> Tuple[Path, Path]:
    return root / "train_stats_neuralode.npz", root / "train_stats_neuralode_meta.json"


@lru_cache(maxsize=8)
def load_meta(root: str) -> pd.DataFrame:
    meta_path = Path(root) / "sequences_meta"
    meta_df = pd.read_parquet(meta_path).drop_duplicates(subset=["seq_id"])
    sort_cols = [col_name for col_name in ["split", "day", "window_id", "seq_id"] if col_name in meta_df.columns]
    if sort_cols:
        meta_df = meta_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return meta_df


@lru_cache(maxsize=8)
def _sequences_dataset(root: str):
    return ds.dataset(Path(root) / "sequences", format="parquet", partitioning="hive")


@lru_cache(maxsize=32)
def load_partition(root: str, split: str, day: str) -> pd.DataFrame:
    filt = (ds.field("split") == split) & (ds.field("day") == day)
    cols = [
        "seq_id",
        "day",
        "window_id",
        "entity",
        "split_group_id",
        "split",
        "seq_y",
        "seq_len",
        "full_seq_len",
        "full_n_attack_events",
        "was_truncated",
        "n_events",
        "n_attack_events",
        "seq_start_ts",
        "seq_end_ts",
        "t",
        "x",
        "y_seq",
    ]
    pdf = _sequences_dataset(root).to_table(filter=filt, columns=cols).to_pandas()
    if pdf.empty:
        return pdf
    return pdf.set_index("seq_id", drop=False)


def infer_default_max_len(
    root: Optional[str] = None,
    percentile: float = 95.0,
    cap: Optional[int] = 512,
) -> int:
    resolved_root = str(resolve_root(root))
    meta_df = load_meta(resolved_root)
    train_lens = meta_df.loc[meta_df["split"] == "train", "seq_len"].to_numpy(dtype=np.int64)
    if train_lens.size == 0:
        raise ValueError("No training trajectories found while inferring max_len.")

    inferred = int(np.percentile(train_lens, percentile))
    inferred = max(inferred, 8)
    if cap is not None:
        inferred = min(inferred, int(cap))
    return inferred


def _to_2d_float32(x_values) -> np.ndarray:
    if isinstance(x_values, np.ndarray):
        if x_values.ndim == 2:
            x = x_values
        elif x_values.ndim == 1:
            x = np.stack(list(x_values), axis=0)
        else:
            raise ValueError(f"Unexpected feature array ndim={x_values.ndim}")
    else:
        if len(x_values) == 0:
            raise ValueError("Empty feature array.")
        if isinstance(x_values[0], np.ndarray):
            x = np.stack(x_values, axis=0)
        else:
            x = np.asarray(x_values, dtype=np.float32)
            if x.ndim == 1:
                x = np.stack(list(x_values), axis=0)

    x = x.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def compute_train_stats(
    meta_df: pd.DataFrame,
    root: Optional[str] = None,
    max_feature_rows: int = 200_000,
    max_dt_samples: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    resolved_root = str(resolve_root(root))
    train_meta = meta_df[meta_df["split"] == "train"]
    if train_meta.empty:
        raise ValueError("No training trajectories found to compute Neural ODE stats.")

    sampled_meta = train_meta.sample(frac=1.0, random_state=0).reset_index(drop=True)

    xs = []
    dt_samples = []
    total_feature_rows = 0
    total_dt = 0

    for _, meta_row in sampled_meta.iterrows():
        day = str(meta_row["day"])
        seq_id = str(meta_row["seq_id"])
        part = load_partition(resolved_root, "train", day)
        if part.empty:
            continue

        if seq_id not in part.index:
            continue
        row = part.loc[seq_id]
        x = _to_2d_float32(row["x"])
        xs.append(x)
        total_feature_rows += int(x.shape[0])

        t = np.asarray(row["t"], dtype=np.float32)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if t.size > 0:
            t_rel = t - float(t[0])
            t_rel = np.clip(t_rel, a_min=0.0, a_max=None)
            t_rel = np.maximum.accumulate(t_rel)
            dt = np.diff(t_rel, prepend=t_rel[0])
            dt[0] = 0.0
            dt = np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)
            dt = np.clip(dt, a_min=0.0, a_max=None)
            pos_dt = dt[dt > 0]
            if pos_dt.size > 0:
                dt_samples.append(pos_dt)
                total_dt += int(pos_dt.size)

        if total_feature_rows >= max_feature_rows and total_dt >= max_dt_samples:
            break

    if not xs:
        raise ValueError("No training trajectories were available to compute feature statistics.")

    X = np.concatenate(xs, axis=0)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)

    if dt_samples:
        dt_arr = np.concatenate(dt_samples, axis=0).astype(np.float32)
        dt_clip = float(np.percentile(dt_arr, 99.0))
    else:
        config = load_sequence_build_config(resolved_root)
        dt_clip = float(config.get("window_seconds", 60))
    dt_clip = max(dt_clip, 1.0)

    stats_path, meta_path = _stats_paths(resolve_root(root))
    config = load_sequence_build_config(resolved_root)
    np.savez(stats_path, mean=mean, std=std, dt_clip=np.asarray([dt_clip], dtype=np.float32))
    save_json(
        meta_path,
        {
            "version": DEFAULT_STATS_VERSION,
            "root": str(resolve_root(root)),
            "feature_columns": config.get("feature_columns", []),
            "sequence_build_version": config.get("sequence_build_version"),
            "event_counting_policy": config.get("event_counting_policy"),
            "variant_id": config.get("variant_id"),
            "max_feature_rows": int(max_feature_rows),
            "max_dt_samples": int(max_dt_samples),
        },
    )
    return mean, std, dt_clip


def load_or_create_train_stats(
    meta_df: pd.DataFrame,
    root: Optional[str] = None,
    max_feature_rows: int = 200_000,
    max_dt_samples: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    resolved_root = resolve_root(root)
    stats_path, meta_path = _stats_paths(resolved_root)
    config = load_sequence_build_config(resolved_root)
    expected_meta = {
        "version": DEFAULT_STATS_VERSION,
        "root": str(resolved_root),
        "feature_columns": config.get("feature_columns", []),
        "sequence_build_version": config.get("sequence_build_version"),
        "event_counting_policy": config.get("event_counting_policy"),
        "variant_id": config.get("variant_id"),
        "max_feature_rows": int(max_feature_rows),
        "max_dt_samples": int(max_dt_samples),
    }

    if stats_path.exists() and meta_path.exists():
        try:
            existing_meta = load_json(meta_path)
            if existing_meta == expected_meta:
                cached = np.load(stats_path)
                return (
                    cached["mean"].astype(np.float32),
                    cached["std"].astype(np.float32),
                    float(cached["dt_clip"][0]),
                )
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

    return compute_train_stats(
        meta_df=meta_df,
        root=str(resolved_root),
        max_feature_rows=max_feature_rows,
        max_dt_samples=max_dt_samples,
    )


def encode_ode_sequence(
    t_list,
    x_list,
    y_list,
    max_len: int,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    dt_clip: float,
) -> Dict[str, np.ndarray]:
    t_seq = list(t_list)[-max_len:]
    x_seq = list(x_list)[-max_len:]
    y_seq_in = list(y_list)[-max_len:] if y_list is not None else None

    seq_len = len(t_seq)
    if seq_len <= 0:
        raise ValueError("Empty trajectory after max_len cropping.")
    if len(x_seq) != seq_len:
        raise ValueError(f"Trajectory feature length mismatch: len(t)={seq_len}, len(x)={len(x_seq)}")
    if y_seq_in is not None and len(y_seq_in) != seq_len:
        raise ValueError(f"Trajectory label length mismatch: len(t)={seq_len}, len(y)={len(y_seq_in)}")

    x = _to_2d_float32(x_seq)
    x = (x - train_mean) / train_std
    x = np.clip(x, -5.0, 5.0)

    t_abs = np.asarray(t_seq, dtype=np.float32)
    t_abs = np.nan_to_num(t_abs, nan=0.0, posinf=0.0, neginf=0.0)
    t_rel = t_abs - float(t_abs[0])
    t_rel = np.clip(t_rel, a_min=0.0, a_max=None)
    t_rel = np.maximum.accumulate(t_rel)

    dt = np.diff(t_rel, prepend=t_rel[0])
    dt[0] = 0.0
    dt = np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)
    dt = np.clip(dt, a_min=0.0, a_max=None)

    dt_clip = max(float(dt_clip), 1.0)
    dt_norm = np.clip(dt, 0.0, dt_clip) / dt_clip
    t_norm = np.clip(t_rel, 0.0, dt_clip * max_len) / dt_clip

    mask = np.ones((seq_len,), dtype=np.int64)

    if y_seq_in is not None:
        y_seq = np.asarray(y_seq_in, dtype=np.int64)
    else:
        y_seq = None

    if seq_len < max_len:
        pad_n = max_len - seq_len
        x = np.concatenate([x, np.zeros((pad_n, x.shape[1]), dtype=np.float32)], axis=0)
        dt_norm = np.concatenate([dt_norm, np.zeros((pad_n,), dtype=np.float32)], axis=0)
        t_norm = np.concatenate([t_norm, np.zeros((pad_n,), dtype=np.float32)], axis=0)
        mask = np.concatenate([mask, np.zeros((pad_n,), dtype=np.int64)], axis=0)

        if y_seq is not None:
            y_seq = np.concatenate(
                [y_seq, np.full((pad_n,), EVENT_IGNORE_INDEX, dtype=np.int64)],
                axis=0,
            )

    payload = {
        "x": x.astype(np.float32),
        "dt": dt_norm.astype(np.float32),
        "t": t_norm.astype(np.float32),
        "mask": mask.astype(np.int64),
    }
    if y_seq is not None:
        payload["y_seq"] = y_seq.astype(np.int64)
    return payload


class CICNeuralODESequenceDataset(Dataset):
    def __init__(
        self,
        split: str,
        meta_df: pd.DataFrame,
        train_mean: np.ndarray,
        train_std: np.ndarray,
        dt_clip: float,
        max_len: int,
        root: Optional[str] = None,
    ):
        self.root = str(resolve_root(root))
        self.split = split
        self.train_mean = train_mean.astype(np.float32)
        self.train_std = train_std.astype(np.float32)
        self.dt_clip = float(dt_clip)
        self.max_len = int(max_len)

        meta = meta_df[meta_df["split"] == split].copy()
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        seq_id = row["seq_id"]
        day = row["day"]
        part = load_partition(self.root, self.split, day)
        seq_row = part.loc[seq_id]

        encoded = encode_ode_sequence(
            t_list=seq_row["t"],
            x_list=seq_row["x"],
            y_list=seq_row["y_seq"],
            max_len=self.max_len,
            train_mean=self.train_mean,
            train_std=self.train_std,
            dt_clip=self.dt_clip,
        )

        return {
            "x": torch.tensor(encoded["x"], dtype=torch.float32),
            "dt": torch.tensor(encoded["dt"], dtype=torch.float32),
            "t": torch.tensor(encoded["t"], dtype=torch.float32),
            "mask": torch.tensor(encoded["mask"], dtype=torch.long),
            "seq_y": torch.tensor(int(seq_row["seq_y"]), dtype=torch.long),
            "y_seq": torch.tensor(encoded["y_seq"], dtype=torch.long),
            "seq_id": seq_row["seq_id"],
            "day": seq_row["day"],
            "entity": seq_row["entity"],
            "split_group_id": seq_row["split_group_id"],
        }


def dataset_metadata(
    root: Optional[str] = None,
    max_len: Optional[int] = None,
    dt_clip: Optional[float] = None,
) -> Dict[str, object]:
    resolved_root = str(resolve_root(root))
    meta_df = load_meta(resolved_root)
    split_summary = load_split_summary(resolved_root)

    train_meta = meta_df[meta_df["split"] == "train"]
    val_meta = meta_df[meta_df["split"] == "val"]
    test_meta = meta_df[meta_df["split"] == "test"]

    feature_columns = load_sequence_build_config(resolved_root).get("feature_columns", [])

    def event_counts(part: pd.DataFrame) -> Tuple[int, int]:
        total_events = int(part["n_events"].sum()) if not part.empty else 0
        attack_events = int(part["n_attack_events"].sum()) if not part.empty else 0
        return total_events, attack_events

    train_events, train_attack_events = event_counts(train_meta)
    val_events, val_attack_events = event_counts(val_meta)
    test_events, test_attack_events = event_counts(test_meta)

    return {
        "root": resolved_root,
        "sequence_build_config": load_sequence_build_config(resolved_root),
        "split_summary": split_summary.to_dict(orient="records") if not split_summary.empty else [],
        "feature_columns": feature_columns,
        "feature_count": int(len(feature_columns)),
        "n_train_total": int(len(train_meta)),
        "n_train_benign": int((train_meta["seq_y"] == 0).sum()),
        "n_train_attack": int((train_meta["seq_y"] == 1).sum()),
        "n_val_total": int(len(val_meta)),
        "n_val_benign": int((val_meta["seq_y"] == 0).sum()),
        "n_val_attack": int((val_meta["seq_y"] == 1).sum()),
        "n_test_total": int(len(test_meta)),
        "n_test_benign": int((test_meta["seq_y"] == 0).sum()),
        "n_test_attack": int((test_meta["seq_y"] == 1).sum()),
        "n_train_events": int(train_events),
        "n_train_attack_events": int(train_attack_events),
        "n_val_events": int(val_events),
        "n_val_attack_events": int(val_attack_events),
        "n_test_events": int(test_events),
        "n_test_attack_events": int(test_attack_events),
        "max_len": int(max_len) if max_len is not None else None,
        "dt_clip": float(dt_clip) if dt_clip is not None else None,
    }


def build_loaders(
    batch_train: int = 64,
    batch_eval: int = 128,
    seed: Optional[int] = None,
    root: Optional[str] = None,
    max_len: Optional[int] = None,
    max_len_percentile: float = 95.0,
    max_len_cap: Optional[int] = 512,
    dt_clip: Optional[float] = None,
    num_workers: int = 0,
    max_feature_rows: int = 200_000,
    max_dt_samples: int = 500_000,
):
    resolved_root = str(resolve_root(root))
    meta_df = load_meta(resolved_root)
    if max_len is None:
        max_len = infer_default_max_len(resolved_root, percentile=max_len_percentile, cap=max_len_cap)

    train_mean, train_std, inferred_dt_clip = load_or_create_train_stats(
        meta_df=meta_df,
        root=resolved_root,
        max_feature_rows=max_feature_rows,
        max_dt_samples=max_dt_samples,
    )
    effective_dt_clip = float(dt_clip) if dt_clip is not None else float(inferred_dt_clip)

    train_ds = CICNeuralODESequenceDataset(
        split="train",
        meta_df=meta_df,
        train_mean=train_mean,
        train_std=train_std,
        dt_clip=effective_dt_clip,
        max_len=max_len,
        root=resolved_root,
    )
    val_ds = CICNeuralODESequenceDataset(
        split="val",
        meta_df=meta_df,
        train_mean=train_mean,
        train_std=train_std,
        dt_clip=effective_dt_clip,
        max_len=max_len,
        root=resolved_root,
    )
    test_ds = CICNeuralODESequenceDataset(
        split="test",
        meta_df=meta_df,
        train_mean=train_mean,
        train_std=train_std,
        dt_clip=effective_dt_clip,
        max_len=max_len,
        root=resolved_root,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader_kwargs = {"num_workers": int(num_workers)}
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["pin_memory"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_eval,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_eval,
        shuffle=False,
        **loader_kwargs,
    )

    data_meta = dataset_metadata(root=resolved_root, max_len=max_len, dt_clip=effective_dt_clip)
    data_meta["train_mean"] = train_mean
    data_meta["train_std"] = train_std
    data_meta["dt_clip"] = float(effective_dt_clip)

    return train_loader, val_loader, test_loader, data_meta


def main() -> None:
    train_loader, val_loader, test_loader, data_meta = build_loaders()

    print("max_len:", data_meta["max_len"])
    print("dt_clip:", data_meta["dt_clip"])
    print("feature_count:", data_meta["feature_count"])
    print("train trajectories:", data_meta["n_train_total"])
    print("val trajectories:", data_meta["n_val_total"])
    print("test trajectories:", data_meta["n_test_total"])

    batch = next(iter(train_loader))
    print("x:", batch["x"].shape)
    print("dt:", batch["dt"].shape)
    print("t:", batch["t"].shape)
    print("mask:", batch["mask"].shape)
    print("seq_y:", batch["seq_y"].shape)
    print("y_seq:", batch["y_seq"].shape)


if __name__ == "__main__":
    main()
