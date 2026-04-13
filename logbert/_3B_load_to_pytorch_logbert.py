from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "traffic_labelled" / "2B_preprocessed_logbert"

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
CLS_TOKEN = "[CLS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN]


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
def load_partition(root: str, split: str, day: str, label_filter: Optional[int] = None) -> pd.DataFrame:
    filt = (ds.field("split") == split) & (ds.field("day") == day)
    if label_filter is not None:
        filt = filt & (ds.field("seq_y") == label_filter)

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
        "was_truncated",
        "n_flows",
        "tokens",
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
    benign_train_lens = meta_df.loc[(meta_df["split"] == "train") & (meta_df["seq_y"] == 0), "seq_len"].to_numpy(dtype=np.int64)
    if benign_train_lens.size == 0:
        raise ValueError("No benign training sequences found while inferring max_len.")

    inferred = int(np.percentile(benign_train_lens, percentile)) + 1  # reserve one position for [CLS]
    inferred = max(inferred, 8)
    if cap is not None:
        inferred = min(inferred, int(cap))
    return inferred


def _vocab_paths(root: Path) -> Tuple[Path, Path]:
    return root / "vocab_token_to_id.json", root / "vocab_metadata.json"


def build_vocab_from_train_benign(
    meta_df: pd.DataFrame,
    root: Optional[str] = None,
    min_freq: int = 2,
) -> Dict[str, int]:
    resolved_root = str(resolve_root(root))
    train_meta = meta_df[(meta_df["split"] == "train") & (meta_df["seq_y"] == 0)]
    counter = Counter()

    for day in sorted(train_meta["day"].unique().tolist()):
        df_day = load_partition(resolved_root, "train", day, label_filter=0)
        if df_day.empty:
            continue
        for tokens in df_day["tokens"].tolist():
            counter.update(tokens)

    token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    return token_to_id


def load_or_create_vocab(
    meta_df: pd.DataFrame,
    root: Optional[str] = None,
    min_freq: int = 2,
) -> Dict[str, int]:
    resolved_root = resolve_root(root)
    vocab_path, meta_path = _vocab_paths(resolved_root)
    expected_meta = {
        "min_freq": int(min_freq),
        "train_scope": "train_benign_only",
        "special_tokens": SPECIAL_TOKENS,
        "root": str(resolved_root),
    }

    if vocab_path.exists() and meta_path.exists():
        existing_meta = load_json(meta_path)
        if existing_meta == expected_meta:
            return load_json(vocab_path)

    token_to_id = build_vocab_from_train_benign(meta_df=meta_df, root=str(resolved_root), min_freq=min_freq)
    save_json(vocab_path, token_to_id)
    save_json(meta_path, expected_meta)
    return token_to_id


def encode_tokens(tokens, token_to_id: Dict[str, int], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_len < 2:
        raise ValueError("max_len must be at least 2 to reserve one position for [CLS].")

    cls_id = token_to_id[CLS_TOKEN]
    pad_id = token_to_id[PAD_TOKEN]
    unk_id = token_to_id[UNK_TOKEN]

    usable_tokens = list(tokens)[-(max_len - 1):]
    ids = [cls_id] + [token_to_id.get(token, unk_id) for token in usable_tokens]
    attn = [1] * len(ids)

    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids.extend([pad_id] * pad_n)
        attn.extend([0] * pad_n)

    return np.asarray(ids, dtype=np.int64), np.asarray(attn, dtype=np.int64)


class CICSequenceDataset(Dataset):
    def __init__(
        self,
        split: str,
        meta_df: pd.DataFrame,
        token_to_id: Dict[str, int],
        max_len: int,
        root: Optional[str] = None,
        label_filter: Optional[int] = None,
    ):
        self.root = str(resolve_root(root))
        self.split = split
        self.token_to_id = token_to_id
        self.max_len = int(max_len)
        self.label_filter = label_filter

        meta = meta_df[meta_df["split"] == split]
        if label_filter is not None:
            meta = meta[meta["seq_y"] == label_filter]
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        seq_id = row["seq_id"]
        day = row["day"]

        part = load_partition(self.root, self.split, day, label_filter=self.label_filter)
        seq_row = part.loc[seq_id]
        input_ids_np, attn_np = encode_tokens(seq_row["tokens"], self.token_to_id, self.max_len)

        return {
            "input_ids": torch.tensor(input_ids_np, dtype=torch.long),
            "attention_mask": torch.tensor(attn_np, dtype=torch.long),
            "seq_y": torch.tensor(int(seq_row["seq_y"]), dtype=torch.long),
            "seq_id": seq_row["seq_id"],
            "day": seq_row["day"],
            "entity": seq_row["entity"],
            "split_group_id": seq_row["split_group_id"],
        }


def dataset_metadata(
    root: Optional[str] = None,
    max_len: Optional[int] = None,
    min_freq: int = 2,
) -> Dict[str, object]:
    resolved_root = str(resolve_root(root))
    meta_df = load_meta(resolved_root)
    split_summary = load_split_summary(resolved_root)

    train_meta = meta_df[meta_df["split"] == "train"]
    val_meta = meta_df[meta_df["split"] == "val"]
    test_meta = meta_df[meta_df["split"] == "test"]

    return {
        "root": resolved_root,
        "sequence_build_config": load_sequence_build_config(resolved_root),
        "split_summary": split_summary.to_dict(orient="records") if not split_summary.empty else [],
        "n_train_total": int(len(train_meta)),
        "n_train_benign": int((train_meta["seq_y"] == 0).sum()),
        "n_train_attack": int((train_meta["seq_y"] == 1).sum()),
        "n_val_total": int(len(val_meta)),
        "n_val_benign": int((val_meta["seq_y"] == 0).sum()),
        "n_val_attack": int((val_meta["seq_y"] == 1).sum()),
        "n_test_total": int(len(test_meta)),
        "n_test_benign": int((test_meta["seq_y"] == 0).sum()),
        "n_test_attack": int((test_meta["seq_y"] == 1).sum()),
        "max_len": int(max_len) if max_len is not None else None,
        "vocab_min_freq": int(min_freq),
        "vocab_train_scope": "train_benign_only",
    }


def build_loaders(
    batch_train: int = 64,
    batch_eval: int = 128,
    seed: Optional[int] = None,
    root: Optional[str] = None,
    max_len: Optional[int] = None,
    max_len_percentile: float = 95.0,
    max_len_cap: Optional[int] = 512,
    min_freq: int = 2,
):
    resolved_root = str(resolve_root(root))
    meta_df = load_meta(resolved_root)
    if max_len is None:
        max_len = infer_default_max_len(resolved_root, percentile=max_len_percentile, cap=max_len_cap)

    token_to_id = load_or_create_vocab(meta_df=meta_df, root=resolved_root, min_freq=min_freq)

    train_ds = CICSequenceDataset("train", meta_df, token_to_id, max_len=max_len, root=resolved_root, label_filter=0)
    val_ds = CICSequenceDataset("val", meta_df, token_to_id, max_len=max_len, root=resolved_root, label_filter=None)
    test_ds = CICSequenceDataset("test", meta_df, token_to_id, max_len=max_len, root=resolved_root, label_filter=None)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_eval, shuffle=False, num_workers=0)

    data_meta = dataset_metadata(root=resolved_root, max_len=max_len, min_freq=min_freq)
    data_meta["vocab_size"] = int(len(token_to_id))
    data_meta["token_to_id"] = token_to_id

    return train_loader, val_loader, test_loader, data_meta


def main() -> None:
    train_loader, val_loader, test_loader, data_meta = build_loaders()
    token_to_id = data_meta["token_to_id"]

    print("max_len:", data_meta["max_len"])
    print("vocab_size:", data_meta["vocab_size"])
    print("train benign samples:", data_meta["n_train_benign"])
    print("val total samples:", data_meta["n_val_total"])
    print("test total samples:", data_meta["n_test_total"])

    batch = next(iter(train_loader))
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("seq_y:", batch["seq_y"].shape)

    unk_id = token_to_id[UNK_TOKEN]
    pad_id = token_to_id[PAD_TOKEN]
    cls_id = token_to_id[CLS_TOKEN]
    unk_pct = (batch["input_ids"] == unk_id).float().mean().item() * 100.0
    pad_pct = (batch["input_ids"] == pad_id).float().mean().item() * 100.0
    cls_pct = (batch["input_ids"] == cls_id).float().mean().item() * 100.0

    print("UNK %:", round(unk_pct, 4))
    print("PAD %:", round(pad_pct, 4))
    print("CLS %:", round(cls_pct, 4))


if __name__ == "__main__":
    main()
