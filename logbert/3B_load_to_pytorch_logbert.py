import os
import json
from functools import lru_cache
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset, DataLoader


ROOT = os.path.join("data", "traffic_labelled", "2B_preprocessed_logbert")
SEQS_DIR = os.path.join(ROOT, "sequences")
META_DIR = os.path.join(ROOT, "sequences_meta")

VOCAB_PATH = os.path.join(ROOT, "vocab_token_to_id.json")

MAX_LEN = 256

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"


seqs_ds = ds.dataset(SEQS_DIR, format="parquet", partitioning="hive")


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_tokens(tokens: List[str], token_to_id: Dict[str, int], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = [token_to_id.get(t, token_to_id[UNK_TOKEN]) for t in tokens[:max_len]]
    attn = [1] * len(ids)

    if len(ids) < max_len:
        pad_id = token_to_id[PAD_TOKEN]
        pad_n = max_len - len(ids)
        ids.extend([pad_id] * pad_n)
        attn.extend([0] * pad_n)

    return np.asarray(ids, dtype=np.int64), np.asarray(attn, dtype=np.int64)


def build_vocab_from_train_benign(meta_df: pd.DataFrame, min_freq: int = 2) -> Dict[str, int]:
    train_benign_meta = meta_df[(meta_df["split"] == "train") & (meta_df["seq_y"] == 0)]
    days = sorted(train_benign_meta["day"].unique().tolist())

    counter = Counter()

    for day in days:
        df_day = load_partition("train", day, benign_only=True)
        for toks in df_day["tokens"].tolist():
            counter.update(toks)

    token_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        MASK_TOKEN: 2,
    }

    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)

    return token_to_id


def load_or_create_vocab(meta_df: pd.DataFrame, min_freq: int = 2) -> Dict[str, int]:
    if os.path.exists(VOCAB_PATH):
        return load_json(VOCAB_PATH)

    token_to_id = build_vocab_from_train_benign(meta_df, min_freq=min_freq)
    save_json(VOCAB_PATH, token_to_id)
    return token_to_id


@lru_cache(maxsize=16)
def load_partition(split: str, day: str, benign_only: bool) -> pd.DataFrame:
    filt = (ds.field("split") == split) & (ds.field("day") == day)
    if benign_only:
        filt = filt & (ds.field("seq_y") == 0)

    cols = ["seq_id", "day", "entity", "split", "seq_y", "seq_len", "tokens"]
    pdf = seqs_ds.to_table(filter=filt, columns=cols).to_pandas()

    if len(pdf) == 0:
        return pdf

    return pdf.set_index("seq_id", drop=False)


class CICSequenceDataset(Dataset):
    def __init__(self, split: str, meta_df: pd.DataFrame, token_to_id: Dict[str, int], benign_only: bool = False):
        self.split = split
        self.token_to_id = token_to_id
        self.benign_only = benign_only

        meta = meta_df[meta_df["split"] == split]
        if benign_only:
            meta = meta[meta["seq_y"] == 0]
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        seq_id = row["seq_id"]
        day = row["day"]

        part = load_partition(self.split, day, self.benign_only)
        s = part.loc[seq_id]

        tokens = s["tokens"]
        input_ids_np, attn_np = encode_tokens(tokens, self.token_to_id, MAX_LEN)

        return {
            "input_ids": torch.tensor(input_ids_np, dtype=torch.long),
            "attention_mask": torch.tensor(attn_np, dtype=torch.long),
            "seq_y": torch.tensor(int(s["seq_y"]), dtype=torch.long),
            "seq_id": s["seq_id"],
            "day": s["day"],
            "entity": s["entity"],
        }


meta_df = pd.read_parquet(META_DIR, columns=["seq_id", "day", "split", "seq_y"]).drop_duplicates()
token_to_id = load_or_create_vocab(meta_df, min_freq=2)

train_ds = CICSequenceDataset("train", meta_df, token_to_id, benign_only=True)
val_ds = CICSequenceDataset("val", meta_df, token_to_id, benign_only=True)
test_ds = CICSequenceDataset("test", meta_df, token_to_id, benign_only=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

batch = next(iter(train_loader))
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("seq_y:", batch["seq_y"].shape)
print("vocab_size:", len(token_to_id))

unk_id = token_to_id["[UNK]"]
unk_pct = (batch["input_ids"] == unk_id).float().mean().item() * 100
pad_id = token_to_id["[PAD]"]
pad_pct = (batch["input_ids"] == pad_id).float().mean().item() * 100
print("UNK %:", round(unk_pct, 4))
print("PAD %:", round(pad_pct, 4))

mask_mismatch = ((batch["attention_mask"] == 0) & (batch["input_ids"] != pad_id)).any().item()
print("mask mismatch:", mask_mismatch)

id_to_token = [""] * len(token_to_id)
for tok, idx in token_to_id.items():
    id_to_token[idx] = tok

sample_ids = batch["input_ids"][0].tolist()
decoded = [id_to_token[i] for i in sample_ids[:30]]
print("decoded first 30:", decoded)

print("train samples:", len(train_ds))
print("val samples:", len(val_ds))
print("test samples:", len(test_ds))

test_meta = pd.read_parquet(META_DIR, columns=["seq_len"]) if "seq_len" else None
meta_test = meta_df[meta_df["split"] == "test"]
print("test attack %:", round((meta_test["seq_y"] == 1).mean() * 100, 2))
