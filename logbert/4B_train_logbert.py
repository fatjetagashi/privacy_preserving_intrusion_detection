import os
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass
class Config:
    root: str = os.path.join("data", "traffic_labelled", "2B_preprocessed_logbert")
    max_len: int = 256
    min_freq: int = 2

    batch_train: int = 64
    batch_eval: int = 128

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    ff_mult: int = 4

    mask_prob: float = 0.15

    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 10
    grad_clip: float = 1.0

    score_passes: int = 5
    threshold_percentile: float = 99.0

    seed: int = 42


CFG = Config()

SEQS_DIR = os.path.join(CFG.root, "sequences")
META_DIR = os.path.join(CFG.root, "sequences_meta")
VOCAB_PATH = os.path.join(CFG.root, "vocab_token_to_id.json")
CKPT_DIR = os.path.join(CFG.root, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)

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


@lru_cache(maxsize=4)
def load_partition(split: str, day: str, benign_only: bool) -> pd.DataFrame:
    filt = (ds.field("split") == split) & (ds.field("day") == day)
    if benign_only:
        filt = filt & (ds.field("seq_y") == 0)

    cols = ["seq_id", "day", "entity", "split", "seq_y", "seq_len", "tokens"]
    pdf = seqs_ds.to_table(filter=filt, columns=cols).to_pandas()
    if len(pdf) == 0:
        return pdf
    return pdf.set_index("seq_id", drop=False)


def build_vocab_from_train_benign(meta_df: pd.DataFrame, min_freq: int) -> Dict[str, int]:
    train_benign = meta_df[(meta_df["split"] == "train") & (meta_df["seq_y"] == 0)]
    days = sorted(train_benign["day"].unique().tolist())

    counter = Counter()
    for day in days:
        df_day = load_partition("train", day, benign_only=True)
        for toks in df_day["tokens"].tolist():
            counter.update(toks)

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1, MASK_TOKEN: 2}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)

    return token_to_id


def load_or_create_vocab(meta_df: pd.DataFrame, min_freq: int) -> Dict[str, int]:
    if os.path.exists(VOCAB_PATH):
        token_to_id = load_json(VOCAB_PATH)
        token_to_id = {k: int(v) for k, v in token_to_id.items()}
        return token_to_id

    token_to_id = build_vocab_from_train_benign(meta_df, min_freq=min_freq)
    save_json(VOCAB_PATH, token_to_id)
    return token_to_id


class CICSequenceDataset(Dataset):
    def __init__(self, split: str, meta_df: pd.DataFrame, token_to_id: Dict[str, int], benign_only: bool):
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

        input_ids_np, attn_np = encode_tokens(s["tokens"], self.token_to_id, CFG.max_len)

        return {
            "input_ids": torch.tensor(input_ids_np, dtype=torch.long),
            "attention_mask": torch.tensor(attn_np, dtype=torch.long),
            "seq_y": torch.tensor(int(s["seq_y"]), dtype=torch.long),
            "seq_id": s["seq_id"],
            "day": s["day"],
            "entity": s["entity"],
        }


def mlm_mask_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
    mask_id: int,
    vocab_size: int,
    mask_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    eligible = (attention_mask == 1) & (input_ids != pad_id)
    probs = torch.rand_like(input_ids.float())
    to_mask = eligible & (probs < mask_prob)

    labels[to_mask] = input_ids[to_mask]

    probs2 = torch.rand_like(input_ids.float())
    mask_80 = to_mask & (probs2 < 0.8)
    rand_10 = to_mask & (probs2 >= 0.8) & (probs2 < 0.9)
    keep_10 = to_mask & (probs2 >= 0.9)

    masked[mask_80] = mask_id

    if rand_10.any():
        rand_ids = torch.randint(low=3, high=vocab_size, size=input_ids.shape, device=input_ids.device)
        masked[rand_10] = rand_ids[rand_10]

    _ = keep_10
    return masked, labels


class LogBERTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        ff_mult: int,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        src_key_padding_mask = attention_mask == 0
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(h)
        return logits


def train_one_epoch(model, loader, optimizer, loss_fn, pad_id, mask_id, vocab_size) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        masked_ids, labels = mlm_mask_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_id=pad_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
            mask_prob=CFG.mask_prob,
        )

        logits = model(masked_ids, attention_mask)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if CFG.grad_clip is not None and CFG.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_mlm_loss(model, loader, loss_fn, pad_id, mask_id, vocab_size) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        masked_ids, labels = mlm_mask_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_id=pad_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
            mask_prob=CFG.mask_prob,
        )

        logits = model(masked_ids, attention_mask)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def score_sequences(model, loader, pad_id, mask_id, vocab_size, K: int) -> pd.DataFrame:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")

    rows = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        seq_loss_sum = torch.zeros(input_ids.shape[0], device=device)

        for _ in range(K):
            masked_ids, labels = mlm_mask_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_id=pad_id,
                mask_id=mask_id,
                vocab_size=vocab_size,
                mask_prob=CFG.mask_prob,
            )

            logits = model(masked_ids, attention_mask)
            per_token_loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1)).view(labels.shape)

            masked_positions = labels != IGNORE_INDEX
            denom = masked_positions.sum(dim=1).clamp(min=1)
            seq_loss = (per_token_loss * masked_positions).sum(dim=1) / denom

            seq_loss_sum += seq_loss

        seq_loss_avg = (seq_loss_sum / float(K)).detach().cpu().numpy().tolist()

        seq_y = batch["seq_y"].detach().cpu().numpy().tolist()
        seq_ids = batch["seq_id"]
        days = batch["day"]
        entities = batch["entity"]

        for i in range(len(seq_ids)):
            rows.append(
                {
                    "seq_id": seq_ids[i],
                    "day": days[i],
                    "entity": entities[i],
                    "seq_y": int(seq_y[i]),
                    "score": float(seq_loss_avg[i]),
                }
            )

    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, object]:
    y_pred = (y_score > thr).astype(int)

    roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": float(thr),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm,
    }


def main():
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())

    meta_df = pd.read_parquet(META_DIR, columns=["seq_id", "day", "split", "seq_y"]).drop_duplicates()
    token_to_id = load_or_create_vocab(meta_df, min_freq=CFG.min_freq)

    vocab_size = len(token_to_id)
    pad_id = token_to_id[PAD_TOKEN]
    mask_id = token_to_id[MASK_TOKEN]

    train_ds = CICSequenceDataset("train", meta_df, token_to_id, benign_only=True)
    val_ds = CICSequenceDataset("val", meta_df, token_to_id, benign_only=True)
    test_ds = CICSequenceDataset("test", meta_df, token_to_id, benign_only=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_train, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_eval, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_eval, shuffle=False, num_workers=0)

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))
    print("test samples:", len(test_ds))
    meta_test = meta_df[meta_df["split"] == "test"]
    print("test attack %:", round((meta_test["seq_y"] == 1).mean() * 100, 2))
    print("vocab_size:", vocab_size)

    model = LogBERTModel(
        vocab_size=vocab_size,
        max_len=CFG.max_len,
        d_model=CFG.d_model,
        n_heads=CFG.n_heads,
        n_layers=CFG.n_layers,
        dropout=CFG.dropout,
        ff_mult=CFG.ff_mult,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_val = float("inf")
    best_path = os.path.join(CKPT_DIR, "logbert_best.pt")

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, pad_id, mask_id, vocab_size)
        val_loss = eval_mlm_loss(model, val_loader, loss_fn, pad_id, mask_id, vocab_size)

        print(f"Epoch {epoch}/{CFG.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CFG.__dict__,
                    "vocab_size": vocab_size,
                },
                best_path,
            )
            print("Saved best model:", best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Scoring val with K={CFG.score_passes} masking passes ...")
    val_scores = score_sequences(model, val_loader, pad_id, mask_id, vocab_size, K=CFG.score_passes)

    print(f"Scoring test with K={CFG.score_passes} masking passes ...")
    test_scores = score_sequences(model, test_loader, pad_id, mask_id, vocab_size, K=CFG.score_passes)

    thr_main = np.percentile(val_scores["score"].values, CFG.threshold_percentile)
    print(f"Threshold (val {CFG.threshold_percentile}th pct): {thr_main:.6f}")

    test_scores["pred"] = (test_scores["score"] > thr_main).astype(int)

    y_true = test_scores["seq_y"].values.astype(int)
    y_score = test_scores["score"].values.astype(float)

    metrics_main = compute_metrics(y_true, y_score, thr_main)

    thr_candidates = [95.0, 97.5, 99.0]
    threshold_sweep = []
    for p in thr_candidates:
        thr = float(np.percentile(val_scores["score"].values, p))
        m = compute_metrics(y_true, y_score, thr)
        m["percentile"] = p
        threshold_sweep.append(m)

    best_by_f1 = max(threshold_sweep, key=lambda x: x["f1"])

    metrics = {
        "best_val_loss": float(best_val),
        "vocab_size": int(vocab_size),
        "mask_prob": float(CFG.mask_prob),
        "score_passes": int(CFG.score_passes),
        "main_threshold_percentile": float(CFG.threshold_percentile),
        "main_metrics": metrics_main,
        "threshold_sweep": threshold_sweep,
        "best_threshold_by_f1": best_by_f1,
    }

    print("Metrics:", json.dumps(metrics, indent=2))

    val_out = os.path.join(CFG.root, "val_scores.csv")
    test_out = os.path.join(CFG.root, "test_scores.csv")
    metrics_out = os.path.join(CFG.root, "metrics_logbert.json")

    val_scores.to_csv(val_out, index=False)
    test_scores.to_csv(test_out, index=False)
    save_json(metrics_out, metrics)

    print("Saved:", val_out)
    print("Saved:", test_out)
    print("Saved:", metrics_out)
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
