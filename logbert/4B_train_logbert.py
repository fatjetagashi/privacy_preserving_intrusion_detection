import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from _3B_load_to_pytorch_logbert import (
    PAD_TOKEN, UNK_TOKEN, MASK_TOKEN,
    save_json, load_json,
    encode_tokens, load_partition,
    build_vocab_from_train_all, load_or_create_vocab,
    CICSequenceDataset,
)

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
    root: str = os.path.join("..", "data", "traffic_labelled", "2B_preprocessed_logbert")
    max_len: int = None
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
    early_stop_patience: int = 4
    early_stop_min_delta: float = 1e-4

    grad_clip: float = 1.0

    score_passes: int = 15
    threshold_percentile: float = 99.0

    seed: int = 42


CFG = Config()

SEQS_DIR = os.path.join(CFG.root, "sequences")
META_DIR = os.path.join(CFG.root, "sequences_meta")
CKPT_DIR = os.path.join(CFG.root, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


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
    meta_len = pd.read_parquet(META_DIR, columns=["seq_len", "split"]).drop_duplicates()
    train_lens = meta_len[meta_len["split"] == "train"]["seq_len"].values

    PCTL = 90
    CFG.max_len = int(np.percentile(train_lens, PCTL))
    print(f"[INFO] Dynamic CFG.max_len set to {CFG.max_len} (train P{PCTL})")

    token_to_id = load_or_create_vocab(meta_df, min_freq=CFG.min_freq)

    vocab_size = len(token_to_id)
    pad_id = token_to_id[PAD_TOKEN]
    mask_id = token_to_id[MASK_TOKEN]

    train_ds = CICSequenceDataset("train", meta_df, token_to_id, max_len=CFG.max_len, benign_only=True)
    val_ds = CICSequenceDataset("val", meta_df, token_to_id, max_len=CFG.max_len, benign_only=True)
    test_ds = CICSequenceDataset("test", meta_df, token_to_id, max_len=CFG.max_len, benign_only=False)

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

    bad_epochs = 0
    patience = CFG.early_stop_patience
    min_delta = CFG.early_stop_min_delta

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, pad_id, mask_id, vocab_size)
        val_loss = eval_mlm_loss(model, val_loader, loss_fn, pad_id, mask_id, vocab_size)

        print(f"Epoch {epoch}/{CFG.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CFG.__dict__,
                    "vocab_size": vocab_size,
                },
                best_path,
            )
            print("Saved best model:", best_path)
        else:
            bad_epochs += 1
            print(f"[EARLYSTOP] no improvement: {bad_epochs}/{patience} (best_val={best_val:.4f})")

            if bad_epochs >= patience:
                print(f"[EARLYSTOP] stopping early at epoch {epoch}. Best val_loss={best_val:.4f}")
                break

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

    thr_candidates = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 97.5, 99.0, 99.5]

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
        "vocab_min_freq": int(CFG.min_freq),
        "vocab_train_scope": "train_all",
        "max_len": int(CFG.max_len),

        "mask_prob": float(CFG.mask_prob),
        "score_passes": int(CFG.score_passes),
        "score_aggregation": "mean",

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
