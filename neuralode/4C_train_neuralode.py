import os
import json
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from _3C_load_to_pytorch_neuralode import (
    META_DIR,
    MAX_LEN,
    EVENT_IGNORE_INDEX,
    save_json,
    load_json,
    CICNeuralODESequenceDataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    root: str = os.path.join("..", "data", "traffic_labelled", "2C_preprocessed_neuralode")
    max_len: int = None

    batch_train: int = 128
    batch_eval: int = 256

    feat_dim: int = None
    x_embed_dim: int = 128
    hidden_dim: int = 128
    ode_hidden: int = 128
    ode_layers: int = 2
    ode_steps: int = 4
    dt_clip: float = 60.0
    dropout: float = 0.1

    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 15
    early_stop_patience: int = 4
    early_stop_min_delta: float = 1e-4
    grad_clip: float = 1.0

    threshold_grid_points: int = 101
    threshold_percentile_min: float = 90.0
    threshold_percentile_max: float = 99.9

    seed: int = 42


CFG = Config()
CFG.max_len = int(MAX_LEN) if CFG.max_len is None else int(CFG.max_len)

CKPT_DIR = os.path.join(CFG.root, "checkpoints_neuralode")
os.makedirs(CKPT_DIR, exist_ok=True)

set_seed(CFG.seed)


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


def best_threshold_by_f1(y_true, y_score, cfg):
    thr_list = np.linspace(0.0, 1.0, 2001)
    best, best_f1 = None, -1.0
    for thr in thr_list:
        m = compute_metrics(y_true, y_score, float(thr))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best = m
    return best


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int, ode_hidden: int, ode_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for i in range(ode_layers - 1):
            layers.append(nn.Linear(in_dim, ode_hidden))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = ode_hidden
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class GRUODEClassifier(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        x_embed_dim: int,
        hidden_dim: int,
        ode_hidden: int,
        ode_layers: int,
        ode_steps: int,
        dropout: float,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.ode_steps = int(ode_steps)

        self.x_proj = nn.Sequential(
            nn.Linear(feat_dim, x_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.odefunc = ODEFunc(hidden_dim=hidden_dim, ode_hidden=ode_hidden, ode_layers=ode_layers, dropout=dropout)
        self.gru = nn.GRUCell(input_size=x_embed_dim, hidden_size=hidden_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def ode_euler(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if self.ode_steps <= 1:
            return h + dt.unsqueeze(-1) * self.odefunc(h)

        step = dt / float(self.ode_steps)
        for _ in range(self.ode_steps):
            h = h + step.unsqueeze(-1) * self.odefunc(h)
        return h

    def forward(self, x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        h = torch.zeros((B, self.hidden_dim), device=x.device, dtype=x.dtype)

        h_all = []
        for i in range(L):
            mi = mask[:, i].float()
            if mi.sum().item() == 0:
                h_all.append(h)
                continue

            hi = self.ode_euler(h, dt[:, i] * mi)
            xi = self.x_proj(x[:, i, :])
            hi = self.gru(xi, hi)
            h = hi
            h_all.append(h)

        H = torch.stack(h_all, dim=1)

        mask_f = mask.float()
        logits_t = self.head(H).squeeze(-1)  # [B, L]

        mask_f = mask.float()
        logits_t = logits_t * mask_f

        sum_logits = logits_t.sum(dim=1)
        len_valid = mask_f.sum(dim=1).clamp(min=1)

        logits = sum_logits / len_valid  # mean pooling

        return logits


def infer_feat_dim_from_one_batch(meta_df: pd.DataFrame) -> int:
    ds_train = CICNeuralODESequenceDataset(
        split="train",
        meta_df=meta_df,
        max_len=CFG.max_len,
        benign_only=False,
        dt_clip=CFG.dt_clip,
        return_event_labels=False,
    )
    loader = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    x = batch["x"]
    return int(x.shape[-1])


def make_loaders(meta_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_train = CICNeuralODESequenceDataset(
        split="train",
        meta_df=meta_df,
        max_len=CFG.max_len,
        benign_only=False,
        dt_clip=CFG.dt_clip,
        return_event_labels=False,
    )
    ds_val = CICNeuralODESequenceDataset(
        split="val",
        meta_df=meta_df,
        max_len=CFG.max_len,
        benign_only=False,
        dt_clip=CFG.dt_clip,
        return_event_labels=False,
    )
    ds_test = CICNeuralODESequenceDataset(
        split="test",
        meta_df=meta_df,
        max_len=CFG.max_len,
        benign_only=False,
        dt_clip=CFG.dt_clip,
        return_event_labels=False,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=CFG.batch_train,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=CFG.batch_eval,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=CFG.batch_eval,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return train_loader, val_loader, test_loader


def compute_pos_weight(meta_df: pd.DataFrame) -> torch.Tensor:
    df = meta_df[meta_df["split"] == "train"][["seq_id", "seq_y"]].drop_duplicates()
    pos = float((df["seq_y"] == 1).sum())
    neg = float((df["seq_y"] == 0).sum())
    if pos <= 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(neg / pos, device=device)


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, loss_fn) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_y = []
    all_s = []
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        dt = batch["dt"].to(device)
        mask = batch["mask"].to(device)
        y = batch["seq_y"].to(device).float()

        logits = model(x, dt, mask)
        loss = loss_fn(logits, y)

        scores = torch.sigmoid(logits).detach().cpu().numpy()
        all_s.append(scores)
        all_y.append(y.detach().cpu().numpy())

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs

    y_true = np.concatenate(all_y, axis=0).astype(np.int64)
    y_score = np.concatenate(all_s, axis=0).astype(np.float64)
    return y_true, y_score, (total_loss / max(n, 1))


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        dt = batch["dt"].to(device)
        mask = batch["mask"].to(device)
        y = batch["seq_y"].to(device).float()

        optim.zero_grad(set_to_none=True)

        logits = model(x, dt, mask)
        loss_seq = loss_fn(logits, y)
        loss = loss_seq

        loss.backward()

        if CFG.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), float(CFG.grad_clip))

        optim.step()

        bs = y.shape[0]
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


def main():
    meta_df = pd.read_parquet(META_DIR)

    CFG.feat_dim = infer_feat_dim_from_one_batch(meta_df)

    train_loader, val_loader, test_loader = make_loaders(meta_df)

    pos_weight = compute_pos_weight(meta_df)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = GRUODEClassifier(
        feat_dim=CFG.feat_dim,
        x_embed_dim=CFG.x_embed_dim,
        hidden_dim=CFG.hidden_dim,
        ode_hidden=CFG.ode_hidden,
        ode_layers=CFG.ode_layers,
        ode_steps=CFG.ode_steps,
        dropout=CFG.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    best_val_f1 = -1.0
    best_epoch = -1
    patience = 0

    cfg_path = os.path.join(CKPT_DIR, "config.json")
    save_json(cfg_path, CFG.__dict__)

    for epoch in range(1, CFG.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, loss_fn)

        yv, sv, val_loss = eval_model(model, val_loader, loss_fn)
        val_best = best_threshold_by_f1(yv, sv, CFG)
        val_f1 = float(val_best["f1"])
        thr = float(val_best["threshold"])
        val_metrics = compute_metrics(yv, sv, thr)

        print(json.dumps({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_metrics_at_best_f1_threshold": val_best,
        }, indent=2))

        val_f1 = float(val_best["f1"])
        improved = (val_f1 - best_val_f1) > CFG.early_stop_min_delta
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience = 0
            torch.save({"model": model.state_dict()}, os.path.join(CKPT_DIR, "best.pt"))
            save_json(os.path.join(CKPT_DIR, "best_val_metrics.json"), val_best)
        else:
            patience += 1
            if patience >= CFG.early_stop_patience:
                print(f"[EARLY STOP] epoch={epoch} best_epoch={best_epoch} best_val_f1={best_val_f1:.6f}")
                break

    ckpt = torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    yt, st, test_loss = eval_model(model, test_loader, loss_fn)
    best_val_saved = load_json(os.path.join(CKPT_DIR, "best_val_metrics.json"))
    thr = float(best_val_saved["threshold"])
    test_metrics = compute_metrics(yt, st, thr)

    out = {
        "test_loss": test_loss,
        "threshold_from_val": best_val_saved,
        "metrics_at_val_threshold": test_metrics,
    }

    save_json(os.path.join(CKPT_DIR, "test_report.json"), out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()