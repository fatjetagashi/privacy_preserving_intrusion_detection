from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.nn import SAGEConv


THRESHOLD_GRID = [round(value, 2) for value in np.arange(0.05, 0.951, 0.01)]


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def save_history_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_run_fragment(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def make_run_dir(base_dir: Path, model_name: str, variant_id: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{sanitize_run_fragment(variant_id)}_seed{seed}"
    run_dir = base_dir / model_name / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class GraphSAGEBinary(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = float(dropout)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x).squeeze(-1)


class NodeMLPBinary(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index=None):
        _ = edge_index
        return self.net(x).squeeze(-1)


@torch.no_grad()
def compute_mean_std(loader, num_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    total_items = 0
    sum_x = torch.zeros(num_features, dtype=torch.float64)
    sum_sq_x = torch.zeros(num_features, dtype=torch.float64)

    for data in loader:
        x = data.x.to(torch.float64)
        sum_x += x.sum(dim=0)
        sum_sq_x += (x * x).sum(dim=0)
        total_items += x.size(0)

    mean = sum_x / max(total_items, 1)
    variance = (sum_sq_x / max(total_items, 1)) - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    return mean.to(torch.float32), std.to(torch.float32)


@torch.no_grad()
def compute_pos_weight(loader) -> torch.Tensor:
    pos = 0
    neg = 0
    for data in loader:
        y = data.y
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())

    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


def normalize_x(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(x.device)) / std.to(x.device)


def threshold_free_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    unique = np.unique(y_true)
    roc_auc = roc_auc_score(y_true, y_prob) if len(unique) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if len(unique) > 1 else float("nan")
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp + eps)),
        "fpr": float(fp / (fp + tn + eps)),
        "fnr": float(fn / (fn + tp + eps)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "support_pos": int((y_true == 1).sum()),
        "support_neg": int((y_true == 0).sum()),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    best_metrics = None
    best_key = None

    for threshold in THRESHOLD_GRID:
        metrics = metrics_at_threshold(y_true, y_prob, threshold)
        key = (
            round(metrics["f1"], 12),
            round(metrics["recall"], 12),
            round(metrics["precision"], 12),
            -threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics

    return best_metrics


@torch.no_grad()
def collect_predictions(model, loader, criterion, device, x_mean, x_std) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_probs = []
    all_y = []
    total_loss = 0.0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        x = normalize_x(data.x, mean=x_mean, std=x_std)
        logits = model(x, data.edge_index)
        y = data.y.float()
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.numel()
        total_nodes += y.numel()
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(data.y.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_probs)
    avg_loss = total_loss / max(total_nodes, 1)
    return y_true, y_prob, float(avg_loss)


def train_one_epoch(model, loader, optimizer, criterion, device, x_mean, x_std, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        x = normalize_x(data.x, mean=x_mean, std=x_std)
        logits = model(x, data.edge_index)
        y = data.y.float()
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * y.numel()
        total_nodes += y.numel()

    return total_loss / max(total_nodes, 1)


def build_epoch_row(
    epoch: int,
    train_loss: float,
    lr: float,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    val_loss: float,
) -> Dict[str, float]:
    threshold_free = threshold_free_metrics(y_true, y_prob)
    metrics_05 = metrics_at_threshold(y_true, y_prob, threshold=0.5)
    tuned_metrics = find_best_threshold(y_true, y_prob)

    return {
        "epoch": epoch,
        "lr": float(lr),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_roc_auc": float(threshold_free["roc_auc"]),
        "val_pr_auc": float(threshold_free["pr_auc"]),
        "val_f1_at_0_5": float(metrics_05["f1"]),
        "val_precision_at_0_5": float(metrics_05["precision"]),
        "val_recall_at_0_5": float(metrics_05["recall"]),
        "val_tuned_threshold": float(tuned_metrics["threshold"]),
        "val_tuned_f1": float(tuned_metrics["f1"]),
        "val_tuned_precision": float(tuned_metrics["precision"]),
        "val_tuned_recall": float(tuned_metrics["recall"]),
    }


def summarize_final_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loss: float,
    tuned_threshold: float,
) -> Dict[str, object]:
    return {
        "threshold_free": {
            "loss": float(loss),
            **threshold_free_metrics(y_true, y_prob),
        },
        "at_threshold_0.5": metrics_at_threshold(y_true, y_prob, 0.5),
        "at_threshold_tuned": metrics_at_threshold(y_true, y_prob, tuned_threshold),
        "tuned_threshold": float(tuned_threshold),
    }


def cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}
