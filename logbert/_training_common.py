from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
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
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def sanitize_run_fragment(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def make_run_dir(base_dir: Path, model_name: str, variant_id: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{sanitize_run_fragment(variant_id)}_seed{seed}"
    run_dir = base_dir / model_name / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def threshold_free_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    unique = np.unique(y_true)
    roc_auc = roc_auc_score(y_true, y_score) if len(unique) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_score) if len(unique) > 1 else float("nan")
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
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


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    best_metrics = None
    best_key = None

    for threshold in THRESHOLD_GRID:
        metrics = metrics_at_threshold(y_true, y_score, threshold)
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


def summarize_final_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    loss: float,
    tuned_threshold: float,
) -> Dict[str, object]:
    return {
        "threshold_free": {
            "loss": float(loss),
            **threshold_free_metrics(y_true, y_score),
        },
        "at_threshold_0.5": metrics_at_threshold(y_true, y_score, 0.5),
        "at_threshold_tuned": metrics_at_threshold(y_true, y_score, tuned_threshold),
        "tuned_threshold": float(tuned_threshold),
    }


def cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}
