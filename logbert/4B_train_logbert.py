from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from _3B_load_to_pytorch_logbert import (
    CLS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    build_loaders,
)
from _training_common import (
    cpu_state_dict,
    make_run_dir,
    save_history_csv,
    save_json,
    set_seed,
)


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"
REPORTS_DIR = THIS_DIR / "reports"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100
SEQ_THRESHOLD_GRID = [round(value, 2) for value in np.arange(0.01, 1.00, 0.01)]
TASK_DEFINITION = (
    "LogBERT-style sequence anomaly detection over time-ordered traffic-flow tokens. "
    "Training uses benign sequences only, with masked language modeling and optional "
    "hypersphere regularization on the [CLS] representation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TASK_DEFINITION)
    parser.add_argument("--data-root", default=None, help="Root of a preprocessed LogBERT sequence dataset.")
    parser.add_argument("--batch-train", type=int, default=64)
    parser.add_argument("--batch-eval", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--mask-prob-train", type=float, default=0.15)
    parser.add_argument("--mask-prob-score", type=float, default=0.50)
    parser.add_argument("--score-passes", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=9)
    parser.add_argument("--vhm-weight", type=float, default=0.10)
    parser.add_argument("--center-warmup-epochs", type=int, default=1)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-len-percentile", type=float, default=95.0)
    parser.add_argument("--max-len-cap", type=int, default=512)
    parser.add_argument(
        "--benign-percentile",
        type=float,
        default=99.0,
        help="Percentile used on train-benign scores/distances for unsupervised thresholds.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def mlm_mask_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
    cls_id: int,
    mask_id: int,
    vocab_size: int,
    mask_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    eligible = (attention_mask == 1) & (input_ids != pad_id) & (input_ids != cls_id)
    probs = torch.rand_like(input_ids.float())
    to_mask = eligible & (probs < mask_prob)

    eligible_counts = eligible.sum(dim=1)
    masked_counts = to_mask.sum(dim=1)
    missing_mask_rows = ((eligible_counts > 0) & (masked_counts == 0)).nonzero(as_tuple=False).view(-1)
    for row_idx in missing_mask_rows.tolist():
        positions = eligible[row_idx].nonzero(as_tuple=False).view(-1)
        if positions.numel() == 0:
            continue
        chosen_pos = positions[torch.randint(0, positions.numel(), (1,), device=input_ids.device)]
        to_mask[row_idx, chosen_pos] = True

    labels[to_mask] = input_ids[to_mask]

    probs2 = torch.rand_like(input_ids.float())
    mask_80 = to_mask & (probs2 < 0.8)
    rand_10 = to_mask & (probs2 >= 0.8) & (probs2 < 0.9)

    masked[mask_80] = mask_id
    if rand_10.any():
        if vocab_size > 4:
            rand_ids = torch.randint(low=4, high=vocab_size, size=input_ids.shape, device=input_ids.device)
            masked[rand_10] = rand_ids[rand_10]
        else:
            masked[rand_10] = mask_id

    return masked, labels


class LogBERTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_id: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        ff_mult: int,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model)
        self.mlm_decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.mlm_decoder.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.tok_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.emb_norm(x)
        x = self.dropout(x)

        key_padding_mask = attention_mask == 0
        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mlm_hidden = self.mlm_norm(torch.nn.functional.gelu(self.mlm_dense(hidden)))
        mlm_logits = self.mlm_decoder(mlm_hidden) + self.mlm_bias

        return {
            "hidden_states": hidden,
            "cls_output": hidden[:, 0, :],
            "mlm_logits": mlm_logits,
        }


def train_one_epoch(
    model,
    loader,
    optimizer,
    mlm_loss_fn,
    center: torch.Tensor | None,
    pad_id: int,
    cls_id: int,
    mask_id: int,
    vocab_size: int,
    mask_prob: float,
    vhm_weight: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    mlm_loss_sum = 0.0
    vhm_loss_sum = 0.0
    total_samples = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        masked_ids, mlm_labels = mlm_mask_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_id=pad_id,
            cls_id=cls_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
            mask_prob=mask_prob,
        )

        outputs = model(masked_ids, attention_mask)
        mlm_loss = mlm_loss_fn(outputs["mlm_logits"].reshape(-1, vocab_size), mlm_labels.reshape(-1))

        if center is not None and vhm_weight > 0:
            dist = torch.sum((outputs["cls_output"] - center.unsqueeze(0)) ** 2, dim=1)
            vhm_loss = torch.mean(dist)
        else:
            vhm_loss = torch.tensor(0.0, device=DEVICE)

        total_loss = mlm_loss + (vhm_weight * vhm_loss)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = input_ids.size(0)
        total_samples += batch_size
        total_loss_sum += float(total_loss.item()) * batch_size
        mlm_loss_sum += float(mlm_loss.item()) * batch_size
        vhm_loss_sum += float(vhm_loss.item()) * batch_size

    denom = max(total_samples, 1)
    return {
        "total": total_loss_sum / denom,
        "mlm": mlm_loss_sum / denom,
        "vhm": vhm_loss_sum / denom,
    }


@torch.no_grad()
def compute_center(model, loader) -> torch.Tensor:
    model.eval()
    cls_sum = None
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        outputs = model(input_ids, attention_mask)
        cls_out = outputs["cls_output"]

        if cls_sum is None:
            cls_sum = torch.zeros(cls_out.size(1), device=DEVICE)
        cls_sum += cls_out.sum(dim=0)
        total += cls_out.size(0)

    if cls_sum is None or total == 0:
        raise ValueError("Cannot compute hypersphere center from an empty loader.")

    center = cls_sum / float(total)
    center[(center.abs() < 1e-6)] = 1e-6
    return center.detach()


@torch.no_grad()
def score_sequences(
    model,
    loader,
    center: torch.Tensor | None,
    pad_id: int,
    cls_id: int,
    mask_id: int,
    vocab_size: int,
    mask_prob: float,
    num_candidates: int,
    score_passes: int,
) -> pd.DataFrame:
    model.eval()
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    rows = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        base_outputs = model(input_ids, attention_mask)
        if center is not None:
            distance = torch.sqrt(torch.sum((base_outputs["cls_output"] - center.unsqueeze(0)) ** 2, dim=1))
        else:
            distance = torch.zeros(input_ids.size(0), device=DEVICE)

        ratio_sum = torch.zeros(input_ids.size(0), device=DEVICE)
        mlm_loss_sum = torch.zeros(input_ids.size(0), device=DEVICE)

        for _ in range(score_passes):
            masked_ids, mlm_labels = mlm_mask_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_id=pad_id,
                cls_id=cls_id,
                mask_id=mask_id,
                vocab_size=vocab_size,
                mask_prob=mask_prob,
            )

            outputs = model(masked_ids, attention_mask)
            logits = outputs["mlm_logits"]
            per_token_loss = token_loss_fn(logits.reshape(-1, vocab_size), mlm_labels.reshape(-1)).view(mlm_labels.shape)
            masked_positions = mlm_labels != IGNORE_INDEX

            for row_idx in range(input_ids.size(0)):
                mask_idx = masked_positions[row_idx]
                masked_count = int(mask_idx.sum().item())
                if masked_count == 0:
                    continue

                masked_logits = logits[row_idx][mask_idx]
                masked_targets = mlm_labels[row_idx][mask_idx]
                candidate_logits = masked_logits.clone()
                if vocab_size > len(SPECIAL_TOKENS):
                    candidate_logits[:, :len(SPECIAL_TOKENS)] = torch.finfo(candidate_logits.dtype).min
                topk = torch.topk(candidate_logits, k=min(num_candidates, vocab_size), dim=-1).indices
                detected = topk.eq(masked_targets.unsqueeze(-1)).any(dim=-1)
                undetected_ratio = float((~detected).float().mean().item())
                mean_loss = float(per_token_loss[row_idx][mask_idx].mean().item())

                ratio_sum[row_idx] += undetected_ratio
                mlm_loss_sum[row_idx] += mean_loss

        ratio_avg = (ratio_sum / float(score_passes)).detach().cpu().numpy()
        mlm_loss_avg = (mlm_loss_sum / float(score_passes)).detach().cpu().numpy()
        distance_np = distance.detach().cpu().numpy()
        seq_y = batch["seq_y"].detach().cpu().numpy()

        for idx in range(input_ids.size(0)):
            rows.append(
                {
                    "seq_id": batch["seq_id"][idx],
                    "day": batch["day"][idx],
                    "entity": batch["entity"][idx],
                    "split_group_id": batch["split_group_id"][idx],
                    "seq_y": int(seq_y[idx]),
                    "mlm_error_ratio": float(ratio_avg[idx]),
                    "mlm_loss": float(mlm_loss_avg[idx]),
                    "distance": float(distance_np[idx]),
                }
            )

    return pd.DataFrame(rows)


def compute_score_stats(benign_scores: pd.DataFrame, benign_percentile: float) -> Dict[str, float]:
    eps = 1e-12
    distance_mean = float(benign_scores["distance"].mean())
    distance_std = float(benign_scores["distance"].std(ddof=0))
    if not np.isfinite(distance_std) or distance_std <= 0:
        distance_std = eps

    mlm_threshold = float(np.percentile(benign_scores["mlm_error_ratio"], benign_percentile))
    distance_threshold = float(np.percentile(benign_scores["distance"], benign_percentile))

    composite_benign = benign_scores["mlm_error_ratio"].to_numpy(dtype=float) + np.clip(
        (benign_scores["distance"].to_numpy(dtype=float) - distance_mean) / distance_std,
        a_min=0.0,
        a_max=None,
    )
    composite_threshold = float(np.percentile(composite_benign, benign_percentile))

    return {
        "mlm_ratio_threshold": mlm_threshold,
        "distance_threshold": distance_threshold,
        "distance_mean": distance_mean,
        "distance_std": distance_std,
        "composite_threshold": composite_threshold,
    }


def attach_composite_score(scores_df: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
    out = scores_df.copy()
    distance_z = np.clip(
        (out["distance"].to_numpy(dtype=float) - stats["distance_mean"]) / max(stats["distance_std"], 1e-12),
        a_min=0.0,
        a_max=None,
    )
    out["distance_z"] = distance_z
    out["composite_score"] = out["mlm_error_ratio"].to_numpy(dtype=float) + distance_z
    return out


def threshold_free_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    unique = np.unique(y_true)
    roc_auc = roc_auc_score(y_true, y_score) if len(unique) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_score) if len(unique) > 1 else float("nan")
    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


def safe_metric_for_selection(value: float, fallback: float = -1.0) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, threshold_meta: Dict[str, float]) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12
    return {
        **threshold_meta,
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


def predict_or_rule(scores_df: pd.DataFrame, seq_threshold: float, distance_threshold: float, use_distance: bool) -> np.ndarray:
    mlm_pred = scores_df["mlm_error_ratio"].to_numpy(dtype=float) > float(seq_threshold)
    if not use_distance:
        return mlm_pred.astype(np.int64)
    dist_pred = scores_df["distance"].to_numpy(dtype=float) > float(distance_threshold)
    return np.logical_or(mlm_pred, dist_pred).astype(np.int64)


def evaluate_with_thresholds(
    scores_df: pd.DataFrame,
    stats: Dict[str, float],
    seq_threshold: float,
    use_distance: bool,
    label: str,
) -> Dict[str, object]:
    y_true = scores_df["seq_y"].to_numpy(dtype=np.int64)
    y_pred = predict_or_rule(
        scores_df=scores_df,
        seq_threshold=seq_threshold,
        distance_threshold=stats["distance_threshold"],
        use_distance=use_distance,
    )
    return metrics_from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        threshold_meta={
            "threshold_type": label,
            "sequence_threshold": float(seq_threshold),
            "distance_threshold": float(stats["distance_threshold"]),
            "use_distance_rule": bool(use_distance),
        },
    )


def find_best_validation_rule(val_scores_df: pd.DataFrame, stats: Dict[str, float], use_distance: bool) -> Dict[str, float]:
    y_true = val_scores_df["seq_y"].to_numpy(dtype=np.int64)
    best_metrics = None
    best_key = None

    for seq_threshold in SEQ_THRESHOLD_GRID:
        y_pred = predict_or_rule(
            scores_df=val_scores_df,
            seq_threshold=seq_threshold,
            distance_threshold=stats["distance_threshold"],
            use_distance=use_distance,
        )
        metrics = metrics_from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            threshold_meta={
                "threshold_type": "validation_tuned_rule",
                "sequence_threshold": float(seq_threshold),
                "distance_threshold": float(stats["distance_threshold"]),
                "use_distance_rule": bool(use_distance),
            },
        )
        key = (
            round(metrics["f1"], 12),
            round(metrics["recall"], 12),
            round(metrics["precision"], 12),
            -float(seq_threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics

    return best_metrics


def build_epoch_row(
    epoch: int,
    lr: float,
    train_losses: Dict[str, float],
    val_scores_df: pd.DataFrame,
    stats: Dict[str, float],
    use_distance: bool,
) -> Dict[str, float]:
    y_true = val_scores_df["seq_y"].to_numpy(dtype=np.int64)
    y_score = val_scores_df["composite_score"].to_numpy(dtype=float)
    threshold_free = threshold_free_metrics(y_true, y_score)
    unsupervised_metrics = evaluate_with_thresholds(
        scores_df=val_scores_df,
        stats=stats,
        seq_threshold=stats["mlm_ratio_threshold"],
        use_distance=use_distance,
        label="train_benign_percentile_rule",
    )
    tuned_metrics = find_best_validation_rule(val_scores_df=val_scores_df, stats=stats, use_distance=use_distance)

    return {
        "epoch": epoch,
        "lr": float(lr),
        "train_total_loss": float(train_losses["total"]),
        "train_mlm_loss": float(train_losses["mlm"]),
        "train_vhm_loss": float(train_losses["vhm"]),
        "val_roc_auc": float(threshold_free["roc_auc"]),
        "val_pr_auc": float(threshold_free["pr_auc"]),
        "val_f1_unsupervised": float(unsupervised_metrics["f1"]),
        "val_precision_unsupervised": float(unsupervised_metrics["precision"]),
        "val_recall_unsupervised": float(unsupervised_metrics["recall"]),
        "val_seq_threshold_unsupervised": float(stats["mlm_ratio_threshold"]),
        "val_distance_threshold_unsupervised": float(stats["distance_threshold"]),
        "val_tuned_sequence_threshold": float(tuned_metrics["sequence_threshold"]),
        "val_f1_tuned": float(tuned_metrics["f1"]),
        "val_precision_tuned": float(tuned_metrics["precision"]),
        "val_recall_tuned": float(tuned_metrics["recall"]),
    }


def summarize_scores(scores_df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "n_sequences": int(len(scores_df)),
        "n_attack": int((scores_df["seq_y"] == 1).sum()),
        "n_benign": int((scores_df["seq_y"] == 0).sum()),
    }

    for score_col in ["mlm_error_ratio", "mlm_loss", "distance", "composite_score"]:
        values = scores_df[score_col].to_numpy(dtype=float)
        summary[score_col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

        for class_label, class_name in [(0, "benign"), (1, "attack")]:
            class_values = scores_df.loc[scores_df["seq_y"] == class_label, score_col].to_numpy(dtype=float)
            if class_values.size == 0:
                summary[score_col][f"{class_name}_mean"] = None
                summary[score_col][f"{class_name}_p95"] = None
            else:
                summary[score_col][f"{class_name}_mean"] = float(np.mean(class_values))
                summary[score_col][f"{class_name}_p95"] = float(np.percentile(class_values, 95))

    return summary


def evaluate_by_day(
    scores_df: pd.DataFrame,
    stats: Dict[str, float],
    tuned_sequence_threshold: float,
    use_distance: bool,
) -> Dict[str, Dict[str, object]]:
    report = {}
    for day, day_df in scores_df.groupby("day", sort=True):
        y_true = day_df["seq_y"].to_numpy(dtype=np.int64)
        y_score = day_df["composite_score"].to_numpy(dtype=float)
        report[str(day)] = {
            "n_sequences": int(len(day_df)),
            "n_attack": int((day_df["seq_y"] == 1).sum()),
            "n_benign": int((day_df["seq_y"] == 0).sum()),
            "threshold_free": threshold_free_metrics(y_true, y_score),
            "at_threshold_train_benign_percentile": evaluate_with_thresholds(
                scores_df=day_df,
                stats=stats,
                seq_threshold=stats["mlm_ratio_threshold"],
                use_distance=use_distance,
                label="train_benign_percentile_rule",
            ),
            "at_threshold_val_tuned": evaluate_with_thresholds(
                scores_df=day_df,
                stats=stats,
                seq_threshold=tuned_sequence_threshold,
                use_distance=use_distance,
                label="validation_tuned_rule",
            ),
        }
    return report


def run_single_seed(args: argparse.Namespace, seed: int) -> Path:
    set_seed(seed)

    train_loader, val_loader, test_loader, data_meta = build_loaders(
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        seed=seed,
        root=args.data_root,
        max_len=args.max_len,
        max_len_percentile=args.max_len_percentile,
        max_len_cap=args.max_len_cap,
        min_freq=args.min_freq,
    )

    token_to_id = data_meta["token_to_id"]
    vocab_size = int(data_meta["vocab_size"])
    max_len = int(data_meta["max_len"])
    pad_id = int(token_to_id[PAD_TOKEN])
    mask_id = int(token_to_id[MASK_TOKEN])
    cls_id = int(token_to_id[CLS_TOKEN])

    model = LogBERTModel(
        vocab_size=vocab_size,
        max_len=max_len,
        pad_id=pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    sequence_build_config = data_meta.get("sequence_build_config", {})
    variant_id = sequence_build_config.get("variant_id", Path(data_meta["root"]).name)
    run_dir = make_run_dir(RUNS_DIR, model_name="logbert", variant_id=variant_id, seed=seed)

    if int(data_meta["n_train_benign"]) <= 0:
        raise ValueError("No benign training sequences were found. LogBERT requires benign-only training data.")
    if int(data_meta["n_val_total"]) <= 0 or int(data_meta["n_test_total"]) <= 0:
        raise ValueError("Validation or test split is empty. Check sequence-building and split configuration.")

    history = []
    center = None
    best_center = None
    best_state = None
    best_epoch = -1
    best_val_pr_auc = float("-inf")
    best_epoch_row = None
    stale_epochs = 0
    use_distance_rule = args.vhm_weight > 0

    for epoch in range(1, args.max_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mlm_loss_fn=mlm_loss_fn,
            center=center if epoch > args.center_warmup_epochs else None,
            pad_id=pad_id,
            cls_id=cls_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
            mask_prob=args.mask_prob_train,
            vhm_weight=args.vhm_weight,
            grad_clip=args.grad_clip,
        )

        if epoch >= args.center_warmup_epochs:
            center = compute_center(model, train_loader)

        train_benign_scores = score_sequences(
            model=model,
            loader=train_loader,
            center=center,
            pad_id=pad_id,
            cls_id=cls_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
            mask_prob=args.mask_prob_score,
            num_candidates=args.num_candidates,
            score_passes=args.score_passes,
        )
        stats = compute_score_stats(train_benign_scores, benign_percentile=args.benign_percentile)

        val_scores_df = score_sequences(
            model=model,
            loader=val_loader,
            center=center,
            pad_id=pad_id,
            cls_id=cls_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
            mask_prob=args.mask_prob_score,
            num_candidates=args.num_candidates,
            score_passes=args.score_passes,
        )
        val_scores_df = attach_composite_score(val_scores_df, stats)

        epoch_row = build_epoch_row(
            epoch=epoch,
            lr=lr_now,
            train_losses=train_losses,
            val_scores_df=val_scores_df,
            stats=stats,
            use_distance=use_distance_rule,
        )
        history.append(epoch_row)

        val_pr_auc_for_selection = safe_metric_for_selection(epoch_row["val_pr_auc"])
        scheduler.step(val_pr_auc_for_selection)

        print(
            f"seed={seed} epoch={epoch:02d} "
            f"train_total={train_losses['total']:.4f} "
            f"train_mlm={train_losses['mlm']:.4f} "
            f"train_vhm={train_losses['vhm']:.4f} "
            f"val_pr_auc={epoch_row['val_pr_auc']:.4f} "
            f"val_f1_unsup={epoch_row['val_f1_unsupervised']:.4f} "
            f"val_f1_tuned={epoch_row['val_f1_tuned']:.4f}"
        )

        improved = best_state is None or val_pr_auc_for_selection > (best_val_pr_auc + args.early_stop_min_delta)
        if improved:
            best_val_pr_auc = val_pr_auc_for_selection
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            best_center = None if center is None else center.detach().cpu()
            best_epoch_row = dict(epoch_row)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print(
                    f"[EARLY STOP] seed={seed} epoch={epoch:02d} "
                    f"best_epoch={best_epoch:02d} best_val_pr_auc={best_val_pr_auc:.6f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_center is not None:
        center = best_center.to(DEVICE)

    train_benign_scores = score_sequences(
        model=model,
        loader=train_loader,
        center=center,
        pad_id=pad_id,
        cls_id=cls_id,
        mask_id=mask_id,
        vocab_size=vocab_size,
        mask_prob=args.mask_prob_score,
        num_candidates=args.num_candidates,
        score_passes=args.score_passes,
    )
    stats = compute_score_stats(train_benign_scores, benign_percentile=args.benign_percentile)
    train_benign_scores = attach_composite_score(train_benign_scores, stats)

    val_scores_df = score_sequences(
        model=model,
        loader=val_loader,
        center=center,
        pad_id=pad_id,
        cls_id=cls_id,
        mask_id=mask_id,
        vocab_size=vocab_size,
        mask_prob=args.mask_prob_score,
        num_candidates=args.num_candidates,
        score_passes=args.score_passes,
    )
    val_scores_df = attach_composite_score(val_scores_df, stats)

    test_scores_df = score_sequences(
        model=model,
        loader=test_loader,
        center=center,
        pad_id=pad_id,
        cls_id=cls_id,
        mask_id=mask_id,
        vocab_size=vocab_size,
        mask_prob=args.mask_prob_score,
        num_candidates=args.num_candidates,
        score_passes=args.score_passes,
    )
    test_scores_df = attach_composite_score(test_scores_df, stats)

    val_y = val_scores_df["seq_y"].to_numpy(dtype=np.int64)
    val_composite = val_scores_df["composite_score"].to_numpy(dtype=float)
    test_y = test_scores_df["seq_y"].to_numpy(dtype=np.int64)
    test_composite = test_scores_df["composite_score"].to_numpy(dtype=float)

    unsupervised_val_metrics = evaluate_with_thresholds(
        scores_df=val_scores_df,
        stats=stats,
        seq_threshold=stats["mlm_ratio_threshold"],
        use_distance=use_distance_rule,
        label="train_benign_percentile_rule",
    )
    tuned_val_metrics = find_best_validation_rule(val_scores_df=val_scores_df, stats=stats, use_distance=use_distance_rule)

    val_metrics = {
        "threshold_free": threshold_free_metrics(val_y, val_composite),
        "at_threshold_train_benign_percentile": unsupervised_val_metrics,
        "at_threshold_val_tuned": tuned_val_metrics,
        "score_statistics": stats,
        "score_summary": summarize_scores(val_scores_df),
    }

    test_unsupervised_metrics = evaluate_with_thresholds(
        scores_df=test_scores_df,
        stats=stats,
        seq_threshold=stats["mlm_ratio_threshold"],
        use_distance=use_distance_rule,
        label="train_benign_percentile_rule",
    )
    test_tuned_metrics = evaluate_with_thresholds(
        scores_df=test_scores_df,
        stats=stats,
        seq_threshold=tuned_val_metrics["sequence_threshold"],
        use_distance=use_distance_rule,
        label="validation_tuned_rule",
    )

    test_metrics = {
        "threshold_free": threshold_free_metrics(test_y, test_composite),
        "at_threshold_train_benign_percentile": test_unsupervised_metrics,
        "at_threshold_val_tuned": test_tuned_metrics,
        "score_statistics": stats,
        "score_summary": summarize_scores(test_scores_df),
    }

    val_metrics_by_day = evaluate_by_day(
        scores_df=val_scores_df,
        stats=stats,
        tuned_sequence_threshold=tuned_val_metrics["sequence_threshold"],
        use_distance=use_distance_rule,
    )
    test_metrics_by_day = evaluate_by_day(
        scores_df=test_scores_df,
        stats=stats,
        tuned_sequence_threshold=tuned_val_metrics["sequence_threshold"],
        use_distance=use_distance_rule,
    )

    run_config = {
        "model_name": "logbert",
        "task_definition": TASK_DEFINITION,
        "task_framing": (
            "Train on benign sequences only. Use masked language modeling and optional hypersphere "
            "regularization. Detect anomalies via masked-token prediction failures and CLS-distance."
        ),
        "prediction_target": "seq_y",
        "seed": seed,
        "data_root": data_meta["root"],
        "batch_train": args.batch_train,
        "batch_eval": args.batch_eval,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "ff_mult": args.ff_mult,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "grad_clip": args.grad_clip,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_min_lr": args.scheduler_min_lr,
        "mask_prob_train": args.mask_prob_train,
        "mask_prob_score": args.mask_prob_score,
        "score_passes": args.score_passes,
        "num_candidates": args.num_candidates,
        "vhm_weight": args.vhm_weight,
        "center_warmup_epochs": args.center_warmup_epochs,
        "min_freq": args.min_freq,
        "max_len": max_len,
        "max_len_percentile": args.max_len_percentile,
        "max_len_cap": args.max_len_cap,
        "benign_percentile": args.benign_percentile,
        "vocab_train_scope": data_meta["vocab_train_scope"],
        "vocab_size": vocab_size,
        "sequence_build_config": sequence_build_config,
        "split_summary": data_meta["split_summary"],
        "special_tokens": [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN],
        "use_distance_rule": use_distance_rule,
        "evaluation_protocol": {
            "threshold_free_score": "composite_score",
            "unsupervised_threshold_source": "train_benign_percentile",
            "validation_tuned_threshold_source": "validation_split_labels",
        },
    }

    summary = {
        "best_epoch": best_epoch,
        "best_val_pr_auc": float(best_val_pr_auc),
        "best_val_row": best_epoch_row,
        "best_unsupervised_val_f1": float(unsupervised_val_metrics["f1"]),
        "best_tuned_val_f1": float(tuned_val_metrics["f1"]),
        "tuned_validation_sequence_threshold": float(tuned_val_metrics["sequence_threshold"]),
        "distance_threshold": float(stats["distance_threshold"]),
        "device": str(DEVICE),
        "train_benign_sequences": int(data_meta["n_train_benign"]),
        "val_total_sequences": int(data_meta["n_val_total"]),
        "test_total_sequences": int(data_meta["n_test_total"]),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "center": None if center is None else center.detach().cpu(),
            "token_to_id": token_to_id,
            "run_config": run_config,
            "summary": summary,
        },
        run_dir / "best_model.pt",
    )

    save_history_csv(run_dir / "history.csv", history)
    save_json(run_dir / "run_config.json", run_config)
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "val_metrics.json", val_metrics)
    save_json(run_dir / "test_metrics.json", test_metrics)
    save_json(run_dir / "val_metrics_by_day.json", val_metrics_by_day)
    save_json(run_dir / "test_metrics_by_day.json", test_metrics_by_day)
    train_benign_scores.to_csv(run_dir / "train_benign_scores.csv", index=False)
    val_scores_df.to_csv(run_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(run_dir / "test_scores.csv", index=False)

    return run_dir


def main() -> None:
    args = parse_args()
    print("Device:", DEVICE)

    run_dirs = []
    for seed in args.seeds:
        run_dir = run_single_seed(args=args, seed=seed)
        run_dirs.append(run_dir)
        print(f"Saved run artifacts to: {run_dir}")

    print("Completed runs:")
    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
