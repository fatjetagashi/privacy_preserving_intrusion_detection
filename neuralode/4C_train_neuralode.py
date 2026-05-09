from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from _3C_load_to_pytorch_neuralode import (
    EVENT_IGNORE_INDEX,
    build_loaders,
)
from _training_common import (
    cpu_state_dict,
    find_best_threshold,
    make_run_dir,
    metrics_at_threshold,
    save_history_csv,
    save_json,
    set_seed,
    summarize_final_metrics,
    threshold_free_metrics,
)


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"
REPORTS_DIR = THIS_DIR / "reports"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_DEFINITION = (
    "Continuous-time intrusion detection over time-stamped traffic-flow trajectories "
    "built from CIC-IDS2017 traffic-labelled records."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TASK_DEFINITION)
    parser.add_argument("--data-root", default=None, help="Root of a preprocessed Neural ODE dataset.")
    parser.add_argument("--batch-train", type=int, default=64)
    parser.add_argument("--batch-eval", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--x-embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ode-hidden", type=int, default=128)
    parser.add_argument("--ode-layers", type=int, default=2)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--solver", choices=["euler", "midpoint", "rk4"], default="rk4")
    parser.add_argument("--time-conditioning", choices=["none", "concat"], default="concat")
    parser.add_argument("--pooling", choices=["mean", "max", "last", "attention"], default="max")
    parser.add_argument("--prediction-level", choices=["sequence", "event"], default="sequence")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-len-percentile", type=float, default=95.0)
    parser.add_argument("--max-len-cap", type=int, default=512)
    parser.add_argument("--dt-clip", type=float, default=None)
    parser.add_argument("--max-feature-rows", type=int, default=200_000)
    parser.add_argument("--max-dt-samples", type=int, default=500_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def safe_metric_for_selection(value: float, fallback: float = -1.0) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def split_coverage_issues(data_meta: Dict[str, object], prediction_level: str) -> list[str]:
    issues = []

    if prediction_level == "sequence":
        counts = {
            "train": (int(data_meta["n_train_attack"]), int(data_meta["n_train_benign"])),
            "val": (int(data_meta["n_val_attack"]), int(data_meta["n_val_benign"])),
            "test": (int(data_meta["n_test_attack"]), int(data_meta["n_test_benign"])),
        }
        unit_name = "sequence"
    else:
        counts = {
            "train": (
                int(data_meta["n_train_attack_events"]),
                int(data_meta["n_train_events"]) - int(data_meta["n_train_attack_events"]),
            ),
            "val": (
                int(data_meta["n_val_attack_events"]),
                int(data_meta["n_val_events"]) - int(data_meta["n_val_attack_events"]),
            ),
            "test": (
                int(data_meta["n_test_attack_events"]),
                int(data_meta["n_test_events"]) - int(data_meta["n_test_attack_events"]),
            ),
        }
        unit_name = "event"

    for split_name, (n_attack, n_benign) in counts.items():
        if n_attack <= 0:
            issues.append(f"{split_name} split has no positive {unit_name} labels.")
        if n_benign <= 0:
            issues.append(f"{split_name} split has no benign {unit_name} labels.")

    return issues


class ODEFunc(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ode_hidden: int,
        ode_layers: int,
        dropout: float,
        time_conditioning: str,
    ):
        super().__init__()
        self.time_conditioning = time_conditioning
        in_dim = hidden_dim + (1 if time_conditioning == "concat" else 0)

        layers = []
        current_dim = in_dim
        for _ in range(max(ode_layers - 1, 0)):
            layers.append(nn.Linear(current_dim, ode_hidden))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            current_dim = ode_hidden
        layers.append(nn.Linear(current_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        if self.time_conditioning == "concat":
            ode_input = torch.cat([h, t_scalar.unsqueeze(-1)], dim=-1)
        else:
            ode_input = h
        return self.net(ode_input)


class ContinuousTimeGRUClassifier(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        x_embed_dim: int,
        hidden_dim: int,
        ode_hidden: int,
        ode_layers: int,
        ode_steps: int,
        solver: str,
        time_conditioning: str,
        pooling: str,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_steps = max(int(ode_steps), 1)
        self.solver = solver
        self.pooling = pooling

        self.x_proj = nn.Sequential(
            nn.Linear(feat_dim, x_embed_dim),
            nn.LayerNorm(x_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.odefunc = ODEFunc(
            hidden_dim=hidden_dim,
            ode_hidden=ode_hidden,
            ode_layers=ode_layers,
            dropout=dropout,
            time_conditioning=time_conditioning,
        )
        self.gru = nn.GRUCell(input_size=x_embed_dim, hidden_size=hidden_dim)
        self.event_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.seq_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        if pooling == "attention":
            self.attn_proj = nn.Linear(hidden_dim, 1)

    def _rhs(self, h: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        return self.odefunc(h, t_scalar)

    def _integrator_step(self, h: torch.Tensor, step_dt: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        if self.solver == "euler":
            return h + step_dt.unsqueeze(-1) * self._rhs(h, t_scalar)

        if self.solver == "midpoint":
            k1 = self._rhs(h, t_scalar)
            h_mid = h + 0.5 * step_dt.unsqueeze(-1) * k1
            k2 = self._rhs(h_mid, t_scalar + 0.5 * step_dt)
            return h + step_dt.unsqueeze(-1) * k2

        if self.solver == "rk4":
            k1 = self._rhs(h, t_scalar)
            k2 = self._rhs(h + 0.5 * step_dt.unsqueeze(-1) * k1, t_scalar + 0.5 * step_dt)
            k3 = self._rhs(h + 0.5 * step_dt.unsqueeze(-1) * k2, t_scalar + 0.5 * step_dt)
            k4 = self._rhs(h + step_dt.unsqueeze(-1) * k3, t_scalar + step_dt)
            return h + (step_dt.unsqueeze(-1) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        raise ValueError(f"Unsupported solver: {self.solver}")

    def _solve_interval(self, h: torch.Tensor, dt: torch.Tensor, t_start: torch.Tensor) -> torch.Tensor:
        dt = torch.clamp(dt, min=0.0)
        sub_dt = dt / float(self.ode_steps)
        current_t = t_start
        next_h = h

        for _ in range(self.ode_steps):
            next_h = self._integrator_step(next_h, sub_dt, current_t)
            current_t = current_t + sub_dt
        return next_h

    def _masked_pool(self, hidden_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float()

        if self.pooling == "mean":
            numer = (hidden_seq * mask_f.unsqueeze(-1)).sum(dim=1)
            denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            return numer / denom

        if self.pooling == "max":
            masked_hidden = hidden_seq.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            pooled = masked_hidden.max(dim=1).values
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            return pooled

        if self.pooling == "last":
            last_idx = (mask.sum(dim=1) - 1).clamp(min=0)
            batch_idx = torch.arange(hidden_seq.size(0), device=hidden_seq.device)
            return hidden_seq[batch_idx, last_idx]

        if self.pooling == "attention":
            scores = self.attn_proj(hidden_seq).squeeze(-1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = torch.softmax(scores, dim=1)
            attn = attn * mask_f
            attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return (hidden_seq * attn.unsqueeze(-1)).sum(dim=1)

        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def forward(self, x: torch.Tensor, dt: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros((batch_size, self.hidden_dim))
        current_t = x.new_zeros((batch_size,))

        hidden_steps = []
        event_logits = []

        for step_idx in range(seq_len):
            step_mask = mask[:, step_idx].float()
            h_propagated = self._solve_interval(h, dt[:, step_idx] * step_mask, current_t)

            x_embed = self.x_proj(x[:, step_idx, :])
            h_updated = self.gru(x_embed, h_propagated)
            h = torch.where(step_mask.unsqueeze(-1) > 0, h_updated, h)
            current_t = torch.where(step_mask > 0, t[:, step_idx], current_t)

            hidden_steps.append(h)
            event_logits.append(self.event_head(h).squeeze(-1))

        hidden_seq = torch.stack(hidden_steps, dim=1)
        event_logits = torch.stack(event_logits, dim=1).masked_fill(mask == 0, 0.0)
        pooled_hidden = self._masked_pool(hidden_seq, mask)
        seq_logits = self.seq_head(pooled_hidden).squeeze(-1)

        return {
            "event_logits": event_logits,
            "seq_logits": seq_logits,
        }


def compute_pos_weight(data_meta: Dict[str, object], prediction_level: str) -> torch.Tensor:
    if prediction_level == "sequence":
        pos = float(data_meta["n_train_attack"])
        neg = float(data_meta["n_train_total"]) - pos
    else:
        pos = float(data_meta["n_train_attack_events"])
        neg = float(data_meta["n_train_events"]) - pos

    if pos <= 0 or neg <= 0:
        return torch.tensor(1.0, device=DEVICE)
    return torch.tensor(neg / pos, device=DEVICE)


def batch_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, object],
    loss_fn,
    prediction_level: str,
) -> torch.Tensor:
    if prediction_level == "sequence":
        y = batch["seq_y"].to(DEVICE).float()
        return loss_fn(outputs["seq_logits"], y)

    y_seq = batch["y_seq"].to(DEVICE)
    valid = y_seq != EVENT_IGNORE_INDEX
    if not bool(valid.any().item()):
        return outputs["event_logits"].sum() * 0.0

    logits = outputs["event_logits"][valid]
    y = y_seq[valid].float()
    return loss_fn(logits, y)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    prediction_level: str,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_targets = 0

    for batch in loader:
        x = batch["x"].to(DEVICE)
        dt = batch["dt"].to(DEVICE)
        t = batch["t"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, dt, t, mask)
        loss = batch_loss(outputs=outputs, batch=batch, loss_fn=loss_fn, prediction_level=prediction_level)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        optimizer.step()

        if prediction_level == "sequence":
            n_targets = int(batch["seq_y"].shape[0])
        else:
            n_targets = int((batch["y_seq"] != EVENT_IGNORE_INDEX).sum().item())

        total_loss += float(loss.item()) * max(n_targets, 1)
        total_targets += n_targets

    return total_loss / max(total_targets, 1)


@torch.no_grad()
def eval_model(model: nn.Module, loader, loss_fn, prediction_level: str) -> Dict[str, object]:
    model.eval()
    all_y = []
    all_s = []
    all_days = []
    total_loss = 0.0
    total_targets = 0

    for batch in loader:
        x = batch["x"].to(DEVICE)
        dt = batch["dt"].to(DEVICE)
        t = batch["t"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        outputs = model(x, dt, t, mask)
        loss = batch_loss(outputs=outputs, batch=batch, loss_fn=loss_fn, prediction_level=prediction_level)

        if prediction_level == "sequence":
            scores = torch.sigmoid(outputs["seq_logits"]).detach().cpu().numpy()
            y = batch["seq_y"].detach().cpu().numpy().astype(np.int64)
            n_targets = int(len(y))
            all_days.extend(list(batch["day"]))
        else:
            y_seq = batch["y_seq"].to(DEVICE)
            valid = y_seq != EVENT_IGNORE_INDEX
            if bool(valid.any().item()):
                scores = torch.sigmoid(outputs["event_logits"][valid]).detach().cpu().numpy()
                y = y_seq[valid].detach().cpu().numpy().astype(np.int64)
                counts = valid.sum(dim=1).detach().cpu().numpy().astype(int)
                for day, count in zip(batch["day"], counts):
                    all_days.extend([day] * int(count))
                n_targets = int(y.shape[0])
            else:
                scores = np.empty((0,), dtype=np.float32)
                y = np.empty((0,), dtype=np.int64)
                n_targets = 0

        if y.size > 0:
            all_y.append(y)
            all_s.append(scores)

        total_loss += float(loss.item()) * max(n_targets, 1)
        total_targets += n_targets

    if not all_y:
        raise ValueError("Evaluation produced no targets. Check the split construction and prediction level.")

    y_true = np.concatenate(all_y, axis=0).astype(np.int64)
    y_score = np.concatenate(all_s, axis=0).astype(np.float64)

    return {
        "y_true": y_true,
        "y_score": y_score,
        "days": np.asarray(all_days, dtype=object),
        "loss": float(total_loss / max(total_targets, 1)),
    }


def build_epoch_row(epoch: int, lr: float, train_loss: float, eval_payload: Dict[str, object]) -> Dict[str, float]:
    y_true = eval_payload["y_true"]
    y_score = eval_payload["y_score"]

    threshold_free = threshold_free_metrics(y_true, y_score)
    tuned = find_best_threshold(y_true, y_score)
    at_05 = metrics_at_threshold(y_true, y_score, 0.5)

    return {
        "epoch": epoch,
        "lr": float(lr),
        "train_loss": float(train_loss),
        "val_loss": float(eval_payload["loss"]),
        "val_roc_auc": float(threshold_free["roc_auc"]),
        "val_pr_auc": float(threshold_free["pr_auc"]),
        "val_f1_at_0_5": float(at_05["f1"]),
        "val_precision_at_0_5": float(at_05["precision"]),
        "val_recall_at_0_5": float(at_05["recall"]),
        "val_tuned_threshold": float(tuned["threshold"]),
        "val_tuned_f1": float(tuned["f1"]),
        "val_tuned_precision": float(tuned["precision"]),
        "val_tuned_recall": float(tuned["recall"]),
    }


def evaluate_by_day(days: np.ndarray, y_true: np.ndarray, y_score: np.ndarray, tuned_threshold: float) -> Dict[str, Dict[str, object]]:
    frame = pd.DataFrame({"day": days, "y_true": y_true, "y_score": y_score})
    report = {}

    for day, part in frame.groupby("day", sort=True):
        day_y = part["y_true"].to_numpy(dtype=np.int64)
        day_s = part["y_score"].to_numpy(dtype=float)
        report[str(day)] = {
            "n_targets": int(len(part)),
            "n_attack": int((day_y == 1).sum()),
            "n_benign": int((day_y == 0).sum()),
            "threshold_free": threshold_free_metrics(day_y, day_s),
            "at_threshold_0.5": metrics_at_threshold(day_y, day_s, 0.5),
            "at_threshold_tuned": metrics_at_threshold(day_y, day_s, tuned_threshold),
        }

    return report


def model_variant_id(data_variant: str, args: argparse.Namespace) -> str:
    parts = [
        data_variant,
        args.prediction_level,
        args.pooling,
        args.solver,
        args.time_conditioning,
    ]
    if args.max_len is not None:
        parts.append(f"maxlen{args.max_len}")
    if args.ode_steps != 4:
        parts.append(f"steps{args.ode_steps}")
    if args.hidden_dim != 128 or args.ode_hidden != 128 or args.x_embed_dim != 128:
        parts.append(f"x{args.x_embed_dim}_h{args.hidden_dim}_ode{args.ode_hidden}")
    return "_".join(str(part) for part in parts)


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
        dt_clip=args.dt_clip,
        num_workers=args.num_workers,
        max_feature_rows=args.max_feature_rows,
        max_dt_samples=args.max_dt_samples,
    )
    coverage_issues = split_coverage_issues(data_meta=data_meta, prediction_level=args.prediction_level)
    if coverage_issues:
        raise ValueError(
            "Neural ODE split coverage checks failed for scientific evaluation:\n- " +
            "\n- ".join(coverage_issues)
        )

    feat_dim = int(data_meta["feature_count"])
    if feat_dim <= 0:
        raise ValueError("Feature count is zero. Check the Neural ODE preprocessing output.")

    pos_weight = compute_pos_weight(data_meta=data_meta, prediction_level=args.prediction_level)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ContinuousTimeGRUClassifier(
        feat_dim=feat_dim,
        x_embed_dim=args.x_embed_dim,
        hidden_dim=args.hidden_dim,
        ode_hidden=args.ode_hidden,
        ode_layers=args.ode_layers,
        ode_steps=args.ode_steps,
        solver=args.solver,
        time_conditioning=args.time_conditioning,
        pooling=args.pooling,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    sequence_build_config = data_meta.get("sequence_build_config", {})
    data_variant = sequence_build_config.get("variant_id", Path(data_meta["root"]).name)
    run_dir = make_run_dir(
        RUNS_DIR,
        model_name="neuralode",
        variant_id=model_variant_id(data_variant, args),
        seed=seed,
    )

    history = []
    best_state = None
    best_epoch = -1
    best_val_pr_auc = float("-inf")
    best_epoch_row = None
    stale_epochs = 0

    for epoch in range(1, args.max_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            prediction_level=args.prediction_level,
            grad_clip=args.grad_clip,
        )

        val_payload = eval_model(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            prediction_level=args.prediction_level,
        )
        epoch_row = build_epoch_row(epoch=epoch, lr=lr_now, train_loss=train_loss, eval_payload=val_payload)
        history.append(epoch_row)

        val_pr_auc_for_selection = safe_metric_for_selection(epoch_row["val_pr_auc"])
        scheduler.step(val_pr_auc_for_selection)

        print(
            f"seed={seed} epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={epoch_row['val_loss']:.4f} "
            f"val_pr_auc={epoch_row['val_pr_auc']:.4f} "
            f"val_f1@0.5={epoch_row['val_f1_at_0_5']:.4f} "
            f"val_f1@tuned={epoch_row['val_tuned_f1']:.4f}"
        )

        improved = best_state is None or val_pr_auc_for_selection > (best_val_pr_auc + args.early_stop_min_delta)
        if improved:
            best_val_pr_auc = val_pr_auc_for_selection
            best_epoch = epoch
            best_state = cpu_state_dict(model)
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

    val_payload = eval_model(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        prediction_level=args.prediction_level,
    )
    tuned_val_metrics = find_best_threshold(val_payload["y_true"], val_payload["y_score"])

    test_payload = eval_model(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        prediction_level=args.prediction_level,
    )

    val_metrics = summarize_final_metrics(
        y_true=val_payload["y_true"],
        y_score=val_payload["y_score"],
        loss=val_payload["loss"],
        tuned_threshold=tuned_val_metrics["threshold"],
    )
    test_metrics = summarize_final_metrics(
        y_true=test_payload["y_true"],
        y_score=test_payload["y_score"],
        loss=test_payload["loss"],
        tuned_threshold=tuned_val_metrics["threshold"],
    )

    val_metrics_by_day = evaluate_by_day(
        days=val_payload["days"],
        y_true=val_payload["y_true"],
        y_score=val_payload["y_score"],
        tuned_threshold=tuned_val_metrics["threshold"],
    )
    test_metrics_by_day = evaluate_by_day(
        days=test_payload["days"],
        y_true=test_payload["y_true"],
        y_score=test_payload["y_score"],
        tuned_threshold=tuned_val_metrics["threshold"],
    )

    run_config = {
        "model_name": "neuralode",
        "task_definition": TASK_DEFINITION,
        "task_framing": (
            "Time-stamped traffic-flow feature vectors are grouped into short entity-centric trajectories. "
            "A continuous-time GRU-ODE-style model evolves hidden states between events and predicts "
            f"{'sequence-level' if args.prediction_level == 'sequence' else 'event-level'} intrusion labels."
        ),
        "prediction_target": "seq_y" if args.prediction_level == "sequence" else "y_seq",
        "prediction_level": args.prediction_level,
        "seed": seed,
        "data_root": data_meta["root"],
        "batch_train": args.batch_train,
        "batch_eval": args.batch_eval,
        "num_workers": args.num_workers,
        "x_embed_dim": args.x_embed_dim,
        "hidden_dim": args.hidden_dim,
        "ode_hidden": args.ode_hidden,
        "ode_layers": args.ode_layers,
        "ode_steps": args.ode_steps,
        "solver": args.solver,
        "time_conditioning": args.time_conditioning,
        "pooling": args.pooling,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "grad_clip": args.grad_clip,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_min_lr": args.scheduler_min_lr,
        "max_len": data_meta["max_len"],
        "max_len_percentile": args.max_len_percentile,
        "max_len_cap": args.max_len_cap,
        "dt_clip": data_meta["dt_clip"],
        "feature_count": feat_dim,
        "feature_columns": data_meta["feature_columns"],
        "sequence_build_config": sequence_build_config,
        "split_summary": data_meta["split_summary"],
        "evaluation_protocol": {
            "threshold_free_score": "sigmoid_probability",
            "validation_tuned_threshold_source": "validation_split_labels",
            "reported_modes": ["threshold_0.5", "threshold_tuned"],
        },
        "split_coverage_checks": {
            "prediction_level": args.prediction_level,
            "issues": coverage_issues,
            "passed": len(coverage_issues) == 0,
        },
    }

    summary = {
        "best_epoch": best_epoch,
        "best_val_pr_auc": float(best_val_pr_auc),
        "best_val_row": best_epoch_row,
        "best_val_threshold": float(tuned_val_metrics["threshold"]),
        "best_val_f1_tuned": float(tuned_val_metrics["f1"]),
        "best_val_precision_tuned": float(tuned_val_metrics["precision"]),
        "best_val_recall_tuned": float(tuned_val_metrics["recall"]),
        "val_loss_at_best_checkpoint": float(val_payload["loss"]),
        "test_loss_at_best_checkpoint": float(test_payload["loss"]),
        "pos_weight": float(pos_weight.item()),
        "device": str(DEVICE),
        "feature_count": feat_dim,
        "n_train_total": int(data_meta["n_train_total"]),
        "n_val_total": int(data_meta["n_val_total"]),
        "n_test_total": int(data_meta["n_test_total"]),
        "n_train_events": int(data_meta["n_train_events"]),
        "n_val_events": int(data_meta["n_val_events"]),
        "n_test_events": int(data_meta["n_test_events"]),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_config": run_config,
            "summary": summary,
            "train_mean": data_meta["train_mean"],
            "train_std": data_meta["train_std"],
            "dt_clip": data_meta["dt_clip"],
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
