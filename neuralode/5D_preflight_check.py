from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _3C_load_to_pytorch_neuralode import build_loaders
from _training_common import save_json

DEFAULT_REPORTS_DIR = THIS_DIR / "reports"
DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight preflight checks for the Neural ODE pipeline.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-len-percentile", type=float, default=95.0)
    parser.add_argument("--max-len-cap", type=int, default=512)
    parser.add_argument("--dt-clip", type=float, default=None)
    parser.add_argument("--inspect-batches", type=int, default=4)
    return parser.parse_args()


def load_train_module():
    module_path = THIS_DIR / "4C_train_neuralode.py"
    spec = importlib.util.spec_from_file_location("neuralode_train_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sampled_time_stats(loader, inspect_batches: int) -> dict[str, float]:
    total_positions = 0
    total_valid = 0
    total_positive_dt = 0
    dt_values = []
    batch_count = 0

    for batch in loader:
        mask = batch["mask"]
        dt = batch["dt"]
        total_positions += int(mask.numel())
        total_valid += int(mask.sum().item())
        valid_dt = dt[mask == 1]
        positive_dt = valid_dt[valid_dt > 0]
        total_positive_dt += int(positive_dt.numel())
        if positive_dt.numel() > 0:
            dt_values.append(positive_dt.numpy())
        batch_count += 1
        if batch_count >= inspect_batches:
            break

    if dt_values:
        merged = np.concatenate(dt_values, axis=0)
        p95 = float(np.percentile(merged, 95))
        p99 = float(np.percentile(merged, 99))
    else:
        p95 = 0.0
        p99 = 0.0

    return {
        "sampled_batches": int(batch_count),
        "sampled_positions": int(total_positions),
        "sampled_valid_positions": int(total_valid),
        "sampled_positive_dt_positions": int(total_positive_dt),
        "dt_p95": p95,
        "dt_p99": p99,
    }


def main() -> None:
    args = parse_args()
    train_module = load_train_module()

    train_loader, val_loader, test_loader, data_meta = build_loaders(
        batch_train=args.batch_size,
        batch_eval=args.batch_size,
        seed=0,
        root=args.data_root,
        max_len=args.max_len,
        max_len_percentile=args.max_len_percentile,
        max_len_cap=args.max_len_cap,
        dt_clip=args.dt_clip,
        num_workers=args.num_workers,
    )

    model = train_module.ContinuousTimeGRUClassifier(
        feat_dim=int(data_meta["feature_count"]),
        x_embed_dim=args.x_embed_dim,
        hidden_dim=args.hidden_dim,
        ode_hidden=args.ode_hidden,
        ode_layers=args.ode_layers,
        ode_steps=args.ode_steps,
        solver=args.solver,
        time_conditioning=args.time_conditioning,
        pooling=args.pooling,
        dropout=args.dropout,
    ).to(train_module.DEVICE)
    model.eval()

    pos_weight = train_module.compute_pos_weight(data_meta=data_meta, prediction_level=args.prediction_level)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_batch = next(iter(train_loader))
    x = train_batch["x"].to(train_module.DEVICE)
    dt = train_batch["dt"].to(train_module.DEVICE)
    t = train_batch["t"].to(train_module.DEVICE)
    mask = train_batch["mask"].to(train_module.DEVICE)

    with torch.no_grad():
        outputs = model(x, dt, t, mask)
        loss = train_module.batch_loss(
            outputs=outputs,
            batch=train_batch,
            loss_fn=loss_fn,
            prediction_level=args.prediction_level,
        )

    sequence_build_config = data_meta.get("sequence_build_config", {})
    variant_id = sequence_build_config.get("variant_id", Path(data_meta["root"]).name)
    coverage_issues = train_module.split_coverage_issues(
        data_meta=data_meta,
        prediction_level=args.prediction_level,
    )
    report_path = DEFAULT_REPORTS_DIR / (
        f"preflight_{variant_id}_{args.prediction_level}_{args.pooling}_{args.solver}_{args.time_conditioning}_"
        f"maxlen{args.max_len if args.max_len is not None else 'auto'}_"
        f"hid{args.hidden_dim}_ode{args.ode_hidden}_steps{args.ode_steps}.json"
    )

    report = {
        "data_root": data_meta["root"],
        "variant_id": variant_id,
        "sequence_build_config": sequence_build_config,
        "counts": {
            "n_train_total": int(data_meta["n_train_total"]),
            "n_train_benign": int(data_meta["n_train_benign"]),
            "n_train_attack": int(data_meta["n_train_attack"]),
            "n_val_total": int(data_meta["n_val_total"]),
            "n_val_benign": int(data_meta["n_val_benign"]),
            "n_val_attack": int(data_meta["n_val_attack"]),
            "n_test_total": int(data_meta["n_test_total"]),
            "n_test_benign": int(data_meta["n_test_benign"]),
            "n_test_attack": int(data_meta["n_test_attack"]),
            "n_train_events": int(data_meta["n_train_events"]),
            "n_train_attack_events": int(data_meta["n_train_attack_events"]),
            "n_val_events": int(data_meta["n_val_events"]),
            "n_val_attack_events": int(data_meta["n_val_attack_events"]),
            "n_test_events": int(data_meta["n_test_events"]),
            "n_test_attack_events": int(data_meta["n_test_attack_events"]),
        },
        "class_coverage": {
            "val_has_both_sequence_classes": bool(data_meta["n_val_benign"] > 0 and data_meta["n_val_attack"] > 0),
            "test_has_both_sequence_classes": bool(data_meta["n_test_benign"] > 0 and data_meta["n_test_attack"] > 0),
            "val_has_both_event_classes": bool(
                (data_meta["n_val_events"] - data_meta["n_val_attack_events"]) > 0 and
                data_meta["n_val_attack_events"] > 0
            ),
            "test_has_both_event_classes": bool(
                (data_meta["n_test_events"] - data_meta["n_test_attack_events"]) > 0 and
                data_meta["n_test_attack_events"] > 0
            ),
        },
        "split_coverage_checks": {
            "prediction_level": args.prediction_level,
            "issues": coverage_issues,
            "ready_for_scientific_training": len(coverage_issues) == 0,
        },
        "trajectory_config": {
            "feature_count": int(data_meta["feature_count"]),
            "max_len": int(data_meta["max_len"]),
            "dt_clip": float(data_meta["dt_clip"]),
        },
        "batch_shapes": {
            "x": list(train_batch["x"].shape),
            "dt": list(train_batch["dt"].shape),
            "t": list(train_batch["t"].shape),
            "mask": list(train_batch["mask"].shape),
            "seq_y": list(train_batch["seq_y"].shape),
            "y_seq": list(train_batch["y_seq"].shape),
            "event_logits": list(outputs["event_logits"].shape),
            "seq_logits": list(outputs["seq_logits"].shape),
        },
        "forward_check": {
            "prediction_level": args.prediction_level,
            "loss": float(loss.item()),
            "loss_is_finite": bool(torch.isfinite(loss).item()),
            "event_logits_finite": bool(torch.isfinite(outputs["event_logits"]).all().item()),
            "seq_logits_finite": bool(torch.isfinite(outputs["seq_logits"]).all().item()),
        },
        "sampled_time_stats": {
            "train": sampled_time_stats(train_loader, inspect_batches=args.inspect_batches),
            "val": sampled_time_stats(val_loader, inspect_batches=args.inspect_batches),
            "test": sampled_time_stats(test_loader, inspect_batches=args.inspect_batches),
        },
    }

    save_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"Saved preflight report to: {report_path}")


if __name__ == "__main__":
    main()
