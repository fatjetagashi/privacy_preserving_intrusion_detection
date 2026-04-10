from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _3A_load_to_pytorch import build_loaders
from _training_common import (
    NodeMLPBinary,
    build_epoch_row,
    collect_predictions,
    compute_mean_std,
    compute_pos_weight,
    cpu_state_dict,
    find_best_threshold,
    make_run_dir,
    save_history_csv,
    save_json,
    set_seed,
    summarize_final_metrics,
    train_one_epoch,
)


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_DEFINITION = "Per-flow binary intrusion detection baseline without graph edges."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TASK_DEFINITION)
    parser.add_argument("--data-root", default=None, help="Root of a preprocessed GNN graph dataset.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def run_single_seed(args: argparse.Namespace, seed: int) -> Path:
    set_seed(seed)

    train_loader, val_loader, test_loader, data_meta = build_loaders(
        batch_size=args.batch_size,
        seed=seed,
        root=args.data_root,
    )

    feature_cols = data_meta["feature_cols"]
    graph_build_config = data_meta["graph_build_config"]
    variant_id = graph_build_config.get("variant_id", Path(data_meta["root"]).name)

    sample = next(iter(train_loader))
    num_features = int(sample.x.size(1))

    x_mean, x_std = compute_mean_std(train_loader, num_features=num_features)
    pos_weight = compute_pos_weight(train_loader).to(DEVICE)

    model = NodeMLPBinary(
        in_dim=num_features,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run_dir = make_run_dir(RUNS_DIR, model_name="mlp", variant_id=variant_id, seed=seed)

    history = []
    best_state = None
    best_epoch = -1
    best_val_pr_auc = -1.0
    best_epoch_row = None
    stale_epochs = 0

    for epoch in range(1, args.max_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            x_mean=x_mean,
            x_std=x_std,
            grad_clip=args.grad_clip,
        )

        val_y, val_prob, val_loss = collect_predictions(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            x_mean=x_mean,
            x_std=x_std,
        )

        epoch_row = build_epoch_row(
            epoch=epoch,
            train_loss=train_loss,
            lr=lr_now,
            y_true=val_y,
            y_prob=val_prob,
            val_loss=val_loss,
        )
        history.append(epoch_row)
        scheduler.step(epoch_row["val_pr_auc"])

        print(
            f"seed={seed} epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_pr_auc={epoch_row['val_pr_auc']:.4f} "
            f"val_f1@0.5={epoch_row['val_f1_at_0_5']:.4f} "
            f"val_f1@tuned={epoch_row['val_tuned_f1']:.4f}"
        )

        improved = epoch_row["val_pr_auc"] > (best_val_pr_auc + args.early_stop_min_delta)
        if improved:
            best_val_pr_auc = epoch_row["val_pr_auc"]
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

    val_y, val_prob, val_loss = collect_predictions(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        x_mean=x_mean,
        x_std=x_std,
    )
    tuned_val_metrics = find_best_threshold(val_y, val_prob)

    test_y, test_prob, test_loss = collect_predictions(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        x_mean=x_mean,
        x_std=x_std,
    )
    test_metrics = summarize_final_metrics(
        y_true=test_y,
        y_prob=test_prob,
        loss=test_loss,
        tuned_threshold=tuned_val_metrics["threshold"],
    )

    run_config = {
        "model_name": "mlp",
        "task_definition": TASK_DEFINITION,
        "task_framing": (
            "Each flow is still evaluated as a node label. The MLP ignores graph edges while "
            "keeping the same 5-minute graph-based split structure for a clean baseline."
        ),
        "prediction_target": "node_y",
        "graph_helper_label_usage": "split_only",
        "seed": seed,
        "data_root": data_meta["root"],
        "batch_size": args.batch_size,
        "hidden": args.hidden,
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
        "feature_columns": feature_cols,
        "graph_build_config": graph_build_config,
        "split_summary": data_meta["split_summary"],
    }

    summary = {
        "best_epoch": best_epoch,
        "best_val_pr_auc": float(best_val_pr_auc),
        "best_val_row": best_epoch_row,
        "best_val_threshold": float(tuned_val_metrics["threshold"]),
        "best_val_f1_tuned": float(tuned_val_metrics["f1"]),
        "best_val_precision_tuned": float(tuned_val_metrics["precision"]),
        "best_val_recall_tuned": float(tuned_val_metrics["recall"]),
        "val_loss_at_best_checkpoint": float(val_loss),
        "test_loss_at_best_checkpoint": float(test_loss),
        "pos_weight": float(pos_weight.item()),
        "device": str(DEVICE),
        "feature_count": len(feature_cols),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "run_config": run_config,
            "summary": summary,
        },
        run_dir / "best_model.pt",
    )

    save_history_csv(run_dir / "history.csv", history)
    save_json(run_dir / "run_config.json", run_config)
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "val_metrics.json", summarize_final_metrics(val_y, val_prob, val_loss, tuned_val_metrics["threshold"]))
    save_json(run_dir / "test_metrics.json", test_metrics)

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
