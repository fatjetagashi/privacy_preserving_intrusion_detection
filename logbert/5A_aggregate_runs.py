from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"
REPORTS_DIR = THIS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

METRIC_COLUMNS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "specificity",
    "fpr",
    "fnr",
    "mcc",
    "selected_sequence_threshold",
    "selected_distance_threshold",
]


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def flatten_run(run_dir: Path) -> List[Dict[str, object]]:
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "test_metrics.json"
    summary_path = run_dir / "summary.json"
    if not (config_path.exists() and metrics_path.exists() and summary_path.exists()):
        return []

    run_config = load_json(config_path)
    test_metrics = load_json(metrics_path)
    summary = load_json(summary_path)

    if "threshold_free" not in test_metrics or "at_threshold_val_tuned" not in test_metrics:
        return []

    seq_build_config = run_config.get("sequence_build_config", {})
    variant_id = seq_build_config.get("variant_id", Path(run_config.get("data_root", run_dir.name)).name)
    threshold_free = test_metrics["threshold_free"]

    rows = []
    for evaluation_mode, metrics_key in [
        ("train_benign_percentile", "at_threshold_train_benign_percentile"),
        ("validation_tuned", "at_threshold_val_tuned"),
    ]:
        threshold_metrics = test_metrics[metrics_key]
        rows.append(
            {
                "model_name": "logbert",
                "run_name": run_dir.name,
                "seed": run_config.get("seed"),
                "data_root": run_config.get("data_root"),
                "variant_id": variant_id,
                "entity_mode": seq_build_config.get("entity_mode"),
                "split_strategy": seq_build_config.get("split_strategy"),
                "window_seconds": seq_build_config.get("window_seconds"),
                "min_len": seq_build_config.get("min_len"),
                "max_len_build": seq_build_config.get("max_len_build"),
                "max_len": run_config.get("max_len"),
                "d_model": run_config.get("d_model"),
                "n_heads": run_config.get("n_heads"),
                "n_layers": run_config.get("n_layers"),
                "dropout": run_config.get("dropout"),
                "lr": run_config.get("lr"),
                "weight_decay": run_config.get("weight_decay"),
                "mask_prob_train": run_config.get("mask_prob_train"),
                "mask_prob_score": run_config.get("mask_prob_score"),
                "score_passes": run_config.get("score_passes"),
                "num_candidates": run_config.get("num_candidates"),
                "benign_percentile": run_config.get("benign_percentile"),
                "vhm_weight": run_config.get("vhm_weight"),
                "evaluation_mode": evaluation_mode,
                "roc_auc": threshold_free.get("roc_auc"),
                "pr_auc": threshold_free.get("pr_auc"),
                "accuracy": threshold_metrics.get("accuracy"),
                "f1": threshold_metrics.get("f1"),
                "precision": threshold_metrics.get("precision"),
                "recall": threshold_metrics.get("recall"),
                "balanced_accuracy": threshold_metrics.get("balanced_accuracy"),
                "specificity": threshold_metrics.get("specificity"),
                "fpr": threshold_metrics.get("fpr"),
                "fnr": threshold_metrics.get("fnr"),
                "mcc": threshold_metrics.get("mcc"),
                "selected_sequence_threshold": threshold_metrics.get("sequence_threshold"),
                "selected_distance_threshold": threshold_metrics.get("distance_threshold"),
                "best_epoch": summary.get("best_epoch"),
                "best_val_pr_auc": summary.get("best_val_pr_auc"),
            }
        )

    return rows


def aggregate_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    group_cols = [
        "model_name",
        "variant_id",
        "entity_mode",
        "split_strategy",
        "window_seconds",
        "min_len",
        "max_len_build",
        "max_len",
        "d_model",
        "n_heads",
        "n_layers",
        "dropout",
        "lr",
        "weight_decay",
        "mask_prob_train",
        "mask_prob_score",
        "score_passes",
        "num_candidates",
        "benign_percentile",
        "vhm_weight",
        "evaluation_mode",
        "data_root",
    ]
    grouped = raw_df.groupby(group_cols, dropna=False, sort=True)

    rows = []
    for group_key, group_df in grouped:
        record = {col_name: value for col_name, value in zip(group_cols, group_key)}
        record["n_runs"] = int(len(group_df))

        for metric in METRIC_COLUMNS:
            metric_values = group_df[metric].dropna().astype(float)
            if metric_values.empty:
                record[f"{metric}_mean"] = None
                record[f"{metric}_std"] = None
                record[f"{metric}_min"] = None
                record[f"{metric}_max"] = None
                continue

            record[f"{metric}_mean"] = float(metric_values.mean())
            record[f"{metric}_std"] = float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
            record[f"{metric}_min"] = float(metric_values.min())
            record[f"{metric}_max"] = float(metric_values.max())

        rows.append(record)

    return pd.DataFrame(rows).sort_values(group_cols, kind="stable").reset_index(drop=True)


def deduplicate_latest_runs(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    dedupe_cols = [
        "model_name",
        "variant_id",
        "seed",
        "evaluation_mode",
        "data_root",
        "max_len",
        "d_model",
        "n_heads",
        "n_layers",
        "dropout",
        "lr",
        "weight_decay",
        "mask_prob_train",
        "mask_prob_score",
        "score_passes",
        "num_candidates",
        "benign_percentile",
        "vhm_weight",
    ]
    return (
        raw_df.sort_values(["model_name", "variant_id", "seed", "evaluation_mode", "run_name"], kind="stable")
        .drop_duplicates(subset=dedupe_cols, keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    raw_rows = []
    model_dir = RUNS_DIR / "logbert"
    if model_dir.exists():
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            raw_rows.extend(flatten_run(run_dir=run_dir))

    raw_df = pd.DataFrame(raw_rows)
    raw_df = deduplicate_latest_runs(raw_df)
    agg_df = aggregate_metrics(raw_df)

    raw_csv = REPORTS_DIR / "raw_run_metrics.csv"
    agg_csv = REPORTS_DIR / "aggregated_metrics.csv"
    raw_json = REPORTS_DIR / "raw_run_metrics.json"
    agg_json = REPORTS_DIR / "aggregated_metrics.json"

    raw_df.to_csv(raw_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    raw_df.to_json(raw_json, orient="records", indent=2)
    agg_df.to_json(agg_json, orient="records", indent=2)

    print(f"Saved raw run metrics to: {raw_csv}")
    print(f"Saved aggregated metrics to: {agg_csv}")


if __name__ == "__main__":
    main()
