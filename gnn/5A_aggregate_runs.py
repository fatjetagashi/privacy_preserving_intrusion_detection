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
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "specificity",
    "fpr",
    "selected_threshold",
]


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def flatten_run(model_name: str, run_dir: Path) -> List[Dict[str, object]]:
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "test_metrics.json"
    summary_path = run_dir / "summary.json"
    if not (config_path.exists() and metrics_path.exists() and summary_path.exists()):
        return []

    run_config = load_json(config_path)
    test_metrics = load_json(metrics_path)
    summary = load_json(summary_path)

    if "threshold_free" not in test_metrics or "at_threshold_tuned" not in test_metrics:
        return []

    graph_build_config = run_config.get("graph_build_config", {})
    variant_id = graph_build_config.get("variant_id", Path(run_config.get("data_root", run_dir.name)).name)
    threshold_free = test_metrics["threshold_free"]

    rows = []
    for evaluation_mode, metrics_key in [
        ("threshold_0_5", "at_threshold_0.5"),
        ("threshold_tuned", "at_threshold_tuned"),
    ]:
        threshold_metrics = test_metrics[metrics_key]
        rows.append(
            {
                "model_name": model_name,
                "run_name": run_dir.name,
                "seed": run_config.get("seed"),
                "data_root": run_config.get("data_root"),
                "variant_id": variant_id,
                "edge_family": graph_build_config.get("edge_family"),
                "chain_k": graph_build_config.get("chain_k"),
                "evaluation_mode": evaluation_mode,
                "roc_auc": threshold_free.get("roc_auc"),
                "pr_auc": threshold_free.get("pr_auc"),
                "f1": threshold_metrics.get("f1"),
                "precision": threshold_metrics.get("precision"),
                "recall": threshold_metrics.get("recall"),
                "balanced_accuracy": threshold_metrics.get("balanced_accuracy"),
                "specificity": threshold_metrics.get("specificity"),
                "fpr": threshold_metrics.get("fpr"),
                "selected_threshold": threshold_metrics.get("threshold"),
                "best_epoch": summary.get("best_epoch"),
                "best_val_pr_auc": summary.get("best_val_pr_auc"),
            }
        )

    return rows


def aggregate_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    group_cols = ["model_name", "variant_id", "edge_family", "chain_k", "evaluation_mode", "data_root"]
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

    dedupe_cols = ["model_name", "variant_id", "seed", "evaluation_mode"]
    present_cols = [col for col in dedupe_cols if col in raw_df.columns]
    if len(present_cols) != len(dedupe_cols):
        return raw_df

    # Run names begin with YYYYMMDD_HHMMSS, so lexical order matches chronology.
    return (
        raw_df.sort_values(["model_name", "variant_id", "seed", "evaluation_mode", "run_name"], kind="stable")
        .drop_duplicates(subset=dedupe_cols, keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    raw_rows = []
    for model_name in ["gnn", "mlp"]:
        model_dir = RUNS_DIR / model_name
        if not model_dir.exists():
            continue

        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            raw_rows.extend(flatten_run(model_name=model_name, run_dir=run_dir))

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
