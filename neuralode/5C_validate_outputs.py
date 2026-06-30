from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "traffic_labelled" / "2C_preprocessed_neuralode"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Neural ODE split integrity and balance artifacts.")
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def score_balance(summary_df: pd.DataFrame) -> dict[str, float]:
    overall = summary_df[summary_df["day"] == "ALL"].copy()
    scores = {}

    for metric in ["n_groups", "n_sequences", "n_attack_sequences", "n_events", "n_attack_events"]:
        total = overall[metric].sum()
        if total <= 0:
            scores[metric] = 0.0
            continue

        abs_error = 0.0
        for split, ratio in SPLIT_RATIOS.items():
            split_value = float(overall.loc[overall["split"] == split, metric].sum())
            abs_error += abs((split_value / total) - ratio)
        scores[metric] = abs_error

    return scores


def validate_sequence_integrity(data_root: Path) -> dict[str, object]:
    sequence_root = data_root / "sequences"
    dataset = ds.dataset(sequence_root, format="parquet", partitioning="hive")
    available_columns = set(dataset.schema.names)
    required_columns = {
        "seq_id",
        "seq_y",
        "seq_len",
        "n_events",
        "n_attack_events",
        "y_seq",
    }
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        return {
            "checked": False,
            "missing_columns": missing_columns,
            "passed": False,
        }

    optional_columns = ["full_seq_len", "full_n_attack_events"]
    columns = sorted(required_columns | {col for col in optional_columns if col in available_columns})
    sequences = dataset.to_table(columns=columns).to_pandas()

    if sequences.empty:
        return {
            "checked": True,
            "n_sequences_checked": 0,
            "passed": False,
            "issue": "No stored sequences found.",
        }

    y_sums = sequences["y_seq"].map(lambda values: int(np.asarray(values, dtype=np.int64).sum()))
    y_max = sequences["y_seq"].map(lambda values: int(np.asarray(values, dtype=np.int64).max()) if len(values) else 0)

    report = {
        "checked": True,
        "n_sequences_checked": int(len(sequences)),
        "n_attack_events_gt_n_events": int((sequences["n_attack_events"] > sequences["n_events"]).sum()),
        "n_events_ne_seq_len": int((sequences["n_events"] != sequences["seq_len"]).sum()),
        "n_attack_events_ne_sum_y_seq": int((sequences["n_attack_events"] != y_sums).sum()),
        "seq_y_ne_max_y_seq": int((sequences["seq_y"] != y_max).sum()),
    }

    if "full_seq_len" in sequences.columns:
        report["full_seq_len_lt_seq_len"] = int((sequences["full_seq_len"] < sequences["seq_len"]).sum())
    if "full_n_attack_events" in sequences.columns:
        report["full_n_attack_events_lt_n_attack_events"] = int(
            (sequences["full_n_attack_events"] < sequences["n_attack_events"]).sum()
        )

    issue_counts = [value for key, value in report.items() if key.startswith("n_") and key != "n_sequences_checked"]
    issue_counts.extend(value for key, value in report.items() if key.endswith("_lt_seq_len") or key.endswith("_lt_n_attack_events"))
    report["passed"] = all(int(count) == 0 for count in issue_counts)
    return report


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()

    assignments = pd.read_csv(data_root / "sequence_assignments.csv")
    split_summary = pd.read_csv(data_root / "split_summary.csv")
    split_comparison = pd.read_csv(data_root / "split_comparison.csv")

    preprocessing_stats_path = data_root / "preprocessing_stats.json"
    preprocessing_stats = {}
    if preprocessing_stats_path.exists():
        with open(preprocessing_stats_path, "r", encoding="utf-8") as file_obj:
            preprocessing_stats = json.load(file_obj)

    group_split_counts = assignments.groupby("split_group_id")["split"].nunique()
    max_split_count = int(group_split_counts.max())
    leaking_groups = int((group_split_counts > 1).sum())

    strategy_scores = {}
    for strategy_label, strategy_df in split_comparison.groupby("summary_label", sort=True):
        strategy_scores[strategy_label] = score_balance(strategy_df)

    sequence_integrity = validate_sequence_integrity(data_root)

    split_class_coverage = {}
    overall_rows = split_summary[split_summary["day"] == "ALL"].copy()
    impossible_summary_rows = split_summary[split_summary["n_attack_events"] > split_summary["n_events"]]
    for split_name in sorted(overall_rows["split"].unique().tolist()):
        split_row = overall_rows[overall_rows["split"] == split_name].iloc[0]
        n_sequences = int(split_row["n_sequences"])
        n_attack_sequences = int(split_row["n_attack_sequences"])
        n_benign_sequences = int(n_sequences - n_attack_sequences)
        n_events = int(split_row["n_events"])
        n_attack_events = int(split_row["n_attack_events"])
        n_benign_events = int(n_events - n_attack_events)
        split_class_coverage[split_name] = {
            "n_sequences": n_sequences,
            "n_attack_sequences": n_attack_sequences,
            "n_benign_sequences": n_benign_sequences,
            "n_events": n_events,
            "n_attack_events": n_attack_events,
            "n_benign_events": n_benign_events,
            "sequence_has_both_classes": bool(n_attack_sequences > 0 and n_benign_sequences > 0),
            "event_has_both_classes": bool(n_attack_events > 0 and n_benign_events > 0),
        }

    report = {
        "data_root": str(data_root),
        "groups_total": int(assignments["split_group_id"].nunique()),
        "max_splits_per_group_id": max_split_count,
        "leaking_group_ids": leaking_groups,
        "split_balance_error": strategy_scores,
        "sequence_integrity_checks": sequence_integrity,
        "split_summary_integrity_checks": {
            "n_attack_events_gt_n_events_rows": int(len(impossible_summary_rows)),
            "passed": bool(impossible_summary_rows.empty),
        },
        "split_class_coverage": split_class_coverage,
        "invalid_timestamp_drop_detected": bool(preprocessing_stats.get("rows_dropped_invalid_timestamp", 0) > 0),
        "preprocessing_stats": preprocessing_stats,
        "split_summary_rows": split_summary.to_dict(orient="records"),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
