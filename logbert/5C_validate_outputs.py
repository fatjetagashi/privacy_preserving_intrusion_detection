from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "traffic_labelled" / "2B_preprocessed_logbert"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LogBERT split integrity and balance artifacts.")
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def score_balance(summary_df: pd.DataFrame) -> dict[str, float]:
    overall = summary_df[summary_df["day"] == "ALL"].copy()
    scores = {}

    for metric in ["n_groups", "n_sequences", "n_attack_sequences", "n_tokens"]:
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


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()

    assignments = pd.read_csv(data_root / "sequence_assignments.csv")
    split_summary = pd.read_csv(data_root / "split_summary.csv")
    split_comparison = pd.read_csv(data_root / "split_comparison.csv")

    group_split_counts = assignments.groupby("split_group_id")["split"].nunique()
    max_split_count = int(group_split_counts.max())
    leaking_groups = int((group_split_counts > 1).sum())

    strategy_scores = {}
    for strategy_label, strategy_df in split_comparison.groupby("summary_label", sort=True):
        strategy_scores[strategy_label] = score_balance(strategy_df)

    split_class_coverage = {}
    overall_rows = split_summary[split_summary["day"] == "ALL"].copy()
    for split_name in sorted(overall_rows["split"].unique().tolist()):
        split_row = overall_rows[overall_rows["split"] == split_name].iloc[0]
        n_attack = int(split_row["n_attack_sequences"])
        n_sequences = int(split_row["n_sequences"])
        n_benign = int(n_sequences - n_attack)
        split_class_coverage[split_name] = {
            "n_sequences": n_sequences,
            "n_attack_sequences": n_attack,
            "n_benign_sequences": n_benign,
            "has_both_classes": bool(n_attack > 0 and n_benign > 0),
        }

    report = {
        "data_root": str(data_root),
        "groups_total": int(assignments["split_group_id"].nunique()),
        "max_splits_per_group_id": max_split_count,
        "leaking_group_ids": leaking_groups,
        "split_balance_error": strategy_scores,
        "split_class_coverage": split_class_coverage,
        "split_summary_rows": split_summary.to_dict(orient="records"),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
