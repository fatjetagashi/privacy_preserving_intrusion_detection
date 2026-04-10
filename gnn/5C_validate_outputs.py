from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "traffic_labelled" / "2A_preprocessed_gnn"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate split integrity and split-balance artifacts.")
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def score_balance(summary_df: pd.DataFrame) -> dict[str, float]:
    overall = summary_df[summary_df["day"] == "ALL"].copy()
    scores = {}

    for metric in ["n_graphs", "n_nodes", "n_pos_nodes"]:
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

    graphs_meta = pd.read_parquet(data_root / "graphs_meta")
    split_summary = pd.read_csv(data_root / "split_summary.csv")
    split_comparison = pd.read_csv(data_root / "split_comparison.csv")

    graph_split_counts = graphs_meta.groupby("graph_id")["split"].nunique()
    max_split_count = int(graph_split_counts.max())
    leaking_graphs = int((graph_split_counts > 1).sum())

    strategy_scores = {}
    for strategy_label, strategy_df in split_comparison.groupby("summary_label", sort=True):
        strategy_scores[strategy_label] = score_balance(strategy_df)

    report = {
        "data_root": str(data_root),
        "graphs_total": int(len(graphs_meta)),
        "max_splits_per_graph_id": max_split_count,
        "leaking_graph_ids": leaking_graphs,
        "split_balance_error": strategy_scores,
        "split_summary_rows": split_summary.to_dict(orient="records"),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
