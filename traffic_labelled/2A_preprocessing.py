from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window, functions as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.schema import CIC_IDS_2017_T_FULL_CLEAN_SCHEMA


INPUT_DIR_PATH = REPO_ROOT / "data" / "traffic_labelled" / "1B_clean_cic_ids_t_2017"
BASE_OUTPUT_DIR = REPO_ROOT / "data" / "traffic_labelled"

WINDOW_SECONDS = 300
DEFAULT_EDGE_FAMILY = "src+dst+svc"
DEFAULT_CHAIN_K = 5
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_STRATEGY = "deterministic_day_graphy_nodes_pos_balanced_v1"
LEGACY_SPLIT_STRATEGY = "legacy_random_percent_rank_by_day_graph_y"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ["train", "val", "test"]

EDGE_FAMILY_GROUPS = {
    "src_only": [
        ("src_ip_chain", ["graph_id", "Source IP"]),
    ],
    "src+dst": [
        ("src_ip_chain", ["graph_id", "Source IP"]),
        ("dst_ip_chain", ["graph_id", "Destination IP"]),
    ],
    "src+dst+svc": [
        ("src_ip_chain", ["graph_id", "Source IP"]),
        ("dst_ip_chain", ["graph_id", "Destination IP"]),
        ("service_chain", ["graph_id", "Destination Port", "Protocol"]),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 5-minute flow graphs for per-flow binary intrusion detection. "
            "Graph-level attack presence is used only for split construction."
        )
    )
    parser.add_argument(
        "--edge-family",
        choices=sorted(EDGE_FAMILY_GROUPS),
        default=DEFAULT_EDGE_FAMILY,
        help="Which edge families to include when building graph connectivity.",
    )
    parser.add_argument(
        "--chain-k",
        type=int,
        choices=[1, 5, 10],
        default=DEFAULT_CHAIN_K,
        help="How many forward neighbors to connect within each edge family.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional explicit output directory for the generated graph dataset.",
    )
    return parser.parse_args()


def sanitize_variant_id(edge_family: str, chain_k: int) -> str:
    safe_family = edge_family.replace("+", "_plus_")
    return f"{safe_family}_k{chain_k}"


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()

    if args.edge_family == DEFAULT_EDGE_FAMILY and args.chain_k == DEFAULT_CHAIN_K:
        return BASE_OUTPUT_DIR / "2A_preprocessed_gnn"

    variant_id = sanitize_variant_id(args.edge_family, args.chain_k)
    return BASE_OUTPUT_DIR / "2A_preprocessed_gnn_variants" / variant_id


def chain_edges(base_df, group_cols: List[str], k: int):
    window = Window.partitionBy(*group_cols).orderBy("Timestamp", "node_id")
    out = None
    for hop in range(1, k + 1):
        tmp = (
            base_df.withColumn("dst", F.lead("node_id", hop).over(window))
            .where(F.col("dst").isNotNull())
            .select("graph_id", F.col("node_id").alias("src"), "dst")
        )
        out = tmp if out is None else out.unionByName(tmp)
    return out


def build_edges(df, edge_family: str, chain_k: int):
    edge_parts = []
    for _, group_cols in EDGE_FAMILY_GROUPS[edge_family]:
        edge_parts.append(chain_edges(df, group_cols=group_cols, k=chain_k))

    edges = edge_parts[0]
    for part in edge_parts[1:]:
        edges = edges.unionByName(part)

    edges_rev = edges.select("graph_id", F.col("dst").alias("src"), F.col("src").alias("dst"))
    return edges.unionByName(edges_rev).dropDuplicates(["graph_id", "src", "dst"])


def build_legacy_split(graphs_pdf: pd.DataFrame, seed: int) -> pd.DataFrame:
    out_parts = []
    rng = np.random.default_rng(seed)
    for (day, graph_y), chunk in graphs_pdf.groupby(["day", "graph_y"], sort=True):
        part = chunk.copy()
        part["legacy_rand"] = rng.random(len(part))
        part = part.sort_values(["legacy_rand", "graph_id"], kind="stable").reset_index(drop=True)

        if len(part) == 1:
            percent_rank = [0.0]
        else:
            percent_rank = [idx / float(len(part) - 1) for idx in range(len(part))]

        part["legacy_split"] = [
            "train" if p < SPLIT_RATIOS["train"] else "val" if p < (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]) else "test"
            for p in percent_rank
        ]
        out_parts.append(part)

    return pd.concat(out_parts, ignore_index=True).drop(columns=["legacy_rand"])


def _metric_deficit(target: float, current: float, delta: float) -> float:
    denom = max(target, 1.0)
    return max(target - current, 0.0) / denom - max((current + delta) - target, 0.0) / denom


def choose_split(
    row: Dict[str, object],
    current: Dict[str, Dict[str, object]],
    targets: Dict[str, Dict[str, float]],
    class_targets: Dict[str, Dict[int, float]],
) -> str:
    best_key = None
    best_split = None

    for split_rank, split in enumerate(SPLIT_ORDER):
        state = current[split]
        graph_y = int(row["graph_y"])

        score_graph_class = _metric_deficit(
            class_targets[split].get(graph_y, 0.0),
            float(state["graphs_by_y"].get(graph_y, 0)),
            1.0,
        )
        score_graph_total = _metric_deficit(targets[split]["graphs"], float(state["graphs_total"]), 1.0)
        score_nodes = _metric_deficit(targets[split]["nodes"], float(state["nodes"]), float(row["n_nodes"]))
        score_pos = _metric_deficit(targets[split]["pos_nodes"], float(state["pos_nodes"]), float(row["n_pos_nodes"]))

        key = (
            round(4.0 * score_graph_class + 2.5 * score_graph_total + 2.0 * score_nodes + 2.0 * score_pos, 12),
            round(score_graph_class, 12),
            round(score_nodes, 12),
            round(score_pos, 12),
            -float(state["nodes"]),
            -float(state["graphs_total"]),
            -split_rank,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_split = split

    return best_split


def assign_balanced_splits(graphs_pdf: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    assignments = []

    for day, day_chunk in graphs_pdf.groupby("day", sort=True):
        day_df = day_chunk.copy()
        total_graphs = float(len(day_df))
        total_nodes = float(day_df["n_nodes"].sum())
        total_pos = float(day_df["n_pos_nodes"].sum())

        targets = {
            split: {
                "graphs": total_graphs * ratio,
                "nodes": total_nodes * ratio,
                "pos_nodes": total_pos * ratio,
            }
            for split, ratio in SPLIT_RATIOS.items()
        }

        class_targets = {
            split: {
                int(graph_y): float(len(class_chunk)) * ratio
                for graph_y, class_chunk in day_df.groupby("graph_y", sort=True)
            }
            for split, ratio in SPLIT_RATIOS.items()
        }

        current = {
            split: {
                "graphs_total": 0,
                "graphs_by_y": {0: 0, 1: 0},
                "nodes": 0.0,
                "pos_nodes": 0.0,
            }
            for split in SPLIT_ORDER
        }

        ordered_rows = (
            day_df.sort_values(
                ["graph_y", "n_pos_nodes", "n_nodes", "attack_ratio", "graph_id"],
                ascending=[False, False, False, False, True],
                kind="stable",
            )
            .to_dict("records")
        )

        for row in ordered_rows:
            split = choose_split(row=row, current=current, targets=targets, class_targets=class_targets)
            graph_y = int(row["graph_y"])

            current[split]["graphs_total"] += 1
            current[split]["graphs_by_y"][graph_y] = current[split]["graphs_by_y"].get(graph_y, 0) + 1
            current[split]["nodes"] += float(row["n_nodes"])
            current[split]["pos_nodes"] += float(row["n_pos_nodes"])

            assignments.append(
                {
                    "graph_id": row["graph_id"],
                    "split": split,
                    "split_strategy": DEFAULT_SPLIT_STRATEGY,
                    "split_seed": split_seed,
                }
            )

    assignments_pdf = pd.DataFrame(assignments)
    return graphs_pdf.merge(assignments_pdf, on="graph_id", how="inner")


def build_split_summary(graphs_pdf: pd.DataFrame, split_col: str, strategy_label: str) -> pd.DataFrame:
    rows = []

    for day, chunk in graphs_pdf.groupby(["day", split_col], sort=True):
        day_name, split = day
        rows.append(
            {
                "summary_label": strategy_label,
                "split": split,
                "day": day_name,
                "n_graphs": int(len(chunk)),
                "n_attack_graphs": int((chunk["graph_y"] == 1).sum()),
                "n_benign_graphs": int((chunk["graph_y"] == 0).sum()),
                "n_nodes": int(chunk["n_nodes"].sum()),
                "n_pos_nodes": int(chunk["n_pos_nodes"].sum()),
                "positive_node_rate": float(chunk["n_pos_nodes"].sum() / max(chunk["n_nodes"].sum(), 1)),
            }
        )

    for split, chunk in graphs_pdf.groupby(split_col, sort=True):
        rows.append(
            {
                "summary_label": strategy_label,
                "split": split,
                "day": "ALL",
                "n_graphs": int(len(chunk)),
                "n_attack_graphs": int((chunk["graph_y"] == 1).sum()),
                "n_benign_graphs": int((chunk["graph_y"] == 0).sum()),
                "n_nodes": int(chunk["n_nodes"].sum()),
                "n_pos_nodes": int(chunk["n_pos_nodes"].sum()),
                "positive_node_rate": float(chunk["n_pos_nodes"].sum() / max(chunk["n_nodes"].sum(), 1)),
            }
        )

    return pd.DataFrame(rows).sort_values(["day", "split"], kind="stable").reset_index(drop=True)


def save_dataframe_artifacts(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)


def expected_graph_build_config(output_dir_path: Path, edge_family: str, chain_k: int, feature_cols: list[str] | None = None) -> Dict[str, object]:
    return {
        "task_definition": "Per-flow binary intrusion detection with 5-minute graphs as relational context.",
        "prediction_target": "node_y",
        "graph_role": "context_only",
        "graph_helper_label": "graph_y",
        "graph_helper_label_usage": "split_construction_and_balance_only",
        "dataset_source": "Traffic Labelled / GeneratedLabelledFlows",
        "window_seconds": WINDOW_SECONDS,
        "undirected": True,
        "edge_family": edge_family,
        "edge_groups": [name for name, _ in EDGE_FAMILY_GROUPS[edge_family]],
        "chain_k": chain_k,
        "variant_id": sanitize_variant_id(edge_family, chain_k),
        "output_root": str(output_dir_path),
        "default_variant": edge_family == DEFAULT_EDGE_FAMILY and chain_k == DEFAULT_CHAIN_K,
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "split_seed": DEFAULT_SPLIT_SEED,
        "legacy_split_strategy": LEGACY_SPLIT_STRATEGY,
        "feature_columns": feature_cols if feature_cols is not None else None,
    }


def can_reuse_existing_output(output_dir_path: Path, edge_family: str, chain_k: int) -> bool:
    config_path = output_dir_path / "graph_build_config.json"
    required_paths = [
        output_dir_path / "nodes",
        output_dir_path / "edges",
        output_dir_path / "graphs_meta",
        output_dir_path / "split_summary.csv",
        output_dir_path / "split_comparison.csv",
        output_dir_path / "graph_assignments.csv",
    ]

    if not config_path.exists() or not all(path.exists() for path in required_paths):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as file_obj:
            existing_config = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return False

    expected_config = expected_graph_build_config(output_dir_path=output_dir_path, edge_family=edge_family, chain_k=chain_k)
    for key, expected_value in expected_config.items():
        if key == "feature_columns":
            continue
        if existing_config.get(key) != expected_value:
            return False

    return True


def main() -> None:
    args = parse_args()
    output_dir_path = resolve_output_root(args)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if can_reuse_existing_output(output_dir_path=output_dir_path, edge_family=args.edge_family, chain_k=args.chain_k):
        print(f"Reusing existing preprocessed graph dataset at: {output_dir_path}")
        return

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.appName("CIC_IDS_T_2017 PREPROCESSING 2A")
        .master("local[4]")
        .config("spark.driver.memory", "12g")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    input_path = INPUT_DIR_PATH / "1B_clean_cic_ids_t_2017.csv"

    df = (
        spark.read.format("csv")
        .schema(CIC_IDS_2017_T_FULL_CLEAN_SCHEMA)
        .option("header", True)
        .option("sep", ",")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .load(str(input_path))
    )

    # Each row is a flow/node. Five-minute graphs are only used as relational context.
    df = df.select([F.col(col_name).alias(col_name.strip()) for col_name in df.columns])
    df = df.withColumn("Timestamp", F.date_format(F.col("Timestamp"), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("y", F.when(F.col("Label") == "BENIGN", F.lit(0)).otherwise(F.lit(1)))
    df = df.withColumn("window_id", F.floor(F.unix_timestamp("Timestamp") / F.lit(WINDOW_SECONDS)))
    df = df.withColumn("graph_id", F.concat_ws("_", F.col("day"), F.col("window_id")))

    window = Window.partitionBy("graph_id").orderBy(F.col("Timestamp"), F.col("Flow ID"))
    df = df.withColumn("node_id", F.row_number().over(window) - 1)

    edges = build_edges(df=df, edge_family=args.edge_family, chain_k=args.chain_k)

    non_features = {
        "graph_id",
        "day",
        "time_of_day",
        "window_id",
        "node_id",
        "y",
        "Label",
        "Flow ID",
        "Source Port",
        "Timestamp",
        "Source IP",
        "Destination IP",
        "Destination Port",
        "Protocol",
    }
    feature_cols = [col_name for col_name, dtype in df.dtypes if col_name not in non_features and dtype in ("double", "int", "bigint", "float")]

    graphs = (
        df.groupBy("graph_id", "day", "window_id")
        .agg(
            F.max("y").alias("graph_y"),
            F.count(F.lit(1)).alias("n_nodes"),
            F.sum("y").alias("n_pos_nodes"),
        )
        .withColumn("attack_ratio", F.col("n_pos_nodes") / F.col("n_nodes"))
    )

    graphs_pdf = graphs.toPandas()
    graphs_pdf["n_nodes"] = graphs_pdf["n_nodes"].astype(int)
    graphs_pdf["n_pos_nodes"] = graphs_pdf["n_pos_nodes"].astype(int)
    graphs_pdf["graph_y"] = graphs_pdf["graph_y"].astype(int)
    graphs_pdf["attack_ratio"] = graphs_pdf["attack_ratio"].astype(float)

    graphs_pdf = build_legacy_split(graphs_pdf=graphs_pdf, seed=DEFAULT_SPLIT_SEED)
    graphs_pdf = assign_balanced_splits(graphs_pdf=graphs_pdf, split_seed=DEFAULT_SPLIT_SEED)

    split_summary_pdf = build_split_summary(
        graphs_pdf=graphs_pdf,
        split_col="split",
        strategy_label=DEFAULT_SPLIT_STRATEGY,
    )
    split_comparison_pdf = pd.concat(
        [
            build_split_summary(
                graphs_pdf=graphs_pdf,
                split_col="legacy_split",
                strategy_label=LEGACY_SPLIT_STRATEGY,
            ),
            split_summary_pdf,
        ],
        ignore_index=True,
    )

    assignment_cols = [
        "graph_id",
        "legacy_split",
        "split",
        "split_strategy",
        "split_seed",
    ]
    assignment_csv_path = output_dir_path / "graph_assignments.csv"
    graphs_pdf[assignment_cols].to_csv(assignment_csv_path, index=False)

    graphs_assignment_df = (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(str(assignment_csv_path))
    )

    graphs = graphs.join(graphs_assignment_df, on="graph_id", how="inner")

    nodes = (
        df.join(graphs.select("graph_id", "split"), on="graph_id", how="inner")
        .select("graph_id", "day", "window_id", "split", "node_id", "y", *feature_cols)
    )

    edge_meta = graphs.select("graph_id", "day", "window_id", "split")
    edges = edges.join(edge_meta, on="graph_id", how="inner")

    output_path_nodes = output_dir_path / "nodes"
    output_path_edges = output_dir_path / "edges"
    output_path_graphs_meta = output_dir_path / "graphs_meta"

    nodes.write.mode("overwrite").partitionBy("split", "day").parquet(str(output_path_nodes))
    edges.write.mode("overwrite").partitionBy("split", "day").parquet(str(output_path_edges))
    graphs.write.mode("overwrite").parquet(str(output_path_graphs_meta))

    graph_build_config = expected_graph_build_config(
        output_dir_path=output_dir_path,
        edge_family=args.edge_family,
        chain_k=args.chain_k,
        feature_cols=feature_cols,
    )

    with open(output_dir_path / "graph_build_config.json", "w", encoding="utf-8") as file_obj:
        json.dump(graph_build_config, file_obj, indent=2)

    save_dataframe_artifacts(
        split_summary_pdf,
        csv_path=output_dir_path / "split_summary.csv",
        json_path=output_dir_path / "split_summary.json",
    )
    save_dataframe_artifacts(
        split_comparison_pdf,
        csv_path=output_dir_path / "split_comparison.csv",
        json_path=output_dir_path / "split_comparison.json",
    )

    spark.stop()


if __name__ == "__main__":
    main()
