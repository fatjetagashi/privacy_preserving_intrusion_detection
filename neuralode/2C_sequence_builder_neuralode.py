from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.schema import CIC_IDS_2017_T_FULL_CLEAN_SCHEMA


INPUT_DIR_PATH = REPO_ROOT / "data" / "traffic_labelled" / "1B_clean_cic_ids_t_2017"
BASE_OUTPUT_DIR = REPO_ROOT / "data" / "traffic_labelled"

DEFAULT_WINDOW_SECONDS = 60
DEFAULT_MIN_LEN = 5
DEFAULT_MAX_LEN_BUILD = 2000
DEFAULT_ENTITY_MODE = "dst_plus_svc"
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_ASSIGNMENT = "balanced"
DEFAULT_SPLIT_STRATEGY = "deterministic_day_entity_balanced_v1"
TEMPORAL_SPLIT_STRATEGY = "global_temporal_group_holdout_v1"
LEGACY_SPLIT_STRATEGY = "legacy_random_percent_rank_by_day_entity_y"
DEFAULT_INVALID_TIMESTAMP_POLICY = "drop"
DEFAULT_SPARK_MASTER = "local[4]"
DEFAULT_SPARK_DRIVER_MEMORY = "12g"
SEQUENCE_BUILD_VERSION = "neuralode_retained_events_v2"
EVENT_COUNTING_POLICY = "retained_events_v2"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ["train", "val", "test"]

ENTITY_MODE_COLUMNS = {
    "dst_plus_svc": ["Destination IP", "Destination Port", "Protocol"],
    "src_plus_svc": ["Source IP", "Destination Port", "Protocol"],
    "src_plus_dst": ["Source IP", "Destination IP", "Protocol"],
    "src_plus_dst_plus_svc": ["Source IP", "Destination IP", "Destination Port", "Protocol"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-stamped Neural ODE trajectories from CIC-IDS2017 traffic-labelled flows."
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help="Sequence window size in seconds.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=DEFAULT_MIN_LEN,
        help="Discard trajectories shorter than this event count after capping.",
    )
    parser.add_argument(
        "--max-len-build",
        type=int,
        default=DEFAULT_MAX_LEN_BUILD,
        help="Cap stored trajectories to the most recent N events.",
    )
    parser.add_argument(
        "--entity-mode",
        choices=sorted(ENTITY_MODE_COLUMNS),
        default=DEFAULT_ENTITY_MODE,
        help="How to define the entity key that anchors each trajectory.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed used for deterministic balanced split assignment.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["balanced", "temporal"],
        default=DEFAULT_SPLIT_ASSIGNMENT,
        help="How to assign day-plus-entity split groups to train/val/test.",
    )
    parser.add_argument(
        "--invalid-timestamp-policy",
        choices=["drop", "error"],
        default=DEFAULT_INVALID_TIMESTAMP_POLICY,
        help="How to handle rows whose Timestamp is null inside the cleaned upstream artifact.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional explicit output directory for the generated Neural ODE dataset.",
    )
    parser.add_argument(
        "--spark-master",
        default=DEFAULT_SPARK_MASTER,
        help="Spark master string for preprocessing execution.",
    )
    parser.add_argument(
        "--spark-driver-memory",
        default=DEFAULT_SPARK_DRIVER_MEMORY,
        help="Optional Spark driver memory value such as 8g or 24g.",
    )
    return parser.parse_args()


def sanitize_variant_id(
    entity_mode: str,
    window_seconds: int,
    min_len: int,
    max_len_build: int,
    split_strategy: str = DEFAULT_SPLIT_ASSIGNMENT,
) -> str:
    variant_id = f"{entity_mode}_win{window_seconds}_min{min_len}_max{max_len_build}"
    if split_strategy != DEFAULT_SPLIT_ASSIGNMENT:
        variant_id = f"{variant_id}_{split_strategy}"
    return variant_id


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()

    if (
        args.window_seconds == DEFAULT_WINDOW_SECONDS and
        args.min_len == DEFAULT_MIN_LEN and
        args.max_len_build == DEFAULT_MAX_LEN_BUILD and
        args.entity_mode == DEFAULT_ENTITY_MODE and
        args.split_strategy == DEFAULT_SPLIT_ASSIGNMENT
    ):
        return BASE_OUTPUT_DIR / "2C_preprocessed_neuralode"

    variant_id = sanitize_variant_id(
        entity_mode=args.entity_mode,
        window_seconds=args.window_seconds,
        min_len=args.min_len,
        max_len_build=args.max_len_build,
        split_strategy=args.split_strategy,
    )
    return BASE_OUTPUT_DIR / "2C_preprocessed_neuralode_variants" / variant_id


def build_entity_expr(entity_mode: str):
    parts = []
    for col_name in ENTITY_MODE_COLUMNS[entity_mode]:
        parts.append(F.coalesce(F.col(col_name).cast("string"), F.lit("NA")))
    return F.concat_ws("::", *parts)


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
        group_y = int(row["group_y"])

        score_group_class = _metric_deficit(
            class_targets[split].get(group_y, 0.0),
            float(state["groups_by_y"].get(group_y, 0)),
            1.0,
        )
        score_group_total = _metric_deficit(targets[split]["groups"], float(state["groups_total"]), 1.0)
        score_sequences = _metric_deficit(targets[split]["sequences"], float(state["sequences"]), float(row["n_sequences"]))
        score_attack_sequences = _metric_deficit(
            targets[split]["attack_sequences"],
            float(state["attack_sequences"]),
            float(row["n_attack_sequences"]),
        )
        score_events = _metric_deficit(targets[split]["events"], float(state["events"]), float(row["n_events"]))
        score_attack_events = _metric_deficit(
            targets[split]["attack_events"],
            float(state["attack_events"]),
            float(row["n_attack_events"]),
        )

        key = (
            round(
                4.0 * score_group_class +
                2.5 * score_group_total +
                2.0 * score_sequences +
                2.0 * score_attack_sequences +
                1.5 * score_attack_events +
                1.0 * score_events,
                12,
            ),
            round(score_group_class, 12),
            round(score_attack_sequences, 12),
            round(score_attack_events, 12),
            round(score_sequences, 12),
            -float(state["sequences"]),
            -float(state["groups_total"]),
            -split_rank,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_split = split

    return best_split


def assign_balanced_splits(groups_pdf: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    assignments = []
    rng = np.random.default_rng(split_seed)

    for day, day_chunk in groups_pdf.groupby("day", sort=True):
        day_df = day_chunk.copy()
        day_df["seed_tie_break"] = rng.random(len(day_df))
        total_groups = float(len(day_df))
        total_sequences = float(day_df["n_sequences"].sum())
        total_attack_sequences = float(day_df["n_attack_sequences"].sum())
        total_events = float(day_df["n_events"].sum())
        total_attack_events = float(day_df["n_attack_events"].sum())

        targets = {
            split: {
                "groups": total_groups * ratio,
                "sequences": total_sequences * ratio,
                "attack_sequences": total_attack_sequences * ratio,
                "events": total_events * ratio,
                "attack_events": total_attack_events * ratio,
            }
            for split, ratio in SPLIT_RATIOS.items()
        }

        class_targets = {
            split: {
                int(group_y): float(len(class_chunk)) * ratio
                for group_y, class_chunk in day_df.groupby("group_y", sort=True)
            }
            for split, ratio in SPLIT_RATIOS.items()
        }

        current = {
            split: {
                "groups_total": 0,
                "groups_by_y": {0: 0, 1: 0},
                "sequences": 0.0,
                "attack_sequences": 0.0,
                "events": 0.0,
                "attack_events": 0.0,
            }
            for split in SPLIT_ORDER
        }

        ordered_rows = (
            day_df.sort_values(
                [
                    "group_y",
                    "n_attack_sequences",
                    "n_attack_events",
                    "n_sequences",
                    "n_events",
                    "positive_event_rate",
                    "attack_ratio",
                    "seed_tie_break",
                    "split_group_id",
                ],
                ascending=[False, False, False, False, False, False, False, True, True],
                kind="stable",
            )
            .to_dict("records")
        )

        for row in ordered_rows:
            split = choose_split(row=row, current=current, targets=targets, class_targets=class_targets)
            group_y = int(row["group_y"])

            current[split]["groups_total"] += 1
            current[split]["groups_by_y"][group_y] = current[split]["groups_by_y"].get(group_y, 0) + 1
            current[split]["sequences"] += float(row["n_sequences"])
            current[split]["attack_sequences"] += float(row["n_attack_sequences"])
            current[split]["events"] += float(row["n_events"])
            current[split]["attack_events"] += float(row["n_attack_events"])

            assignments.append(
                {
                    "day": row["day"],
                    "split_group_id": row["split_group_id"],
                    "split": split,
                    "split_strategy": DEFAULT_SPLIT_STRATEGY,
                    "split_seed": split_seed,
                }
            )

    assignments_pdf = pd.DataFrame(assignments)
    return groups_pdf.merge(assignments_pdf, on=["day", "split_group_id"], how="inner")


def build_temporal_split(groups_pdf: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    part = groups_pdf.copy()
    part = part.sort_values(["group_start_ts", "day", "split_group_id"], kind="stable").reset_index(drop=True)

    if len(part) == 1:
        percent_rank = [0.0]
    else:
        percent_rank = [idx / float(len(part) - 1) for idx in range(len(part))]

    part["split"] = [
        "train"
        if p < SPLIT_RATIOS["train"]
        else "val"
        if p < (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])
        else "test"
        for p in percent_rank
    ]
    part["split_strategy"] = TEMPORAL_SPLIT_STRATEGY
    part["split_seed"] = split_seed
    return part


def build_legacy_split(groups_pdf: pd.DataFrame, seed: int) -> pd.DataFrame:
    out_parts = []
    rng = np.random.default_rng(seed)

    for (day, group_y), chunk in groups_pdf.groupby(["day", "group_y"], sort=True):
        part = chunk.copy()
        part["legacy_rand"] = rng.random(len(part))
        part = part.sort_values(["legacy_rand", "split_group_id"], kind="stable").reset_index(drop=True)

        if len(part) == 1:
            percent_rank = [0.0]
        else:
            percent_rank = [idx / float(len(part) - 1) for idx in range(len(part))]

        part["legacy_split"] = [
            "train"
            if p < SPLIT_RATIOS["train"]
            else "val"
            if p < (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])
            else "test"
            for p in percent_rank
        ]
        out_parts.append(part)

    return pd.concat(out_parts, ignore_index=True).drop(columns=["legacy_rand"])


def build_split_summary(groups_pdf: pd.DataFrame, split_col: str, strategy_label: str) -> pd.DataFrame:
    rows = []

    for (day, split), chunk in groups_pdf.groupby(["day", split_col], sort=True):
        rows.append(
            {
                "summary_label": strategy_label,
                "split": split,
                "day": day,
                "n_groups": int(len(chunk)),
                "n_attack_groups": int((chunk["group_y"] == 1).sum()),
                "n_benign_groups": int((chunk["group_y"] == 0).sum()),
                "n_sequences": int(chunk["n_sequences"].sum()),
                "n_attack_sequences": int(chunk["n_attack_sequences"].sum()),
                "n_events": int(chunk["n_events"].sum()),
                "n_attack_events": int(chunk["n_attack_events"].sum()),
                "positive_sequence_rate": float(chunk["n_attack_sequences"].sum() / max(chunk["n_sequences"].sum(), 1)),
                "positive_event_rate": float(chunk["n_attack_events"].sum() / max(chunk["n_events"].sum(), 1)),
            }
        )

    for split, chunk in groups_pdf.groupby(split_col, sort=True):
        rows.append(
            {
                "summary_label": strategy_label,
                "split": split,
                "day": "ALL",
                "n_groups": int(len(chunk)),
                "n_attack_groups": int((chunk["group_y"] == 1).sum()),
                "n_benign_groups": int((chunk["group_y"] == 0).sum()),
                "n_sequences": int(chunk["n_sequences"].sum()),
                "n_attack_sequences": int(chunk["n_attack_sequences"].sum()),
                "n_events": int(chunk["n_events"].sum()),
                "n_attack_events": int(chunk["n_attack_events"].sum()),
                "positive_sequence_rate": float(chunk["n_attack_sequences"].sum() / max(chunk["n_sequences"].sum(), 1)),
                "positive_event_rate": float(chunk["n_attack_events"].sum() / max(chunk["n_events"].sum(), 1)),
            }
        )

    return pd.DataFrame(rows).sort_values(["day", "split"], kind="stable").reset_index(drop=True)


def save_dataframe_artifacts(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)


def main() -> None:
    args = parse_args()
    output_dir_path = resolve_output_root(args)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    spark_builder = (
        SparkSession.builder
        .appName("CIC_IDS_T_2017 NEURALODE SEQUENCE BUILDER")
        .master(args.spark_master)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.shuffle.partitions", "64")
    )
    if args.spark_driver_memory:
        spark_builder = spark_builder.config("spark.driver.memory", args.spark_driver_memory)
    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    input_path = INPUT_DIR_PATH / "1B_clean_cic_ids_t_2017.csv"
    df = (
        spark.read
        .format("csv")
        .schema(CIC_IDS_2017_T_FULL_CLEAN_SCHEMA)
        .option("header", True)
        .option("sep", ",")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .load(str(input_path))
    )

    df = df.select([F.col(col_name).alias(col_name.strip()) for col_name in df.columns])
    input_rows = df.count()

    null_timestamp_rows = df.filter(F.col("Timestamp").isNull()).count()
    if null_timestamp_rows > 0 and args.invalid_timestamp_policy == "error":
        raise ValueError(
            f"Found {null_timestamp_rows} rows with null Timestamp values in the cleaned upstream artifact."
        )

    if args.invalid_timestamp_policy == "drop":
        df = df.filter(F.col("Timestamp").isNotNull())

    rows_after_timestamp_filter = df.count()
    rows_dropped_invalid_timestamp = input_rows - rows_after_timestamp_filter

    if rows_after_timestamp_filter <= 0:
        raise ValueError("No rows remain after timestamp handling. Neural ODE trajectories require valid timestamps.")

    df = df.withColumn(
        "Flow ID",
        F.coalesce(
            F.col("Flow ID"),
            F.concat_ws(
                "-",
                F.coalesce(F.col("Source IP"), F.lit("NA")),
                F.coalesce(F.col("Destination IP"), F.lit("NA")),
                F.coalesce(F.col("Source Port").cast("string"), F.lit("NA")),
                F.coalesce(F.col("Destination Port").cast("string"), F.lit("NA")),
                F.coalesce(F.col("Protocol").cast("string"), F.lit("NA")),
            ),
        ),
    )
    df = df.withColumn("Destination Port", F.coalesce(F.col("Destination Port"), F.lit(-1)))
    df = df.withColumn("Protocol", F.coalesce(F.col("Protocol"), F.lit(-1)))
    df = df.withColumn("y", F.when(F.trim(F.col("Label")) == "BENIGN", F.lit(0)).otherwise(F.lit(1)))
    df = df.withColumn("ts_unix", F.col("Timestamp").cast("long"))
    df = df.withColumn("window_id", F.floor(F.col("ts_unix") / F.lit(args.window_seconds)))
    df = df.withColumn("entity", build_entity_expr(args.entity_mode))
    df = df.withColumn("split_group_id", F.concat_ws("::", F.col("day"), F.col("entity")))
    df = df.withColumn("seq_id", F.concat_ws("_", F.col("day"), F.col("window_id").cast("string"), F.col("entity")))

    non_features = {
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        "Timestamp",
        "Label",
        "day",
        "time_of_day",
        "y",
        "ts_unix",
        "window_id",
        "entity",
        "split_group_id",
        "seq_id",
    }
    feature_cols = [
        col_name
        for col_name, dtype in df.dtypes
        if col_name not in non_features and dtype in ("double", "int", "bigint", "float")
    ]

    df = df.withColumn("x", F.array([F.col(col_name).cast("double") for col_name in feature_cols]))

    full_sequence_stats = (
        df.groupBy("seq_id", "day", "window_id", "entity", "split_group_id")
        .agg(
            F.count("*").alias("full_seq_len"),
            F.sum("y").alias("full_n_attack_events"),
        )
        .withColumn("was_truncated", F.col("full_seq_len") > F.lit(args.max_len_build))
    )

    tail_window = Window.partitionBy("seq_id").orderBy(
        F.col("ts_unix").desc(),
        F.col("Flow ID").desc(),
    )
    capped_events = (
        df.withColumn("_tail_rank", F.row_number().over(tail_window))
        .filter(F.col("_tail_rank") <= F.lit(args.max_len_build))
        .drop("_tail_rank")
    )

    sequence_payload = (
        capped_events.groupBy("seq_id", "day", "window_id", "entity", "split_group_id")
        .agg(
            F.min("Timestamp").alias("seq_start_ts"),
            F.max("Timestamp").alias("seq_end_ts"),
            F.collect_list(
                F.struct(
                    F.col("ts_unix").alias("t"),
                    F.col("Flow ID").alias("flow_id"),
                    F.col("x").alias("x"),
                    F.col("y").alias("y"),
                )
            ).alias("events"),
        )
        .withColumn("events_sorted", F.sort_array(F.col("events"), asc=True))
        .drop("events")
        .withColumn("t", F.expr("transform(events_sorted, e -> e.t)"))
        .withColumn("x", F.expr("transform(events_sorted, e -> e.x)"))
        .withColumn("y_seq", F.expr("transform(events_sorted, e -> e.y)"))
        .drop("events_sorted")
        .withColumn("seq_len", F.size("t"))
        .withColumn("n_events", F.col("seq_len"))
        .withColumn("n_attack_events", F.expr("aggregate(y_seq, 0, (acc, y) -> acc + y)"))
        .withColumn("seq_y", F.when(F.col("n_attack_events") > F.lit(0), F.lit(1)).otherwise(F.lit(0)))
    )

    seqs = full_sequence_stats.join(
        sequence_payload,
        on=["seq_id", "day", "window_id", "entity", "split_group_id"],
        how="inner",
    )

    sequences_before_min_filter = seqs.count()
    seqs = seqs.filter(F.col("seq_len") >= F.lit(args.min_len))
    sequences_after_min_filter = seqs.count()
    sequences_dropped_short = sequences_before_min_filter - sequences_after_min_filter
    truncated_sequences = seqs.filter(F.col("was_truncated")).count()
    truncated_attack_events_removed = (
        seqs
        .select((F.col("full_n_attack_events") - F.col("n_attack_events")).alias("n_removed_attack_events"))
        .agg(F.coalesce(F.sum("n_removed_attack_events"), F.lit(0)).alias("total"))
        .collect()[0]["total"]
    )

    groups = (
        seqs.groupBy("day", "split_group_id")
        .agg(
            F.first("entity", ignorenulls=True).alias("entity"),
            F.max("seq_y").alias("group_y"),
            F.count("*").alias("n_sequences"),
            F.sum("seq_y").alias("n_attack_sequences"),
            F.sum("seq_len").alias("n_events"),
            F.sum("n_attack_events").alias("n_attack_events"),
            F.sum(F.col("was_truncated").cast("int")).alias("n_truncated_sequences"),
            F.min("seq_start_ts").alias("group_start_ts"),
            F.max("seq_end_ts").alias("group_end_ts"),
        )
        .withColumn("attack_ratio", F.col("n_attack_sequences") / F.greatest(F.col("n_sequences"), F.lit(1)))
        .withColumn("positive_event_rate", F.col("n_attack_events") / F.greatest(F.col("n_events"), F.lit(1)))
    )

    groups_pdf = groups.toPandas()
    if args.split_strategy == "balanced":
        assignments_pdf = assign_balanced_splits(groups_pdf, split_seed=args.split_seed)
        applied_split_strategy = DEFAULT_SPLIT_STRATEGY
    else:
        assignments_pdf = build_temporal_split(groups_pdf, split_seed=args.split_seed)
        applied_split_strategy = TEMPORAL_SPLIT_STRATEGY
    legacy_pdf = build_legacy_split(groups_pdf, seed=args.split_seed)

    split_summary = build_split_summary(assignments_pdf, split_col="split", strategy_label=applied_split_strategy)
    legacy_summary = build_split_summary(legacy_pdf, split_col="legacy_split", strategy_label=LEGACY_SPLIT_STRATEGY)
    split_comparison = pd.concat([legacy_summary, split_summary], ignore_index=True)

    sequence_assignments = assignments_pdf.merge(
        legacy_pdf[["day", "split_group_id", "legacy_split"]],
        on=["day", "split_group_id"],
        how="left",
    )

    assignments_join_path = output_dir_path / "sequence_assignments_for_join.tmp.csv"
    assignments_pdf[["day", "split_group_id", "split", "split_strategy", "split_seed"]].to_csv(
        assignments_join_path,
        index=False,
    )
    assignments_spark = (
        spark.read
        .option("header", True)
        .schema("day string, split_group_id string, split string, split_strategy string, split_seed int")
        .csv(str(assignments_join_path))
    )
    seqs = seqs.join(assignments_spark, on=["day", "split_group_id"], how="inner")

    out_seq = output_dir_path / "sequences"
    out_meta = output_dir_path / "sequences_meta"

    (
        seqs.select(
            "seq_id",
            "day",
            "window_id",
            "entity",
            "split_group_id",
            "split",
            "split_strategy",
            "split_seed",
            "seq_y",
            "seq_len",
            "full_seq_len",
            "full_n_attack_events",
            "was_truncated",
            "n_events",
            "n_attack_events",
            "seq_start_ts",
            "seq_end_ts",
            "t",
            "x",
            "y_seq",
        )
        .write.mode("overwrite")
        .partitionBy("split", "day")
        .parquet(str(out_seq))
    )

    (
        seqs.select(
            "seq_id",
            "day",
            "window_id",
            "entity",
            "split_group_id",
            "split",
            "split_strategy",
            "split_seed",
            "seq_y",
            "seq_len",
            "full_seq_len",
            "full_n_attack_events",
            "was_truncated",
            "n_events",
            "n_attack_events",
            "seq_start_ts",
            "seq_end_ts",
        )
        .write.mode("overwrite")
        .parquet(str(out_meta))
    )

    save_dataframe_artifacts(
        sequence_assignments,
        output_dir_path / "sequence_assignments.csv",
        output_dir_path / "sequence_assignments.json",
    )
    save_dataframe_artifacts(
        split_summary,
        output_dir_path / "split_summary.csv",
        output_dir_path / "split_summary.json",
    )
    save_dataframe_artifacts(
        split_comparison,
        output_dir_path / "split_comparison.csv",
        output_dir_path / "split_comparison.json",
    )

    sequence_build_config = {
        "dataset_source": "Traffic Labelled / GeneratedLabelledFlows",
        "sequence_build_version": SEQUENCE_BUILD_VERSION,
        "event_counting_policy": EVENT_COUNTING_POLICY,
        "output_root": str(output_dir_path),
        "variant_id": sanitize_variant_id(
            entity_mode=args.entity_mode,
            window_seconds=args.window_seconds,
            min_len=args.min_len,
            max_len_build=args.max_len_build,
            split_strategy=args.split_strategy,
        ),
        "window_seconds": args.window_seconds,
        "entity_mode": args.entity_mode,
        "entity_columns": ENTITY_MODE_COLUMNS[args.entity_mode],
        "split_grouping": "day_plus_entity",
        "split_strategy": applied_split_strategy,
        "split_assignment": args.split_strategy,
        "legacy_split_strategy": LEGACY_SPLIT_STRATEGY,
        "split_seed": args.split_seed,
        "split_ratios": SPLIT_RATIOS,
        "invalid_timestamp_policy": args.invalid_timestamp_policy,
        "min_len": args.min_len,
        "max_len_build": args.max_len_build,
        "capping_policy": "tail",
        "time_representation": "event_timestamp_seconds",
        "label_scope": "retained_capped_trajectory",
        "prediction_targets_supported": ["seq_y", "y_seq"],
        "feature_columns": feature_cols,
        "spark_master": args.spark_master,
        "spark_driver_memory": args.spark_driver_memory,
    }

    preprocessing_stats = {
        "input_path": str(input_path),
        "rows_input": int(input_rows),
        "rows_after_timestamp_filter": int(rows_after_timestamp_filter),
        "rows_dropped_invalid_timestamp": int(rows_dropped_invalid_timestamp),
        "null_timestamp_rows_seen": int(null_timestamp_rows),
        "sequences_before_min_len_filter": int(sequences_before_min_filter),
        "sequences_after_min_len_filter": int(sequences_after_min_filter),
        "sequences_dropped_short": int(sequences_dropped_short),
        "sequences_truncated_after_filter": int(truncated_sequences),
        "truncated_attack_events_removed_after_capping": int(truncated_attack_events_removed),
        "n_split_groups": int(len(groups_pdf)),
        "n_attack_split_groups": int((groups_pdf["group_y"] == 1).sum()) if not groups_pdf.empty else 0,
        "n_benign_split_groups": int((groups_pdf["group_y"] == 0).sum()) if not groups_pdf.empty else 0,
        "days_present_after_timestamp_filter": sorted(groups_pdf["day"].dropna().unique().tolist()) if not groups_pdf.empty else [],
        "timestamp_warning": (
            "Rows with null timestamps were dropped inside the Neural ODE branch. "
            "This leaves upstream artifacts unchanged while preventing invalid continuous-time trajectories."
        ),
    }

    with open(output_dir_path / "sequence_build_config.json", "w", encoding="utf-8") as file_obj:
        json.dump(sequence_build_config, file_obj, indent=2)

    with open(output_dir_path / "preprocessing_stats.json", "w", encoding="utf-8") as file_obj:
        json.dump(preprocessing_stats, file_obj, indent=2)

    spark.stop()

    try:
        assignments_join_path.unlink(missing_ok=True)
    except PermissionError:
        pass


if __name__ == "__main__":
    main()
