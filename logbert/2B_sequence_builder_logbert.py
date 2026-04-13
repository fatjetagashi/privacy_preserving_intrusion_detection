from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.schema import CIC_IDS_2017_T_FULL_CLEAN_SCHEMA


INPUT_DIR_PATH = REPO_ROOT / "data" / "traffic_labelled" / "1B_clean_cic_ids_t_2017"
BASE_OUTPUT_DIR = REPO_ROOT / "data" / "traffic_labelled"

DEFAULT_WINDOW_SECONDS = 300
DEFAULT_MIN_LEN = 5
DEFAULT_MAX_LEN_BUILD = 2048
DEFAULT_ENTITY_MODE = "src_plus_dst_plus_svc"
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_ASSIGNMENT = "balanced"
DEFAULT_SPLIT_STRATEGY = "deterministic_day_entity_balanced_v1"
TEMPORAL_SPLIT_STRATEGY = "global_temporal_group_holdout_v1"
LEGACY_SPLIT_STRATEGY = "legacy_random_percent_rank_by_day_entity_y"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ["train", "val", "test"]

ENTITY_MODE_COLUMNS = {
    "src_plus_svc": ["Source IP", "Destination Port", "Protocol"],
    "src_plus_dst": ["Source IP", "Destination IP", "Protocol"],
    "src_plus_dst_plus_svc": ["Source IP", "Destination IP", "Destination Port", "Protocol"],
}

DUR_EDGES = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 120_000_000]
PKT_EDGES = [0, 1, 2, 5, 10, 20, 50, 100, 1_000, 10_000, 100_000, 300_000]
BYTES_EDGES = [0, 1, 64, 512, 4_096, 65_536, 1_048_576, 10_485_760, 104_857_600, 524_288_000, 700_000_000]
PPS_EDGES = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

FLAG_COLUMNS = [
    ("FIN Flag Count", "F"),
    ("SYN Flag Count", "S"),
    ("RST Flag Count", "R"),
    ("PSH Flag Count", "P"),
    ("ACK Flag Count", "A"),
    ("URG Flag Count", "U"),
]

TOKEN_INPUT_NUMERIC_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Packets/s",
    *[col_name for col_name, _ in FLAG_COLUMNS],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-ordered LogBERT sequences from CIC-IDS2017 traffic-labelled flows."
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
        help="Discard sequences shorter than this token count after capping.",
    )
    parser.add_argument(
        "--max-len-build",
        type=int,
        default=DEFAULT_MAX_LEN_BUILD,
        help="Cap stored token sequences to the most recent N tokens.",
    )
    parser.add_argument(
        "--entity-mode",
        choices=sorted(ENTITY_MODE_COLUMNS),
        default=DEFAULT_ENTITY_MODE,
        help="How to define the sequence entity key.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed used for the balanced split assignment.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["balanced", "temporal"],
        default=DEFAULT_SPLIT_ASSIGNMENT,
        help="How to assign split_group_id groups to train/val/test.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional explicit output directory for the generated LogBERT dataset.",
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
        return BASE_OUTPUT_DIR / "2B_preprocessed_logbert"

    variant_id = sanitize_variant_id(
        entity_mode=args.entity_mode,
        window_seconds=args.window_seconds,
        min_len=args.min_len,
        max_len_build=args.max_len_build,
        split_strategy=args.split_strategy,
    )
    return BASE_OUTPUT_DIR / "2B_preprocessed_logbert_variants" / variant_id


def bucketize_fixed(col_name: str, edges: list[float], prefix: str):
    expr = None
    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1]
        cond = (F.col(col_name) >= F.lit(lo)) & (F.col(col_name) < F.lit(hi))
        expr = F.when(cond, F.lit(f"{prefix}B{idx}")) if expr is None else expr.when(cond, F.lit(f"{prefix}B{idx}"))
    return expr.otherwise(F.lit(f"{prefix}B{len(edges) - 1}"))


def build_entity_expr(entity_mode: str):
    cols = []
    for col_name in ENTITY_MODE_COLUMNS[entity_mode]:
        cols.append(F.coalesce(F.col(col_name).cast("string"), F.lit("NA")))
    return F.concat_ws("::", *cols)


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
        score_tokens = _metric_deficit(targets[split]["tokens"], float(state["tokens"]), float(row["n_tokens"]))

        key = (
            round(4.0 * score_group_class + 2.5 * score_group_total + 2.0 * score_sequences + 2.0 * score_attack_sequences + 1.5 * score_tokens, 12),
            round(score_group_class, 12),
            round(score_attack_sequences, 12),
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
        total_tokens = float(day_df["n_tokens"].sum())

        targets = {
            split: {
                "groups": total_groups * ratio,
                "sequences": total_sequences * ratio,
                "attack_sequences": total_attack_sequences * ratio,
                "tokens": total_tokens * ratio,
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
                "tokens": 0.0,
            }
            for split in SPLIT_ORDER
        }

        ordered_rows = (
            day_df.sort_values(
                ["group_y", "n_attack_sequences", "n_sequences", "n_tokens", "attack_ratio", "seed_tie_break", "split_group_id"],
                ascending=[False, False, False, False, False, True, True],
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
            current[split]["tokens"] += float(row["n_tokens"])

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
                "n_tokens": int(chunk["n_tokens"].sum()),
                "positive_sequence_rate": float(chunk["n_attack_sequences"].sum() / max(chunk["n_sequences"].sum(), 1)),
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
                "n_tokens": int(chunk["n_tokens"].sum()),
                "positive_sequence_rate": float(chunk["n_attack_sequences"].sum() / max(chunk["n_sequences"].sum(), 1)),
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

    spark = (
        SparkSession.builder
        .appName("CIC_IDS_T_2017 LOGBERT SEQUENCE BUILDER")
        .master("local[*]")
        .getOrCreate()
    )
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

    df = df.withColumn("Timestamp", F.date_format(F.col("Timestamp"), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("ts", F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss"))
    df = df.filter(F.col("ts").isNotNull())
    rows_after_timestamp_filter = df.count()

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

    for col_name in TOKEN_INPUT_NUMERIC_COLUMNS:
        df = df.withColumn(
            col_name,
            F.when(
                F.col(col_name).isNull() | F.isnan(F.col(col_name)) | (F.col(col_name) < 0),
                F.lit(0.0),
            ).otherwise(F.col(col_name)),
        )

    df = df.withColumn("y", F.when(F.trim(F.col("Label")) == "BENIGN", F.lit(0)).otherwise(F.lit(1)))
    df = df.withColumn("window_id", F.floor(F.unix_timestamp("ts") / F.lit(args.window_seconds)))
    df = df.withColumn("entity", build_entity_expr(args.entity_mode))
    df = df.withColumn("split_group_id", F.concat_ws("::", F.col("day"), F.col("entity")))
    df = df.withColumn("seq_id", F.concat_ws("_", F.col("day"), F.col("window_id").cast("string"), F.col("entity")))

    protocol_tok = F.concat(F.lit("PR_"), F.col("Protocol").cast("string"))
    dport_tok = F.concat(F.lit("DP_"), F.col("Destination Port").cast("string"))

    dur_bin = bucketize_fixed("Flow Duration", DUR_EDGES, "DUR_")
    fwdp_bin = bucketize_fixed("Total Fwd Packets", PKT_EDGES, "FWD_")
    bwdp_bin = bucketize_fixed("Total Backward Packets", PKT_EDGES, "BWD_")

    df = df.withColumn("BYTES_TOTAL", F.col("Total Length of Fwd Packets") + F.col("Total Length of Bwd Packets"))
    byt_bin = bucketize_fixed("BYTES_TOTAL", BYTES_EDGES, "BYT_")
    pps_bin = bucketize_fixed("Flow Packets/s", PPS_EDGES, "PPS_")

    flag_parts = [F.when(F.col(col_name) > 0, F.lit(flag_letter)).otherwise(F.lit("")) for col_name, flag_letter in FLAG_COLUMNS]
    flag_sig = F.concat(*flag_parts)
    flag_sig = F.when(flag_sig == F.lit(""), F.lit("N")).otherwise(flag_sig)
    flag_tok = F.concat(F.lit("FL_"), flag_sig)

    df = df.withColumn(
        "token",
        F.concat_ws("|", protocol_tok, dport_tok, dur_bin, fwdp_bin, bwdp_bin, byt_bin, pps_bin, flag_tok),
    )

    seqs = (
        df.groupBy("seq_id", "day", "window_id", "entity", "split_group_id")
        .agg(
            F.max("y").alias("seq_y"),
            F.count("*").alias("n_flows"),
            F.min("ts").alias("seq_start_ts"),
            F.max("ts").alias("seq_end_ts"),
            F.collect_list(
                F.struct(
                    F.col("ts").alias("ts"),
                    F.col("Flow ID").alias("flow_id"),
                    F.col("token").alias("tok"),
                )
            ).alias("tok_structs"),
        )
        .withColumn("tok_sorted", F.sort_array(F.col("tok_structs"), asc=True))
        .withColumn("tokens_full", F.expr("transform(tok_sorted, x -> x.tok)"))
        .drop("tok_structs", "tok_sorted")
        .withColumn("full_seq_len", F.size("tokens_full"))
        .withColumn("was_truncated", F.col("full_seq_len") > F.lit(args.max_len_build))
        .withColumn(
            "tokens",
            F.expr(
                f"slice(tokens_full, greatest(size(tokens_full) - {args.max_len_build} + 1, 1), {args.max_len_build})"
            ),
        )
        .drop("tokens_full")
        .withColumn("seq_len", F.size("tokens"))
    )

    sequences_before_min_filter = seqs.count()
    seqs = seqs.filter(F.col("seq_len") >= F.lit(args.min_len))
    sequences_after_min_filter = seqs.count()
    sequences_dropped_short = sequences_before_min_filter - sequences_after_min_filter
    truncated_sequences = seqs.filter(F.col("was_truncated")).count()

    groups = (
        seqs.groupBy("day", "split_group_id")
        .agg(
            F.first("entity", ignorenulls=True).alias("entity"),
            F.max("seq_y").alias("group_y"),
            F.count("*").alias("n_sequences"),
            F.sum("seq_len").alias("n_tokens"),
            F.sum("seq_y").alias("n_attack_sequences"),
            F.sum(F.col("was_truncated").cast("int")).alias("n_truncated_sequences"),
            F.min("seq_start_ts").alias("group_start_ts"),
            F.max("seq_end_ts").alias("group_end_ts"),
        )
        .withColumn("attack_ratio", F.col("n_attack_sequences") / F.greatest(F.col("n_sequences"), F.lit(1)))
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

    assignments_spark = spark.createDataFrame(
        assignments_pdf[["day", "split_group_id", "split", "split_strategy", "split_seed"]]
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
            "was_truncated",
            "n_flows",
            "seq_start_ts",
            "seq_end_ts",
            "tokens",
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
            "was_truncated",
            "n_flows",
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
    save_dataframe_artifacts(split_summary, output_dir_path / "split_summary.csv", output_dir_path / "split_summary.json")
    save_dataframe_artifacts(
        split_comparison,
        output_dir_path / "split_comparison.csv",
        output_dir_path / "split_comparison.json",
    )

    sequence_build_config = {
        "dataset_source": "Traffic Labelled / GeneratedLabelledFlows",
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
        "token_schema": {
            "base_tokens": ["Protocol", "Destination Port"],
            "bucketized_tokens": [
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "BYTES_TOTAL",
                "Flow Packets/s",
            ],
            "flag_signature_columns": [col_name for col_name, _ in FLAG_COLUMNS],
        },
        "min_len": args.min_len,
        "max_len_build": args.max_len_build,
        "capping_policy": "tail",
    }

    preprocessing_stats = {
        "input_path": str(input_path),
        "rows_input": int(input_rows),
        "rows_after_timestamp_filter": int(rows_after_timestamp_filter),
        "rows_dropped_invalid_timestamp": int(input_rows - rows_after_timestamp_filter),
        "sequences_before_min_len_filter": int(sequences_before_min_filter),
        "sequences_after_min_len_filter": int(sequences_after_min_filter),
        "sequences_dropped_short": int(sequences_dropped_short),
        "sequences_truncated_after_filter": int(truncated_sequences),
        "n_split_groups": int(len(groups_pdf)),
        "n_attack_split_groups": int((groups_pdf["group_y"] == 1).sum()),
        "n_benign_split_groups": int((groups_pdf["group_y"] == 0).sum()),
    }

    with open(output_dir_path / "sequence_build_config.json", "w", encoding="utf-8") as file_obj:
        json.dump(sequence_build_config, file_obj, indent=2)

    with open(output_dir_path / "preprocessing_stats.json", "w", encoding="utf-8") as file_obj:
        json.dump(preprocessing_stats, file_obj, indent=2)

    spark.stop()


if __name__ == "__main__":
    main()
