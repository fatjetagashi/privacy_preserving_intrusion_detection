from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

PREPROCESS_SCRIPT = THIS_DIR / "2B_sequence_builder_logbert.py"
VALIDATE_SCRIPT = THIS_DIR / "5C_validate_outputs.py"
PREFLIGHT_SCRIPT = THIS_DIR / "5D_preflight_check.py"
TRAIN_SCRIPT = THIS_DIR / "4B_train_logbert.py"
AGGREGATE_SCRIPT = THIS_DIR / "5A_aggregate_runs.py"


def resolve_python_executable(explicit_python: str | None) -> str:
    if explicit_python:
        return str(Path(explicit_python).resolve())
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def variant_output_root(
    entity_mode: str,
    window_seconds: int,
    min_len: int,
    max_len_build: int,
    split_strategy: str,
) -> Path:
    if (
        entity_mode == "src_plus_dst_plus_svc" and
        window_seconds == 300 and
        min_len == 5 and
        max_len_build == 2048 and
        split_strategy == "balanced"
    ):
        return REPO_ROOT / "data" / "traffic_labelled" / "2B_preprocessed_logbert"

    variant_id = f"{entity_mode}_win{window_seconds}_min{min_len}_max{max_len_build}"
    if split_strategy != "balanced":
        variant_id = f"{variant_id}_{split_strategy}"
    return REPO_ROOT / "data" / "traffic_labelled" / "2B_preprocessed_logbert_variants" / variant_id


def experiment_matrix():
    return [
        {
            "label": "logbert_exact_default",
            "entity_mode": "src_plus_dst_plus_svc",
            "window_seconds": 300,
            "min_len": 5,
            "max_len_build": 2048,
            "split_strategy": "balanced",
            "vhm_weight": 0.10,
        },
        {
            "label": "logbert_mlm_only_default",
            "entity_mode": "src_plus_dst_plus_svc",
            "window_seconds": 300,
            "min_len": 5,
            "max_len_build": 2048,
            "split_strategy": "balanced",
            "vhm_weight": 0.0,
        },
        {
            "label": "logbert_exact_window120",
            "entity_mode": "src_plus_dst_plus_svc",
            "window_seconds": 120,
            "min_len": 5,
            "max_len_build": 2048,
            "split_strategy": "balanced",
            "vhm_weight": 0.10,
        },
        {
            "label": "logbert_exact_entity_src_svc",
            "entity_mode": "src_plus_svc",
            "window_seconds": 300,
            "min_len": 5,
            "max_len_build": 2048,
            "split_strategy": "balanced",
            "vhm_weight": 0.10,
        },
        {
            "label": "logbert_exact_temporal_holdout",
            "entity_mode": "src_plus_dst_plus_svc",
            "window_seconds": 300,
            "min_len": 5,
            "max_len_build": 2048,
            "split_strategy": "temporal",
            "vhm_weight": 0.10,
        },
    ]


def command_strings(seeds: list[int], python_executable: str, selected_labels: set[str] | None = None) -> list[list[str]]:
    commands = []
    for experiment in experiment_matrix():
        if selected_labels is not None and experiment["label"] not in selected_labels:
            continue

        output_root = variant_output_root(
            entity_mode=experiment["entity_mode"],
            window_seconds=experiment["window_seconds"],
            min_len=experiment["min_len"],
            max_len_build=experiment["max_len_build"],
            split_strategy=experiment["split_strategy"],
        )

        commands.append(
            [
                python_executable,
                str(PREPROCESS_SCRIPT),
                "--entity-mode",
                experiment["entity_mode"],
                "--window-seconds",
                str(experiment["window_seconds"]),
                "--min-len",
                str(experiment["min_len"]),
                "--max-len-build",
                str(experiment["max_len_build"]),
                "--split-strategy",
                experiment["split_strategy"],
                "--output-root",
                str(output_root),
            ]
        )
        commands.append([python_executable, str(VALIDATE_SCRIPT), "--data-root", str(output_root)])
        commands.append([python_executable, str(PREFLIGHT_SCRIPT), "--data-root", str(output_root)])
        commands.append(
            [
                python_executable,
                str(TRAIN_SCRIPT),
                "--data-root",
                str(output_root),
                "--vhm-weight",
                str(experiment["vhm_weight"]),
                "--seeds",
                *[str(seed) for seed in seeds],
            ]
        )

    commands.append([python_executable, str(AGGREGATE_SCRIPT)])
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or execute the LogBERT paper experiment matrix."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the full command matrix. Without this flag, the script only prints commands.",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Optional Python executable to use for subprocess commands. Defaults to the repo .venv if present.",
    )
    parser.add_argument(
        "--only-labels",
        nargs="+",
        default=None,
        help="Optional subset of experiment labels to run. Use labels from experiment_matrix().",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_executable = resolve_python_executable(args.python_executable)
    selected_labels = set(args.only_labels) if args.only_labels else None
    commands = command_strings(
        seeds=args.seeds,
        python_executable=python_executable,
        selected_labels=selected_labels,
    )

    for command in commands:
        print(" ".join(command))

    if not args.execute:
        return

    for command in commands:
        subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
