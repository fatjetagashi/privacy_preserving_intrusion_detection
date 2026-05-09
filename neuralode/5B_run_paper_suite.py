from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

PREPROCESS_SCRIPT = THIS_DIR / "2C_sequence_builder_neuralode.py"
VALIDATE_SCRIPT = THIS_DIR / "5C_validate_outputs.py"
PREFLIGHT_SCRIPT = THIS_DIR / "5D_preflight_check.py"
TRAIN_SCRIPT = THIS_DIR / "4C_train_neuralode.py"
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
        entity_mode == "dst_plus_svc" and
        window_seconds == 60 and
        min_len == 5 and
        max_len_build == 2000 and
        split_strategy == "balanced"
    ):
        return REPO_ROOT / "data" / "traffic_labelled" / "2C_preprocessed_neuralode"

    variant_id = f"{entity_mode}_win{window_seconds}_min{min_len}_max{max_len_build}"
    if split_strategy != "balanced":
        variant_id = f"{variant_id}_{split_strategy}"
    return REPO_ROOT / "data" / "traffic_labelled" / "2C_preprocessed_neuralode_variants" / variant_id


def cli_args_from_mapping(options: dict[str, object] | None) -> list[str]:
    if not options:
        return []

    args = []
    for key, value in options.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args


def experiment_matrix():
    return [
        {
            "label": "neuralode_exact_current60_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_current60_seq_mean",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "mean",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_current60_seq_last",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "last",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_current60_seq_attention",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "attention",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_current60_event",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "event",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_src_plus_svc_seq_max",
            "entity_mode": "src_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_src_plus_dst_seq_max",
            "entity_mode": "src_plus_dst",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_src_plus_dst_plus_svc_seq_max",
            "entity_mode": "src_plus_dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_window30_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 30,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_window120_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 120,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_window300_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 300,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_temporal_holdout_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "temporal",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_midpoint_solver_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "midpoint",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_euler_solver_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "euler",
            "time_conditioning": "concat",
        },
        {
            "label": "neuralode_no_time_conditioning_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "none",
        },
        {
            "label": "neuralode_maxlen128_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
            "runtime_args": {"max_len": 128},
        },
        {
            "label": "neuralode_maxlen256_seq_max",
            "entity_mode": "dst_plus_svc",
            "window_seconds": 60,
            "min_len": 5,
            "max_len_build": 2000,
            "split_strategy": "balanced",
            "prediction_level": "sequence",
            "pooling": "max",
            "solver": "rk4",
            "time_conditioning": "concat",
            "runtime_args": {"max_len": 256},
        },
    ]


def command_strings(seeds: list[int], python_executable: str, selected_labels: set[str] | None = None) -> list[list[str]]:
    commands = []
    seen_preprocess_roots = set()

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
        preprocess_extra_args = cli_args_from_mapping(experiment.get("preprocess_args"))
        runtime_args = cli_args_from_mapping(experiment.get("runtime_args"))
        preflight_extra_args = cli_args_from_mapping(experiment.get("preflight_args"))
        train_extra_args = cli_args_from_mapping(experiment.get("train_args"))

        preprocess_key = str(output_root)
        if preprocess_key not in seen_preprocess_roots:
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
                    *preprocess_extra_args,
                ]
            )
            commands.append([python_executable, str(VALIDATE_SCRIPT), "--data-root", str(output_root)])
            seen_preprocess_roots.add(preprocess_key)

        commands.append(
            [
                python_executable,
                str(PREFLIGHT_SCRIPT),
                "--data-root",
                str(output_root),
                "--prediction-level",
                experiment["prediction_level"],
                "--pooling",
                experiment["pooling"],
                "--solver",
                experiment["solver"],
                "--time-conditioning",
                experiment["time_conditioning"],
                *runtime_args,
                *preflight_extra_args,
            ]
        )
        commands.append(
            [
                python_executable,
                str(TRAIN_SCRIPT),
                "--data-root",
                str(output_root),
                "--prediction-level",
                experiment["prediction_level"],
                "--pooling",
                experiment["pooling"],
                "--solver",
                experiment["solver"],
                "--time-conditioning",
                experiment["time_conditioning"],
                *runtime_args,
                *train_extra_args,
                "--seeds",
                *[str(seed) for seed in seeds],
            ]
        )

    commands.append([python_executable, str(AGGREGATE_SCRIPT)])
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print or execute the Neural ODE paper experiment matrix.")
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
