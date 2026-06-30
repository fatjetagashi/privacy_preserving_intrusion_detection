from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

PREPROCESS_SCRIPT = THIS_DIR / "2C_sequence_builder_neuralode.py"
VALIDATE_SCRIPT = THIS_DIR / "5C_validate_outputs.py"
PREFLIGHT_SCRIPT = THIS_DIR / "5D_preflight_check.py"
TRAIN_SCRIPT = THIS_DIR / "4C_train_neuralode.py"
AGGREGATE_SCRIPT = THIS_DIR / "5A_aggregate_runs.py"
RUNS_DIR = THIS_DIR / "runs" / "neuralode"

PREPROCESSED_REQUIRED_FILES = [
    "sequence_assignments.csv",
    "sequence_assignments.json",
    "split_summary.csv",
    "split_summary.json",
    "split_comparison.csv",
    "split_comparison.json",
    "sequence_build_config.json",
    "preprocessing_stats.json",
]
PREPROCESSED_REQUIRED_DIRS = ["sequences", "sequences_meta"]
RUN_REQUIRED_FILES = [
    "run_config.json",
    "summary.json",
    "history.csv",
    "val_metrics.json",
    "test_metrics.json",
    "val_metrics_by_day.json",
    "test_metrics_by_day.json",
    "best_model.pt",
]
REQUIRED_SEQUENCE_BUILD_VERSION = "neuralode_retained_events_v2"
REQUIRED_EVENT_COUNTING_POLICY = "retained_events_v2"
CORE_MATRIX_LABELS = [
    "neuralode_exact_current60_seq_max",
    "neuralode_src_plus_svc_seq_max",
    "neuralode_src_plus_dst_plus_svc_seq_max",
    "neuralode_window120_seq_max",
    "neuralode_temporal_holdout_seq_max",
]

DEFAULT_TRAIN_ARGS = {
    "batch_train": 64,
    "batch_eval": 128,
    "num_workers": 0,
    "x_embed_dim": 128,
    "hidden_dim": 128,
    "ode_hidden": 128,
    "ode_layers": 2,
    "ode_steps": 4,
    "dropout": 0.1,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "max_epochs": 25,
    "grad_clip": 1.0,
    "early_stop_patience": 5,
    "early_stop_min_delta": 1e-4,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-5,
    "max_len": None,
    "max_len_percentile": 95.0,
    "max_len_cap": 512,
    "dt_clip": None,
    "max_feature_rows": 200_000,
    "max_dt_samples": 500_000,
}


def resolve_python_executable(explicit_python: str | None) -> str:
    if explicit_python:
        return str(Path(explicit_python).resolve())
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def python_script_command(python_executable: str, script_path: Path, *args: str) -> list[str]:
    return [python_executable, "-u", str(script_path), *args]


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


def load_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def paths_equal(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def has_parquet_file(root: Path) -> bool:
    try:
        next(root.rglob("*.parquet"))
        return True
    except StopIteration:
        return False


def preprocessed_root_is_complete(output_root: Path, experiment: dict[str, object]) -> bool:
    if not output_root.exists():
        return False
    if any(not (output_root / file_name).is_file() for file_name in PREPROCESSED_REQUIRED_FILES):
        return False
    if any(not (output_root / dir_name).is_dir() for dir_name in PREPROCESSED_REQUIRED_DIRS):
        return False
    if not has_parquet_file(output_root / "sequences") or not has_parquet_file(output_root / "sequences_meta"):
        return False

    try:
        config = load_json(output_root / "sequence_build_config.json")
    except (json.JSONDecodeError, OSError):
        return False

    expected_values = {
        "sequence_build_version": REQUIRED_SEQUENCE_BUILD_VERSION,
        "event_counting_policy": REQUIRED_EVENT_COUNTING_POLICY,
        "entity_mode": experiment["entity_mode"],
        "window_seconds": int(experiment["window_seconds"]),
        "min_len": int(experiment["min_len"]),
        "max_len_build": int(experiment["max_len_build"]),
        "split_assignment": experiment["split_strategy"],
    }
    for key, expected_value in expected_values.items():
        if config.get(key) != expected_value:
            return False

    return True


def merged_runtime_train_args(experiment: dict[str, object]) -> dict[str, object]:
    merged = dict(DEFAULT_TRAIN_ARGS)
    for section_name in ["runtime_args", "train_args"]:
        section = experiment.get(section_name)
        if isinstance(section, dict):
            merged.update(section)
    return merged


def model_variant_id(data_variant: str, experiment: dict[str, object], merged_args: dict[str, object]) -> str:
    parts = [
        data_variant,
        experiment["prediction_level"],
        experiment["pooling"],
        experiment["solver"],
        experiment["time_conditioning"],
    ]
    if merged_args.get("max_len") is not None:
        parts.append(f"maxlen{merged_args['max_len']}")
    if int(merged_args["ode_steps"]) != 4:
        parts.append(f"steps{merged_args['ode_steps']}")
    if (
        int(merged_args["hidden_dim"]) != 128
        or int(merged_args["ode_hidden"]) != 128
        or int(merged_args["x_embed_dim"]) != 128
    ):
        parts.append(f"x{merged_args['x_embed_dim']}_h{merged_args['hidden_dim']}_ode{merged_args['ode_hidden']}")
    return "_".join(str(part) for part in parts)


def run_config_matches(
    run_config: dict[str, object],
    output_root: Path,
    experiment: dict[str, object],
    merged_args: dict[str, object],
    seed: int,
) -> bool:
    if not paths_equal(str(run_config.get("data_root", "")), output_root):
        return False
    if int(run_config.get("seed", -1)) != int(seed):
        return False

    try:
        dataset_config = load_json(output_root / "sequence_build_config.json")
    except (json.JSONDecodeError, OSError):
        return False

    run_sequence_config = run_config.get("sequence_build_config", {})
    if not isinstance(run_sequence_config, dict):
        return False
    required_dataset_values = {
        "sequence_build_version": REQUIRED_SEQUENCE_BUILD_VERSION,
        "event_counting_policy": REQUIRED_EVENT_COUNTING_POLICY,
    }
    for key, expected_value in required_dataset_values.items():
        if dataset_config.get(key) != expected_value:
            return False
        if run_sequence_config.get(key) != dataset_config.get(key):
            return False

    expected_values = {
        "prediction_level": experiment["prediction_level"],
        "pooling": experiment["pooling"],
        "solver": experiment["solver"],
        "time_conditioning": experiment["time_conditioning"],
        "x_embed_dim": int(merged_args["x_embed_dim"]),
        "hidden_dim": int(merged_args["hidden_dim"]),
        "ode_hidden": int(merged_args["ode_hidden"]),
        "ode_layers": int(merged_args["ode_layers"]),
        "ode_steps": int(merged_args["ode_steps"]),
        "max_epochs": int(merged_args["max_epochs"]),
        "max_len_percentile": float(merged_args["max_len_percentile"]),
        "max_len_cap": int(merged_args["max_len_cap"]) if merged_args["max_len_cap"] is not None else None,
    }
    for key, expected_value in expected_values.items():
        if run_config.get(key) != expected_value:
            return False

    for key in ["dropout", "lr", "weight_decay", "grad_clip", "early_stop_min_delta"]:
        if abs(float(run_config.get(key, float("nan"))) - float(merged_args[key])) > 1e-12:
            return False

    if merged_args["max_len"] is not None and int(run_config.get("max_len", -1)) != int(merged_args["max_len"]):
        return False
    if merged_args["dt_clip"] is not None and abs(float(run_config.get("dt_clip", float("nan"))) - float(merged_args["dt_clip"])) > 1e-12:
        return False

    return True


def complete_run_exists(output_root: Path, experiment: dict[str, object], seed: int) -> bool:
    if not RUNS_DIR.exists():
        return False

    merged_args = merged_runtime_train_args(experiment)
    data_variant = output_root.name
    try:
        config = load_json(output_root / "sequence_build_config.json")
        data_variant = str(config.get("variant_id", data_variant))
    except (json.JSONDecodeError, OSError):
        pass

    expected_fragment = model_variant_id(data_variant, experiment, merged_args)
    candidate_dirs = sorted(RUNS_DIR.glob(f"*_{expected_fragment}_seed{seed}"), reverse=True)
    for run_dir in candidate_dirs:
        if any(not (run_dir / file_name).is_file() for file_name in RUN_REQUIRED_FILES):
            continue

        try:
            run_config = load_json(run_dir / "run_config.json")
            load_json(run_dir / "summary.json")
            load_json(run_dir / "val_metrics.json")
            load_json(run_dir / "test_metrics.json")
        except (json.JSONDecodeError, OSError):
            continue

        if run_config_matches(run_config, output_root, experiment, merged_args, seed):
            return True

    return False


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
            if not preprocessed_root_is_complete(output_root, experiment):
                commands.append(
                    python_script_command(
                        python_executable,
                        PREPROCESS_SCRIPT,
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
                    )
                )
            commands.append(python_script_command(python_executable, VALIDATE_SCRIPT, "--data-root", str(output_root)))
            seen_preprocess_roots.add(preprocess_key)

        commands.append(
            python_script_command(
                python_executable,
                PREFLIGHT_SCRIPT,
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
            )
        )
        for seed in seeds:
            if complete_run_exists(output_root=output_root, experiment=experiment, seed=seed):
                continue
            commands.append(
                python_script_command(
                    python_executable,
                    TRAIN_SCRIPT,
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
                    str(seed),
                )
            )

    commands.append(python_script_command(python_executable, AGGREGATE_SCRIPT))
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
    parser.add_argument(
        "--core-matrix",
        action="store_true",
        help="Run the paper core matrix labels only.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry each failed subprocess this many times before stopping.",
    )
    return parser.parse_args()


def command_stage(command: list[str]) -> str:
    script_name = Path(command[2]).name if len(command) > 2 else "unknown"
    stage_by_script = {
        PREPROCESS_SCRIPT.name: "preprocess",
        VALIDATE_SCRIPT.name: "validate",
        PREFLIGHT_SCRIPT.name: "preflight",
        TRAIN_SCRIPT.name: "train",
        AGGREGATE_SCRIPT.name: "aggregate",
    }
    return stage_by_script.get(script_name, script_name)


def command_target(command: list[str]) -> str:
    for flag in ["--data-root", "--output-root"]:
        if flag in command:
            flag_idx = command.index(flag)
            if flag_idx + 1 < len(command):
                return Path(command[flag_idx + 1]).name
    return "all"


def run_command_with_retries(command: list[str], max_retries: int) -> None:
    attempts = max(int(max_retries), 0) + 1
    stage = command_stage(command)
    target = command_target(command)
    for attempt_idx in range(1, attempts + 1):
        print(f"[RUN] stage={stage} target={target} attempt={attempt_idx}/{attempts}", flush=True)
        print(" ".join(command), flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode == 0:
            return

        print(f"[WARN] command failed with returncode={result.returncode}", flush=True)
        if attempt_idx >= attempts:
            raise subprocess.CalledProcessError(result.returncode, command)

        time.sleep(min(60, 5 * attempt_idx))


def main() -> None:
    args = parse_args()
    python_executable = resolve_python_executable(args.python_executable)
    if args.core_matrix and args.only_labels:
        raise ValueError("Use either --core-matrix or --only-labels, not both.")
    selected_labels = set(CORE_MATRIX_LABELS) if args.core_matrix else set(args.only_labels) if args.only_labels else None
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
        run_command_with_retries(command, max_retries=args.max_retries)


if __name__ == "__main__":
    main()
