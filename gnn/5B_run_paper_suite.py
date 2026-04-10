from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

PREPROCESS_SCRIPT = REPO_ROOT / "traffic_labelled" / "2A_preprocessing.py"
GNN_SCRIPT = THIS_DIR / "4A_gnn_model.py"
MLP_SCRIPT = THIS_DIR / "4B_mlp_baseline.py"
AGGREGATE_SCRIPT = THIS_DIR / "5A_aggregate_runs.py"


def resolve_python_executable(explicit_python: str | None) -> str:
    if explicit_python:
        return str(Path(explicit_python).resolve())
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def variant_output_root(edge_family: str, chain_k: int) -> Path:
    if edge_family == "src+dst+svc" and chain_k == 5:
        return REPO_ROOT / "data" / "traffic_labelled" / "2A_preprocessed_gnn"

    variant_id = edge_family.replace("+", "_plus_")
    return REPO_ROOT / "data" / "traffic_labelled" / "2A_preprocessed_gnn_variants" / f"{variant_id}_k{chain_k}"


def experiment_matrix():
    return [
        {"label": "gnn_default", "model": "gnn", "edge_family": "src+dst+svc", "chain_k": 5},
        {"label": "gnn_chain_k1", "model": "gnn", "edge_family": "src+dst+svc", "chain_k": 1},
        {"label": "gnn_chain_k10", "model": "gnn", "edge_family": "src+dst+svc", "chain_k": 10},
        {"label": "gnn_edge_src_only", "model": "gnn", "edge_family": "src_only", "chain_k": 5},
        {"label": "gnn_edge_src_dst", "model": "gnn", "edge_family": "src+dst", "chain_k": 5},
        {"label": "mlp_baseline", "model": "mlp", "edge_family": "src+dst+svc", "chain_k": 5},
    ]


def command_strings(seeds: list[int], python_executable: str, selected_labels: set[str] | None = None) -> list[list[str]]:
    commands = []
    for experiment in experiment_matrix():
        if selected_labels is not None and experiment["label"] not in selected_labels:
            continue
        output_root = variant_output_root(experiment["edge_family"], experiment["chain_k"])
        commands.append(
            [
                python_executable,
                str(PREPROCESS_SCRIPT),
                "--edge-family",
                experiment["edge_family"],
                "--chain-k",
                str(experiment["chain_k"]),
                "--output-root",
                str(output_root),
            ]
        )

        train_script = GNN_SCRIPT if experiment["model"] == "gnn" else MLP_SCRIPT
        commands.append(
            [
                python_executable,
                str(train_script),
                "--data-root",
                str(output_root),
                "--seeds",
                *[str(seed) for seed in seeds],
            ]
        )

    commands.append([python_executable, str(AGGREGATE_SCRIPT)])
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or execute the paper experiment matrix for the GNN and MLP baseline."
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
