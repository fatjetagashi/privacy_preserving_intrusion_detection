from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "traffic_labelled" / "2A_preprocessed_gnn"

NON_FEATURES = {
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
    "split",
    "graph_y",
    "n_nodes",
    "n_pos_nodes",
    "attack_ratio",
    "legacy_split",
    "split_strategy",
    "split_seed",
}


def resolve_root(root: Optional[str] = None) -> Path:
    if root is None:
        return DEFAULT_ROOT
    return Path(root).resolve()


def load_graph_build_config(root: Optional[str] = None) -> Dict[str, object]:
    resolved_root = resolve_root(root)
    config_path = resolved_root / "graph_build_config.json"
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_split_summary(root: Optional[str] = None) -> pd.DataFrame:
    resolved_root = resolve_root(root)
    summary_path = resolved_root / "split_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


@lru_cache(maxsize=8)
def load_graphs_meta(root: str) -> pd.DataFrame:
    graphs_meta = pd.read_parquet(Path(root) / "graphs_meta")
    sort_cols = [col for col in ["split", "day", "window_id", "graph_id"] if col in graphs_meta.columns]
    if sort_cols:
        graphs_meta = graphs_meta.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return graphs_meta


@lru_cache(maxsize=8)
def _nodes_dataset(root: str):
    return ds.dataset(Path(root) / "nodes", format="parquet", partitioning="hive")


@lru_cache(maxsize=8)
def _edges_dataset(root: str):
    return ds.dataset(Path(root) / "edges", format="parquet", partitioning="hive")


def infer_feature_cols(root: Optional[str] = None) -> list[str]:
    resolved_root = str(resolve_root(root))
    node_schema = _nodes_dataset(resolved_root).schema

    feature_cols = []
    for field in node_schema:
        if field.name in NON_FEATURES:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            feature_cols.append(field.name)
    return feature_cols


@lru_cache(maxsize=64)
def load_partition(root: str, split: str, day: str):
    filt = (ds.field("split") == split) & (ds.field("day") == day)
    nodes = _nodes_dataset(root).to_table(filter=filt).to_pandas()
    edges = _edges_dataset(root).to_table(filter=filt).to_pandas()
    nodes = nodes.sort_values(["graph_id", "node_id"], kind="stable")
    return nodes, edges


class CICWindowGraphDataset(Dataset):
    def __init__(self, split: str, root: Optional[str] = None):
        self.root = str(resolve_root(root))
        self.split = split
        self.graphs_meta = load_graphs_meta(self.root)
        self.meta = self.graphs_meta[self.graphs_meta["split"] == split].reset_index(drop=True)
        self.feature_cols = infer_feature_cols(self.root)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Data:
        row = self.meta.iloc[idx]
        graph_id = row["graph_id"]
        day = row["day"]

        nodes_df, edges_df = load_partition(self.root, self.split, day)
        graph_nodes = nodes_df[nodes_df["graph_id"] == graph_id].copy()
        graph_edges = edges_df[edges_df["graph_id"] == graph_id].copy()

        graph_nodes = graph_nodes.sort_values("node_id", kind="stable")
        node_ids = graph_nodes["node_id"].to_numpy()
        if len(node_ids) > 0:
            assert node_ids.min() == 0 and node_ids.max() == len(node_ids) - 1, f"node_id not contiguous for {graph_id}"

        x = torch.tensor(graph_nodes[self.feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(graph_nodes["y"].to_numpy(dtype=np.int64), dtype=torch.long)

        if graph_edges.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(
                graph_edges[["src", "dst"]].to_numpy(dtype=np.int64).T,
                dtype=torch.long,
            )

        data = Data(x=x, edge_index=edge_index, y=y)
        data.graph_id = graph_id
        data.day = day
        return data


def dataset_metadata(root: Optional[str] = None) -> Dict[str, object]:
    resolved_root = resolve_root(root)
    graph_build_config = load_graph_build_config(str(resolved_root))
    feature_cols = infer_feature_cols(str(resolved_root))
    split_summary = load_split_summary(str(resolved_root))

    return {
        "root": str(resolved_root),
        "feature_cols": feature_cols,
        "graph_build_config": graph_build_config,
        "split_summary": split_summary.to_dict(orient="records") if not split_summary.empty else [],
        "graphs_meta_columns": list(load_graphs_meta(str(resolved_root)).columns),
    }


def build_loaders(batch_size: int = 1, seed: Optional[int] = None, root: Optional[str] = None):
    resolved_root = str(resolve_root(root))
    train_ds = CICWindowGraphDataset("train", root=resolved_root)
    val_ds = CICWindowGraphDataset("val", root=resolved_root)
    test_ds = CICWindowGraphDataset("test", root=resolved_root)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset_metadata(resolved_root)
