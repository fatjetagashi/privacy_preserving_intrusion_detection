import os
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

ROOT = os.path.join("..", "data", "traffic_labelled", "2A_preprocessed_gnn")
NODES_DIR = os.path.join(ROOT, "nodes")
EDGES_DIR = os.path.join(ROOT, "edges")
META_DIR  = os.path.join(ROOT, "graphs_meta")

graphs_meta = pd.read_parquet(META_DIR)[["graph_id", "day", "split"]].drop_duplicates().reset_index(drop=True)

nodes_ds = ds.dataset(NODES_DIR, format="parquet", partitioning="hive")
edges_ds = ds.dataset(EDGES_DIR, format="parquet", partitioning="hive")

NON_FEATURES = {"graph_id","day","time_of_day","window_id","node_id","y","Label","Flow ID", "Source Port",
                "Timestamp","Source IP","Destination IP","Destination Port","Protocol"}

@lru_cache(maxsize=32)
def load_partition(split: str, day: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    filt = (ds.field("split") == split) & (ds.field("day") == day)

    nodes = nodes_ds.to_table(filter=filt).to_pandas()
    edges = edges_ds.to_table(filter=filt).to_pandas()

    nodes = nodes.sort_values(["graph_id", "node_id"], kind="stable")
    return nodes, edges

def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols



class CICWindowGraphDataset(Dataset):
    def __init__(self, split: str):
        self.split = split
        self.meta = graphs_meta[graphs_meta["split"] == split].reset_index(drop=True)
        self.feature_cols = None  # inferred lazily

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Data:
        row = self.meta.iloc[idx]
        gid = row["graph_id"]
        day = row["day"]

        nodes_df, edges_df = load_partition(self.split, day)

        g_nodes = nodes_df[nodes_df["graph_id"] == gid].copy()
        g_edges = edges_df[edges_df["graph_id"] == gid].copy()

        # Infer feature columns from actual data once
        if self.feature_cols is None:
            self.feature_cols = infer_feature_cols(g_nodes)

        # Ensure node_id order is 0..N-1
        g_nodes = g_nodes.sort_values("node_id", kind="stable")
        node_ids = g_nodes["node_id"].to_numpy()

        if len(node_ids) > 0:
            # This should pass because of your row_number() logic
            assert node_ids.min() == 0 and node_ids.max() == len(node_ids) - 1, \
                f"node_id not contiguous for {gid}"

        x = torch.tensor(g_nodes[self.feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(g_nodes["y"].to_numpy(dtype=np.int64), dtype=torch.long)

        edge_index = torch.tensor(
            g_edges[["src", "dst"]].to_numpy(dtype=np.int64).T,
            dtype=torch.long
        )

        data = Data(x=x, edge_index=edge_index, y=y)
        data.graph_id = gid  # optional, helpful for debugging
        return data


train_ds = CICWindowGraphDataset("train")
val_ds   = CICWindowGraphDataset("val")
test_ds  = CICWindowGraphDataset("test")

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)


batch = next(iter(train_loader))
print(batch)
print("x:", batch.x.shape)
print("edge_index:", batch.edge_index.shape)
print("y:", batch.y.shape)