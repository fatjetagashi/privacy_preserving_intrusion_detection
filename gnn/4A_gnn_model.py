import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np



# Optional but recommended for metrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from _3A_load_to_pytorch import build_loaders

train_loader, val_loader, test_loader = build_loaders(batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- 1) Compute feature mean/std on train split (for scaling) ----------
@torch.no_grad()
def compute_mean_std(loader, num_features: int):
    n = 0
    s1 = torch.zeros(num_features, dtype=torch.float64)
    s2 = torch.zeros(num_features, dtype=torch.float64)
    for data in loader:
        x = data.x.to(torch.float64)  # keep precision
        s1 += x.sum(dim=0)
        s2 += (x * x).sum(dim=0)
        n += x.size(0)
    mean = s1 / n
    var = (s2 / n) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.to(torch.float32), std.to(torch.float32)

# Infer num_features from one batch
sample = next(iter(train_loader))
num_features = sample.x.size(1)

print("Computing train mean/std...")
x_mean, x_std = compute_mean_std(train_loader, num_features)

# ---------- 2) Compute pos_weight for BCEWithLogitsLoss ----------
@torch.no_grad()
def compute_pos_weight(loader):
    pos = 0
    neg = 0
    for data in loader:
        y = data.y
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())
    # pos_weight = neg/pos (how much to upweight positives)
    pos_weight = (neg / max(pos, 1))
    return torch.tensor([pos_weight], dtype=torch.float32)

print("Computing pos_weight...")
pos_weight = compute_pos_weight(train_loader).to(device)
print("pos_weight:", pos_weight.item())

# ---------- 3) Model ----------
class GraphSAGEBinary(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.mlp(x).squeeze(-1)  # [num_nodes]
        return logits

model = GraphSAGEBinary(num_features, hidden=128, dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ---------- 4) Helpers: apply scaling + eval metrics ----------
def normalize_x(x: torch.Tensor) -> torch.Tensor:
    return (x - x_mean.to(x.device)) / x_std.to(x.device)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_probs = []
    all_y = []

    total_loss = 0.0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        x = normalize_x(data.x)
        logits = model(x, data.edge_index)

        y = data.y.float()
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.numel()
        total_nodes += y.numel()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(data.y.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    # Metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # AUCs require both classes present
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    avg_loss = total_loss / max(total_nodes, 1)
    return {"loss": avg_loss, "f1": f1, "precision": prec, "recall": rec, "roc_auc": roc, "pr_auc": pr}

# ---------- 5) Training loop ----------
def train_one_epoch():
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for data in train_loader:
        data = data.to(device)
        x = normalize_x(data.x)
        logits = model(x, data.edge_index)

        y = data.y.float()
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * y.numel()
        total_nodes += y.numel()

    return total_loss / max(total_nodes, 1)

best_val_f1 = -1.0
best_state = None

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()
    val_metrics = evaluate(val_loader)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
          f"val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f} | "
          f"val_roc={val_metrics['roc_auc']:.4f}")

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

# Load best model and test
if best_state is not None:
    model.load_state_dict(best_state)

test_metrics = evaluate(test_loader)
print("TEST:", test_metrics)
