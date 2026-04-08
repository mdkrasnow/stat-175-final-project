import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


class QCTRDataset(Dataset):
    """Dataset for query-conditioned transition retrieval training data."""

    def __init__(self, data_dir, split="train"):
        npz = np.load(f"{data_dir}/{split}_features.npz")
        self.features = torch.tensor(npz["features"], dtype=torch.float32)
        self.edge_types = torch.tensor(npz["edge_types"], dtype=torch.long)
        self.labels = torch.tensor(npz["labels"], dtype=torch.float32)

        with open(f"{data_dir}/{split}_metadata.json", "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.edge_types[idx], self.labels[idx]

    def get_hop_mask(self, hop_count):
        """Return a boolean tensor selecting samples with the given hop count."""
        return torch.tensor(
            [m["hop_count"] == hop_count for m in self.metadata], dtype=torch.bool
        )


class TransitionScorer(nn.Module):
    """MLP that scores graph transitions for query-conditioned retrieval."""

    def __init__(
        self,
        input_dim=1155,
        num_edge_types=19,
        edge_embedding_dim=16,
        hidden_dims=(512, 256),
        dropout=0.3,
    ):
        super().__init__()
        # Edge types are -1..17, shift by +1 to get indices 0..18 (19 entries)
        self.edge_embedding = nn.Embedding(num_edge_types, edge_embedding_dim)

        layers = []
        in_dim = input_dim + edge_embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features, edge_types):
        # Shift edge_types so -1 -> 0, 0 -> 1, ..., 17 -> 18
        edge_idx = edge_types + 1
        edge_emb = self.edge_embedding(edge_idx)
        x = torch.cat([features, edge_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataloader. Returns dict with loss, accuracy, auc."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for features, edge_types, labels in dataloader:
            features = features.to(device)
            edge_types = edge_types.to(device)
            labels = labels.to(device)

            logits = model(features, edge_types)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    preds = (all_logits > 0).float()
    accuracy = (preds == all_labels).float().mean().item()

    labels_np = all_labels.numpy()
    probs_np = torch.sigmoid(all_logits).numpy()
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy,
        "auc": auc,
    }
