"""GraphSAGE unsupervised embeddings, trained per schema via link prediction.

GraphSAGE probes *neighborhood feature aggregation* — distinct from the
random-walk samplers (proximity/roles) and useful as a 4th structural
hypothesis in the sampler-as-probe study.

Features: learned per-node embeddings (Embedding table initialized randomly).
Keeping node features purely structural (no text) is deliberate: we want
this sampler's residual τ̂_s to isolate *aggregation-structure* signal rather
than text it picked up through node attributes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from ..data.primekg_loader import load_primekg
from ..data.schemas import ALL_SCHEMAS, Schema, induce_schema_subgraph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "graphsage"


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, num_nodes: int, feat_dim: int = 64, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.node_emb = torch.nn.Embedding(num_nodes, feat_dim)
        torch.nn.init.normal_(self.node_emb.weight, std=0.1)
        self.conv1 = SAGEConv(feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


def _nx_to_edge_index(graph: nx.Graph, node_to_idx: dict[int, int]) -> torch.Tensor:
    """Convert undirected networkx graph to symmetric edge_index tensor."""
    rows, cols = [], []
    for u, v in graph.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        rows.extend([ui, vi])
        cols.extend([vi, ui])
    return torch.tensor([rows, cols], dtype=torch.long)


def _sample_edge_batch(
    pos_edges: torch.Tensor,   # shape [E, 2]
    num_nodes: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of positive edges and matching random negatives."""
    idx = rng.choice(len(pos_edges), size=batch_size, replace=False)
    pos = pos_edges[idx]
    neg_v = rng.integers(0, num_nodes, size=batch_size)
    neg = torch.stack([pos[:, 0], torch.tensor(neg_v, dtype=torch.long)], dim=1)
    labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)])
    return pos, neg, labels


def train_graphsage(
    graph: nx.Graph,
    out_dim: int = 128,
    hidden: int = 128,
    feat_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 0.005,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Train 2-layer GraphSAGE with link-prediction loss on ``graph``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    node_ids = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    edge_index = _nx_to_edge_index(graph, node_to_idx)

    pos_edges = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in graph.edges()],
        dtype=torch.long,
    )
    num_nodes = len(node_ids)

    model = GraphSAGEEncoder(num_nodes, feat_dim=feat_dim, hidden=hidden, out_dim=out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = max(1, len(pos_edges) // batch_size)
    if verbose:
        print(f"  GraphSAGE: {num_nodes:,} nodes, {len(pos_edges):,} edges, "
              f"{epochs} epochs x {steps_per_epoch} steps")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            pos, neg, labels = _sample_edge_batch(pos_edges, num_nodes, batch_size, rng)
            z = model(edge_index)
            pos_score = (z[pos[:, 0]] * z[pos[:, 1]]).sum(dim=-1)
            neg_score = (z[neg[:, 0]] * z[neg[:, 1]]).sum(dim=-1)
            scores = torch.cat([pos_score, neg_score])
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={total_loss/steps_per_epoch:.4f}")

    model.eval()
    with torch.no_grad():
        z = model(edge_index).detach().cpu().numpy().astype(np.float32)
    return z, node_ids


def train_per_schema(
    kg,
    schemas: list[Schema] | None = None,
    out_dir: Path | str = DEFAULT_EMBEDDING_DIR,
    **train_kwargs,
) -> None:
    if schemas is None:
        schemas = ALL_SCHEMAS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for schema in schemas:
        print(f"\n--- graphsage on {schema.name} ---")
        subgraph = induce_schema_subgraph(kg, schema)
        print(f"  subgraph: {subgraph.number_of_nodes():,} nodes, "
              f"{subgraph.number_of_edges():,} edges")
        emb, node_ids = train_graphsage(subgraph, **train_kwargs)
        path = out_dir / f"{schema.name}.npz"
        np.savez_compressed(
            path,
            embeddings=emb,
            node_ids=np.array(node_ids, dtype=np.int64),
        )
        print(f"  saved: {path}")


if __name__ == "__main__":
    kg = load_primekg()
    train_per_schema(kg)
