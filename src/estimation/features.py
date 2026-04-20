"""Pair-feature construction for link-prediction DML.

Standard trick: represent an undirected node pair (u,v) by the Hadamard
(elementwise) product of endpoint embeddings. This gives a permutation-
invariant feature vector that link-prediction classifiers work well with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_embedding(path: Path | str) -> tuple[np.ndarray, dict[int, int]]:
    """Load an embedding .npz file. Returns (emb, node_id -> row_index)."""
    data = np.load(Path(path))
    emb = data["embeddings"].astype(np.float32)
    node_ids = data["node_ids"].tolist()
    idx = {int(n): i for i, n in enumerate(node_ids)}
    return emb, idx


def hadamard_features(
    pairs: np.ndarray,              # shape [N, 2]
    emb: np.ndarray,                # shape [num_nodes, dim]
    node_to_row: dict[int, int],
) -> np.ndarray:
    """Elementwise product of endpoint embeddings. Returns [N, dim] float32.

    Pairs whose endpoints are missing from the embedding get a zero row.
    """
    N, dim = len(pairs), emb.shape[1]
    out = np.zeros((N, dim), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        ru = node_to_row.get(int(u))
        rv = node_to_row.get(int(v))
        if ru is None or rv is None:
            continue
        out[i] = emb[ru] * emb[rv]
    return out


def build_pair_features(
    pairs: np.ndarray,
    text_emb: np.ndarray,
    text_node_to_row: dict[int, int],
    struct_emb: np.ndarray | None = None,
    struct_node_to_row: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build (X_T, X_TS) for a batch of node pairs.

    X_T  = Hadamard(text(u), text(v))
    X_TS = concat(X_T, Hadamard(struct(u), struct(v)))   (or None if struct_emb is None)
    """
    X_T = hadamard_features(pairs, text_emb, text_node_to_row)
    if struct_emb is None or struct_node_to_row is None:
        return X_T, None
    X_S = hadamard_features(pairs, struct_emb, struct_node_to_row)
    X_TS = np.concatenate([X_T, X_S], axis=1)
    return X_T, X_TS


def stack_pos_neg(
    pos: np.ndarray,
    neg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack positive and negative pairs with 0/1 labels. Returns (pairs, labels)."""
    pairs = np.concatenate([pos, neg], axis=0) if len(neg) else pos
    labels = np.concatenate([
        np.ones(len(pos), dtype=np.float32),
        np.zeros(len(neg), dtype=np.float32),
    ]) if len(neg) else np.ones(len(pos), dtype=np.float32)
    return pairs, labels
