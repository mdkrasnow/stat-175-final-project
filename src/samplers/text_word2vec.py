"""Word2Vec text embeddings for PrimeKG nodes (schema-agnostic).

We train a skip-gram Word2Vec on the corpus of all node descriptions, then
compute each node's embedding as the mean of its word vectors (weighted by
IDF to down-weight common tokens). These embeddings serve as the text
baseline T in the DML setup — they are trained once on the full graph and
reused across schemas (text is schema-agnostic).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from ..data.primekg_loader import PrimeKG, load_primekg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "will", "with", "which", "but", "not", "also",
})


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def build_corpus(kg: PrimeKG) -> tuple[list[list[str]], list[int]]:
    """Tokenize each node's description. Returns (sentences, node_ids_in_order).

    Nodes with empty text still get an entry (empty list) so row indices line
    up with the node_id ordering.
    """
    node_ids = sorted(kg.node_texts.keys())
    sentences = [tokenize(kg.node_texts.get(n, "")) for n in node_ids]
    return sentences, node_ids


def _compute_idf(sentences: list[list[str]]) -> dict[str, float]:
    doc_count = len(sentences)
    df: Counter[str] = Counter()
    for sent in sentences:
        df.update(set(sent))
    # Smoothed IDF
    return {
        w: float(np.log((doc_count + 1) / (c + 1)) + 1.0)
        for w, c in df.items()
    }


def aggregate_node_embedding(
    tokens: list[str],
    model: Word2Vec,
    idf: dict[str, float],
    dim: int,
) -> np.ndarray:
    if not tokens:
        return np.zeros(dim, dtype=np.float32)
    vecs, weights = [], []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])
            weights.append(idf.get(t, 1.0))
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    V = np.stack(vecs).astype(np.float32)
    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    return (V * w[:, None]).sum(axis=0)


def train_word2vec(
    kg: PrimeKG,
    dim: int = 200,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 10,
    workers: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int], Word2Vec]:
    """Train Word2Vec on the node-description corpus and return per-node embeddings.

    Returns:
        embeddings: (num_nodes, dim) float32 array, row i = embedding of node_ids[i]
        node_ids:   list of node ids in row order (monotonic)
        model:      trained gensim Word2Vec model
    """
    sentences, node_ids = build_corpus(kg)
    if verbose:
        total_tokens = sum(len(s) for s in sentences)
        print(f"Corpus: {len(sentences):,} nodes, {total_tokens:,} tokens "
              f"({total_tokens / max(len(sentences), 1):.1f} avg tokens/node)")

    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,              # skip-gram
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    if verbose:
        print(f"Trained: vocab {len(model.wv):,} words")

    idf = _compute_idf(sentences)
    emb = np.zeros((len(node_ids), dim), dtype=np.float32)
    for i, tokens in enumerate(sentences):
        emb[i] = aggregate_node_embedding(tokens, model, idf, dim)

    # Sanity: fraction of rows that are non-zero
    nonzero = int((emb.any(axis=1)).sum())
    if verbose:
        print(f"Per-node embeddings: {nonzero:,}/{len(node_ids):,} non-zero "
              f"({nonzero / len(node_ids) * 100:.1f}%)")

    return emb, node_ids, model


def save_embeddings(
    embeddings: np.ndarray,
    node_ids: list[int],
    out_path: Path | str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        node_ids=np.array(node_ids, dtype=np.int64),
    )


def load_embeddings(path: Path | str) -> tuple[np.ndarray, list[int]]:
    data = np.load(Path(path))
    return data["embeddings"], data["node_ids"].tolist()


if __name__ == "__main__":
    kg = load_primekg()
    emb, node_ids, _ = train_word2vec(kg, dim=200)
    out = DEFAULT_EMBEDDING_DIR / "word2vec_200d.npz"
    save_embeddings(emb, node_ids, out)
    print(f"\nSaved to {out}")
    print(f"Shape: {emb.shape}, dtype: {emb.dtype}")
    print(f"Mean norm: {np.linalg.norm(emb, axis=1).mean():.3f}")
