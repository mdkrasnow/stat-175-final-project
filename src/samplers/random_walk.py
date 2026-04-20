"""DeepWalk and node2vec structural embeddings via random walks + skip-gram.

DeepWalk uses uniform random walks; node2vec uses biased walks controlled
by (p, q). Both are trained with gensim Word2Vec skip-gram over the walk
corpus, giving us a shared implementation that differs only in walk sampling.

Each sampler is a probe for a distinct structural signal:
  - DeepWalk:              proximity / homophily
  - node2vec (p=1, q=0.25): BFS-biased, structural-role
  - node2vec (p=1, q=4):    DFS-biased, community membership
"""

from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from ..data.primekg_loader import load_primekg
from ..data.schemas import ALL_SCHEMAS, Schema, induce_schema_subgraph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"


def _uniform_walks(
    graph: nx.Graph,
    num_walks: int,
    walk_length: int,
    seed: int,
) -> list[list[str]]:
    """DeepWalk: uniform random walks."""
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    neighbors = {n: list(graph.neighbors(n)) for n in nodes}
    walks: list[list[str]] = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                nbrs = neighbors[cur]
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            walks.append([str(n) for n in walk])
    return walks


def _node2vec_walks(
    graph: nx.Graph,
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    seed: int,
) -> list[list[str]]:
    """node2vec: second-order biased walks.

    Transition from t -> v -> x weighted by:
      1/p   if x == t         (return)
      1     if x is neighbor of t (BFS-like)
      1/q   otherwise          (DFS-like)
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    neighbors: dict[int, list[int]] = {n: list(graph.neighbors(n)) for n in nodes}
    neighbor_sets: dict[int, set[int]] = {n: set(nbrs) for n, nbrs in neighbors.items()}

    walks: list[list[str]] = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            if not neighbors[start]:
                walks.append([str(start)])
                continue
            walk.append(rng.choice(neighbors[start]))
            for _ in range(walk_length - 2):
                cur = walk[-1]
                prev = walk[-2]
                nbrs = neighbors[cur]
                if not nbrs:
                    break
                prev_nbrs = neighbor_sets[prev]
                weights = [
                    (1.0 / p) if x == prev
                    else 1.0 if x in prev_nbrs
                    else (1.0 / q)
                    for x in nbrs
                ]
                total = sum(weights)
                r = rng.random() * total
                acc = 0.0
                chosen = nbrs[-1]
                for x, w in zip(nbrs, weights):
                    acc += w
                    if acc >= r:
                        chosen = x
                        break
                walk.append(chosen)
            walks.append([str(n) for n in walk])
    return walks


def train_random_walk_embedding(
    graph: nx.Graph,
    sampler: str,
    dim: int = 128,
    walk_length: int = 40,
    num_walks: int = 10,
    window: int = 5,
    p: float = 1.0,
    q: float = 1.0,
    epochs: int = 5,
    workers: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Train DeepWalk or node2vec on ``graph`` and return per-node embeddings.

    Args:
        sampler: "deepwalk", "node2vec_bfs", or "node2vec_dfs"
                 (latter two use p=1, q=0.25 and p=1, q=4 by default — override
                  via the p, q kwargs).

    Returns:
        embeddings: (num_nodes_in_subgraph, dim) float32, row i = embedding of node_ids[i]
        node_ids:   list of subgraph node ids, sorted.
    """
    if sampler == "deepwalk":
        if verbose:
            print(f"  DeepWalk: generating {num_walks} walks x {graph.number_of_nodes()} nodes (len {walk_length})")
        walks = _uniform_walks(graph, num_walks, walk_length, seed)
    elif sampler in ("node2vec_bfs", "node2vec_dfs"):
        default_pq = {"node2vec_bfs": (1.0, 0.25), "node2vec_dfs": (1.0, 4.0)}
        p_eff, q_eff = default_pq[sampler]
        # Only override if caller passed non-default values
        if p != 1.0:
            p_eff = p
        if q != 1.0:
            q_eff = q
        if verbose:
            print(f"  node2vec p={p_eff}, q={q_eff}: "
                  f"{num_walks} walks x {graph.number_of_nodes()} nodes")
        walks = _node2vec_walks(graph, num_walks, walk_length, p_eff, q_eff, seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    if verbose:
        print(f"  Generated {len(walks):,} walks, training Word2Vec (dim={dim})...")
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    node_ids = sorted(graph.nodes())
    emb = np.zeros((len(node_ids), dim), dtype=np.float32)
    for i, n in enumerate(node_ids):
        key = str(n)
        if key in model.wv:
            emb[i] = model.wv[key]
    nonzero = int(emb.any(axis=1).sum())
    if verbose:
        print(f"  {nonzero:,}/{len(node_ids):,} non-zero embeddings")
    return emb, node_ids


def train_per_schema(
    kg,
    sampler: str,
    schemas: list[Schema] | None = None,
    out_dir: Path | str = DEFAULT_EMBEDDING_DIR,
    **train_kwargs,
) -> None:
    """Train ``sampler`` on each schema and save embeddings to
    ``{out_dir}/{sampler}/{schema_name}.npz``.
    """
    if schemas is None:
        schemas = ALL_SCHEMAS
    out_dir = Path(out_dir) / sampler
    out_dir.mkdir(parents=True, exist_ok=True)

    for schema in schemas:
        print(f"\n--- {sampler} on {schema.name} ---")
        subgraph = induce_schema_subgraph(kg, schema)
        print(f"  subgraph: {subgraph.number_of_nodes():,} nodes, "
              f"{subgraph.number_of_edges():,} edges")
        emb, node_ids = train_random_walk_embedding(subgraph, sampler, **train_kwargs)
        path = out_dir / f"{schema.name}.npz"
        np.savez_compressed(
            path,
            embeddings=emb,
            node_ids=np.array(node_ids, dtype=np.int64),
        )
        print(f"  saved: {path}")


if __name__ == "__main__":
    kg = load_primekg()
    for sampler in ("deepwalk", "node2vec_bfs", "node2vec_dfs"):
        print(f"\n{'='*60}\n{sampler}\n{'='*60}")
        train_per_schema(kg, sampler)
