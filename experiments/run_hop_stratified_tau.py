"""Phase 3.3: hop-stratified residual value of structure.

For each held-out pair (u, v), compute the shortest-path distance in the
schema subgraph *with the direct edge (u,v) removed if present* — this is
the "structural gap" the sampler would have to bridge. Bin pairs into
{2, 3, 4+} hops and report τ̄_{s, h} per bin.

Our phase-transition intuition: residual value of structure should grow
with hop distance. Proximity-based samplers (DeepWalk) should show the
strongest h-dependence; role-based ones may flatten.

Output: results/hop_stratified_tau.json + heatmap data to plot later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ALL_SCHEMAS, Schema, load_primekg, induce_schema_subgraph
from src.data.splits import load_fold
from src.estimation.dml import cross_fit_dml
from src.estimation.features import (
    hadamard_features,
    load_embedding,
    stack_pos_neg,
)
from src.estimation.inference import cluster_bootstrap_ci


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "hop_stratified_tau.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
SAMPLERS = ("deepwalk", "node2vec_bfs", "node2vec_dfs", "graphsage")
HOP_CUTOFF = 5


def _pool_folds(schema_name: str, num_folds: int = 5):
    all_pairs, all_labels, all_folds = [], [], []
    for k in range(num_folds):
        fold = load_fold(SPLITS_DIR / f"{schema_name}_fold{k}.npz")
        pairs, labels = stack_pos_neg(fold.test_pos, fold.test_neg)
        all_pairs.append(pairs)
        all_labels.append(labels)
        all_folds.append(np.full(len(pairs), k, dtype=np.int64))
    return (
        np.concatenate(all_pairs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_folds, axis=0),
    )


def _hop_distance(G: nx.Graph, u: int, v: int, cutoff: int = HOP_CUTOFF) -> int:
    """Shortest path length in G with edge (u,v) removed. Returns cutoff+1 if
    no path within cutoff is found or graph disconnected."""
    had_edge = G.has_edge(u, v)
    if had_edge:
        G.remove_edge(u, v)
    try:
        d = nx.shortest_path_length(G, u, v)
        if d is None or d > cutoff:
            d = cutoff + 1
    except nx.NetworkXNoPath:
        d = cutoff + 1
    except nx.NodeNotFound:
        d = cutoff + 1
    if had_edge:
        G.add_edge(u, v)
    return int(d)


def _bin_hop(d: int) -> str:
    if d == 2:
        return "2"
    if d == 3:
        return "3"
    if d >= 4:
        return "4+"
    return "direct"  # d==1 means adjacent even after removal (multi-edge), shouldn't happen


def compute_hop_distances(schema: Schema, kg, pairs: np.ndarray) -> np.ndarray:
    """Compute the shortest-path distance for each pair in the schema subgraph,
    with the direct edge (u,v) removed if present. Returns shape [N]."""
    sub = induce_schema_subgraph(kg, schema)
    # Use the largest connected component for distances; isolated pairs go to 4+
    lcc = max(nx.connected_components(sub), key=len)
    lcc_set = set(lcc)

    hops = np.full(len(pairs), HOP_CUTOFF + 1, dtype=np.int16)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        if u not in lcc_set or v not in lcc_set:
            continue
        hops[i] = _hop_distance(sub, u, v)
        if (i + 1) % 5000 == 0:
            print(f"    hop-dist progress: {i+1}/{len(pairs)}")
    return hops


def main() -> None:
    kg = load_primekg()
    text_emb, text_idx = load_embedding(EMBEDDING_DIR / "word2vec_200d.npz")

    results: dict = {}
    for schema in ALL_SCHEMAS:
        print(f"\n=== {schema.name} ===")
        pairs, labels, folds = _pool_folds(schema.name)
        print(f"  pooled: {len(pairs):,} pairs")

        hops = compute_hop_distances(schema, kg, pairs)
        bins = np.array([_bin_hop(int(h)) for h in hops])
        unique, counts = np.unique(bins, return_counts=True)
        print(f"  hop bins: {dict(zip(unique.tolist(), counts.tolist()))}")

        X_T = hadamard_features(pairs, text_emb, text_idx)
        schema_results: dict[str, dict] = {}

        for sampler in SAMPLERS:
            emb_path = EMBEDDING_DIR / sampler / f"{schema.name}.npz"
            if not emb_path.exists():
                continue
            struct_emb, struct_idx = load_embedding(emb_path)
            X_S = hadamard_features(pairs, struct_emb, struct_idx)
            X_TS = np.concatenate([X_T, X_S], axis=1)
            dml = cross_fit_dml(X_T, X_TS, labels, folds)
            tau = dml.tau_per_pair

            per_bin: dict[str, dict] = {}
            for b in ("2", "3", "4+"):
                mask = bins == b
                n = int(mask.sum())
                if n < 30:
                    per_bin[b] = {"n": n, "tau_bar": float("nan")}
                    continue
                ci = cluster_bootstrap_ci(
                    tau[mask],
                    node_u=pairs[mask, 0],
                    node_v=pairs[mask, 1],
                    B=300,
                )
                per_bin[b] = {
                    "n": n,
                    "tau_bar": float(tau[mask].mean()),
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                }
            schema_results[sampler] = per_bin
            print(f"  {sampler}: "
                  f"h=2 τ̄={per_bin['2'].get('tau_bar', float('nan')):+.3f} | "
                  f"h=3 τ̄={per_bin['3'].get('tau_bar', float('nan')):+.3f} | "
                  f"h=4+ τ̄={per_bin['4+'].get('tau_bar', float('nan')):+.3f}")

        results[schema.name] = schema_results

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
