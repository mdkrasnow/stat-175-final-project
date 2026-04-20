"""Node-disjoint 5-fold splits + degree-matched negative sampling for link prediction.

Why node-disjoint?
    For double ML, a node's embedding must not appear in both the nuisance-
    training fold and the evaluation fold — otherwise the "held-out" τ̂ is
    contaminated by in-sample fit. We partition nodes into K groups and, for
    fold k, define:
        TEST  = edges with BOTH endpoints in group k
        TRAIN = edges with NEITHER endpoint in group k
    Edges with exactly one endpoint in group k are dropped (they leak).

Why degree-matched negatives?
    Uniform negative sampling makes link prediction trivially easy: any high-
    degree node will look "linked to everything". Matching the negative's
    target-side degree to the positive's target-side degree forces the
    classifier to use structure/text, not degree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random

import networkx as nx
import numpy as np

from .primekg_loader import PrimeKG
from .schemas import Schema, induce_schema_subgraph, target_edges


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


@dataclass
class FoldSplit:
    schema_name: str
    fold_idx: int
    num_folds: int
    # node_group: node_id -> which fold (0..num_folds-1) this node belongs to
    node_group: dict[int, int]
    # positives/negatives per split: each is a (N, 2) int array of (u, v) pairs with u<=v
    train_pos: np.ndarray
    train_neg: np.ndarray
    test_pos: np.ndarray
    test_neg: np.ndarray

    def summary(self) -> dict:
        return {
            "schema": self.schema_name,
            "fold_idx": self.fold_idx,
            "num_folds": self.num_folds,
            "num_train_pos": int(len(self.train_pos)),
            "num_train_neg": int(len(self.train_neg)),
            "num_test_pos": int(len(self.test_pos)),
            "num_test_neg": int(len(self.test_neg)),
            "num_test_nodes": sum(1 for g in self.node_group.values() if g == self.fold_idx),
        }


def _assign_node_groups(nodes: list[int], num_folds: int, seed: int) -> dict[int, int]:
    """Randomly partition nodes into ``num_folds`` groups of ~equal size."""
    rng = random.Random(seed)
    shuffled = nodes.copy()
    rng.shuffle(shuffled)
    return {n: i % num_folds for i, n in enumerate(shuffled)}


def _degree_matched_negative(
    u: int,
    deg_v: int,
    degree_buckets: dict[int, list[int]],
    bucket_keys: np.ndarray,
    all_positive_edges: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    allowed_nodes: set[int],
    rng: np.random.Generator,
    tolerance: float = 2.0,
    max_tries: int = 50,
) -> tuple[int, int] | None:
    """Sample a negative partner for u whose degree is within [deg_v/tol, deg_v*tol]."""
    lo, hi = max(1, int(deg_v / tolerance)), max(1, int(deg_v * tolerance))
    candidate_degs = bucket_keys[(bucket_keys >= lo) & (bucket_keys <= hi)]
    if len(candidate_degs) == 0:
        return None

    for _ in range(max_tries):
        d = int(rng.choice(candidate_degs))
        bucket = degree_buckets[d]
        if not bucket:
            continue
        v_neg = int(rng.choice(bucket))
        if v_neg == u or v_neg not in allowed_nodes:
            continue
        pair = (u, v_neg) if u <= v_neg else (v_neg, u)
        if pair in all_positive_edges or pair in drawn:
            continue
        return pair
    return None


def _sample_negatives(
    positives: np.ndarray,
    graph: nx.Graph,
    allowed_nodes: set[int],
    all_positive_edges: set[tuple[int, int]],
    seed: int,
) -> np.ndarray:
    """For each positive (u, v), sample a degree-matched non-edge (u, v')."""
    rng = np.random.default_rng(seed)

    degree_buckets: dict[int, list[int]] = {}
    for n in allowed_nodes:
        d = int(graph.degree(n))
        degree_buckets.setdefault(d, []).append(int(n))
    bucket_keys = np.array(sorted(degree_buckets.keys()))

    negatives: list[tuple[int, int]] = []
    drawn: set[tuple[int, int]] = set()
    for u, v in positives:
        u, v = int(u), int(v)
        deg_v = int(graph.degree(v))
        neg = _degree_matched_negative(
            u, deg_v, degree_buckets, bucket_keys,
            all_positive_edges, drawn, allowed_nodes, rng,
        )
        if neg is None:
            deg_u = int(graph.degree(u))
            neg = _degree_matched_negative(
                v, deg_u, degree_buckets, bucket_keys,
                all_positive_edges, drawn, allowed_nodes, rng,
            )
        if neg is None:
            continue
        negatives.append(neg)
        drawn.add(neg)

    if not negatives:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(negatives, dtype=np.int64)


def build_fold_splits(
    kg: PrimeKG,
    schema: Schema,
    num_folds: int = 5,
    seed: int = 42,
) -> list[FoldSplit]:
    """Build all K folds of a node-disjoint link-prediction split for one schema.

    Positives are the schema's target edges (``schema.target_relations``).
    Negatives are degree-matched non-edges over the full schema subgraph.
    """
    subgraph = induce_schema_subgraph(kg, schema)
    positives_list = target_edges(kg, schema, subgraph=subgraph)
    if not positives_list:
        raise ValueError(f"Schema {schema.name} has no target edges.")
    positives = np.array(positives_list, dtype=np.int64)

    schema_nodes = sorted(subgraph.nodes())
    node_group = _assign_node_groups(schema_nodes, num_folds, seed)

    all_positive_edges: set[tuple[int, int]] = {
        (a, b) if a <= b else (b, a)
        for a, b in kg.graph.edges()
    }

    folds: list[FoldSplit] = []
    for k in range(num_folds):
        test_group_nodes = {n for n, g in node_group.items() if g == k}
        train_allowed = {n for n in schema_nodes if n not in test_group_nodes}

        groups = np.array([node_group[int(u)] for u, _ in positives])
        groups_v = np.array([node_group[int(v)] for _, v in positives])

        is_test = (groups == k) & (groups_v == k)
        is_train = (groups != k) & (groups_v != k)
        # edges with exactly one endpoint in test group are dropped (leakage).

        test_pos = positives[is_test]
        train_pos = positives[is_train]

        train_neg = _sample_negatives(
            train_pos, subgraph, train_allowed, all_positive_edges, seed + k
        )
        test_neg = _sample_negatives(
            test_pos, subgraph, test_group_nodes, all_positive_edges, seed + 1000 + k
        )

        folds.append(FoldSplit(
            schema_name=schema.name,
            fold_idx=k,
            num_folds=num_folds,
            node_group=node_group,
            train_pos=train_pos,
            train_neg=train_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        ))

    return folds


def save_folds(folds: list[FoldSplit], out_dir: Path | str = DEFAULT_SPLITS_DIR) -> None:
    """Serialize each fold to ``{schema}_fold{k}.npz`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        path = out_dir / f"{fold.schema_name}_fold{fold.fold_idx}.npz"
        node_ids = np.array(sorted(fold.node_group.keys()), dtype=np.int64)
        node_groups = np.array(
            [fold.node_group[int(n)] for n in node_ids], dtype=np.int8
        )
        np.savez_compressed(
            path,
            train_pos=fold.train_pos,
            train_neg=fold.train_neg,
            test_pos=fold.test_pos,
            test_neg=fold.test_neg,
            node_ids=node_ids,
            node_groups=node_groups,
            fold_idx=np.int32(fold.fold_idx),
            num_folds=np.int32(fold.num_folds),
        )


def load_fold(path: Path | str) -> FoldSplit:
    """Inverse of ``save_folds`` for a single file."""
    path = Path(path)
    data = np.load(path)
    node_group = {int(n): int(g) for n, g in zip(data["node_ids"], data["node_groups"])}
    # Parse schema name from filename: {schema}_fold{k}.npz
    stem = path.stem
    schema_name = stem.rsplit("_fold", 1)[0]
    return FoldSplit(
        schema_name=schema_name,
        fold_idx=int(data["fold_idx"]),
        num_folds=int(data["num_folds"]),
        node_group=node_group,
        train_pos=data["train_pos"],
        train_neg=data["train_neg"],
        test_pos=data["test_pos"],
        test_neg=data["test_neg"],
    )


def verify_no_node_leak(fold: FoldSplit) -> dict:
    """Check that no node appears in both train and test edge sets."""
    test_nodes = {n for n, g in fold.node_group.items() if g == fold.fold_idx}
    train_nodes = set(fold.train_pos[:, 0].tolist()) | set(fold.train_pos[:, 1].tolist())
    train_nodes |= set(fold.train_neg[:, 0].tolist()) | set(fold.train_neg[:, 1].tolist())
    leak = train_nodes & test_nodes
    return {
        "num_train_edges_pos": int(len(fold.train_pos)),
        "num_test_edges_pos": int(len(fold.test_pos)),
        "num_train_nodes_touched": len(train_nodes),
        "num_test_group_nodes": len(test_nodes),
        "leaked_nodes": len(leak),
    }


if __name__ == "__main__":
    from .primekg_loader import load_primekg
    from .schemas import ALL_SCHEMAS

    kg = load_primekg()
    summary: dict = {"schemas": {}}

    for schema in ALL_SCHEMAS:
        print(f"\n=== {schema.name} ===")
        folds = build_fold_splits(kg, schema, num_folds=5, seed=42)
        save_folds(folds)

        per_fold = []
        for fold in folds:
            s = fold.summary()
            v = verify_no_node_leak(fold)
            s["leaked_nodes"] = v["leaked_nodes"]
            per_fold.append(s)
            print(
                f"  fold {fold.fold_idx}: "
                f"train pos/neg={s['num_train_pos']}/{s['num_train_neg']} "
                f"test pos/neg={s['num_test_pos']}/{s['num_test_neg']} "
                f"leak={s['leaked_nodes']}"
            )
        summary["schemas"][schema.name] = per_fold

    out_path = PROJECT_ROOT / "results" / "splits_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {out_path}")
