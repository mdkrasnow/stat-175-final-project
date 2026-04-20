"""Phase 3.2: paired comparisons between samplers.

For each schema and each (s, s') sampler pair, we re-use the same fold
assignment and same held-out pairs so τ̂_s(i) and τ̂_s'(i) are paired per
pair i. We then bootstrap over node clusters on the difference
τ̂_s - τ̂_s' to get a CI on the contrast.

Output: results/sampler_contrasts.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ALL_SCHEMAS
from src.estimation.dml import cross_fit_dml
from src.estimation.features import (
    hadamard_features,
    load_embedding,
    stack_pos_neg,
)
from src.estimation.inference import cluster_bootstrap_ci
from src.data.splits import load_fold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "sampler_contrasts.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

SAMPLERS = ("deepwalk", "node2vec_bfs", "node2vec_dfs", "graphsage")


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


def main() -> None:
    text_emb, text_idx = load_embedding(EMBEDDING_DIR / "word2vec_200d.npz")

    results: dict = {}
    for schema in ALL_SCHEMAS:
        print(f"\n=== {schema.name} ===")
        pairs, labels, folds = _pool_folds(schema.name)
        X_T = hadamard_features(pairs, text_emb, text_idx)

        taus_per_sampler: dict[str, np.ndarray] = {}
        for sampler in SAMPLERS:
            emb_path = EMBEDDING_DIR / sampler / f"{schema.name}.npz"
            if not emb_path.exists():
                continue
            struct_emb, struct_idx = load_embedding(emb_path)
            X_S = hadamard_features(pairs, struct_emb, struct_idx)
            X_TS = np.concatenate([X_T, X_S], axis=1)
            dml = cross_fit_dml(X_T, X_TS, labels, folds)
            taus_per_sampler[sampler] = dml.tau_per_pair
            print(f"  {sampler}: τ̄={dml.tau_bar:+.4f}")

        contrasts: dict[str, dict] = {}
        for s, s2 in combinations(taus_per_sampler.keys(), 2):
            diff = taus_per_sampler[s] - taus_per_sampler[s2]
            ci = cluster_bootstrap_ci(
                diff, node_u=pairs[:, 0], node_v=pairs[:, 1], B=500
            )
            key = f"{s}__minus__{s2}"
            contrasts[key] = {
                "mean_diff": float(diff.mean()),
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "ci_se": ci.se,
                "excludes_zero": bool(ci.lower > 0 or ci.upper < 0),
            }
            sig = "*" if (ci.lower > 0 or ci.upper < 0) else " "
            print(f"  {sig} {key}: {diff.mean():+.4f} "
                  f"[{ci.lower:+.4f}, {ci.upper:+.4f}]")

        results[schema.name] = contrasts

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
