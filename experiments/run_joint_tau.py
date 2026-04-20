"""Phase 3.4: joint-sampler DML — redundancy vs complementarity test.

If the samplers mostly capture overlapping structural information, then
τ̄_all (DML with all structural embeddings concatenated) ≈ max_s τ̄_s.

If they capture *complementary* structural primitives, τ̄_all >> max_s τ̄_s
and the delta quantifies how much more residual signal stacking adds.

Output: results/joint_tau.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ALL_SCHEMAS
from src.data.splits import load_fold
from src.estimation.dml import cross_fit_dml
from src.estimation.features import (
    hadamard_features,
    load_embedding,
    stack_pos_neg,
)
from src.estimation.inference import cluster_bootstrap_ci


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "joint_tau.json"
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

        # Per-sampler τ̄
        individual: dict[str, float] = {}
        per_sampler_X_S: list[np.ndarray] = []
        for sampler in SAMPLERS:
            emb_path = EMBEDDING_DIR / sampler / f"{schema.name}.npz"
            if not emb_path.exists():
                continue
            struct_emb, struct_idx = load_embedding(emb_path)
            X_S = hadamard_features(pairs, struct_emb, struct_idx)
            X_TS = np.concatenate([X_T, X_S], axis=1)
            dml = cross_fit_dml(X_T, X_TS, labels, folds)
            individual[sampler] = dml.tau_bar
            per_sampler_X_S.append(X_S)
            print(f"  {sampler}: τ̄={dml.tau_bar:+.4f}")

        if not per_sampler_X_S:
            continue

        # Joint DML with all samplers stacked
        X_all_struct = np.concatenate(per_sampler_X_S, axis=1)
        X_TS_all = np.concatenate([X_T, X_all_struct], axis=1)
        dml_all = cross_fit_dml(X_T, X_TS_all, labels, folds)

        ci_all = cluster_bootstrap_ci(
            dml_all.tau_per_pair, node_u=pairs[:, 0], node_v=pairs[:, 1], B=500
        )

        max_individual = max(individual.values())
        complementarity_gap = dml_all.tau_bar - max_individual

        print(f"  JOINT: τ̄_all={dml_all.tau_bar:+.4f} "
              f"CI=[{ci_all.lower:+.4f}, {ci_all.upper:+.4f}]")
        print(f"  max_individual={max_individual:+.4f}  "
              f"complementarity_gap={complementarity_gap:+.4f}")

        results[schema.name] = {
            "individual": individual,
            "joint_tau_bar": dml_all.tau_bar,
            "joint_ci_lower": ci_all.lower,
            "joint_ci_upper": ci_all.upper,
            "max_individual_tau_bar": max_individual,
            "complementarity_gap": complementarity_gap,
            "interpretation": (
                "complementary" if complementarity_gap > 0.02
                else "redundant" if complementarity_gap < 0.005
                else "mixed"
            ),
            "auc_T": dml_all.auc_T,
            "auc_TS_all": dml_all.auc_TS,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
