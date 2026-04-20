"""Phase 3.1: per-sampler in-distribution residual structural value on PrimeKG.

For each (schema, sampler) pair, we:
  1. Load the schema's 5-fold node-disjoint splits (built in Phase 1.3).
  2. Build per-pair features: X_T = Hadamard(text, text), X_TS = concat(X_T, S).
  3. Pool held-out τ̂ across folds via cross-fit DML (re-using our 5-fold
     assignment — each pair's τ̂ is predicted by a nuisance model that never
     saw that pair's nodes).
  4. Compute cluster-bootstrap CIs over nodes and a permutation null over
     the structural feature block.
  5. Apply Holm correction across the sampler family per schema.

Output: results/in_dist_tau.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ALL_SCHEMAS, Schema, load_primekg
from src.data.splits import load_fold
from src.estimation.dml import cross_fit_dml
from src.estimation.features import (
    hadamard_features,
    load_embedding,
    stack_pos_neg,
)
from src.estimation.inference import (
    cluster_bootstrap_ci,
    holm_correction,
    permutation_null,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "in_dist_tau.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

SAMPLERS = ("deepwalk", "node2vec_bfs", "node2vec_dfs", "graphsage")


def _pool_folds(
    schema_name: str,
    num_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool test pairs across all folds. Returns (pairs, labels, fold_ids)."""
    all_pairs, all_labels, all_folds = [], [], []
    for k in range(num_folds):
        path = SPLITS_DIR / f"{schema_name}_fold{k}.npz"
        fold = load_fold(path)
        pairs, labels = stack_pos_neg(fold.test_pos, fold.test_neg)
        all_pairs.append(pairs)
        all_labels.append(labels)
        all_folds.append(np.full(len(pairs), k, dtype=np.int64))
    return (
        np.concatenate(all_pairs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_folds, axis=0),
    )


def _build_feature_matrices(
    pairs: np.ndarray,
    text_emb: np.ndarray,
    text_idx: dict[int, int],
    struct_emb: np.ndarray,
    struct_idx: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    X_T = hadamard_features(pairs, text_emb, text_idx)
    X_S = hadamard_features(pairs, struct_emb, struct_idx)
    X_TS = np.concatenate([X_T, X_S], axis=1)
    return X_T, X_TS


def run_schema(
    schema: Schema,
    text_emb: np.ndarray,
    text_idx: dict[int, int],
    samplers: tuple[str, ...] = SAMPLERS,
    permutation_B: int = 50,
    bootstrap_B: int = 500,
    results_path: Path | None = None,
    accumulator: dict | None = None,
) -> dict:
    pairs, labels, folds = _pool_folds(schema.name)
    print(f"\n=== {schema.name}: {len(pairs):,} pooled pairs "
          f"({int(labels.sum()):,} pos) ===")

    per_sampler: dict[str, dict] = {}
    p_values: dict[str, float] = {}

    for sampler in samplers:
        emb_path = EMBEDDING_DIR / sampler / f"{schema.name}.npz"
        if not emb_path.exists():
            print(f"  {sampler}: missing embeddings at {emb_path}, skipping")
            continue

        struct_emb, struct_idx = load_embedding(emb_path)
        X_T, X_TS = _build_feature_matrices(
            pairs, text_emb, text_idx, struct_emb, struct_idx
        )
        print(f"  {sampler}: X_T={X_T.shape}, X_TS={X_TS.shape}")

        dml = cross_fit_dml(X_T, X_TS, labels, folds)
        ci = cluster_bootstrap_ci(
            dml.tau_per_pair,
            node_u=pairs[:, 0],
            node_v=pairs[:, 1],
            B=bootstrap_B,
        )
        perm = permutation_null(
            X_T, X_TS, labels, folds,
            struct_slice=slice(X_T.shape[1], None),
            B=permutation_B,
            verbose=False,
        )

        per_sampler[sampler] = {
            "tau_bar": dml.tau_bar,
            "auc_T": dml.auc_T,
            "auc_TS": dml.auc_TS,
            "auc_gap": dml.auc_TS - dml.auc_T,
            "ci_lower": ci.lower,
            "ci_upper": ci.upper,
            "ci_se": ci.se,
            "permutation_p_auc": perm["p_value_auc_one_sided"],
            "permutation_p_tau": perm["p_value_tau_two_sided"],
            "n_pairs": int(len(pairs)),
        }
        # Use AUC-gap p-value as the primary significance statistic
        p_values[sampler] = perm["p_value_auc_one_sided"]

        print(f"    τ̄={dml.tau_bar:+.4f} "
              f"CI=[{ci.lower:+.4f}, {ci.upper:+.4f}] "
              f"AUC_T={dml.auc_T:.3f} AUC_TS={dml.auc_TS:.3f} "
              f"(+{dml.auc_TS - dml.auc_T:.4f}) "
              f"p_AUC={perm['p_value_auc_one_sided']:.3f} "
              f"p_τ={perm['p_value_tau_two_sided']:.3f}")

        # Write partial results after each sampler so a crash doesn't lose everything
        if results_path is not None and accumulator is not None:
            accumulator[schema.name] = per_sampler
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(accumulator, indent=2))

    # Holm correction within schema on AUC-gap p-values
    if p_values:
        corrected = holm_correction(p_values)
        for k, p_adj in corrected.items():
            per_sampler[k]["permutation_p_auc_holm"] = p_adj

    return per_sampler


def main() -> None:
    kg = load_primekg()
    del kg  # only needed to warm the cache; we don't use it directly

    print("Loading Word2Vec text embeddings...")
    text_emb, text_idx = load_embedding(EMBEDDING_DIR / "word2vec_200d.npz")
    print(f"  text_emb shape: {text_emb.shape}")

    results: dict = {}
    for schema in ALL_SCHEMAS:
        results[schema.name] = run_schema(
            schema, text_emb, text_idx,
            results_path=RESULTS_PATH,
            accumulator=results,
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
