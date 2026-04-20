"""Phase 4.1: schema-generalization — residual value of structure when the
DML nuisance is trained on one schema and evaluated on another.

Protocol:
  For each held-out schema G_test and each sampler s:
    1. Fit η̂_T, η̂_{T, S_s} on pooled pairs from the OTHER three schemas
       (cross-fit over node-disjoint folds within that pooled training
       set).
    2. Evaluate on G_test's pairs using the sampler's embedding learned
       on G_test's own topology (sampler runs on G_test; the nuisance
       model is transferred).
    3. Report τ̄_s^OOD and the schema-transfer gap Δ_s = τ̄_s^OOD - τ̄_s^ID.

Notes:
  - This implementation uses a single train/test partition rather than
    K-fold within the combined train set (which would require node-disjoint
    folds *across* schemas). Simpler and still honest because the test set
    comes from a schema the model never trained on.
  - For each held-out schema we use a single combined train set; the
    embeddings for the nuisance inputs are drawn from each source schema's
    own sampler output.

Output: results/schema_transfer.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ALL_SCHEMAS, Schema
from src.data.splits import load_fold
from src.estimation.features import (
    hadamard_features,
    load_embedding,
    stack_pos_neg,
)
from src.estimation.inference import cluster_bootstrap_ci

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "schema_transfer.json"
IN_DIST_RESULTS = PROJECT_ROOT / "results" / "in_dist_tau.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
SAMPLERS = ("deepwalk", "node2vec_bfs", "node2vec_dfs", "graphsage")


def _pool_all_folds(schema_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Pool every positive and negative edge from all 5 folds of a schema —
    gives the full labeled pair set for that schema."""
    all_pairs, all_labels = [], []
    for k in range(5):
        fold = load_fold(SPLITS_DIR / f"{schema_name}_fold{k}.npz")
        # Use both train and test partitions — for transfer we want maximum
        # training signal from source schemas. (A node seen in one source
        # schema is fine to include, since evaluation is on a *different*
        # schema's held-out pairs.)
        pairs_tr, labels_tr = stack_pos_neg(fold.train_pos, fold.train_neg)
        pairs_te, labels_te = stack_pos_neg(fold.test_pos, fold.test_neg)
        all_pairs.extend([pairs_tr, pairs_te])
        all_labels.extend([labels_tr, labels_te])
    return np.concatenate(all_pairs, axis=0), np.concatenate(all_labels, axis=0)


def _build_features_for(
    schema: Schema,
    sampler: str,
    pairs: np.ndarray,
    text_emb: np.ndarray,
    text_idx: dict[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    emb_path = EMBEDDING_DIR / sampler / f"{schema.name}.npz"
    if not emb_path.exists():
        return None
    struct_emb, struct_idx = load_embedding(emb_path)
    X_T = hadamard_features(pairs, text_emb, text_idx)
    X_S = hadamard_features(pairs, struct_emb, struct_idx)
    X_TS = np.concatenate([X_T, X_S], axis=1)
    return X_T, X_TS


def _fit_learner(X, y):
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=4, verbosity=0,
    )
    clf.fit(X, y)
    return clf


def transfer_for_sampler(
    sampler: str,
    test_schema: Schema,
    train_schemas: list[Schema],
    text_emb: np.ndarray,
    text_idx: dict[int, int],
) -> dict | None:
    # Build training set: pool pairs + features from all three source schemas,
    # each using that schema's own structural embedding.
    train_X_T_list, train_X_TS_list, train_y_list = [], [], []
    for s in train_schemas:
        pairs, labels = _pool_all_folds(s.name)
        feats = _build_features_for(s, sampler, pairs, text_emb, text_idx)
        if feats is None:
            return None
        X_T, X_TS = feats
        train_X_T_list.append(X_T)
        train_X_TS_list.append(X_TS)
        train_y_list.append(labels)

    X_T_train = np.concatenate(train_X_T_list, axis=0)
    X_TS_train = np.concatenate(train_X_TS_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    eta_T = _fit_learner(X_T_train, y_train)
    eta_TS = _fit_learner(X_TS_train, y_train)

    # Evaluate on test schema
    test_pairs, test_labels = _pool_all_folds(test_schema.name)
    feats = _build_features_for(test_schema, sampler, test_pairs, text_emb, text_idx)
    if feats is None:
        return None
    X_T_te, X_TS_te = feats

    # Feature-dim mismatch check: struct dims might differ across schemas if we
    # trained different-dim samplers. Our samplers are all 128-d so this should
    # hold, but assert it.
    if X_T_te.shape[1] != X_T_train.shape[1] or X_TS_te.shape[1] != X_TS_train.shape[1]:
        print(f"    feature-dim mismatch, skipping: "
              f"T {X_T_train.shape[1]} vs {X_T_te.shape[1]}, "
              f"TS {X_TS_train.shape[1]} vs {X_TS_te.shape[1]}")
        return None

    p_T = eta_T.predict_proba(X_T_te)[:, 1]
    p_TS = eta_TS.predict_proba(X_TS_te)[:, 1]
    tau = p_TS - p_T

    ci = cluster_bootstrap_ci(
        tau, node_u=test_pairs[:, 0], node_v=test_pairs[:, 1], B=500,
    )

    from sklearn.metrics import roc_auc_score
    auc_T = float(roc_auc_score(test_labels, p_T)) if len(np.unique(test_labels)) > 1 else float("nan")
    auc_TS = float(roc_auc_score(test_labels, p_TS)) if len(np.unique(test_labels)) > 1 else float("nan")

    return {
        "tau_bar_ood": float(tau.mean()),
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
        "auc_T_ood": auc_T,
        "auc_TS_ood": auc_TS,
        "n_test_pairs": int(len(test_pairs)),
        "n_train_pairs": int(len(y_train)),
    }


def main() -> None:
    text_emb, text_idx = load_embedding(EMBEDDING_DIR / "word2vec_200d.npz")

    # Pre-load in-distribution τ̄_s for the transfer-gap Δ_s
    in_dist: dict = {}
    if IN_DIST_RESULTS.exists():
        in_dist = json.loads(IN_DIST_RESULTS.read_text())

    results: dict = {}
    for test_schema in ALL_SCHEMAS:
        print(f"\n=== held-out: {test_schema.name} ===")
        train_schemas = [s for s in ALL_SCHEMAS if s.name != test_schema.name]

        per_sampler: dict[str, dict] = {}
        for sampler in SAMPLERS:
            out = transfer_for_sampler(
                sampler, test_schema, train_schemas, text_emb, text_idx,
            )
            if out is None:
                print(f"  {sampler}: skipped (missing data)")
                continue
            tau_ood = out["tau_bar_ood"]
            tau_id = (in_dist.get(test_schema.name, {}).get(sampler, {}).get("tau_bar"))
            if tau_id is not None:
                out["tau_bar_in_dist"] = tau_id
                out["transfer_gap"] = tau_ood - tau_id
            per_sampler[sampler] = out
            gap_str = (f" Δ={out['transfer_gap']:+.4f}" if "transfer_gap" in out else "")
            print(f"  {sampler}: τ̄^OOD={tau_ood:+.4f} "
                  f"[{out['ci_lower']:+.4f}, {out['ci_upper']:+.4f}] "
                  f"AUC_T={out['auc_T_ood']:.3f} AUC_TS={out['auc_TS_ood']:.3f}"
                  f"{gap_str}")
        results[test_schema.name] = per_sampler

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
