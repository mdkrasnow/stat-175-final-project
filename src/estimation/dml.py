"""Cross-fit double-ML estimator for residual predictive value of graph structure.

Estimand (per pair):
    τ_i = E[Y | text_i, struct_i] - E[Y | text_i]

We fit two nuisance models:
    η_T  : predicts Y from text-only features X_T
    η_TS : predicts Y from concat(X_T, X_S)

with K-fold cross-fitting. For each held-out fold, τ̂_i = η_TS(x_i) - η_T(x_i)
is computed on pairs not seen by the model during training. Aggregate
τ̄ = mean_i τ̂_i is an unbiased plug-in estimator of E[τ] under standard DML
regularity conditions.

Folds must be **node-disjoint** (see src/data/splits.py): a node's embedding
cannot appear in both the fold's training and test edge sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

try:
    from xgboost import XGBClassifier
    _DEFAULT_LEARNER = lambda: XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        verbosity=0,
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _DEFAULT_LEARNER = lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1
    )


@runtime_checkable
class BinaryClassifier(Protocol):
    def fit(self, X, y): ...
    def predict_proba(self, X) -> np.ndarray: ...


@dataclass
class DMLResult:
    tau_per_pair: np.ndarray        # shape [N], τ̂_i per pair (in original order)
    tau_bar: float                  # mean τ̂
    eta_T: np.ndarray               # shape [N], η̂_T predictions
    eta_TS: np.ndarray              # shape [N], η̂_TS predictions
    fold_assignments: np.ndarray    # shape [N], int fold each pair was held out in
    tau_bar_by_fold: list[float]    # per-fold means (useful for CIs)
    auc_T: float                    # out-of-fold AUC of η̂_T alone
    auc_TS: float                   # out-of-fold AUC of η̂_TS


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _clip_logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cross_fit_dml(
    X_T: np.ndarray,
    X_TS: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,            # shape [N], integer fold index per pair
    make_learner=None,
    scale: str = "logit",            # "logit" or "probability"
) -> DMLResult:
    """Cross-fit DML.

    Fits ``make_learner()`` twice per fold (once for η_T, once for η_TS) on
    non-held-out pairs, predicts on the held-out pairs. Returns DMLResult.

    ``scale``:
        "logit"       — τ̂_i = logit(η̂_TS) - logit(η̂_T). More stable under
                        miscalibrated probability estimates (default).
        "probability" — τ̂_i = η̂_TS - η̂_T. Keeps the natural probability
                        scale but can be biased toward zero when two
                        independently-fit classifiers have different
                        calibration profiles.
    """
    if make_learner is None:
        make_learner = _DEFAULT_LEARNER

    N = len(y)
    assert X_T.shape[0] == N == X_TS.shape[0] == len(fold_ids)

    eta_T = np.zeros(N, dtype=np.float32)
    eta_TS = np.zeros(N, dtype=np.float32)
    fold_means: list[float] = []

    unique_folds = sorted(np.unique(fold_ids).tolist())
    for k in unique_folds:
        train_mask = fold_ids != k
        test_mask = fold_ids == k
        if not test_mask.any() or not train_mask.any():
            continue

        learner_T = make_learner()
        learner_T.fit(X_T[train_mask], y[train_mask])
        eta_T[test_mask] = learner_T.predict_proba(X_T[test_mask])[:, 1]

        learner_TS = make_learner()
        learner_TS.fit(X_TS[train_mask], y[train_mask])
        eta_TS[test_mask] = learner_TS.predict_proba(X_TS[test_mask])[:, 1]

        if scale == "logit":
            fold_tau = _clip_logit(eta_TS[test_mask]) - _clip_logit(eta_T[test_mask])
        else:
            fold_tau = eta_TS[test_mask] - eta_T[test_mask]
        fold_means.append(float(fold_tau.mean()))

    if scale == "logit":
        tau_per_pair = _clip_logit(eta_TS) - _clip_logit(eta_T)
    elif scale == "probability":
        tau_per_pair = eta_TS - eta_T
    else:
        raise ValueError(f"Unknown scale: {scale}")

    return DMLResult(
        tau_per_pair=tau_per_pair,
        tau_bar=float(tau_per_pair.mean()),
        eta_T=eta_T,
        eta_TS=eta_TS,
        fold_assignments=fold_ids,
        tau_bar_by_fold=fold_means,
        auc_T=_auc(y, eta_T),
        auc_TS=_auc(y, eta_TS),
    )
