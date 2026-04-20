"""Uncertainty quantification for DML τ̂ estimates.

Three tools:
  1. cluster_bootstrap_ci: CIs that account for nodes appearing in many pairs
  2. permutation_null:     sharp null H0 via structural-embedding shuffle
  3. holm_correction:      multi-sampler family-wise error control

All inputs are vectors of per-pair τ̂ plus the node IDs of each pair's endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    se: float
    level: float = 0.95


def cluster_bootstrap_ci(
    tau: np.ndarray,                # shape [N], per-pair τ̂
    node_u: np.ndarray,             # shape [N]
    node_v: np.ndarray,             # shape [N]
    B: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Node-cluster bootstrap: resample nodes with replacement, aggregate τ̂ over
    pairs whose endpoints are both in the resampled node set.

    This is a conservative approach — some resamples will drop many pairs. An
    alternative is weighted bootstrap where each pair gets weight proportional
    to how many times each endpoint was resampled. We implement the simpler
    node-subset variant here.
    """
    rng = np.random.default_rng(seed)
    nodes = np.unique(np.concatenate([node_u, node_v]))
    n_nodes = len(nodes)
    node_to_pos = {int(n): i for i, n in enumerate(nodes)}
    u_pos = np.array([node_to_pos[int(n)] for n in node_u])
    v_pos = np.array([node_to_pos[int(n)] for n in node_v])

    means = np.empty(B, dtype=np.float64)
    for b in range(B):
        resampled = rng.choice(n_nodes, size=n_nodes, replace=True)
        in_sample = np.zeros(n_nodes, dtype=bool)
        in_sample[resampled] = True
        mask = in_sample[u_pos] & in_sample[v_pos]
        if mask.any():
            means[b] = float(tau[mask].mean())
        else:
            means[b] = float("nan")

    means = means[~np.isnan(means)]
    alpha = 1.0 - level
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return BootstrapCI(
        mean=float(tau.mean()),
        lower=lower,
        upper=upper,
        se=float(means.std(ddof=1)),
        level=level,
    )


def permutation_null(
    X_T: np.ndarray,
    X_TS: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    struct_slice: slice,             # which columns of X_TS are structural
    B: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Permutation null for H0: structural features carry no residual signal
    beyond text.

    We shuffle the rows of the structural block of X_TS across pairs
    (breaking the link between text and structure), re-run cross-fit DML,
    and record TWO test statistics:

      - τ̄ (mean probability-scale contrast)
      - AUC gap (AUC_TS - AUC_T)

    Observed values are in the tail of their respective null distributions
    when structure adds real signal. We report p-values from both.

    The AUC-gap p-value is the primary significance metric because τ̄ is
    sensitive to per-classifier calibration drift (can be tiny and near-zero
    even when structure clearly improves ranking), producing inflated
    p-values. AUC gap is calibration-invariant.
    """
    from .dml import cross_fit_dml

    rng = np.random.default_rng(seed)

    obs = cross_fit_dml(X_T, X_TS, y, fold_ids)
    obs_tau = obs.tau_bar
    obs_auc_gap = obs.auc_TS - obs.auc_T

    X_TS_perm = X_TS.copy()
    taus = np.empty(B, dtype=np.float64)
    auc_gaps = np.empty(B, dtype=np.float64)
    for b in range(B):
        order = rng.permutation(X_TS.shape[0])
        X_TS_perm[:, struct_slice] = X_TS[order][:, struct_slice]
        res = cross_fit_dml(X_T, X_TS_perm, y, fold_ids)
        taus[b] = res.tau_bar
        auc_gaps[b] = res.auc_TS - res.auc_T
        if verbose and (b + 1) % max(1, B // 5) == 0:
            print(f"  perm {b+1}/{B}: null τ̄={taus[b]:+.4f}, "
                  f"ΔAUC={auc_gaps[b]:+.4f}")

    # Two-sided p on τ̄ and one-sided p on AUC gap (does struct improve ranking?)
    p_tau = float(((np.abs(taus) >= abs(obs_tau)).sum() + 1) / (B + 1))
    p_auc = float(((auc_gaps >= obs_auc_gap).sum() + 1) / (B + 1))
    return {
        "observed_tau_bar": float(obs_tau),
        "observed_auc_gap": float(obs_auc_gap),
        "null_tau_bars": taus.tolist(),
        "null_auc_gaps": auc_gaps.tolist(),
        "p_value_tau_two_sided": p_tau,
        "p_value_auc_one_sided": p_auc,
        "p_value_two_sided": p_tau,    # kept for back-compat; prefer p_value_auc_one_sided
        "B": B,
    }


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down FWER correction for a family of p-values.

    Returns a dict mapping each key to its corrected p-value, clipped to [0, 1].
    """
    keys = list(p_values.keys())
    raw = np.array([p_values[k] for k in keys], dtype=np.float64)
    m = len(raw)
    order = np.argsort(raw)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        p_adj = min(1.0, raw[idx] * factor)
        running_max = max(running_max, p_adj)
        adjusted[idx] = running_max
    return {k: float(a) for k, a in zip(keys, adjusted)}
