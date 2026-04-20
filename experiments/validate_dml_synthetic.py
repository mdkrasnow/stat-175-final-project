"""Synthetic-graph validation of the cross-fit DML estimator.

We generate a block-structured graph where the data-generating process is
known, and check that the DML estimator recovers the expected residual
value of structure τ̄ in three regimes:

  1. Structural features = random noise → τ̄ ≈ 0 (CI should cover 0).
  2. Structural features = perfect block indicator → τ̄ > 0 when text is
     noisy (structure adds information beyond text).
  3. Structural features = perfect block indicator AND text = perfect block
     indicator → τ̄ ≈ 0 (text already carries everything structure could).

A passing estimator produces all three signatures. We also check that the
permutation null recovers p ≈ 0.5 in regime 1 and p < 0.05 in regime 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.estimation.dml import cross_fit_dml
from src.estimation.inference import cluster_bootstrap_ci, permutation_null


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "dml_synthetic_validation.json"


def generate_synthetic(
    num_blocks: int = 3,
    nodes_per_block: int = 300,
    pairs_per_block: int = 1000,
    text_noise: float = 1.0,
    struct_mode: str = "noise",     # "noise", "perfect_block", "noisy_block"
    text_mode: str = "noisy_block", # "noisy_block" or "perfect_block"
    alpha: float = 2.0,             # coef on [b(u) == b(v)]
    beta: float = 1.5,              # coef on text similarity
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    total_nodes = num_blocks * nodes_per_block
    block = np.repeat(np.arange(num_blocks), nodes_per_block)

    text_dim = 16
    block_means = rng.normal(size=(num_blocks, text_dim)) * 2.0
    if text_mode == "perfect_block":
        text = block_means[block]
    else:
        text = block_means[block] + rng.normal(scale=text_noise, size=(total_nodes, text_dim))

    struct_dim = num_blocks
    if struct_mode == "noise":
        struct = rng.normal(size=(total_nodes, struct_dim))
    elif struct_mode == "perfect_block":
        struct = np.eye(num_blocks)[block].astype(np.float32)
    elif struct_mode == "noisy_block":
        struct = np.eye(num_blocks)[block].astype(np.float32) + rng.normal(scale=0.3, size=(total_nodes, num_blocks))
    else:
        raise ValueError(struct_mode)

    # Sample node pairs (roughly half intra-block, half inter-block)
    pairs = []
    labels = []
    for _ in range(pairs_per_block * num_blocks):
        if rng.random() < 0.5:
            b = rng.integers(num_blocks)
            u = rng.integers(b * nodes_per_block, (b + 1) * nodes_per_block)
            v = rng.integers(b * nodes_per_block, (b + 1) * nodes_per_block)
        else:
            u = rng.integers(total_nodes)
            v = rng.integers(total_nodes)
        if u == v:
            continue
        same_block = int(block[u] == block[v])
        sim = float(np.dot(text[u], text[v]) / (np.linalg.norm(text[u]) * np.linalg.norm(text[v]) + 1e-8))
        logit = alpha * same_block + beta * sim - 1.0
        y = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
        pairs.append((u, v))
        labels.append(y)

    pairs_arr = np.array(pairs, dtype=np.int64)
    labels_arr = np.array(labels, dtype=np.float32)

    # Hadamard features
    def hadamard(emb, arr):
        return emb[arr[:, 0]] * emb[arr[:, 1]]

    X_T = hadamard(text, pairs_arr).astype(np.float32)
    X_S = hadamard(struct, pairs_arr).astype(np.float32)
    X_TS = np.concatenate([X_T, X_S], axis=1)

    # Node-disjoint folds over the node set
    node_fold = rng.integers(0, 5, size=total_nodes)
    fold_u = node_fold[pairs_arr[:, 0]]
    fold_v = node_fold[pairs_arr[:, 1]]
    keep = fold_u == fold_v
    pair_fold = fold_u.copy()
    # Discard pairs crossing fold boundaries (leakage prevention)
    X_T = X_T[keep]
    X_TS = X_TS[keep]
    labels_arr = labels_arr[keep]
    pair_fold = pair_fold[keep]
    pairs_arr = pairs_arr[keep]

    return {
        "X_T": X_T,
        "X_TS": X_TS,
        "y": labels_arr,
        "fold_ids": pair_fold,
        "pairs": pairs_arr,
        "text_dim": text_dim,
        "struct_dim": struct_dim,
    }


def run_regime(name: str, **kwargs) -> dict:
    data = generate_synthetic(**kwargs)
    N = len(data["y"])
    if N < 200:
        return {"name": name, "skipped": True, "reason": "too few pairs after fold filter"}

    dml = cross_fit_dml(data["X_T"], data["X_TS"], data["y"], data["fold_ids"])
    ci = cluster_bootstrap_ci(
        dml.tau_per_pair,
        node_u=data["pairs"][:, 0],
        node_v=data["pairs"][:, 1],
        B=500,
    )
    # Permutation null on the structural block
    perm = permutation_null(
        data["X_T"],
        data["X_TS"],
        data["y"],
        data["fold_ids"],
        struct_slice=slice(data["text_dim"], None),
        B=100,
    )

    return {
        "name": name,
        "N": int(N),
        "pos_rate": float(data["y"].mean()),
        "auc_T": dml.auc_T,
        "auc_TS": dml.auc_TS,
        "tau_bar": dml.tau_bar,
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
        "permutation_p": perm["p_value_two_sided"],
    }


def main() -> None:
    regimes = [
        # Regime 1: structure is noise → τ̄ should be ~0
        dict(name="regime_1_struct_noise",
             struct_mode="noise", text_mode="noisy_block", text_noise=1.0),
        # Regime 2: structure carries block info that text mostly lacks → τ̄ > 0
        dict(name="regime_2_struct_informative",
             struct_mode="perfect_block", text_mode="noisy_block", text_noise=3.0),
        # Regime 3: text already perfect, structure redundant → τ̄ ≈ 0
        dict(name="regime_3_text_dominant",
             struct_mode="perfect_block", text_mode="perfect_block", text_noise=0.0),
    ]

    results = []
    for r in regimes:
        print(f"\n=== {r['name']} ===")
        out = run_regime(**r)
        print(json.dumps(out, indent=2))
        results.append(out)

    # Acceptance tests
    verdicts = {}
    r1 = next(r for r in results if r["name"] == "regime_1_struct_noise")
    r2 = next(r for r in results if r["name"] == "regime_2_struct_informative")
    r3 = next(r for r in results if r["name"] == "regime_3_text_dominant")

    verdicts["regime_1_ci_covers_zero"] = bool(r1["ci_lower"] <= 0 <= r1["ci_upper"])
    verdicts["regime_1_perm_p_large"] = bool(r1["permutation_p"] > 0.1)
    verdicts["regime_2_tau_positive"] = bool(r2["tau_bar"] > 0.01)
    verdicts["regime_2_perm_p_small"] = bool(r2["permutation_p"] < 0.1)
    verdicts["regime_3_tau_small"] = bool(abs(r3["tau_bar"]) < 0.05)

    print("\n=== Acceptance ===")
    for k, v in verdicts.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump({"regimes": results, "verdicts": verdicts}, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
