"""Final theory-vs-empirical comparison.

Key insight from previous runs:
- δ_raw decreases with hop count (bridge entity effect)
- The real M is the index size (129K), not d̄^h
- Noise σ ≈ 0.108 (empirically measured)

The MAFC model should use:
- Per-bin δ (measured gold - random noise similarity)
- M = index size (129,375) — the actual competition
- σ = empirically measured noise std

This gives the most honest theory-empirical comparison.
"""

import sys
import os
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def mafc_probability(delta_norm, M):
    """P(correct) in M-AFC with normalized signal delta_norm."""
    def integrand(x):
        return ndtr(x + delta_norm) ** (M - 1) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    result, _ = quad(integrand, -15, 15 + delta_norm, limit=300)
    return min(max(result, 0), 1)


def main():
    # Load theory analysis from first run
    with open(os.path.join(RESULTS_DIR, "theory_analysis.json")) as f:
        theory = json.load(f)

    # Load experiment results
    import glob
    exp_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "prime_experiment_*.json")))
    with open(exp_files[-1]) as f:
        exp = json.load(f)

    D = theory["embedding_dim"]
    d_bar = theory["mean_degree"]
    N_index = 129375  # actual index size

    hop_labels = ["1-hop", "2-hop", "3-hop", "4-hop", "5-hop", "6+-hop"]
    hop_nums = [1, 2, 3, 4, 5, 6]

    # Empirical Hit@1 (node-centric only — fairest comparison to MAFC)
    empirical_node_hit1 = []
    empirical_avg_hit1 = []
    for h in hop_labels:
        if h in exp["results"]:
            empirical_node_hit1.append(exp["results"][h]["node_centric"]["Hit@1"])
            vals = [exp["results"][h][s]["Hit@1"] for s in exp["results"][h]]
            empirical_avg_hit1.append(np.mean(vals))
        else:
            empirical_node_hit1.append(0)
            empirical_avg_hit1.append(0)

    # Per-bin δ from the first measurement (gold vs random noise)
    delta_per_bin = theory["delta_per_bin"]

    print("=" * 70)
    print("  Theory vs Empirical (Final)")
    print("=" * 70)
    print(f"  D = {D}, d̄ = {d_bar}, N_index = {N_index}")
    print()

    # --- Model 1: MAFC with M = N_index, per-bin δ ---
    print("Model 1: MAFC with M = N_index, per-bin δ")
    model1 = []
    for h_label in hop_labels:
        if h_label in delta_per_bin:
            dn = delta_per_bin[h_label]["delta_normalized"]
        else:
            dn = 0
        p = mafc_probability(dn, N_index)
        model1.append(p)
        print(f"  {h_label}: δ_norm={dn:.2f}, P(correct, M={N_index})={p:.4f}, "
              f"empirical node Hit@1={empirical_node_hit1[hop_labels.index(h_label)]:.4f}")

    # --- Model 2: MAFC with M = d̄^h, fixed δ from 1-hop ---
    print("\nModel 2: MAFC with M = d̄^h, fixed δ from 1-hop")
    delta_1hop = delta_per_bin["1-hop"]["delta_normalized"]
    model2 = []
    for h_label, h_num in zip(hop_labels, hop_nums):
        M = d_bar ** h_num
        p = mafc_probability(delta_1hop, max(2, int(M)))
        model2.append(p)
        print(f"  {h_label}: M={int(M):>12}, P(correct)={p:.4f}")

    # --- Model 3: MAFC with M = d̄^h, per-bin δ (accounts for bridge entity effect) ---
    print("\nModel 3: MAFC with M = d̄^h, per-bin δ (bridge entity + neighborhood explosion)")
    model3 = []
    for h_label, h_num in zip(hop_labels, hop_nums):
        M = d_bar ** h_num
        if h_label in delta_per_bin:
            dn = delta_per_bin[h_label]["delta_normalized"]
        else:
            dn = 0
        p = mafc_probability(dn, max(2, int(M)))
        model3.append(p)
        print(f"  {h_label}: M={int(M):>12}, δ_norm={dn:.2f}, P(correct)={p:.4f}, "
              f"empirical={empirical_node_hit1[hop_labels.index(h_label)]:.4f}")

    # --- h* predictions ---
    print("\n--- Critical hop count h* ---")
    for label, dn in [("1-hop δ", delta_1hop),
                       ("2-hop δ", delta_per_bin["2-hop"]["delta_normalized"]),
                       ("overall δ", theory["overall_delta_normalized"])]:
        hstar = dn**2 / (2 * np.log(d_bar) / D)  # = δ_norm² * D / (2 log d̄)... wait
        # h* = δ²D / (2 log d̄) where δ is raw, or h* = δ_norm² / (2 log d̄) * (σ²D)
        # With δ_norm = δ_raw/σ and σ = 1/√D: δ_norm = δ_raw * √D
        # h* condition: δ_norm > √(2h log d̄) → h* = δ_norm² / (2 log d̄)
        hstar = dn**2 / (2 * np.log(d_bar))
        print(f"  {label}: δ_norm={dn:.2f}, h* = {hstar:.1f}")

    # --- Main figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Three models vs empirical
    ax = axes[0]
    ax.plot(hop_nums, empirical_node_hit1, "o-", label="Empirical Hit@1 (node-centric)",
            color="#e74c3c", linewidth=2.5, markersize=8, zorder=5)
    ax.plot(hop_nums, model3, "s--",
            label="Theorem 1 (per-bin δ, M=d̄ʰ)",
            color="#2ecc71", linewidth=2, markersize=7)
    ax.plot(hop_nums, model1, "^:",
            label="Theorem 1 (per-bin δ, M=N)",
            color="#9b59b6", linewidth=2, markersize=7)
    ax.plot(hop_nums, model2, "d-.",
            label=f"Theorem 1 (fixed δ={delta_1hop:.1f}, M=d̄ʰ)",
            color="#3498db", linewidth=1.5, markersize=6, alpha=0.6)
    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("P(correct retrieval at rank 1)", fontsize=12)
    ax.set_title("MAFC Theorem vs Empirical Hit@1", fontsize=13)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.7)

    # Right: δ decay = the bridge entity effect
    ax = axes[1]
    deltas_raw = [delta_per_bin[h]["delta_raw"] if h in delta_per_bin else 0 for h in hop_labels]
    deltas_norm = [delta_per_bin[h]["delta_normalized"] if h in delta_per_bin else 0 for h in hop_labels]
    gold_means = [delta_per_bin[h]["gold_mean"] if h in delta_per_bin else 0 for h in hop_labels]
    noise_means = [delta_per_bin[h]["noise_mean"] if h in delta_per_bin else 0 for h in hop_labels]

    ax.plot(hop_nums, gold_means, "o-", label="Gold answer similarity", color="#e74c3c", linewidth=2, markersize=8)
    ax.plot(hop_nums, noise_means, "s-", label="Random noise similarity", color="#3498db", linewidth=2, markersize=7)
    ax.fill_between(hop_nums, noise_means, gold_means, alpha=0.15, color="#2ecc71")
    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Query", fontsize=12)
    ax.set_title("Signal Margin δ Decays with Hop Count\n(Bridge Entity Effect)", fontsize=12)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate δ values
    for i, (h, dr) in enumerate(zip(hop_nums, deltas_raw)):
        mid = (gold_means[i] + noise_means[i]) / 2
        ax.annotate(f"δ={dr:.3f}", xy=(h, mid), fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "theory_final.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")

    # Save
    output = {
        "models": {
            "model1_M_index_perbin_delta": model1,
            "model2_M_dh_fixed_delta": model2,
            "model3_M_dh_perbin_delta": model3,
        },
        "empirical_node_hit1": empirical_node_hit1,
        "empirical_avg_hit1": empirical_avg_hit1,
        "delta_per_bin": {h: delta_per_bin[h] for h in hop_labels if h in delta_per_bin},
    }
    with open(os.path.join(RESULTS_DIR, "theory_final.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {os.path.join(RESULTS_DIR, 'theory_final.json')}")


if __name__ == "__main__":
    main()
