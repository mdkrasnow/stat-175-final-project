"""Estimate the signal margin δ from empirical data.

For each QA pair, compute:
- The cosine similarity between the query embedding and the gold answer node embedding
- The cosine similarity between the query embedding and random noise candidates
- δ = mean(gold similarity) - mean(noise similarity), normalized by noise std

Then use δ to predict h* and plot the theoretical MAFC curve against empirical Hit@1.
"""

import sys
import os
import json
import time

import numpy as np
from scipy import stats as sp_stats
from scipy.special import ndtr  # Phi (normal CDF)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def mafc_probability(delta_normalized, M):
    """Compute P(correct retrieval) for the M-alternative forced choice.

    The signal candidate has similarity delta_normalized above the noise mean,
    measured in units of the noise standard deviation. There are M-1 noise candidates.

    P(correct) = integral Phi(x + delta_normalized)^{M-1} * phi(x) dx

    Uses numerical integration.
    """
    from scipy.integrate import quad

    def integrand(x):
        return ndtr(x + delta_normalized) ** (M - 1) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    result, _ = quad(integrand, -10, 10 + delta_normalized, limit=200)
    return result


def main():
    # Load PrimeKG data
    from src.data.stark_loader import load_stark_dataset
    from src.retrievers.shared_index import SharedIndex

    print("Loading PrimeKG...")
    graph, qa_dataset = load_stark_dataset("prime")

    print("Building shared index...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Index built in {time.time()-t0:.0f}s")

    # Load hop stratification
    with open(os.path.join(RESULTS_DIR, "prime_hop_stratification.json")) as f:
        hop_bins = json.load(f)

    # --- Estimate δ per hop bin ---
    print("\n--- Estimating δ (signal margin) ---")
    D = shared.embeddings.shape[1]  # embedding dimension
    print(f"Embedding dimension D = {D}")

    delta_per_bin = {}
    noise_std_per_bin = {}
    gold_sims_per_bin = {}
    noise_sims_per_bin = {}

    MAX_PER_BIN = 200

    for bin_name, items in hop_bins.items():
        if not items:
            continue
        items = items[:MAX_PER_BIN]

        gold_similarities = []
        noise_similarities = []

        for item in items:
            idx = item["index"]
            query, _, gold_ids, _ = qa_dataset[idx]
            q_emb = shared.encode_query(query)

            # Gold answer similarity
            for gid in gold_ids:
                if gid in shared.node_id_to_idx:
                    g_emb = shared.embeddings[shared.node_id_to_idx[gid]:shared.node_id_to_idx[gid]+1]
                    sim = float(q_emb @ g_emb.T)
                    gold_similarities.append(sim)

            # Random noise similarities (sample 50 random nodes)
            random_indices = np.random.choice(len(shared.node_ids), size=50, replace=False)
            for ri in random_indices:
                n_emb = shared.embeddings[ri:ri+1]
                sim = float(q_emb @ n_emb.T)
                noise_similarities.append(sim)

        gold_sims = np.array(gold_similarities)
        noise_sims = np.array(noise_similarities)

        noise_mean = np.mean(noise_sims)
        noise_std = np.std(noise_sims)
        gold_mean = np.mean(gold_sims)

        # δ in raw similarity units
        delta_raw = gold_mean - noise_mean
        # δ normalized by noise std (this is the "delta_normalized" for MAFC)
        delta_norm = delta_raw / noise_std if noise_std > 0 else 0

        delta_per_bin[bin_name] = {
            "delta_raw": float(delta_raw),
            "delta_normalized": float(delta_norm),
            "gold_mean": float(gold_mean),
            "noise_mean": float(noise_mean),
            "noise_std": float(noise_std),
            "n_gold": len(gold_sims),
            "n_noise": len(noise_sims),
        }
        gold_sims_per_bin[bin_name] = gold_sims
        noise_sims_per_bin[bin_name] = noise_sims

        print(f"  {bin_name}: δ_raw={delta_raw:.4f}, δ_norm={delta_norm:.2f}, "
              f"gold_mean={gold_mean:.4f}, noise_mean={noise_mean:.4f}, noise_std={noise_std:.4f}")

    # --- Compute overall δ (averaged across bins, weighted by n) ---
    all_gold = np.concatenate([gold_sims_per_bin[b] for b in gold_sims_per_bin])
    all_noise = np.concatenate([noise_sims_per_bin[b] for b in noise_sims_per_bin])
    overall_delta_raw = np.mean(all_gold) - np.mean(all_noise)
    overall_noise_std = np.std(all_noise)
    overall_delta_norm = overall_delta_raw / overall_noise_std
    d_bar = 62.6  # PrimeKG mean degree

    print(f"\n  Overall: δ_raw={overall_delta_raw:.4f}, δ_norm={overall_delta_norm:.2f}, "
          f"noise_std={overall_noise_std:.4f}")
    print(f"  Predicted h* = δ_norm² * D / (2 * log(d̄)) = "
          f"{overall_delta_norm**2 * D / (2 * np.log(d_bar)):.2f}")

    # --- Compute per-hop-count δ for the theoretical curve ---
    # Use the 1-hop δ as the "true" signal margin (least contaminated by noise)
    delta_1hop = delta_per_bin["1-hop"]["delta_normalized"]
    print(f"\n  Using 1-hop δ_norm = {delta_1hop:.2f} for theoretical predictions")
    print(f"  Predicted h* = {delta_1hop**2 * D / (2 * np.log(d_bar)):.2f}")

    # --- Plot: Theoretical MAFC curve vs empirical Hit@1 ---
    print("\n--- Computing MAFC predictions ---")

    hop_labels = ["1-hop", "2-hop", "3-hop", "4-hop", "5-hop", "6+-hop"]
    hop_nums = [1, 2, 3, 4, 5, 6]

    # Load empirical Hit@1 from experiment results
    import glob
    exp_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "prime_experiment_*.json")))
    with open(exp_files[-1]) as f:
        exp = json.load(f)

    # Empirical Hit@1 (averaged across strategies)
    empirical_hit1 = []
    for h in hop_labels:
        if h in exp["results"]:
            vals = [exp["results"][h][s]["Hit@1"] for s in exp["results"][h]]
            empirical_hit1.append(np.mean(vals))
        else:
            empirical_hit1.append(0)

    # Theoretical MAFC P(correct) for each hop count
    # Use per-bin δ for a more accurate comparison
    theoretical_pcorrect_perbin = []
    theoretical_pcorrect_fixed = []

    for i, (h_label, h_num) in enumerate(zip(hop_labels, hop_nums)):
        M = d_bar ** h_num  # number of candidates

        # Per-bin δ
        if h_label in delta_per_bin:
            dn = delta_per_bin[h_label]["delta_normalized"]
        else:
            dn = 0
        p_perbin = mafc_probability(dn, max(2, int(M)))
        theoretical_pcorrect_perbin.append(p_perbin)

        # Fixed δ (from 1-hop)
        p_fixed = mafc_probability(delta_1hop, max(2, int(M)))
        theoretical_pcorrect_fixed.append(p_fixed)

        print(f"  {h_label}: M={int(M):>10}, δ_norm={dn:.2f}, "
              f"P(correct|per-bin δ)={p_perbin:.4f}, P(correct|fixed δ)={p_fixed:.4f}, "
              f"empirical Hit@1={empirical_hit1[i]:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Theory vs Empirical
    ax = axes[0]
    ax.plot(hop_nums, empirical_hit1, "o-", label="Empirical Hit@1 (avg across strategies)",
            color="#e74c3c", linewidth=2.5, markersize=8)
    ax.plot(hop_nums, theoretical_pcorrect_fixed, "s--",
            label=f"MAFC theory (fixed δ={delta_1hop:.2f})",
            color="#3498db", linewidth=2, markersize=7)
    ax.plot(hop_nums, theoretical_pcorrect_perbin, "^:",
            label="MAFC theory (per-bin δ)",
            color="#2ecc71", linewidth=2, markersize=7)
    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("P(correct retrieval)", fontsize=12)
    ax.set_title("Theoretical vs Empirical: PrimeKG", fontsize=13)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, max(max(empirical_hit1), max(theoretical_pcorrect_fixed)) * 1.2)

    # Right: δ decay across hop counts
    ax = axes[1]
    deltas = [delta_per_bin[h]["delta_normalized"] if h in delta_per_bin else 0 for h in hop_labels]
    noise_floors = [np.sqrt(2 * h * np.log(d_bar) / D) for h in hop_nums]
    ax.plot(hop_nums, deltas, "o-", label="Measured δ (normalized)", color="#e74c3c", linewidth=2, markersize=8)
    ax.plot(hop_nums, noise_floors, "s--", label="Noise floor √(2h log d̄ / D)", color="#3498db", linewidth=2, markersize=7)
    ax.axhline(y=delta_1hop, color="gray", linestyle=":", alpha=0.5, label=f"1-hop δ = {delta_1hop:.2f}")
    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("Signal / Noise (normalized)", fontsize=12)
    ax.set_title("Signal Margin vs Noise Floor", fontsize=13)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "theory_vs_empirical.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    # --- Save all results ---
    theory_results = {
        "embedding_dim": D,
        "mean_degree": d_bar,
        "delta_per_bin": delta_per_bin,
        "overall_delta_raw": float(overall_delta_raw),
        "overall_delta_normalized": float(overall_delta_norm),
        "overall_noise_std": float(overall_noise_std),
        "delta_1hop": float(delta_1hop),
        "predicted_hstar": float(delta_1hop**2 * D / (2 * np.log(d_bar))),
        "theoretical_pcorrect_fixed": theoretical_pcorrect_fixed,
        "theoretical_pcorrect_perbin": theoretical_pcorrect_perbin,
        "empirical_hit1": empirical_hit1,
    }
    with open(os.path.join(RESULTS_DIR, "theory_analysis.json"), "w") as f:
        json.dump(theory_results, f, indent=2)
    print(f"Saved: {os.path.join(RESULTS_DIR, 'theory_analysis.json')}")


if __name__ == "__main__":
    main()
