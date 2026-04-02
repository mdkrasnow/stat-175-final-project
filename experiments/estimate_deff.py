"""Estimate the effective embedding dimension D_eff per hop count.

The MAFC model assumes independent noise with variance 1/D. In reality,
candidates in an h-hop neighborhood are correlated. We measure the
average pairwise cosine similarity (ρ_h) among candidates and compute:

D_eff = D * (1 - ρ_h)

This corrects the noise variance to σ²_eff = 1/D_eff, giving a
tighter theoretical prediction.
"""

import sys
import os
import json
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def mafc_probability(delta_normalized, M):
    """P(correct) for MAFC with M candidates."""
    def integrand(x):
        return ndtr(x + delta_normalized) ** (M - 1) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    result, _ = quad(integrand, -10, 10 + delta_normalized, limit=200)
    return result


def main():
    from src.data.stark_loader import load_stark_dataset
    from src.retrievers.shared_index import SharedIndex

    print("Loading PrimeKG...")
    graph, qa_dataset = load_stark_dataset("prime")

    print("Building shared index...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Index built in {time.time()-t0:.0f}s")

    with open(os.path.join(RESULTS_DIR, "prime_hop_stratification.json")) as f:
        hop_bins = json.load(f)

    D = shared.embeddings.shape[1]
    d_bar = 62.6
    MAX_PER_BIN = 100  # fewer samples since we're computing pairwise sims

    print(f"\nD = {D}, d̄ = {d_bar}")
    print("\n--- Measuring candidate correlations ρ_h ---")

    results = {}

    for bin_name, items in hop_bins.items():
        if not items:
            continue
        items = items[:MAX_PER_BIN]

        all_rho = []
        all_delta_raw = []
        all_noise_var = []

        for item in items:
            idx = item["index"]
            query, _, gold_ids, _ = qa_dataset[idx]
            q_emb = shared.encode_query(query)

            # Get top-50 candidates from FAISS (these are the actual competitors)
            scores_arr, indices_arr = shared.index.search(q_emb, 50)
            candidate_ids = [shared.node_ids[i] for i in indices_arr[0]]
            candidate_embs = shared.embeddings[indices_arr[0]]

            # Pairwise cosine similarity among candidates
            # (candidates are already unit-normalized)
            pairwise = candidate_embs @ candidate_embs.T
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones(pairwise.shape, dtype=bool), k=1)
            pairwise_vals = pairwise[mask]
            avg_rho = float(np.mean(pairwise_vals))
            all_rho.append(avg_rho)

            # Also measure variance of candidate scores
            candidate_scores = scores_arr[0]
            all_noise_var.append(float(np.var(candidate_scores)))

            # Signal margin
            for gid in gold_ids:
                if gid in shared.node_id_to_idx:
                    g_emb = shared.embeddings[shared.node_id_to_idx[gid]:shared.node_id_to_idx[gid]+1]
                    gold_sim = float(q_emb @ g_emb.T)
                    noise_mean = float(np.mean(candidate_scores))
                    all_delta_raw.append(gold_sim - noise_mean)

        avg_rho_bin = float(np.mean(all_rho))
        avg_noise_var = float(np.mean(all_noise_var))
        avg_delta = float(np.mean(all_delta_raw)) if all_delta_raw else 0
        noise_std = float(np.sqrt(avg_noise_var))
        delta_norm = avg_delta / noise_std if noise_std > 0 else 0

        # Effective dimension
        D_eff = D * (1 - avg_rho_bin) if avg_rho_bin < 1 else 1
        sigma_eff = 1 / np.sqrt(D_eff)
        delta_norm_eff = avg_delta / sigma_eff if sigma_eff > 0 else 0

        results[bin_name] = {
            "avg_rho": avg_rho_bin,
            "D_eff": float(D_eff),
            "avg_delta_raw": avg_delta,
            "noise_std_empirical": noise_std,
            "delta_norm_standard": delta_norm,
            "delta_norm_corrected": delta_norm_eff,
        }

        print(f"  {bin_name}: ρ={avg_rho_bin:.4f}, D_eff={D_eff:.1f}, "
              f"δ_raw={avg_delta:.4f}, δ_norm(D)={delta_norm:.2f}, δ_norm(D_eff)={delta_norm_eff:.2f}")

    # --- Compute corrected h* and MAFC curves ---
    print("\n--- Corrected MAFC predictions ---")
    hop_labels = ["1-hop", "2-hop", "3-hop", "4-hop", "5-hop", "6+-hop"]
    hop_nums = [1, 2, 3, 4, 5, 6]

    # Load empirical data
    import glob
    exp_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "prime_experiment_*.json")))
    with open(exp_files[-1]) as f:
        exp = json.load(f)

    empirical_hit1 = []
    for h in hop_labels:
        if h in exp["results"]:
            vals = [exp["results"][h][s]["Hit@1"] for s in exp["results"][h]]
            empirical_hit1.append(np.mean(vals))
        else:
            empirical_hit1.append(0)

    # Use 1-hop δ_raw and per-bin ρ for corrected predictions
    delta_raw_1hop = results["1-hop"]["avg_delta_raw"]

    theory_standard = []
    theory_corrected = []
    theory_empirical_sigma = []

    for i, (h_label, h_num) in enumerate(zip(hop_labels, hop_nums)):
        M = d_bar ** h_num

        # Standard model: σ = 1/√D
        delta_norm_std = delta_raw_1hop * np.sqrt(D)
        p_std = mafc_probability(delta_norm_std, max(2, int(M)))
        theory_standard.append(p_std)

        # Corrected model: σ = 1/√D_eff
        if h_label in results:
            D_eff = results[h_label]["D_eff"]
            delta_norm_corr = delta_raw_1hop * np.sqrt(D_eff)
        else:
            delta_norm_corr = delta_norm_std
        p_corr = mafc_probability(delta_norm_corr, max(2, int(M)))
        theory_corrected.append(p_corr)

        # Model with empirically measured noise σ
        if h_label in results:
            sigma_emp = results[h_label]["noise_std_empirical"]
            delta_norm_emp = delta_raw_1hop / sigma_emp if sigma_emp > 0 else 0
        else:
            delta_norm_emp = delta_norm_std
        # Use actual top-50 candidates as M (more realistic)
        p_emp = mafc_probability(delta_norm_emp, 50)
        theory_empirical_sigma.append(p_emp)

        print(f"  {h_label}: M={int(M):>12}, "
              f"P(std)={p_std:.4f}, P(D_eff)={p_corr:.4f}, P(emp σ, M=50)={p_emp:.4f}, "
              f"empirical={empirical_hit1[i]:.4f}")

    # Compute h* for corrected model
    rho_avg = np.mean([results[h]["avg_rho"] for h in hop_labels if h in results])
    D_eff_avg = D * (1 - rho_avg)
    hstar_std = (delta_raw_1hop**2 * D) / (2 * np.log(d_bar))
    hstar_corr = (delta_raw_1hop**2 * D_eff_avg) / (2 * np.log(d_bar))

    print(f"\n  Average ρ across bins: {rho_avg:.4f}")
    print(f"  Average D_eff: {D_eff_avg:.1f}")
    print(f"  h* (standard, D={D}): {hstar_std:.1f}")
    print(f"  h* (corrected, D_eff={D_eff_avg:.0f}): {hstar_corr:.1f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Left: Theory vs Empirical
    ax = axes[0]
    ax.plot(hop_nums, empirical_hit1, "o-", label="Empirical Hit@1",
            color="#e74c3c", linewidth=2.5, markersize=8)
    ax.plot(hop_nums, theory_empirical_sigma, "s--",
            label="MAFC (empirical σ, M=50)",
            color="#2ecc71", linewidth=2, markersize=7)
    ax.plot(hop_nums, theory_corrected, "^:",
            label=f"MAFC (D_eff corrected)",
            color="#9b59b6", linewidth=2, markersize=7)
    ax.plot(hop_nums, theory_standard, "d-.",
            label=f"MAFC (independent noise, M=d̄ʰ)",
            color="#3498db", linewidth=1.5, markersize=6, alpha=0.5)
    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("P(correct retrieval)", fontsize=12)
    ax.set_title("Theory vs Empirical: PrimeKG", fontsize=13)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: Correlation by hop count
    ax = axes[1]
    rhos = [results[h]["avg_rho"] for h in hop_labels if h in results]
    d_effs = [results[h]["D_eff"] for h in hop_labels if h in results]
    valid_hops = [h for h in hop_nums if hop_labels[h-1] in results]
    ax.plot(valid_hops, rhos, "o-", color="#e74c3c", linewidth=2, markersize=8, label="Avg ρ (pairwise cosine sim)")
    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("Average Pairwise Correlation ρ", fontsize=12)
    ax.set_title("Candidate Correlation by Hop Count", fontsize=13)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: D_eff by hop count
    ax = axes[2]
    ax.plot(valid_hops, d_effs, "s-", color="#3498db", linewidth=2, markersize=8, label="D_eff = D(1-ρ)")
    ax.axhline(y=D, color="gray", linestyle=":", alpha=0.5, label=f"D = {D}")
    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("Effective Dimension", fontsize=12)
    ax.set_title("Effective Embedding Dimension", fontsize=13)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "theory_corrected.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    # Save results
    output = {
        "D": D, "d_bar": d_bar,
        "delta_raw_1hop": float(delta_raw_1hop),
        "rho_avg": float(rho_avg),
        "D_eff_avg": float(D_eff_avg),
        "hstar_standard": float(hstar_std),
        "hstar_corrected": float(hstar_corr),
        "per_bin": results,
        "empirical_hit1": empirical_hit1,
        "theory_standard": theory_standard,
        "theory_corrected": theory_corrected,
        "theory_empirical_sigma": theory_empirical_sigma,
    }
    with open(os.path.join(RESULTS_DIR, "theory_deff_analysis.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {os.path.join(RESULTS_DIR, 'theory_deff_analysis.json')}")


if __name__ == "__main__":
    main()
