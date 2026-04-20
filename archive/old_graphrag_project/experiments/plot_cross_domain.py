"""Generate cross-domain comparison plots: PrimeKG vs Amazon."""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_experiment(dataset):
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{dataset}_experiment_*.json")))
    with open(files[-1]) as f:
        return json.load(f)


def main():
    prime = load_experiment("prime")
    amazon = load_experiment("amazon")

    hop_labels = ["1-hop", "2-hop", "3-hop", "4-hop", "5-hop", "6+-hop"]
    hop_nums = [1, 2, 3, 4, 5, 6]
    strategies = ["node_centric", "subgraph_centric"]
    strat_labels = {"node_centric": "Node-centric", "subgraph_centric": "Subgraph-centric"}

    # --- Figure 1: Cross-domain Hit@10 comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    colors = {"node_centric": "#e74c3c", "subgraph_centric": "#2ecc71",
              "path_centric": "#3498db", "hybrid_mor": "#9b59b6"}
    all_strats = ["node_centric", "path_centric", "subgraph_centric", "hybrid_mor"]
    all_labels = {"node_centric": "Node-centric", "path_centric": "Path-centric",
                  "subgraph_centric": "Subgraph-centric", "hybrid_mor": "Hybrid"}

    for ax, (data, title, mean_deg) in zip(axes, [
        (prime, "STaRK-PrimeKG (mean degree 62.6)", 62.6),
        (amazon, "STaRK-Amazon (mean degree 9.1)", 9.1),
    ]):
        for strat in all_strats:
            vals = []
            for h in hop_labels:
                if h in data["results"] and strat in data["results"][h]:
                    vals.append(data["results"][h][strat].get("Hit@10", 0))
                else:
                    vals.append(0)
            ax.plot(hop_nums[:len(vals)], vals, "o-", label=all_labels[strat],
                    color=colors[strat], linewidth=2, markersize=6)
        ax.set_xlabel("Query Hop Count")
        ax.set_ylabel("Hit@10")
        ax.set_title(title)
        ax.set_xticks(hop_nums)
        ax.set_xticklabels(hop_labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Cliff Severity Depends on Graph Density", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cross_domain_hit10.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # --- Figure 2: Normalized drop from 1-hop ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for data, label, color, ls in [
        (prime, "PrimeKG (dense, d=62.6)", "#e74c3c", "-"),
        (amazon, "Amazon (sparse, d=9.1)", "#3498db", "--"),
    ]:
        # Average Hit@10 across all strategies
        avg_vals = []
        for h in hop_labels:
            if h in data["results"]:
                strat_vals = [data["results"][h][s].get("Hit@10", 0) for s in all_strats if s in data["results"][h]]
                avg_vals.append(np.mean(strat_vals))
            else:
                avg_vals.append(0)
        # Normalize to 1-hop
        if avg_vals[0] > 0:
            normalized = [v / avg_vals[0] for v in avg_vals]
        else:
            normalized = avg_vals
        ax.plot(hop_nums[:len(normalized)], normalized, "o-", label=label,
                color=color, linewidth=2.5, markersize=7, linestyle=ls)

    ax.set_xlabel("Query Hop Count", fontsize=12)
    ax.set_ylabel("Relative Performance (normalized to 1-hop)", fontsize=12)
    ax.set_title("Performance Degradation: Dense vs Sparse Graphs", fontsize=13)
    ax.set_xticks(hop_nums)
    ax.set_xticklabels(hop_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.annotate("50% retention", xy=(5.5, 0.52), fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cross_domain_normalized.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- Figure 3: Amazon phase transition (same format as PrimeKG) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metrics = ["Hit@1", "Hit@5", "Hit@10", "MRR"]

    for ax, metric in zip(axes, metrics):
        for strat in all_strats:
            vals = []
            for h in hop_labels:
                if h in amazon["results"] and strat in amazon["results"][h]:
                    vals.append(amazon["results"][h][strat].get(metric, 0))
                else:
                    vals.append(0)
            ax.plot(hop_nums[:len(vals)], vals, "o-", label=all_labels[strat],
                    color=colors[strat], linewidth=2, markersize=6)
        ax.set_xlabel("Query Hop Count")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Query Complexity")
        ax.legend(fontsize=8)
        ax.set_xticks(hop_nums)
        ax.set_xticklabels(hop_labels, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Retrieval Performance — STaRK-Amazon", fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "amazon_phase_transition.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
