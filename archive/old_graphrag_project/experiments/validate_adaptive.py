"""Validate the AdaptiveRetriever across a sweep of skew_threshold values.

For each threshold we evaluate on hop-stratified bins and track what
fraction of queries are routed to the focused vs. broad sub-retriever.
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.retrievers.shared_index import SharedIndex
from src.retrievers.adaptive_retriever import AdaptiveRetriever
from src.evaluation.metrics import hit_at_k, mrr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
KS = [1, 5, 10]
MAX_PER_BIN = 200
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]


def evaluate_adaptive_on_bins(retriever, qa_dataset, hop_bins):
    """Evaluate AdaptiveRetriever on each hop bin.

    Returns per-bin metrics *and* routing statistics.
    """
    results = {}
    for bin_name, items in hop_bins.items():
        if not items:
            continue
        items = items[:MAX_PER_BIN]

        hit_scores = {k: [] for k in KS}
        mrr_scores = []
        route_counts = {"focused": 0, "broad": 0}

        for item in items:
            query, _, gold_ids, _ = qa_dataset[item["index"]]
            predicted = retriever.retrieve_ids(query, top_k=max(KS))

            for k in KS:
                hit_scores[k].append(hit_at_k(predicted, gold_ids, k))
            mrr_scores.append(mrr(predicted, gold_ids))

            route_counts[retriever.last_route] += 1

        n = len(items)
        results[bin_name] = {
            f"Hit@{k}": float(np.mean(hit_scores[k])) for k in KS
        }
        results[bin_name]["MRR"] = float(np.mean(mrr_scores))
        results[bin_name]["n"] = n
        results[bin_name]["pct_focused"] = route_counts["focused"] / n
        results[bin_name]["pct_broad"] = route_counts["broad"] / n

    return results


def print_summary_table(all_results):
    """Print a readable summary across thresholds and hop bins."""
    # Collect all bin names across thresholds.
    bin_names = []
    for res in all_results.values():
        for b in res:
            if b not in bin_names:
                bin_names.append(b)

    header = f"{'Threshold':>10} | {'Bin':>12} | {'Hit@10':>7} | {'MRR':>7} | {'%Focused':>9} | {'n':>4}"
    print("\n" + "=" * len(header))
    print("ADAPTIVE RETRIEVER — THRESHOLD SWEEP")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for thresh, res in sorted(all_results.items()):
        for bin_name in bin_names:
            if bin_name not in res:
                continue
            m = res[bin_name]
            print(
                f"{thresh:>10} | {bin_name:>12} | {m['Hit@10']:>7.4f} | "
                f"{m['MRR']:>7.4f} | {m['pct_focused']:>8.1%} | {m['n']:>4}"
            )
        print("-" * len(header))


def main():
    print("Loading STaRK-PrimeKG...")
    graph, qa_dataset = load_stark_dataset("prime")

    # Load hop stratification produced by earlier experiments.
    strat_path = os.path.join(RESULTS_DIR, "prime_hop_stratification.json")
    with open(strat_path) as f:
        hop_bins = json.load(f)

    print("\nBuilding shared embedding index...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Index built in {time.time() - t0:.1f}s")

    all_results = {}

    for thresh in THRESHOLDS:
        print(f"\n--- skew_threshold = {thresh} ---")
        retriever = AdaptiveRetriever(graph, shared, skew_threshold=thresh)

        bin_results = evaluate_adaptive_on_bins(retriever, qa_dataset, hop_bins)
        all_results[str(thresh)] = bin_results

        for bin_name, metrics in bin_results.items():
            print(
                f"  {bin_name}: Hit@10={metrics['Hit@10']:.4f}, "
                f"MRR={metrics['MRR']:.4f}, "
                f"focused={metrics['pct_focused']:.1%}"
            )

    print_summary_table(all_results)

    # Save results.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"prime_adaptive_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
