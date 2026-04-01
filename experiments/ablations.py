"""Ablation studies: vary hyperparameters per strategy to understand sensitivity."""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.retrievers.shared_index import SharedIndex
from src.retrievers.node_retriever import NodeRetriever
from src.retrievers.path_retriever import PathRetriever
from src.retrievers.subgraph_retriever import SubgraphRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.evaluation.metrics import hit_at_k, mrr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def evaluate_on_bins(retriever, qa_dataset, hop_bins, ks=[1, 5, 10], max_per_bin=200):
    """Evaluate a retriever on a subset of each hop bin. Returns per-bin metrics."""
    results = {}
    for bin_name, items in hop_bins.items():
        if not items:
            continue
        items = items[:max_per_bin]

        hit_scores = {k: [] for k in ks}
        mrr_scores = []

        for item in items:
            query, _, gold_ids, _ = qa_dataset[item["index"]]
            predicted = retriever.retrieve_ids(query, top_k=max(ks))
            for k in ks:
                hit_scores[k].append(hit_at_k(predicted, gold_ids, k))
            mrr_scores.append(mrr(predicted, gold_ids))

        results[bin_name] = {
            f"Hit@{k}": float(np.mean(hit_scores[k])) for k in ks
        }
        results[bin_name]["MRR"] = float(np.mean(mrr_scores))
        results[bin_name]["n"] = len(items)

    return results


def ablation_top_k(graph, shared, qa_dataset, hop_bins):
    """Ablation 1: Effect of top-k budget on node-centric retrieval."""
    print("\n" + "=" * 70)
    print("ABLATION 1: Effect of top-k budget (Node-centric)")
    print("=" * 70)

    top_k_values = [1, 3, 5, 10, 20, 50]
    all_results = {}

    for top_k in top_k_values:
        print(f"\n  top_k = {top_k}")
        ret = NodeRetriever(graph, shared)

        results = {}
        for bin_name, items in hop_bins.items():
            if not items:
                continue
            items_sub = items[:200]
            hits = []
            mrr_scores = []
            for item in items_sub:
                query, _, gold_ids, _ = qa_dataset[item["index"]]
                predicted = ret.retrieve_ids(query, top_k=top_k)
                hits.append(hit_at_k(predicted, gold_ids, top_k))
                mrr_scores.append(mrr(predicted, gold_ids))
            results[bin_name] = {
                f"Hit@{top_k}": float(np.mean(hits)),
                "MRR": float(np.mean(mrr_scores)),
                "n": len(items_sub),
            }
            print(f"    {bin_name}: Hit@{top_k}={np.mean(hits):.4f}, MRR={np.mean(mrr_scores):.4f}")

        all_results[f"top_k={top_k}"] = results

    return all_results


def ablation_k_hops(graph, shared, qa_dataset, hop_bins):
    """Ablation 2: Effect of subgraph expansion depth (k_hops)."""
    print("\n" + "=" * 70)
    print("ABLATION 2: Effect of k-hop depth (Subgraph-centric)")
    print("=" * 70)

    k_hops_values = [1, 2, 3]
    max_subgraph_sizes = [200, 500, 1000]
    all_results = {}

    for k_hops, max_nodes in zip(k_hops_values, max_subgraph_sizes):
        print(f"\n  k_hops = {k_hops} (max_subgraph_nodes = {max_nodes})")
        ret = SubgraphRetriever(graph, shared, num_seeds=3, k_hops=k_hops, max_subgraph_nodes=max_nodes)

        results = evaluate_on_bins(ret, qa_dataset, hop_bins)
        for bin_name, metrics in results.items():
            print(f"    {bin_name}: Hit@10={metrics['Hit@10']:.4f}, MRR={metrics['MRR']:.4f}")

        all_results[f"k_hops={k_hops}"] = results

    return all_results


def ablation_path_length(graph, shared, qa_dataset, hop_bins):
    """Ablation 3: Effect of max path length (Path-centric)."""
    print("\n" + "=" * 70)
    print("ABLATION 3: Effect of max path length (Path-centric)")
    print("=" * 70)

    path_lengths = [2, 3, 4, 5]
    all_results = {}

    for max_len in path_lengths:
        print(f"\n  max_path_length = {max_len}")
        ret = PathRetriever(graph, shared, num_seeds=5, max_path_length=max_len)

        results = evaluate_on_bins(ret, qa_dataset, hop_bins)
        for bin_name, metrics in results.items():
            print(f"    {bin_name}: Hit@10={metrics['Hit@10']:.4f}, MRR={metrics['MRR']:.4f}")

        all_results[f"max_path_length={max_len}"] = results

    return all_results


def ablation_num_seeds(graph, shared, qa_dataset, hop_bins):
    """Ablation 4: Effect of number of seed nodes."""
    print("\n" + "=" * 70)
    print("ABLATION 4: Effect of num_seeds (Path + Subgraph)")
    print("=" * 70)

    seed_values = [1, 3, 5, 10]
    all_results = {}

    for num_seeds in seed_values:
        print(f"\n  num_seeds = {num_seeds}")

        # Path
        path_ret = PathRetriever(graph, shared, num_seeds=num_seeds, max_path_length=4)
        path_results = evaluate_on_bins(path_ret, qa_dataset, hop_bins)
        print(f"  Path-centric:")
        for bin_name, metrics in path_results.items():
            print(f"    {bin_name}: Hit@10={metrics['Hit@10']:.4f}, MRR={metrics['MRR']:.4f}")

        # Subgraph
        sub_ret = SubgraphRetriever(graph, shared, num_seeds=num_seeds, k_hops=1, max_subgraph_nodes=200)
        sub_results = evaluate_on_bins(sub_ret, qa_dataset, hop_bins)
        print(f"  Subgraph-centric:")
        for bin_name, metrics in sub_results.items():
            print(f"    {bin_name}: Hit@10={metrics['Hit@10']:.4f}, MRR={metrics['MRR']:.4f}")

        all_results[f"num_seeds={num_seeds}"] = {
            "path_centric": path_results,
            "subgraph_centric": sub_results,
        }

    return all_results


def ablation_text_weight(graph, shared, qa_dataset, hop_bins):
    """Ablation 5: Effect of text_weight in hybrid retriever."""
    print("\n" + "=" * 70)
    print("ABLATION 5: Effect of text_weight (Hybrid)")
    print("=" * 70)

    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    all_results = {}

    for w in weights:
        print(f"\n  text_weight = {w}")
        ret = HybridRetriever(graph, shared, text_weight=w)

        results = evaluate_on_bins(ret, qa_dataset, hop_bins)
        for bin_name, metrics in results.items():
            print(f"    {bin_name}: Hit@10={metrics['Hit@10']:.4f}, MRR={metrics['MRR']:.4f}")

        all_results[f"text_weight={w}"] = results

    return all_results


def main():
    print("Loading STaRK-PrimeKG...")
    graph, qa_dataset = load_stark_dataset("prime")

    # Load hop stratification
    strat_path = os.path.join(RESULTS_DIR, "prime_hop_stratification.json")
    with open(strat_path) as f:
        hop_bins = json.load(f)

    print("\nBuilding shared embedding index...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Index built in {time.time() - t0:.1f}s")

    # Run all ablations
    all_ablations = {}

    all_ablations["top_k"] = ablation_top_k(graph, shared, qa_dataset, hop_bins)
    all_ablations["k_hops"] = ablation_k_hops(graph, shared, qa_dataset, hop_bins)
    all_ablations["path_length"] = ablation_path_length(graph, shared, qa_dataset, hop_bins)
    all_ablations["num_seeds"] = ablation_num_seeds(graph, shared, qa_dataset, hop_bins)
    all_ablations["text_weight"] = ablation_text_weight(graph, shared, qa_dataset, hop_bins)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"prime_ablations_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(all_ablations, f, indent=2)
    print(f"\nAll ablation results saved to {output_path}")


if __name__ == "__main__":
    main()
