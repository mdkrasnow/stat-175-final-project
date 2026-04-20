"""Main experiment runner: evaluate all retrieval strategies on STaRK, stratified by hop count."""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.retrievers.shared_index import SharedIndex
from src.retrievers.node_retriever import NodeRetriever
from src.retrievers.path_retriever import PathRetriever
from src.retrievers.subgraph_retriever import SubgraphRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.evaluation.metrics import hit_at_k, mrr


def run_all(dataset_name: str = "prime", max_samples: int | None = None):
    """Run all retrieval strategies, stratified by hop count."""

    # Load data
    graph, qa_dataset = load_stark_dataset(dataset_name)

    # Load hop stratification
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    strat_path = os.path.join(results_dir, f"{dataset_name}_hop_stratification.json")
    if not os.path.exists(strat_path):
        print(f"ERROR: Run phase1_analysis.py first to generate {strat_path}")
        sys.exit(1)

    with open(strat_path) as f:
        hop_bins = json.load(f)

    # Build shared index
    print("\nBuilding shared embedding index...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Index built in {time.time() - t0:.1f}s\n")

    # Initialize retrievers
    retrievers = {
        "node_centric": NodeRetriever(graph, shared),
        "path_centric": PathRetriever(graph, shared),
        "subgraph_centric": SubgraphRetriever(graph, shared),
        "hybrid_mor": HybridRetriever(graph, shared),
    }

    ks = [1, 5, 10]
    all_results = {}

    # Evaluate per hop-count bin
    for bin_name, items in hop_bins.items():
        if not items:
            continue

        if max_samples:
            items = items[:max_samples]

        print(f"\n{'='*60}")
        print(f"  {bin_name} (n={len(items)})")
        print(f"{'='*60}")

        bin_results = {}
        for ret_name, retriever in retrievers.items():
            hit_scores = {k: [] for k in ks}
            mrr_scores = []

            t0 = time.time()
            for item in items:
                idx = item["index"]
                query, _, gold_ids, _ = qa_dataset[idx]
                predicted = retriever.retrieve_ids(query, top_k=max(ks))

                for k in ks:
                    hit_scores[k].append(hit_at_k(predicted, gold_ids, k))
                mrr_scores.append(mrr(predicted, gold_ids))

            elapsed = time.time() - t0
            import numpy as np
            results = {}
            for k in ks:
                results[f"Hit@{k}"] = float(np.mean(hit_scores[k]))
            results["MRR"] = float(np.mean(mrr_scores))
            results["n"] = len(items)
            results["time_s"] = round(elapsed, 1)

            # Store per-query results for statistical tests
            results["per_query_hit1"] = hit_scores[1]
            results["per_query_mrr"] = mrr_scores

            bin_results[ret_name] = results

        all_results[bin_name] = bin_results

        # Print summary for this bin
        print(f"\n{'Strategy':<20} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8} {'Time':>8}")
        print("-" * 60)
        for ret_name, results in bin_results.items():
            print(f"{ret_name:<20} {results['Hit@1']:>8.4f} {results['Hit@5']:>8.4f} {results['Hit@10']:>8.4f} {results['MRR']:>8.4f} {results['time_s']:>7.1f}s")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)

    # Strip per-query arrays for the summary file
    summary = {}
    for bin_name, bin_results in all_results.items():
        summary[bin_name] = {}
        for ret_name, results in bin_results.items():
            summary[bin_name][ret_name] = {
                k: v for k, v in results.items()
                if k not in ("per_query_hit1", "per_query_mrr")
            }

    output_path = os.path.join(results_dir, f"{dataset_name}_experiment_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump({"dataset": dataset_name, "timestamp": timestamp, "results": summary}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save per-query results separately for statistical tests
    perquery_path = os.path.join(results_dir, f"{dataset_name}_perquery_{timestamp}.json")
    perquery = {}
    for bin_name, bin_results in all_results.items():
        perquery[bin_name] = {}
        for ret_name, results in bin_results.items():
            perquery[bin_name][ret_name] = {
                "per_query_hit1": results["per_query_hit1"],
                "per_query_mrr": results["per_query_mrr"],
            }
    with open(perquery_path, "w") as f:
        json.dump(perquery, f)
    print(f"Per-query results saved to {perquery_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphRAG retrieval experiments")
    parser.add_argument("--dataset", default="prime", choices=["prime", "amazon", "mag"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit QA samples per bin for testing")
    args = parser.parse_args()

    run_all(args.dataset, args.max_samples)
