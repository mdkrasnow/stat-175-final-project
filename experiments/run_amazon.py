"""Run full pipeline on STaRK-Amazon: Phase 1 (graph analysis + stratification) then Phase 3 (experiment)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.data.graph_analysis import characterize_graph, stratify_by_hop_count
from src.retrievers.shared_index import SharedIndex
from src.retrievers.node_retriever import NodeRetriever
from src.retrievers.path_retriever import PathRetriever
from src.retrievers.subgraph_retriever import SubgraphRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.evaluation.metrics import hit_at_k, mrr

import json
import time
import numpy as np
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATASET = "amazon"


def main():
    print("=" * 60)
    print(f"  Full Pipeline: STaRK-{DATASET}")
    print("=" * 60)

    # Phase 1: Load + characterize + stratify
    graph, qa_dataset = load_stark_dataset(DATASET)

    print("\n--- Graph Characterization ---")
    summary = characterize_graph(graph, DATASET)

    print("\n--- Hop Count Stratification ---")
    bins = stratify_by_hop_count(graph, qa_dataset, DATASET)

    # Statistical power check
    print("\n--- Statistical Power Check ---")
    for bin_name, items in bins.items():
        n = len(items)
        if 0 < n < 50:
            print(f"  WARNING: {bin_name} has only {n} samples")
        elif n == 0:
            print(f"  NOTE: {bin_name} is empty")

    # Phase 3: Build index + run experiment
    print("\n--- Building Shared Index ---")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Index built in {time.time() - t0:.1f}s")

    # Load stratification from saved file
    strat_path = os.path.join(RESULTS_DIR, f"{DATASET}_hop_stratification.json")
    with open(strat_path) as f:
        hop_bins = json.load(f)

    retrievers = {
        "node_centric": NodeRetriever(graph, shared),
        "path_centric": PathRetriever(graph, shared),
        "subgraph_centric": SubgraphRetriever(graph, shared),
        "hybrid_mor": HybridRetriever(graph, shared),
    }

    ks = [1, 5, 10]
    all_results = {}
    all_perquery = {}

    for bin_name, items in hop_bins.items():
        if not items:
            continue

        print(f"\n{'='*60}")
        print(f"  {bin_name} (n={len(items)})")
        print(f"{'='*60}")

        bin_results = {}
        bin_perquery = {}

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
            results = {f"Hit@{k}": float(np.mean(hit_scores[k])) for k in ks}
            results["MRR"] = float(np.mean(mrr_scores))
            results["n"] = len(items)
            results["time_s"] = round(elapsed, 1)

            bin_results[ret_name] = results
            bin_perquery[ret_name] = {
                "per_query_hit1": hit_scores[1],
                "per_query_mrr": mrr_scores,
            }

        all_results[bin_name] = bin_results
        all_perquery[bin_name] = bin_perquery

        print(f"\n{'Strategy':<20} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8} {'Time':>8}")
        print("-" * 60)
        for ret_name, results in bin_results.items():
            print(f"{ret_name:<20} {results['Hit@1']:>8.4f} {results['Hit@5']:>8.4f} {results['Hit@10']:>8.4f} {results['MRR']:>8.4f} {results['time_s']:>7.1f}s")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    exp_path = os.path.join(RESULTS_DIR, f"{DATASET}_experiment_{timestamp}.json")
    with open(exp_path, "w") as f:
        json.dump({"dataset": DATASET, "timestamp": timestamp, "results": all_results}, f, indent=2)

    pq_path = os.path.join(RESULTS_DIR, f"{DATASET}_perquery_{timestamp}.json")
    with open(pq_path, "w") as f:
        json.dump(all_perquery, f)

    print(f"\nResults saved to {exp_path}")
    print(f"Per-query results saved to {pq_path}")

    print("\n" + "=" * 60)
    print(f"  STaRK-{DATASET} pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
