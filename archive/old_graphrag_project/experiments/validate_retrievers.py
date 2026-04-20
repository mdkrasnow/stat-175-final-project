"""Phase 2: Validate all retrieval strategies on a small sample."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.retrievers.shared_index import SharedIndex
from src.retrievers.node_retriever import NodeRetriever
from src.retrievers.path_retriever import PathRetriever
from src.retrievers.subgraph_retriever import SubgraphRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.evaluation.metrics import hit_at_k, evaluate_retrieval


def main():
    sample_size = 50

    print("Loading STaRK-PrimeKG...")
    graph, qa_dataset = load_stark_dataset("prime")

    print("\nBuilding shared embedding index (one-time cost)...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Shared index built in {time.time() - t0:.1f}s\n")

    # Initialize all retrievers
    retrievers = {
        "Node-centric": NodeRetriever(graph, shared),
        "Path-centric": PathRetriever(graph, shared),
        "Subgraph-centric": SubgraphRetriever(graph, shared),
        "Hybrid (MoR)": HybridRetriever(graph, shared),
    }

    # Quick sanity check on first query
    query, _, gold_ids, _ = qa_dataset[0]
    print(f"Sanity check query: {query[:80]}...")
    print(f"Gold answer IDs: {gold_ids}\n")

    for name, ret in retrievers.items():
        retrieved = ret.retrieve_ids(query, top_k=10)
        h1 = hit_at_k(retrieved, gold_ids, 1)
        h10 = hit_at_k(retrieved, gold_ids, 10)
        print(f"  {name}: top-10={retrieved[:5]}... Hit@1={h1} Hit@10={h10}")

    # Batch evaluation
    print(f"\n{'='*60}")
    print(f"Batch evaluation on {sample_size} QA pairs")
    print(f"{'='*60}\n")

    all_results = {}
    for name, ret in retrievers.items():
        print(f"Evaluating {name}...")
        t0 = time.time()
        results = evaluate_retrieval(ret, qa_dataset, ks=[1, 5, 10], max_samples=sample_size)
        elapsed = time.time() - t0
        all_results[name] = results
        print(f"  Done in {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY (n={sample_size})")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8}")
    print("-" * 52)
    for name, results in all_results.items():
        print(f"{name:<20} {results['Hit@1']:>8.4f} {results['Hit@5']:>8.4f} {results['Hit@10']:>8.4f} {results['MRR']:>8.4f}")


if __name__ == "__main__":
    main()
