"""Main experiment runner: evaluate all retrieval strategies on STaRK."""

import argparse
import json
import os
from datetime import datetime

from src.data import load_stark_dataset
from src.retrievers import (
    NodeRetriever,
    PathRetriever,
    SubgraphRetriever,
    HybridRetriever,
)
from src.evaluation.metrics import evaluate_retrieval


def run_all(dataset_name: str = "prime", max_samples: int | None = None):
    """Run all retrieval strategies and save results."""

    # Load data
    graph, qa_dataset = load_stark_dataset(dataset_name)

    if max_samples:
        qa_dataset = qa_dataset[:max_samples]

    # Initialize retrievers
    print("\n--- Initializing retrievers ---")
    retrievers = {
        "node_centric": NodeRetriever(graph),
        "path_centric": PathRetriever(graph),
        "subgraph_centric": SubgraphRetriever(graph),
        "hybrid_mor": HybridRetriever(graph),
    }

    # Evaluate each
    all_results = {}
    for name, retriever in retrievers.items():
        print(f"\n=== Evaluating: {name} ===")
        results = evaluate_retrieval(retriever, qa_dataset, ks=[1, 5, 10])
        all_results[name] = results
        print(f"  Hit@1:  {results['Hit@1']:.4f}")
        print(f"  Hit@5:  {results['Hit@5']:.4f}")
        print(f"  Hit@10: {results['Hit@10']:.4f}")
        print(f"  MRR:    {results['MRR']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, f"{dataset_name}_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(
            {"dataset": dataset_name, "timestamp": timestamp, "results": all_results},
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphRAG retrieval experiments")
    parser.add_argument("--dataset", default="prime", choices=["prime", "amazon", "mag"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit QA samples for testing")
    args = parser.parse_args()

    run_all(args.dataset, args.max_samples)
