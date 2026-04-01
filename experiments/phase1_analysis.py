"""Phase 1: Load STaRK, characterize graph, stratify QA pairs by hop count."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.data.graph_analysis import characterize_graph, stratify_by_hop_count


def main(dataset_name: str = "prime"):
    print(f"\n{'='*60}")
    print(f"  Phase 1 Analysis: STaRK-{dataset_name}")
    print(f"{'='*60}\n")

    # Step 1: Load dataset
    graph, qa_dataset = load_stark_dataset(dataset_name)

    # Step 2: Characterize graph properties
    print("\n--- Graph Characterization ---")
    summary = characterize_graph(graph, dataset_name)

    # Step 3: Stratify QA pairs by hop count
    print("\n--- Hop Count Stratification ---")
    bins = stratify_by_hop_count(graph, qa_dataset, dataset_name)

    # Step 4: Statistical power check
    print("\n--- Statistical Power Check ---")
    sufficient = True
    for bin_name, items in bins.items():
        n = len(items)
        if n < 50 and n > 0:
            print(f"  WARNING: {bin_name} has only {n} samples (target >= 50)")
            sufficient = False
        elif n == 0:
            print(f"  NOTE: {bin_name} is empty — will exclude from analysis")
    if sufficient:
        print("  All non-empty bins have >= 50 samples. Good to proceed.")

    print(f"\n{'='*60}")
    print(f"  Phase 1 Complete!")
    print(f"  Results saved to results/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Graph analysis and QA stratification")
    parser.add_argument("--dataset", default="prime", choices=["prime", "amazon", "mag"])
    args = parser.parse_args()
    main(args.dataset)
