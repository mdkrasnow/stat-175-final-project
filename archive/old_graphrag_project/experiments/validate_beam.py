"""Validate BeamRetriever: sweep beam_width and max_hops, compare to baselines.

Evaluates the embedding-guided beam search retriever across configurations
and compares against node-centric and subgraph-centric baselines. Results
are stratified by hop distance using pre-computed hop bins.
"""

import sys
import os
import json
import time
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.retrievers.shared_index import SharedIndex
from src.retrievers.beam_retriever import BeamRetriever
from src.retrievers.node_retriever import NodeRetriever
from src.retrievers.subgraph_retriever import SubgraphRetriever
from src.evaluation.metrics import hit_at_k, mrr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hop_bins(path: str = "results/prime_hop_stratification.json") -> dict:
    """Load pre-computed hop-distance bins for stratified evaluation."""
    abs_path = os.path.join(os.path.dirname(__file__), "..", path)
    with open(abs_path) as f:
        bins = json.load(f)
    print(f"Loaded hop bins from {path}:")
    for name, indices in bins.items():
        print(f"  {name}: {len(indices)} samples")
    return bins


def evaluate_retriever_on_bin(
    retriever,
    qa_dataset,
    sample_indices: list[int],
    max_samples: int = 200,
    ks: list[int] = [1, 5, 10, 20],
) -> dict[str, float]:
    """Evaluate a retriever on a specific subset of the QA dataset.

    Args:
        retriever: Any BaseRetriever instance.
        qa_dataset: Full STaRK QA dataset.
        sample_indices: Indices into qa_dataset for this hop bin.
        max_samples: Cap on number of samples to evaluate.
        ks: Values of k for Hit@k.

    Returns:
        Dict with Hit@k and MRR scores.
    """
    indices = sample_indices[:max_samples]
    hit_scores = {k: [] for k in ks}
    mrr_scores = []

    for item in indices:
        idx = item["index"] if isinstance(item, dict) else item
        query, query_id, gold_ids, meta = qa_dataset[idx]
        predicted_ids = retriever.retrieve_ids(query, top_k=max(ks))

        for k in ks:
            hit_scores[k].append(hit_at_k(predicted_ids, gold_ids, k))
        mrr_scores.append(mrr(predicted_ids, gold_ids))

    results = {}
    for k in ks:
        results[f"Hit@{k}"] = float(np.mean(hit_scores[k])) if hit_scores[k] else 0.0
    results["MRR"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    results["n_samples"] = len(indices)
    return results


def evaluate_across_bins(
    retriever,
    qa_dataset,
    hop_bins: dict,
    max_samples: int = 200,
    label: str = "",
) -> dict[str, dict]:
    """Evaluate a retriever across all hop bins and return structured results."""
    results = {}
    for bin_name, indices in hop_bins.items():
        if not indices:
            continue
        t0 = time.time()
        scores = evaluate_retriever_on_bin(retriever, qa_dataset, indices, max_samples)
        elapsed = time.time() - t0
        scores["time_sec"] = round(elapsed, 2)
        results[bin_name] = scores
        print(f"  {label} | {bin_name}: Hit@1={scores['Hit@1']:.3f}  "
              f"Hit@5={scores['Hit@5']:.3f}  MRR={scores['MRR']:.3f}  "
              f"({scores['n_samples']} samples, {elapsed:.1f}s)")
    return results


def print_summary_table(all_results: dict[str, dict[str, dict]], metric: str = "Hit@5"):
    """Print a compact summary table comparing methods across hop bins."""
    configs = list(all_results.keys())
    # Collect all bin names across configs
    bin_names = []
    for config_results in all_results.values():
        for bn in config_results:
            if bn not in bin_names:
                bin_names.append(bn)

    header = f"{'Config':<35}" + "".join(f"{bn:>12}" for bn in bin_names)
    print(f"\n{'='*len(header)}")
    print(f"  Summary: {metric}")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for config_name in configs:
        row = f"{config_name:<35}"
        for bn in bin_names:
            val = all_results[config_name].get(bn, {}).get(metric, float("nan"))
            row += f"{val:>12.3f}"
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print(f"  Beam Retriever Validation")
    print(f"{'='*60}\n")

    # Load dataset
    graph, qa_dataset = load_stark_dataset("prime")

    # Build shared index (one-time cost)
    shared_index = SharedIndex(graph)

    # Load hop stratification
    hop_bins = load_hop_bins()

    all_results = {}
    max_samples = 200

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("\n--- Baseline: NodeRetriever ---")
    node_ret = NodeRetriever(graph, shared_index)
    all_results["node_centric"] = evaluate_across_bins(
        node_ret, qa_dataset, hop_bins, max_samples, label="NodeCentric"
    )

    print("\n--- Baseline: SubgraphRetriever (seeds=3, hops=1) ---")
    subgraph_ret = SubgraphRetriever(graph, shared_index, num_seeds=3, k_hops=1)
    all_results["subgraph_s3_h1"] = evaluate_across_bins(
        subgraph_ret, qa_dataset, hop_bins, max_samples, label="Subgraph(s3,h1)"
    )

    print("\n--- Baseline: SubgraphRetriever (seeds=3, hops=2) ---")
    subgraph_ret2 = SubgraphRetriever(graph, shared_index, num_seeds=3, k_hops=2)
    all_results["subgraph_s3_h2"] = evaluate_across_bins(
        subgraph_ret2, qa_dataset, hop_bins, max_samples, label="Subgraph(s3,h2)"
    )

    # ------------------------------------------------------------------
    # Beam width sweep (fix max_hops=3, num_seeds=3)
    # ------------------------------------------------------------------
    print("\n--- Beam Width Sweep (max_hops=3, num_seeds=3) ---")
    for bw in [3, 5, 10, 20]:
        label = f"Beam(bw={bw},hops=3)"
        print(f"\n  Config: {label}")
        ret = BeamRetriever(graph, shared_index, num_seeds=3, beam_width=bw, max_hops=3)
        all_results[f"beam_bw{bw}_h3"] = evaluate_across_bins(
            ret, qa_dataset, hop_bins, max_samples, label=label
        )

    # ------------------------------------------------------------------
    # Max hops sweep (fix beam_width=10, num_seeds=3)
    # ------------------------------------------------------------------
    print("\n--- Max Hops Sweep (beam_width=10, num_seeds=3) ---")
    for mh in [1, 2, 3, 4]:
        label = f"Beam(bw=10,hops={mh})"
        print(f"\n  Config: {label}")
        ret = BeamRetriever(graph, shared_index, num_seeds=3, beam_width=10, max_hops=mh)
        all_results[f"beam_bw10_h{mh}"] = evaluate_across_bins(
            ret, qa_dataset, hop_bins, max_samples, label=label
        )

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    for metric in ["Hit@1", "Hit@5", "Hit@10", "MRR"]:
        print_summary_table(all_results, metric)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", f"prime_beam_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
