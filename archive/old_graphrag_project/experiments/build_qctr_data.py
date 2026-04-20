#!/usr/bin/env python3
"""Build QCTR training data from STaRK-PrimeKG."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.data.qctr_data import (
    build_edge_type_lookup,
    extract_trajectories,
    sample_negatives,
    build_feature_vectors,
    split_dataset,
    save_dataset,
)
from src.retrievers.shared_index import SharedIndex


def main(
    dataset_name="prime",
    max_samples=None,
    num_random_neg=5,
    num_hard_neg=3,
    max_paths=5,
    seed=42,
):
    print(f"\n{'='*60}")
    print(f"  Build QCTR Training Data: STaRK-{dataset_name}")
    print(f"{'='*60}\n")

    random.seed(seed)
    t0 = time.time()

    # Step 1: Load dataset
    print("--- Step 1: Load dataset ---")
    graph, qa_dataset = load_stark_dataset(dataset_name)
    print(f"  Loaded graph and QA dataset in {time.time() - t0:.1f}s")

    # Step 2: Load hop stratification
    print("\n--- Step 2: Load hop stratification ---")
    strat_path = f"results/{dataset_name}_hop_stratification.json"
    with open(strat_path) as f:
        stratification = json.load(f)
    print(f"  Loaded stratification from {strat_path}")
    for hop, indices in sorted(stratification.items(), key=lambda x: x[0]):
        print(f"    {hop}: {len(indices)} queries")

    # Step 3: Build edge type lookup
    print("\n--- Step 3: Build edge type lookup ---")
    edge_type_lookup = build_edge_type_lookup(graph.skb)
    edge_type_dict = graph.skb.edge_type_dict
    print(f"  Found {len(edge_type_dict)} relation types:")
    for rt_id, rt_name in sorted(edge_type_dict.items()):
        print(f"    - [{rt_id}] {rt_name}")

    # Step 4: Build SharedIndex
    print("\n--- Step 4: Build SharedIndex ---")
    shared_index = SharedIndex(graph)
    print("  SharedIndex built")

    # Step 5: Extract trajectories
    print("\n--- Step 5: Extract trajectories ---")
    if max_samples is not None:
        print(f"  Truncating each hop bin to {max_samples} samples")
        stratification = {
            hop: indices[:max_samples]
            for hop, indices in stratification.items()
        }
    trajectories = extract_trajectories(
        graph, stratification, edge_type_lookup, max_paths_per_query=max_paths
    )
    print(f"  Total positive transitions: {len(trajectories)}")

    hop_counts = Counter(t["hop_count"] for t in trajectories)
    print("  Transitions by hop count:")
    for hop, count in sorted(hop_counts.items()):
        print(f"    {hop}-hop: {count}")

    step_counts = Counter(t["step_position"] for t in trajectories)
    print("  Transitions by step position:")
    for step, count in sorted(step_counts.items()):
        print(f"    step {step}: {count}")

    edge_type_dist = Counter(t["edge_type"] for t in trajectories)
    print("  Distribution of edge types in positives:")
    for et, count in edge_type_dist.most_common(20):
        name = edge_type_dict.get(et, "unknown") if et >= 0 else "unknown"
        print(f"    [{et}] {name}: {count}")

    # Step 6: Sample negatives
    print("\n--- Step 6: Sample negatives ---")
    samples = sample_negatives(
        graph, trajectories, shared_index, edge_type_lookup,
        num_random_neg, num_hard_neg
    )
    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = len(samples) - n_pos
    print(f"  Total samples (positives + negatives): {len(samples)}")
    print(f"  Label balance: {n_pos / len(samples) * 100:.1f}% positive, {n_neg / len(samples) * 100:.1f}% negative")

    neg_type_dist = Counter(
        s["neg_type"] for s in samples if s["label"] == 0
    )
    print("  Negative type distribution:")
    for nt, count in neg_type_dist.most_common():
        print(f"    {nt}: {count}")

    # Step 7: Build feature vectors
    print("\n--- Step 7: Build feature vectors ---")
    features_dict = build_feature_vectors(samples, shared_index, qa_dataset)
    X = features_dict["features"]
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Feature matrix dtype: {X.dtype}")

    # Step 8: Split dataset
    print("\n--- Step 8: Split dataset ---")
    dataset_dict = split_dataset(features_dict)
    for split_name, split_data in dataset_dict.items():
        print(f"  {split_name}: {split_data['features'].shape[0]} samples")

    # Step 9: Save dataset
    print("\n--- Step 9: Save dataset ---")
    save_path = f"data/qctr/{dataset_name}"
    save_dataset(dataset_dict, save_path)
    print(f"  Saved to {save_path}")

    # Step 10: Final summary / checkpoint validation
    print(f"\n{'='*60}")
    print("  Final Summary")
    print(f"{'='*60}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    passed = True

    # Check: at least 1000 training transitions for 2-hop queries
    train_data = dataset_dict.get("train", {})
    train_meta = train_data.get("metadata", [])
    n_train_2hop = sum(1 for m in train_meta if m["hop_count"] == 2)
    if n_train_2hop < 1000:
        print(f"  FAIL: Only {n_train_2hop} training transitions for 2-hop (need >= 1000)")
        passed = False
    else:
        print(f"  OK: {n_train_2hop} training transitions for 2-hop")

    # Check: at least 100 validation samples for 2-hop
    val_data = dataset_dict.get("val", {})
    val_meta = val_data.get("metadata", [])
    n_val_2hop = sum(1 for m in val_meta if m["hop_count"] == 2)
    if n_val_2hop < 100:
        print(f"  FAIL: Only {n_val_2hop} validation samples for 2-hop (need >= 100)")
        passed = False
    else:
        print(f"  OK: {n_val_2hop} validation samples for 2-hop")

    # Check: feature dimensionality = 384*3 + 3 = 1155
    expected_cols = 384 * 3 + 3  # 1155
    actual_cols = X.shape[1] if len(X.shape) > 1 else X.shape[0]
    if actual_cols != expected_cols:
        print(f"  FAIL: Feature matrix has {actual_cols} columns (expected {expected_cols})")
        passed = False
    else:
        print(f"  OK: Feature matrix has {actual_cols} columns")

    if passed:
        print("\n  CHECKPOINT PASSED")
    else:
        print("\n  CHECKPOINT FAILED")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build QCTR training data from STaRK")
    parser.add_argument("--dataset", type=str, default="prime", help="Dataset name (default: prime)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per hop bin (for testing)")
    parser.add_argument("--num-random-neg", type=int, default=5, help="Number of random negatives per positive")
    parser.add_argument("--num-hard-neg", type=int, default=3, help="Number of hard negatives per positive")
    parser.add_argument("--max-paths", type=int, default=5, help="Max paths per query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        num_random_neg=args.num_random_neg,
        num_hard_neg=args.num_hard_neg,
        max_paths=args.max_paths,
        seed=args.seed,
    )
