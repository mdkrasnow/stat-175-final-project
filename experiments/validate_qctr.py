"""Validate the QCTR retriever on a small sample and compare against baselines."""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.data.qctr_data import build_edge_type_lookup
from src.retrievers.shared_index import SharedIndex
from src.retrievers.qctr_retriever import QCTRRetriever
from src.retrievers.path_retriever import PathRetriever
from src.retrievers.node_retriever import NodeRetriever
from src.evaluation.metrics import hit_at_k, mrr


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def main(dataset_name="prime", n_samples=50, beam_width=10, max_hops=4, device="cpu"):
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Loading STaRK-{dataset_name}...")
    print("=" * 60)
    graph, qa_dataset = load_stark_dataset(dataset_name)

    strat_path = os.path.join(
        PROJECT_ROOT, "results", f"{dataset_name}_hop_stratification.json"
    )
    print(f"Loading hop stratification from {strat_path}")
    with open(strat_path) as f:
        stratification = json.load(f)

    for hop_bin, items in stratification.items():
        print(f"  {hop_bin}: {len(items)} items")

    # ------------------------------------------------------------------
    # 2. Build SharedIndex and edge_type_lookup
    # ------------------------------------------------------------------
    print("\nBuilding shared embedding index...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Shared index built in {time.time() - t0:.1f}s")

    print("\nBuilding edge type lookup...")
    t0 = time.time()
    edge_type_lookup = build_edge_type_lookup(graph.skb)
    print(f"Edge type lookup built in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Initialize retrievers
    # ------------------------------------------------------------------
    model_path = os.path.join(PROJECT_ROOT, "models", "qctr", "best_model.pt")
    print(f"\nLoading QCTR model from {model_path}")

    retrievers = {
        "node": NodeRetriever(graph, shared),
        "path": PathRetriever(graph, shared),
        "QCTR": QCTRRetriever(
            graph,
            shared,
            model_path=model_path,
            edge_type_lookup=edge_type_lookup,
            beam_width=beam_width,
            max_hops=max_hops,
            device=device,
        ),
    }

    # ------------------------------------------------------------------
    # 4. Sanity check on first 3 items from 2-hop bin
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SANITY CHECK: First 3 queries from 2-hop bin")
    print("=" * 60)

    two_hop_items = stratification.get("2-hop", [])
    for idx, item in enumerate(two_hop_items[:3]):
        query, query_id, gold_ids, meta = qa_dataset[item["index"]]
        print(f"\n--- Query {idx + 1} ---")
        print(f"  Text:     {query[:100]}...")
        print(f"  Gold IDs: {gold_ids}")

        for rname in ["QCTR", "path"]:
            ret = retrievers[rname]
            retrieved = ret.retrieve_ids(query, top_k=10)
            h1 = hit_at_k(retrieved, gold_ids, 1)
            h10 = hit_at_k(retrieved, gold_ids, 10)
            print(f"  {rname:>6} top-10: {retrieved}")
            print(f"  {rname:>6} Hit@1={h1:.0f}  Hit@10={h10:.0f}")

    # ------------------------------------------------------------------
    # 5. Quantitative comparison on n_samples per hop bin
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"QUANTITATIVE COMPARISON (up to {n_samples} per hop bin)")
    print("=" * 60)

    all_rows = []  # (hop_bin, method, h1, h5, h10, mrr_val, avg_time)

    for hop_bin in sorted(stratification.keys(), key=_hop_sort_key):
        items = stratification[hop_bin]
        n = min(n_samples, len(items))
        print(f"\nEvaluating {hop_bin} ({n} queries)...")

        for rname, ret in retrievers.items():
            h1_scores, h5_scores, h10_scores, mrr_scores = [], [], [], []
            total_time = 0.0
            non_empty = 0

            for i, item in enumerate(items[:n]):
                query, query_id, gold_ids, _meta = qa_dataset[item["index"]]

                t0 = time.time()
                retrieved = ret.retrieve_ids(query, top_k=10)
                elapsed = time.time() - t0
                total_time += elapsed

                h1_scores.append(hit_at_k(retrieved, gold_ids, 1))
                h5_scores.append(hit_at_k(retrieved, gold_ids, 5))
                h10_scores.append(hit_at_k(retrieved, gold_ids, 10))
                mrr_scores.append(mrr(retrieved, gold_ids))

                if len(retrieved) > 0:
                    non_empty += 1

                if (i + 1) % 10 == 0:
                    print(f"  Evaluating {hop_bin}: {i + 1}/{n}...")

            avg_h1 = sum(h1_scores) / len(h1_scores) if h1_scores else 0.0
            avg_h5 = sum(h5_scores) / len(h5_scores) if h5_scores else 0.0
            avg_h10 = sum(h10_scores) / len(h10_scores) if h10_scores else 0.0
            avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
            avg_time = total_time / n if n > 0 else 0.0

            all_rows.append((hop_bin, rname, avg_h1, avg_h5, avg_h10, avg_mrr, avg_time, non_empty, n))

    # Print results table
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"{'Hop':<8} | {'Method':<12} | {'Hit@1':>7} | {'Hit@5':>7} | {'Hit@10':>7} | {'MRR':>7} | {'avg_time':>9}")
    print("-" * 90)
    for hop_bin, method, h1, h5, h10, mrr_val, avg_t, _ne, _n in all_rows:
        print(
            f"{hop_bin:<8} | {method:<12} | {h1:>7.3f} | {h5:>7.3f} | "
            f"{h10:>7.3f} | {mrr_val:>7.3f} | {avg_t:>8.3f}s"
        )

    # ------------------------------------------------------------------
    # 6. Checkpoint validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECKPOINT VALIDATION")
    print("=" * 60)

    # Gather QCTR-specific rows
    qctr_rows = [(h, m, h1, h5, h10, mr, t, ne, n)
                 for h, m, h1, h5, h10, mr, t, ne, n in all_rows if m == "QCTR"]

    # Check 1: QCTR runtime < 5s per query (average)
    total_qctr_time = sum(t * n for _, _, _, _, _, _, t, _, n in qctr_rows)
    total_qctr_queries = sum(n for _, _, _, _, _, _, _, _, n in qctr_rows)
    avg_qctr_time = total_qctr_time / total_qctr_queries if total_qctr_queries > 0 else 0.0
    runtime_ok = avg_qctr_time < 5.0
    print(f"  QCTR avg runtime: {avg_qctr_time:.3f}s  {'PASS' if runtime_ok else 'FAIL'} (< 5s)")

    # Check 2: QCTR produces non-empty results for all queries
    total_non_empty = sum(ne for _, _, _, _, _, _, _, ne, _ in qctr_rows)
    all_non_empty = total_non_empty == total_qctr_queries
    print(f"  QCTR non-empty results: {total_non_empty}/{total_qctr_queries}  "
          f"{'PASS' if all_non_empty else 'FAIL'}")

    # Check 3: QCTR beats path-centric on 2-hop Hit@10
    qctr_2hop = [r for r in all_rows if r[0] == "2-hop" and r[1] == "QCTR"]
    path_2hop = [r for r in all_rows if r[0] == "2-hop" and r[1] == "path"]

    if qctr_2hop and path_2hop:
        qctr_h10 = qctr_2hop[0][4]
        path_h10 = path_2hop[0][4]
        beats_path = qctr_h10 > path_h10
        print(f"  QCTR vs path on 2-hop Hit@10: QCTR={qctr_h10:.3f} path={path_h10:.3f}  "
              f"{'PASS' if beats_path else 'FAIL'}")
    else:
        print("  QCTR vs path on 2-hop Hit@10: no 2-hop data available  SKIP")

    print()


def _hop_sort_key(hop_bin: str) -> tuple:
    """Sort hop bins numerically, with '6+-hop' going last."""
    prefix = hop_bin.split("-")[0].rstrip("+")
    try:
        return (int(prefix), 0)
    except ValueError:
        return (999, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate QCTR retriever")
    parser.add_argument("--dataset", default="prime", help="Dataset name (default: prime)")
    parser.add_argument("--n-samples", type=int, default=50, help="Samples per hop bin (default: 50)")
    parser.add_argument("--beam-width", type=int, default=10, help="QCTR beam width (default: 10)")
    parser.add_argument("--max-hops", type=int, default=4, help="QCTR max hops (default: 4)")
    parser.add_argument("--device", default="cpu", help="Device for QCTR model (default: cpu)")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        beam_width=args.beam_width,
        max_hops=args.max_hops,
        device=args.device,
    )
