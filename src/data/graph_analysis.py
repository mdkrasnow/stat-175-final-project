"""Characterize graph properties and stratify QA pairs by hop count."""

import json
import os
import random
from collections import Counter

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.data.stark_loader import StarkGraphWrapper


def characterize_graph(graph: StarkGraphWrapper, dataset_name: str, output_dir: str = "results"):
    """Compute and save graph properties: degree distribution, diameter, clustering."""
    G = graph.graph
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Graph Properties: STaRK-{dataset_name} ===")

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Density: {nx.density(G):.6f}")

    degrees = [d for _, d in G.degree()]
    print(f"Mean degree: {np.mean(degrees):.2f}")
    print(f"Median degree: {np.median(degrees):.1f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")

    # Clustering coefficient (sample if graph is large)
    if n_nodes > 10000:
        sample_nodes = random.sample(list(G.nodes()), 5000)
        avg_clustering = nx.average_clustering(G, nodes=sample_nodes)
        print(f"Avg clustering coefficient (sampled 5000 nodes): {avg_clustering:.4f}")
    else:
        avg_clustering = nx.average_clustering(G)
        print(f"Avg clustering coefficient: {avg_clustering:.4f}")

    # Connected components
    components = list(nx.connected_components(G))
    print(f"Connected components: {len(components)}")
    largest_cc = max(components, key=len)
    print(f"Largest component: {len(largest_cc)} nodes ({100*len(largest_cc)/n_nodes:.1f}%)")

    # Approximate diameter via BFS from a few random nodes with cutoff
    largest_subgraph = G.subgraph(largest_cc)
    sample = random.sample(list(largest_cc), min(10, len(largest_cc)))
    approx_diameter = 0
    for node in sample:
        lengths = nx.single_source_shortest_path_length(largest_subgraph, node, cutoff=15)
        if lengths:
            approx_diameter = max(approx_diameter, max(lengths.values()))
    print(f"Approximate diameter (sampled 10 nodes, cutoff=15): {approx_diameter}")

    # Save degree distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(degrees, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Degree Distribution — STaRK-{dataset_name}")

    degree_counts = Counter(degrees)
    degs = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degs]
    axes[1].loglog(degs, counts, "o", markersize=3, alpha=0.7)
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title(f"Log-Log Degree Distribution — STaRK-{dataset_name}")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset_name}_graph_properties.png"), dpi=150)
    plt.close()
    print(f"Saved degree distribution plot to {output_dir}/{dataset_name}_graph_properties.png")

    summary = {
        "dataset": dataset_name,
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": nx.density(G),
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(max(degrees)),
        "avg_clustering": float(avg_clustering),
        "num_components": len(components),
        "largest_component_size": len(largest_cc),
        "largest_component_pct": 100 * len(largest_cc) / n_nodes,
        "approx_diameter": approx_diameter,
    }
    with open(os.path.join(output_dir, f"{dataset_name}_graph_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def stratify_by_hop_count(
    graph: StarkGraphWrapper,
    qa_dataset,
    dataset_name: str,
    output_dir: str = "results",
    max_hops: int = 6,
) -> dict[str, list]:
    """Stratify QA pairs by shortest path from query-relevant node to answer.

    Approach: embed the query, find the top-1 closest node in the graph
    (the "query entity"), then measure shortest path distance from that
    node to each gold answer node. The minimum such distance is the hop count.
    """
    os.makedirs(output_dir, exist_ok=True)
    G = graph.graph

    # Build FAISS index for fast query-to-node matching
    from sentence_transformers import SentenceTransformer
    import faiss

    print("Building embedding index for query-entity matching...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    node_ids = sorted(graph.node_texts.keys())
    node_texts_list = [graph.node_texts[nid] for nid in node_ids]

    print(f"  Encoding {len(node_texts_list)} node texts...")
    node_embeddings = encoder.encode(
        node_texts_list, show_progress_bar=True, normalize_embeddings=True, batch_size=256
    ).astype(np.float32)

    index = faiss.IndexFlatIP(node_embeddings.shape[1])
    index.add(node_embeddings)
    print(f"  FAISS index built with {index.ntotal} vectors.")

    # Stratify
    bins = {f"{h}-hop": [] for h in range(1, max_hops)}
    bins[f"{max_hops}+-hop"] = []
    skipped = 0

    print(f"Stratifying {len(qa_dataset)} QA pairs by hop count...")

    # Batch encode all queries
    all_queries = [qa_dataset[i][0] for i in range(len(qa_dataset))]
    print(f"  Encoding {len(all_queries)} queries...")
    query_embeddings = encoder.encode(
        all_queries, show_progress_bar=True, normalize_embeddings=True, batch_size=256
    ).astype(np.float32)

    # For each query, find the closest node, then shortest path to gold
    for i in range(len(qa_dataset)):
        query, query_id, gold_ids, _ = qa_dataset[i]

        # Find top-1 node closest to query by embedding
        q_emb = query_embeddings[i:i+1]
        _, top_indices = index.search(q_emb, 1)
        query_node = node_ids[top_indices[0][0]]

        # Shortest path from query_node to any gold answer node
        min_dist = float("inf")
        for gold_id in gold_ids:
            if gold_id not in G:
                continue
            try:
                dist = nx.shortest_path_length(G, query_node, gold_id)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue

        if min_dist == float("inf"):
            skipped += 1
            continue

        hop_count = int(min_dist)
        if hop_count == 0:
            hop_count = 1  # query node is the answer itself; treat as 1-hop

        if hop_count < max_hops:
            bin_key = f"{hop_count}-hop"
        else:
            bin_key = f"{max_hops}+-hop"

        bins[bin_key].append({
            "index": i,
            "query": query,
            "query_id": query_id,
            "gold_ids": gold_ids,
            "hop_count": hop_count,
            "query_node": query_node,
        })

        if (i + 1) % 1000 == 0:
            print(f"    ...processed {i + 1}/{len(qa_dataset)} queries")

    # Print summary
    print(f"\n=== Hop Count Stratification: STaRK-{dataset_name} ===")
    print(f"Total QA pairs: {len(qa_dataset)}")
    print(f"Skipped (no path to answer): {skipped}")
    for bin_name, items in bins.items():
        status = "OK" if len(items) >= 50 else "LOW — may lack statistical power"
        print(f"  {bin_name}: {len(items)} queries [{status}]")

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_names = list(bins.keys())
    bin_counts = [len(bins[b]) for b in bin_names]
    ax.bar(bin_names, bin_counts, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Hop Count")
    ax.set_ylabel("Number of QA Pairs")
    ax.set_title(f"QA Distribution by Hop Count — STaRK-{dataset_name}")
    for j, count in enumerate(bin_counts):
        ax.text(j, count + 1, str(count), ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{dataset_name}_hop_distribution.png"), dpi=150)
    plt.close()
    print(f"Saved hop distribution plot to {output_dir}/{dataset_name}_hop_distribution.png")

    # Save stratification as JSON
    serializable_bins = {}
    for bin_name, items in bins.items():
        serializable_bins[bin_name] = [
            {"index": it["index"], "query_id": it["query_id"],
             "gold_ids": [int(g) for g in it["gold_ids"]], "hop_count": it["hop_count"],
             "query_node": it["query_node"]}
            for it in items
        ]
    with open(os.path.join(output_dir, f"{dataset_name}_hop_stratification.json"), "w") as f:
        json.dump(serializable_bins, f, indent=2)

    return bins
