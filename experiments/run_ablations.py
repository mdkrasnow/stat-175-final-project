"""Phase 5: QCTR ablation study on a sample of queries per hop bin."""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.data.qctr_data import build_edge_type_lookup
from src.retrievers.shared_index import SharedIndex
from src.retrievers.qctr_retriever import QCTRRetriever
from src.evaluation.metrics import hit_at_k, mrr

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

ABLATIONS = [
    {"name": "QCTR (full)",       "scoring_mode": "learned", "use_edge_types": True,  "beam_width": 10, "max_hops": 4},
    {"name": "cosine-only beam",  "scoring_mode": "cosine",  "use_edge_types": True,  "beam_width": 10, "max_hops": 4},
    {"name": "no edge types",     "scoring_mode": "learned", "use_edge_types": False, "beam_width": 10, "max_hops": 4},
    {"name": "beam_width=5",      "scoring_mode": "learned", "use_edge_types": True,  "beam_width": 5,  "max_hops": 4},
    {"name": "beam_width=20",     "scoring_mode": "learned", "use_edge_types": True,  "beam_width": 20, "max_hops": 4},
    {"name": "beam_width=30",     "scoring_mode": "learned", "use_edge_types": True,  "beam_width": 30, "max_hops": 4},
    {"name": "max_hops=2",        "scoring_mode": "learned", "use_edge_types": True,  "beam_width": 10, "max_hops": 2},
    {"name": "max_hops=6",        "scoring_mode": "learned", "use_edge_types": True,  "beam_width": 10, "max_hops": 6},
]


def _hop_sort_key(hop_bin):
    """Sort hop bins numerically, with '6+-hop' going last."""
    prefix = hop_bin.split("-")[0].rstrip("+")
    try:
        return (int(prefix), 0)
    except ValueError:
        return (999, 0)


def evaluate_config(config, graph, shared, model_path, edge_type_lookup,
                    qa_dataset, hop_bins, n_samples, device):
    """Evaluate one ablation config across all hop bins.

    Returns dict: {hop_bin: {"Hit@1": x, "Hit@5": x, "Hit@10": x, "MRR": x, "n": n}}
    """
    retriever = QCTRRetriever(
        graph, shared,
        model_path=model_path,
        edge_type_lookup=edge_type_lookup,
        beam_width=config["beam_width"],
        max_hops=config["max_hops"],
        device=device,
        scoring_mode=config["scoring_mode"],
        use_edge_types=config["use_edge_types"],
    )

    ks = [1, 5, 10]
    results = {}

    for hop_bin in sorted(hop_bins.keys(), key=_hop_sort_key):
        items = hop_bins[hop_bin]
        if not items:
            continue

        sample = items[:min(n_samples, len(items))]
        n = len(sample)
        print(f"    {hop_bin} ({n} queries) ...", end=" ", flush=True)

        hit_scores = {k: [] for k in ks}
        mrr_scores = []

        for item in sample:
            query, _, gold_ids, _ = qa_dataset[item["index"]]
            predicted = retriever.retrieve_ids(query, top_k=max(ks))
            for k in ks:
                hit_scores[k].append(hit_at_k(predicted, gold_ids, k))
            mrr_scores.append(mrr(predicted, gold_ids))

        agg = {}
        for k in ks:
            agg[f"Hit@{k}"] = float(np.mean(hit_scores[k]))
        agg["MRR"] = float(np.mean(mrr_scores))
        agg["n"] = n

        results[hop_bin] = agg
        print(f"Hit@10={agg['Hit@10']:.3f}  MRR={agg['MRR']:.3f}")

    return results


def bridge_entity_recall(graph, shared, model_path, edge_type_lookup,
                         qa_dataset, hop_bins, n_samples, device):
    """Measure whether correct intermediate nodes appear in the beam after hop 1 for 2-hop queries."""
    if "2-hop" not in hop_bins or not hop_bins["2-hop"]:
        print("No 2-hop queries available for bridge entity recall analysis.")
        return

    items = hop_bins["2-hop"][:min(n_samples, len(hop_bins["2-hop"]))]

    for mode_label, scoring_mode in [("learned", "learned"), ("cosine", "cosine")]:
        retriever = QCTRRetriever(
            graph, shared,
            model_path=model_path,
            edge_type_lookup=edge_type_lookup,
            beam_width=10,
            max_hops=4,
            device=device,
            scoring_mode=scoring_mode,
            use_edge_types=True,
        )

        retained = 0
        total = 0

        for item in items:
            query, _, gold_ids, _ = qa_dataset[item["index"]]
            query_node = item["query_node"]

            # Find bridge nodes from gold paths
            bridge_nodes = set()
            for gold_id in gold_ids:
                try:
                    path = nx.shortest_path(graph.graph, query_node, gold_id)
                except nx.NetworkXNoPath:
                    continue
                if len(path) == 3:  # query_node -> bridge -> gold
                    bridge_nodes.add(path[1])

            if not bridge_nodes:
                continue

            total += 1

            # Run one hop of beam search manually
            query_emb = shared.encode_query(query).flatten()
            start = retriever._get_query_entity(query_emb.reshape(1, -1))

            neighbors = list(graph.graph.neighbors(start))
            if not neighbors:
                continue

            scored = retriever._score_transitions(query_emb, start, neighbors)
            top_candidates = {node_id for node_id, _ in scored[:retriever.beam_width]}

            if bridge_nodes & top_candidates:
                retained += 1

        recall = retained / total if total > 0 else 0.0
        print(f"  {mode_label} scoring: bridge recall = {recall:.3f} ({retained}/{total})")


def density_analysis_with_retriever(graph, shared, model_path, edge_type_lookup,
                                    qa_dataset, hop_bins, n_samples, device):
    """Run density analysis with actual per-query evaluation split by degree."""
    retriever = QCTRRetriever(
        graph, shared,
        model_path=model_path,
        edge_type_lookup=edge_type_lookup,
        beam_width=10,
        max_hops=4,
        device=device,
        scoring_mode="learned",
        use_edge_types=True,
    )

    for hop_label in ["2-hop", "3-hop"]:
        if hop_label not in hop_bins or not hop_bins[hop_label]:
            continue

        items = hop_bins[hop_label][:min(n_samples, len(hop_bins[hop_label]))]
        degrees = [graph.graph.degree(item["query_node"]) for item in items]
        median_deg = float(np.median(degrees))

        low_hits = []
        high_hits = []

        for item, deg in zip(items, degrees):
            query, _, gold_ids, _ = qa_dataset[item["index"]]
            predicted = retriever.retrieve_ids(query, top_k=10)
            h10 = hit_at_k(predicted, gold_ids, 10)

            if deg <= median_deg:
                low_hits.append(h10)
            else:
                high_hits.append(h10)

        low_mean = float(np.mean(low_hits)) if low_hits else 0.0
        high_mean = float(np.mean(high_hits)) if high_hits else 0.0

        print(f"  {hop_label}: Low degree: Hit@10={low_mean:.3f} (n={len(low_hits)}), "
              f"High degree: Hit@10={high_mean:.3f} (n={len(high_hits)})")


def main(dataset_name: str = "prime", n_samples: int = 200, device: str = "cpu"):
    t_start = time.time()

    # Estimate runtime
    est_seconds = n_samples * 6 * len(ABLATIONS) * 0.6
    print(f"Estimated runtime: ~{est_seconds / 60:.0f} min "
          f"({n_samples} samples x 6 bins x {len(ABLATIONS)} configs x ~0.6s/query)")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Step 1: Loading STaRK-{dataset_name} and building indices")
    print("=" * 70)

    graph, qa_dataset = load_stark_dataset(dataset_name)

    strat_path = os.path.join(RESULTS_DIR, f"{dataset_name}_hop_stratification.json")
    if not os.path.exists(strat_path):
        print(f"ERROR: Run phase1_analysis.py first to generate {strat_path}")
        sys.exit(1)

    with open(strat_path) as f:
        hop_bins = json.load(f)

    for hop_bin in sorted(hop_bins.keys(), key=_hop_sort_key):
        print(f"  {hop_bin}: {len(hop_bins[hop_bin])} queries")

    print("\nBuilding shared embedding index...")
    t0 = time.time()
    shared = SharedIndex(graph)
    print(f"Shared index built in {time.time() - t0:.1f}s")

    print("\nBuilding edge type lookup...")
    t0 = time.time()
    edge_type_lookup = build_edge_type_lookup(graph.skb)
    print(f"Edge type lookup built in {time.time() - t0:.1f}s")

    model_path = os.path.join(PROJECT_ROOT, "models", "qctr", "best_model.pt")

    # ------------------------------------------------------------------
    # Step 2: Run all ablation configs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2: Running ablation configs")
    print("=" * 70)

    results_by_config = {}

    for i, config in enumerate(ABLATIONS):
        print(f"\n  [{i + 1}/{len(ABLATIONS)}] {config['name']}")
        t0 = time.time()
        results_by_config[config["name"]] = evaluate_config(
            config, graph, shared, model_path, edge_type_lookup,
            qa_dataset, hop_bins, n_samples, device,
        )
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Print results tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3: Results tables")
    print("=" * 70)

    all_hop_bins = sorted(
        set(h for r in results_by_config.values() for h in r.keys()),
        key=_hop_sort_key,
    )

    # Hit@10 table
    header_bins = "".join(f" | {h:>7}" for h in all_hop_bins)
    print(f"\n{'Config':<22}{header_bins}")
    sep = "-" * 22 + "".join(" | -------" for _ in all_hop_bins)
    print(sep)

    for config in ABLATIONS:
        name = config["name"]
        row = f"{name:<22}"
        for hop_bin in all_hop_bins:
            val = results_by_config.get(name, {}).get(hop_bin, {}).get("Hit@10", float("nan"))
            row += f" | {val:>7.3f}"
        print(row)

    # MRR table
    print(f"\n{'Config':<22}" + "".join(f" | {h:>7}" for h in all_hop_bins))
    print(sep)

    for config in ABLATIONS:
        name = config["name"]
        row = f"{name:<22}"
        for hop_bin in all_hop_bins:
            val = results_by_config.get(name, {}).get(hop_bin, {}).get("MRR", float("nan"))
            row += f" | {val:>7.3f}"
        print(row)

    # ------------------------------------------------------------------
    # Step 4: Bridge entity recall
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Bridge entity recall (2-hop queries)")
    print("=" * 70)

    bridge_entity_recall(graph, shared, model_path, edge_type_lookup,
                         qa_dataset, hop_bins, n_samples, device)

    # ------------------------------------------------------------------
    # Step 5: Density analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5: Density analysis (degree vs performance)")
    print("=" * 70)

    density_analysis_with_retriever(graph, shared, model_path, edge_type_lookup,
                                    qa_dataset, hop_bins, n_samples, device)

    # ------------------------------------------------------------------
    # Step 6: Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 6: Saving results")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "ablations": {},
    }
    for config in ABLATIONS:
        output["ablations"][config["name"]] = {
            "config": {k: v for k, v in config.items() if k != "name"},
            "results": results_by_config.get(config["name"], {}),
        }

    out_path = os.path.join(RESULTS_DIR, "prime_ablations_qctr.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Step 7: Ablation plot
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 7: Generating ablation plot")
    print("=" * 70)

    focus_hops = [h for h in ["2-hop", "3-hop", "4-hop"] if h in all_hop_bins]
    config_names = [c["name"] for c in ABLATIONS]
    n_configs = len(config_names)
    n_hops = len(focus_hops)

    if n_hops > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n_configs)
        bar_width = 0.8 / max(n_hops, 1)

        cmap = plt.colormaps["Set2"]
        colors = [cmap(i / max(n_hops - 1, 1)) for i in range(n_hops)]

        for j, hop_bin in enumerate(focus_hops):
            values = []
            for config in ABLATIONS:
                val = results_by_config.get(config["name"], {}).get(hop_bin, {}).get("Hit@10", 0.0)
                values.append(val)
            offset = (j - n_hops / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=hop_bin, color=colors[j])

        ax.set_xlabel("Ablation Configuration")
        ax.set_ylabel("Hit@10")
        ax.set_title(f"QCTR Ablation Study -- Hit@10 by Hop Count (STaRK-{dataset_name})")
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        plot_path = os.path.join(RESULTS_DIR, "prime_ablation_qctr.png")
        fig.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved ablation plot to {plot_path}")
    else:
        print("No focus hops available for plotting.")

    # ------------------------------------------------------------------
    # Total runtime
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: QCTR ablation study")
    parser.add_argument("--dataset", default="prime", help="Dataset name (default: prime)")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per hop bin (default: 200)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    main(dataset_name=args.dataset, n_samples=args.n_samples, device=args.device)
