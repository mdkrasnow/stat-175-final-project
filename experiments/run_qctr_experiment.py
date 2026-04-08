"""Full QCTR evaluation on all hop bins with statistical tests and phase transition plot."""

import argparse
import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.stark_loader import load_stark_dataset
from src.data.qctr_data import build_edge_type_lookup
from src.retrievers.shared_index import SharedIndex
from src.retrievers.qctr_retriever import QCTRRetriever
from src.evaluation.metrics import hit_at_k, mrr

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _hop_sort_key(hop_bin: str) -> tuple:
    """Sort hop bins numerically, with '6+-hop' going last."""
    prefix = hop_bin.split("-")[0].rstrip("+")
    try:
        return (int(prefix), 0)
    except ValueError:
        return (999, 0)


def _hop_x_position(hop_bin: str) -> int:
    """Map hop bin name to x-axis position for plotting."""
    mapping = {"1-hop": 1, "2-hop": 2, "3-hop": 3, "4-hop": 4, "5-hop": 5, "6+-hop": 6}
    return mapping.get(hop_bin, 7)


def paired_bootstrap(a, b, n_bootstrap=10000, seed=42):
    """Test if mean(a) > mean(b) via paired bootstrap."""
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    observed_diff = np.mean(a) - np.mean(b)
    n = len(a)
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diff = np.mean(a[idx]) - np.mean(b[idx])
        if boot_diff <= 0:
            count += 1
    p_value = count / n_bootstrap
    return observed_diff, p_value


def load_baseline_results(dataset_name: str):
    """Load the latest baseline experiment and per-query results."""
    exp_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{dataset_name}_experiment_*.json")))
    pq_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{dataset_name}_perquery_*.json")))

    if not exp_files or not pq_files:
        print("ERROR: No baseline experiment results found. Run run_experiment.py first.")
        sys.exit(1)

    with open(exp_files[-1]) as f:
        experiment = json.load(f)
    with open(pq_files[-1]) as f:
        perquery = json.load(f)

    print(f"Loaded baseline: {exp_files[-1]}")
    print(f"Loaded per-query: {pq_files[-1]}")
    return experiment, perquery


def main(dataset_name: str = "prime", beam_width: int = 10, max_hops: int = 4, device: str = "cpu"):
    t_start = time.time()

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
    print(f"\nLoading QCTR model from {model_path}")
    retriever = QCTRRetriever(
        graph,
        shared,
        model_path=model_path,
        edge_type_lookup=edge_type_lookup,
        beam_width=beam_width,
        max_hops=max_hops,
        device=device,
    )
    print("QCTR retriever initialized.\n")

    # ------------------------------------------------------------------
    # Step 2: Run QCTR on all queries
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 2: Running QCTR evaluation on all hop bins")
    print("=" * 70)

    ks = [1, 5, 10]
    qctr_aggregate = {}  # bin_name -> {Hit@1, Hit@5, Hit@10, MRR, n, time_s}
    qctr_perquery = {}   # bin_name -> {QCTR: {per_query_hit1, per_query_mrr}}

    for hop_bin in sorted(hop_bins.keys(), key=_hop_sort_key):
        items = hop_bins[hop_bin]
        if not items:
            continue

        n = len(items)
        print(f"\n  {hop_bin} ({n} queries)")

        hit_scores = {k: [] for k in ks}
        mrr_scores = []
        t0 = time.time()

        for i, item in enumerate(items):
            query, _, gold_ids, _ = qa_dataset[item["index"]]
            predicted = retriever.retrieve_ids(query, top_k=max(ks))

            for k in ks:
                hit_scores[k].append(hit_at_k(predicted, gold_ids, k))
            mrr_scores.append(mrr(predicted, gold_ids))

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {hop_bin}: {i + 1}/{n} queries ({elapsed:.0f}s)")

        elapsed = time.time() - t0
        agg = {}
        for k in ks:
            agg[f"Hit@{k}"] = float(np.mean(hit_scores[k]))
        agg["MRR"] = float(np.mean(mrr_scores))
        agg["n"] = n
        agg["time_s"] = round(elapsed, 1)

        qctr_aggregate[hop_bin] = agg
        qctr_perquery[hop_bin] = {
            "QCTR": {
                "per_query_hit1": [float(x) for x in hit_scores[1]],
                "per_query_mrr": [float(x) for x in mrr_scores],
            }
        }

        print(f"  {hop_bin} done: Hit@1={agg['Hit@1']:.4f}  Hit@5={agg['Hit@5']:.4f}  "
              f"Hit@10={agg['Hit@10']:.4f}  MRR={agg['MRR']:.4f}  ({elapsed:.1f}s)")

    # Save QCTR results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    qctr_exp_path = os.path.join(RESULTS_DIR, f"{dataset_name}_qctr_experiment.json")
    with open(qctr_exp_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "beam_width": beam_width,
            "max_hops": max_hops,
            "results": qctr_aggregate,
        }, f, indent=2)
    print(f"\nQCTR aggregate results saved to {qctr_exp_path}")

    qctr_pq_path = os.path.join(RESULTS_DIR, f"{dataset_name}_qctr_perquery.json")
    with open(qctr_pq_path, "w") as f:
        json.dump(qctr_perquery, f)
    print(f"QCTR per-query results saved to {qctr_pq_path}")

    # ------------------------------------------------------------------
    # Step 3: Load existing baseline results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3: Loading baseline results")
    print("=" * 70)

    baseline_exp, baseline_pq = load_baseline_results(dataset_name)
    baseline_results = baseline_exp["results"]

    # ------------------------------------------------------------------
    # Step 4: Merged results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Merged comparison table")
    print("=" * 70)

    all_hop_bins = sorted(
        set(list(baseline_results.keys()) + list(qctr_aggregate.keys())),
        key=_hop_sort_key,
    )

    baseline_methods = []
    if all_hop_bins:
        first_bin = all_hop_bins[0]
        if first_bin in baseline_results:
            baseline_methods = list(baseline_results[first_bin].keys())

    print(f"\n{'Hop':<8} | {'Method':<18} | {'Hit@1':>7} | {'Hit@5':>7} | {'Hit@10':>7} | {'MRR':>7}")
    print("-" * 75)

    for hop_bin in all_hop_bins:
        # Baseline methods
        if hop_bin in baseline_results:
            for method in baseline_methods:
                if method in baseline_results[hop_bin]:
                    r = baseline_results[hop_bin][method]
                    print(f"{hop_bin:<8} | {method:<18} | {r['Hit@1']:>7.3f} | "
                          f"{r['Hit@5']:>7.3f} | {r['Hit@10']:>7.3f} | {r['MRR']:>7.3f}")

        # QCTR
        if hop_bin in qctr_aggregate:
            r = qctr_aggregate[hop_bin]
            print(f"{hop_bin:<8} | {'QCTR':<18} | {r['Hit@1']:>7.3f} | "
                  f"{r['Hit@5']:>7.3f} | {r['Hit@10']:>7.3f} | {r['MRR']:>7.3f}")

    # ------------------------------------------------------------------
    # Step 5: Paired bootstrap test (QCTR vs path_centric)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5: Paired bootstrap tests (QCTR vs path_centric)")
    print("=" * 70)

    print(f"\n{'Hop':<8} | {'Metric':<7} | {'QCTR':>8} | {'path':>8} | {'diff':>8} | {'p-value':>8} | sig?")
    print("-" * 70)

    bootstrap_results = {}

    for hop_bin in all_hop_bins:
        if hop_bin not in qctr_perquery or hop_bin not in baseline_pq:
            continue
        if "QCTR" not in qctr_perquery[hop_bin] or "path_centric" not in baseline_pq[hop_bin]:
            continue

        qctr_pq_data = qctr_perquery[hop_bin]["QCTR"]
        path_pq_data = baseline_pq[hop_bin]["path_centric"]

        bootstrap_results[hop_bin] = {}

        for metric_key, label in [("per_query_hit1", "Hit@1"), ("per_query_mrr", "MRR")]:
            a = qctr_pq_data[metric_key]
            b = path_pq_data[metric_key]

            if len(a) != len(b):
                print(f"  WARNING: {hop_bin} {label} length mismatch: QCTR={len(a)} path={len(b)}")
                continue

            diff, p_val = paired_bootstrap(a, b)
            mean_a = float(np.mean(a))
            mean_b = float(np.mean(b))

            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = ""

            print(f"{hop_bin:<8} | {label:<7} | {mean_a:>8.3f} | {mean_b:>8.3f} | "
                  f"{diff:>+8.3f} | {p_val:>8.4f} | {sig}")

            bootstrap_results[hop_bin][label] = {
                "qctr_mean": mean_a,
                "path_mean": mean_b,
                "diff": float(diff),
                "p_value": float(p_val),
            }

    # ------------------------------------------------------------------
    # Step 6: Phase transition plot
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 6: Generating phase transition plot with QCTR")
    print("=" * 70)

    # Merge all methods into one structure for plotting
    merged = {}
    for hop_bin in all_hop_bins:
        merged[hop_bin] = {}
        if hop_bin in baseline_results:
            for method, r in baseline_results[hop_bin].items():
                merged[hop_bin][method] = r
        if hop_bin in qctr_aggregate:
            merged[hop_bin]["QCTR"] = qctr_aggregate[hop_bin]

    # Determine methods and hop positions
    hop_labels = sorted(all_hop_bins, key=_hop_sort_key)
    hop_positions = [_hop_x_position(h) for h in hop_labels]

    strategy_labels = {
        "node_centric": "Node-centric (text-only)",
        "path_centric": "Path-centric",
        "subgraph_centric": "Subgraph-centric",
        "hybrid_mor": "Hybrid (MoR)",
        "QCTR": "QCTR (ours)",
    }
    colors = {
        "node_centric": "#e74c3c",
        "path_centric": "#3498db",
        "subgraph_centric": "#2ecc71",
        "hybrid_mor": "#9b59b6",
        "QCTR": "red",
    }
    markers = {
        "node_centric": "o",
        "path_centric": "s",
        "subgraph_centric": "^",
        "hybrid_mor": "v",
        "QCTR": "D",
    }

    all_methods = baseline_methods + ["QCTR"]

    metrics = ["Hit@1", "Hit@5", "Hit@10", "MRR"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for method in all_methods:
            values = []
            xs = []
            for hop_bin, xpos in zip(hop_labels, hop_positions):
                if hop_bin in merged and method in merged[hop_bin] and metric in merged[hop_bin][method]:
                    values.append(merged[hop_bin][method][metric])
                    xs.append(xpos)

            if not values:
                continue

            is_qctr = method == "QCTR"
            ax.plot(
                xs, values,
                marker=markers.get(method, "o"),
                linestyle="--" if is_qctr else "-",
                label=strategy_labels.get(method, method),
                color=colors.get(method, None),
                linewidth=2.5 if is_qctr else 1.5,
                markersize=8 if is_qctr else 6,
            )

        ax.set_xlabel("Query Hop Count")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Query Complexity")
        ax.legend(fontsize=8)
        ax.set_xticks(hop_positions)
        ax.set_xticklabels(hop_labels, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Phase Transition in Retrieval Performance (with QCTR) -- STaRK-{dataset_name}", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase_transition_qctr.png")
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved phase transition plot to {plot_path}")

    # ------------------------------------------------------------------
    # Step 7: Checkpoint validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 7: Checkpoint validation")
    print("=" * 70)

    # Check 1: QCTR 2-hop Hit@10 > path_centric 2-hop Hit@10
    qctr_2hop_h10 = qctr_aggregate.get("2-hop", {}).get("Hit@10", 0)
    path_2hop_h10 = baseline_results.get("2-hop", {}).get("path_centric", {}).get("Hit@10", 0)
    check1 = qctr_2hop_h10 > path_2hop_h10
    print(f"  QCTR 2-hop Hit@10 ({qctr_2hop_h10:.3f}) > path_centric 2-hop Hit@10 ({path_2hop_h10:.3f}): "
          f"{'PASS' if check1 else 'FAIL'}")

    # Check 2: QCTR 2-hop Hit@10 > 0
    check2 = qctr_2hop_h10 > 0
    print(f"  QCTR 2-hop Hit@10 ({qctr_2hop_h10:.3f}) > 0 (non-trivial): "
          f"{'PASS' if check2 else 'FAIL'}")

    # Check 3: Bootstrap p < 0.05 for QCTR vs path on 2-hop Hit@1
    p_val_2hop_hit1 = bootstrap_results.get("2-hop", {}).get("Hit@1", {}).get("p_value", 1.0)
    check3 = p_val_2hop_hit1 < 0.05
    print(f"  Bootstrap p-value for 2-hop Hit@1 ({p_val_2hop_hit1:.4f}) < 0.05: "
          f"{'PASS' if check3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Total runtime
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full QCTR evaluation experiment")
    parser.add_argument("--dataset", default="prime", help="Dataset name (default: prime)")
    parser.add_argument("--beam-width", type=int, default=10, help="QCTR beam width (default: 10)")
    parser.add_argument("--max-hops", type=int, default=4, help="QCTR max hops (default: 4)")
    parser.add_argument("--device", default="cpu", help="Device for QCTR model (default: cpu)")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        beam_width=args.beam_width,
        max_hops=args.max_hops,
        device=args.device,
    )
