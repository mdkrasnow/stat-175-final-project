"""Phase 4: Statistical analysis of phase transition in retrieval performance."""

import json
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_latest_results(dataset_name: str = "prime"):
    """Load the most recent experiment + per-query results."""
    exp_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{dataset_name}_experiment_*.json")))
    pq_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{dataset_name}_perquery_*.json")))

    if not exp_files or not pq_files:
        print("No experiment results found. Run run_experiment.py first.")
        sys.exit(1)

    with open(exp_files[-1]) as f:
        experiment = json.load(f)
    with open(pq_files[-1]) as f:
        perquery = json.load(f)

    print(f"Loaded: {exp_files[-1]}")
    return experiment, perquery


def plot_phase_transition(experiment: dict, dataset_name: str = "prime"):
    """Plot Hit@k and MRR as a function of hop count for each strategy — the main figure."""
    results = experiment["results"]

    # Parse hop bins and sort
    hop_labels = []
    hop_nums = []
    for bin_name in results.keys():
        hop_labels.append(bin_name)
        if "+" in bin_name:
            hop_nums.append(int(bin_name.split("+")[0]))
        else:
            hop_nums.append(int(bin_name.split("-")[0]))

    order = np.argsort(hop_nums)
    hop_labels = [hop_labels[i] for i in order]
    hop_nums = [hop_nums[i] for i in order]

    strategies = list(next(iter(results.values())).keys())
    strategy_labels = {
        "node_centric": "Node-centric (text-only)",
        "path_centric": "Path-centric",
        "subgraph_centric": "Subgraph-centric",
        "hybrid_mor": "Hybrid (MoR)",
    }
    colors = {
        "node_centric": "#e74c3c",
        "path_centric": "#3498db",
        "subgraph_centric": "#2ecc71",
        "hybrid_mor": "#9b59b6",
    }

    # Plot Hit@1, Hit@5, Hit@10, MRR
    metrics = ["Hit@1", "Hit@5", "Hit@10", "MRR"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for strat in strategies:
            values = []
            for label in hop_labels:
                values.append(results[label][strat][metric])
            ax.plot(hop_nums, values, "o-", label=strategy_labels.get(strat, strat),
                    color=colors.get(strat, None), linewidth=2, markersize=6)

        ax.set_xlabel("Query Hop Count")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Query Complexity")
        ax.legend(fontsize=8)
        ax.set_xticks(hop_nums)
        ax.set_xticklabels(hop_labels, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Phase Transition in Retrieval Performance — STaRK-{dataset_name}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase_transition.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved phase transition plot to {path}")


def paired_bootstrap_test(a: list, b: list, n_bootstrap: int = 10000) -> dict:
    """Paired bootstrap test: is mean(a) significantly different from mean(b)?"""
    a = np.array(a)
    b = np.array(b)
    observed_diff = np.mean(a) - np.mean(b)

    diffs = a - b
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        boot_diffs.append(np.mean(sample))

    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(boot_diffs <= 0) if observed_diff > 0 else np.mean(boot_diffs >= 0)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95": (float(ci_lower), float(ci_upper)),
    }


def cohens_d(a: list, b: list) -> float:
    """Cohen's d effect size."""
    a = np.array(a)
    b = np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def run_statistical_tests(perquery: dict, dataset_name: str = "prime"):
    """Run pairwise statistical tests at each hop count."""
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: Structural vs Node-centric (baseline)")
    print("=" * 70)

    structural_strategies = ["path_centric", "subgraph_centric", "hybrid_mor"]
    all_test_results = {}

    for bin_name, bin_data in perquery.items():
        if "node_centric" not in bin_data:
            continue

        baseline_hit1 = bin_data["node_centric"]["per_query_hit1"]
        baseline_mrr = bin_data["node_centric"]["per_query_mrr"]
        n = len(baseline_hit1)

        print(f"\n--- {bin_name} (n={n}) ---")
        bin_tests = {}

        for strat in structural_strategies:
            if strat not in bin_data:
                continue

            strat_hit1 = bin_data[strat]["per_query_hit1"]
            strat_mrr = bin_data[strat]["per_query_mrr"]

            # Hit@1 bootstrap test
            hit1_test = paired_bootstrap_test(strat_hit1, baseline_hit1)
            hit1_d = cohens_d(strat_hit1, baseline_hit1)

            # MRR bootstrap test
            mrr_test = paired_bootstrap_test(strat_mrr, baseline_mrr)
            mrr_d = cohens_d(strat_mrr, baseline_mrr)

            sig_hit1 = "*" if hit1_test["p_value"] < 0.05 else ""
            sig_mrr = "*" if mrr_test["p_value"] < 0.05 else ""

            print(f"  {strat} vs node_centric:")
            print(f"    Hit@1: diff={hit1_test['observed_diff']:+.4f}, p={hit1_test['p_value']:.4f}{sig_hit1}, d={hit1_d:.3f}")
            print(f"    MRR:   diff={mrr_test['observed_diff']:+.4f}, p={mrr_test['p_value']:.4f}{sig_mrr}, d={mrr_d:.3f}")

            bin_tests[strat] = {
                "hit1": {**hit1_test, "cohens_d": hit1_d},
                "mrr": {**mrr_test, "cohens_d": mrr_d},
            }

        all_test_results[bin_name] = bin_tests

    # Save
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_statistical_tests.json")
    with open(path, "w") as f:
        json.dump(all_test_results, f, indent=2)
    print(f"\nSaved statistical tests to {path}")

    return all_test_results


def interaction_analysis(perquery: dict, dataset_name: str = "prime"):
    """Logistic regression: strategy x hop-count interaction on Hit@1."""
    print("\n" + "=" * 70)
    print("INTERACTION ANALYSIS: Strategy x Hop-Count on Hit@1")
    print("=" * 70)

    # Build design matrix
    X_rows = []
    y_rows = []

    for bin_name, bin_data in perquery.items():
        if "+" in bin_name:
            hop = int(bin_name.split("+")[0])
        else:
            hop = int(bin_name.split("-")[0])

        for strat_name, strat_data in bin_data.items():
            is_structural = 1 if strat_name != "node_centric" else 0
            for hit1 in strat_data["per_query_hit1"]:
                X_rows.append([hop, is_structural, hop * is_structural])
                y_rows.append(hit1)

    X = np.array(X_rows)
    y = np.array(y_rows)

    # Check if y has both classes
    if len(np.unique(y)) < 2:
        print("  Cannot fit logistic regression — only one class in y.")
        return

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    feature_names = ["hop_count", "is_structural", "hop_x_structural"]
    print(f"\n  Logistic Regression coefficients:")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"    {name}: {coef:.4f}")
    print(f"    intercept: {model.intercept_[0]:.4f}")

    # The interaction term (hop_x_structural) tells us whether structural
    # methods gain/lose more per hop relative to text-only
    interaction_coef = model.coef_[0][2]
    print(f"\n  Interaction (hop x structural): {interaction_coef:.4f}")
    if interaction_coef > 0:
        print("  -> Structural methods GAIN relative advantage as hop count increases")
    else:
        print("  -> Structural methods LOSE relative advantage as hop count increases")


def main():
    experiment, perquery = load_latest_results("prime")

    # 1. Phase transition plots
    plot_phase_transition(experiment, "prime")

    # 2. Pairwise statistical tests
    run_statistical_tests(perquery, "prime")

    # 3. Interaction analysis
    interaction_analysis(perquery, "prime")

    print("\n" + "=" * 70)
    print("Phase 4 analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
