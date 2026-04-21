"""Generate paper figures from results/*.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGS = PROJECT_ROOT / "results"

SCHEMAS_SHORT = {
    "A_drug_gene_disease": "A: drug–gene–disease",
    "B_drug_pathway_disease": "B: +pathway",
    "C_disease_phenotype_gene": "C: disease–phenotype–gene",
    "D_drug_disease_direct": "D: drug–disease direct",
}
SAMPLERS = ["deepwalk", "node2vec_bfs", "node2vec_dfs", "graphsage"]
SAMPLER_LABELS = {
    "deepwalk": "DeepWalk",
    "node2vec_bfs": "n2v-BFS",
    "node2vec_dfs": "n2v-DFS",
    "graphsage": "GraphSAGE",
}
SAMPLER_COLORS = {
    "deepwalk": "#1f77b4",
    "node2vec_bfs": "#ff7f0e",
    "node2vec_dfs": "#2ca02c",
    "graphsage": "#d62728",
}


def fig_in_dist_auc_gap():
    data = json.loads((RESULTS / "in_dist_tau.json").read_text())
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    schema_keys = list(SCHEMAS_SHORT.keys())
    x = np.arange(len(schema_keys))
    width = 0.2
    for i, sampler in enumerate(SAMPLERS):
        gaps = [data[s][sampler]["auc_gap"] for s in schema_keys]
        ax.bar(x + i * width - 1.5 * width, gaps, width,
               label=SAMPLER_LABELS[sampler], color=SAMPLER_COLORS[sampler])
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEMAS_SHORT[s] for s in schema_keys], rotation=10)
    ax.set_ylabel(r"$\Delta$AUC = AUC(text+structure) – AUC(text)")
    ax.set_title("In-distribution residual AUC from each structural sampler")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "fig_in_dist_auc_gap.png", dpi=150)
    plt.close(fig)


def fig_text_strength_vs_structural_value():
    data = json.loads((RESULTS / "in_dist_tau.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for sampler in SAMPLERS:
        xs, ys = [], []
        for s in SCHEMAS_SHORT:
            xs.append(data[s][sampler]["auc_T"])
            ys.append(data[s][sampler]["auc_gap"])
        ax.scatter(xs, ys, label=SAMPLER_LABELS[sampler],
                   color=SAMPLER_COLORS[sampler], s=80, alpha=0.85)
    ax.set_xlabel(r"AUC$_T$ (text-only baseline)")
    ax.set_ylabel(r"$\Delta$AUC from structure")
    ax.set_title("Structural value anti-correlates with text adequacy")
    ax.invert_xaxis()  # so left = weaker text
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "fig_text_vs_structure.png", dpi=150)
    plt.close(fig)


def fig_hop_stratified():
    data = json.loads((RESULTS / "hop_stratified_tau.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    hops = ["2", "3", "4+"]
    for ax, schema in zip(axes, SCHEMAS_SHORT.keys()):
        for sampler in SAMPLERS:
            ys = []
            for h in hops:
                d = data[schema][sampler][h]
                ys.append(d.get("tau_bar", float("nan")))
            ax.plot(hops, ys, marker="o", label=SAMPLER_LABELS[sampler],
                    color=SAMPLER_COLORS[sampler])
        ax.set_title(SCHEMAS_SHORT[schema], fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("hop distance")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\bar\tau$")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("Hop-stratified residual structural value", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS / "fig_hop_stratified.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_schema_transfer():
    data = json.loads((RESULTS / "schema_transfer.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    schema_keys = list(SCHEMAS_SHORT.keys())
    x = np.arange(len(schema_keys))
    width = 0.16

    # Text-only AUC (single bar per schema)
    text_aucs = [data[s][SAMPLERS[0]]["auc_T_ood"] for s in schema_keys]
    ax.bar(x - 2 * width, text_aucs, width, label="Text only",
           color="lightgray", edgecolor="black")

    # Per-sampler AUC_TS
    for i, sampler in enumerate(SAMPLERS):
        aucs = [data[s][sampler]["auc_TS_ood"] for s in schema_keys]
        offset = (i - 1) * width
        ax.bar(x + offset, aucs, width, label=SAMPLER_LABELS[sampler],
               color=SAMPLER_COLORS[sampler])

    ax.set_xticks(x)
    ax.set_xticklabels([SCHEMAS_SHORT[s] for s in schema_keys], rotation=10)
    ax.set_ylabel("AUC on held-out schema (OOD)")
    ax.set_title("Schema-transfer AUC: text vs each structural sampler")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "fig_schema_transfer.png", dpi=150)
    plt.close(fig)


def fig_joint_redundancy():
    in_dist = json.loads((RESULTS / "in_dist_tau.json").read_text())
    joint = json.loads((RESULTS / "joint_tau.json").read_text())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    schemas = list(SCHEMAS_SHORT.keys())
    x = np.arange(len(schemas))
    width = 0.18
    for i, sampler in enumerate(SAMPLERS):
        ax.bar(x + (i - 1.5) * width,
               [in_dist[s][sampler]["auc_TS"] for s in schemas],
               width, label=SAMPLER_LABELS[sampler],
               color=SAMPLER_COLORS[sampler], alpha=0.7)
    joint_aucs = [joint[s]["auc_TS_all"] for s in schemas]
    ax.plot(x, joint_aucs, "k*", markersize=14, label="Joint (all 4)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEMAS_SHORT[s] for s in schemas], rotation=10)
    ax.set_ylabel(r"AUC$_{TS}$")
    ax.set_title("Joint stacking adds little over best single sampler")
    ax.set_ylim(0.94, 1.0)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "fig_joint_redundancy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating figures...")
    fig_in_dist_auc_gap()
    fig_text_strength_vs_structural_value()
    fig_hop_stratified()
    fig_schema_transfer()
    fig_joint_redundancy()
    print("Done. See results/fig_*.png")
