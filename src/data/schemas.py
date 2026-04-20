"""Schema definitions and schema-subgraph induction for PrimeKG.

A "schema" is a subset of (node types, relation types). Each schema defines
a sub-KB with a distinct relational flavor. We train structural samplers per
schema and transfer the learned DML nuisance models across schemas to test
generalization of different structural signals.

PrimeKG vocabulary (from stark_qa.skb.prime.PrimeSKB):
  NODE_TYPES (10): disease, gene/protein, molecular_function, drug, pathway,
                   anatomy, effect/phenotype, biological_process,
                   cellular_component, exposure
  RELATION_TYPES (18): ppi, carrier, enzyme, target, transporter,
                       contraindication, indication, off-label use,
                       synergistic interaction, associated with, parent-child,
                       phenotype absent, phenotype present, side effect,
                       interacts with, linked to, expression present,
                       expression absent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
import json

import networkx as nx

from .primekg_loader import PrimeKG


@dataclass(frozen=True)
class Schema:
    name: str
    description: str
    node_types: frozenset[str]
    relation_types: frozenset[str]
    target_relations: frozenset[str] = field(default_factory=frozenset)
    # target_relations: the relations we predict links for (the outcome Y).
    # If empty, use all relations in the schema as candidate targets.


SCHEMA_A = Schema(
    name="A_drug_gene_disease",
    description="Drug–gene/protein–disease triangles: mechanism + indication",
    node_types=frozenset({"drug", "gene/protein", "disease"}),
    relation_types=frozenset({
        "target", "carrier", "enzyme", "transporter",           # drug ↔ gene/protein
        "associated with",                                      # gene/protein ↔ disease
        "indication", "contraindication", "off-label use",      # drug ↔ disease
        "ppi",                                                  # gene/protein ↔ gene/protein (internal)
    }),
    target_relations=frozenset({"indication"}),
)

SCHEMA_B = Schema(
    name="B_drug_pathway_disease",
    description="Drug–gene/protein–pathway–disease chains: mechanism-of-action subgraph",
    node_types=frozenset({"drug", "gene/protein", "pathway", "disease"}),
    relation_types=frozenset({
        "target", "carrier", "enzyme", "transporter",           # drug ↔ gene/protein
        "interacts with",                                       # gene/protein ↔ pathway
        "associated with",                                      # gene/protein ↔ disease
        "indication", "contraindication", "off-label use",      # drug ↔ disease
        "ppi",
    }),
    target_relations=frozenset({"indication"}),
)

SCHEMA_C = Schema(
    name="C_disease_phenotype_gene",
    description="Disease–phenotype–gene/protein: clinical presentation subgraph",
    node_types=frozenset({"disease", "effect/phenotype", "gene/protein"}),
    relation_types=frozenset({
        "phenotype present", "phenotype absent",                # disease ↔ effect/phenotype
        "associated with",                                      # gene/protein ↔ disease/phenotype
        "linked to",                                            # effect/phenotype ↔ gene/protein
        "ppi",
    }),
    target_relations=frozenset({"associated with"}),
)

SCHEMA_D = Schema(
    name="D_drug_disease_direct",
    description="Held-out: drug–disease direct relations only (no intermediates)",
    node_types=frozenset({"drug", "disease"}),
    relation_types=frozenset({"indication", "contraindication", "off-label use"}),
    target_relations=frozenset({"indication"}),
)

ALL_SCHEMAS: list[Schema] = [SCHEMA_A, SCHEMA_B, SCHEMA_C, SCHEMA_D]
SCHEMAS_BY_NAME: dict[str, Schema] = {s.name: s for s in ALL_SCHEMAS}


def induce_schema_subgraph(kg: PrimeKG, schema: Schema) -> nx.Graph:
    """Induce the subgraph of ``kg`` containing only the schema's nodes and edges.

    - Keeps nodes whose type ∈ schema.node_types.
    - Keeps edges where at least one relation on the pair ∈ schema.relation_types
      and both endpoints survived the node filter. The edge's 'relations'
      attribute stores the intersection (all schema-relevant relations).
    """
    allowed_nodes = {n for n, t in kg.node_types.items() if t in schema.node_types}

    H = nx.Graph()
    for n in allowed_nodes:
        H.add_node(n, node_type=kg.node_types[n], text=kg.node_texts.get(n, ""))

    schema_rels = schema.relation_types
    for u, v in kg.graph.edges():
        if u not in allowed_nodes or v not in allowed_nodes:
            continue
        key = (u, v) if u <= v else (v, u)
        pair_rels = kg.relations_by_pair.get(key, frozenset())
        kept = pair_rels & schema_rels
        if not kept:
            continue
        H.add_edge(u, v, relations=kept, primary_relation=min(kept))

    return H


def target_edges(
    kg: PrimeKG,
    schema: Schema,
    subgraph: nx.Graph | None = None,
) -> list[tuple[int, int]]:
    """Return the edges whose relation set intersects schema.target_relations.

    These are the positive examples for the link-prediction task. Returned
    as sorted (u<=v) pairs in lexicographic order for reproducibility.
    """
    targets = schema.target_relations or schema.relation_types
    G = subgraph if subgraph is not None else kg.graph
    edges: list[tuple[int, int]] = []
    for u, v, data in G.edges(data=True):
        rels = data.get("relations")
        if rels is None:
            key = (u, v) if u <= v else (v, u)
            rels = kg.relations_by_pair.get(key, frozenset())
        if rels & targets:
            a, b = (u, v) if u <= v else (v, u)
            edges.append((a, b))
    edges.sort()
    return edges


def schema_inventory(kg: PrimeKG) -> dict:
    """Produce a summary of each schema's size + component structure.

    Writes to results/schema_inventory.json when run as __main__.
    """
    node_type_counts = Counter(kg.node_types.values())
    relation_counts: Counter[str] = Counter()
    for rels in kg.relations_by_pair.values():
        relation_counts.update(rels)

    inventory = {
        "full_graph": {
            "num_nodes": kg.num_nodes(),
            "num_edges": kg.num_edges(),
            "node_type_counts": dict(node_type_counts.most_common()),
            "relation_counts": dict(relation_counts.most_common()),
        },
        "schemas": {},
    }

    for schema in ALL_SCHEMAS:
        H = induce_schema_subgraph(kg, schema)
        if H.number_of_nodes() == 0:
            components = []
        else:
            components = sorted(
                (len(c) for c in nx.connected_components(H)), reverse=True
            )[:5]
        targets = target_edges(kg, schema, subgraph=H)
        schema_node_types = Counter(H.nodes[n]["node_type"] for n in H.nodes)
        schema_rel_types: Counter[str] = Counter()
        for _, _, d in H.edges(data=True):
            schema_rel_types.update(d["relations"])
        inventory["schemas"][schema.name] = {
            "description": schema.description,
            "num_nodes": H.number_of_nodes(),
            "num_edges": H.number_of_edges(),
            "num_target_edges": len(targets),
            "top5_component_sizes": components,
            "node_type_counts": dict(schema_node_types),
            "relation_counts": dict(schema_rel_types),
        }
    return inventory


if __name__ == "__main__":
    from .primekg_loader import load_primekg

    kg = load_primekg()
    inv = schema_inventory(kg)

    results_path = Path(__file__).resolve().parents[2] / "results" / "schema_inventory.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(inv, f, indent=2)

    print(f"\nWrote inventory to {results_path}\n")
    print(f"Full graph: {inv['full_graph']['num_nodes']:,} nodes, "
          f"{inv['full_graph']['num_edges']:,} edges")
    for name, info in inv["schemas"].items():
        print(f"\n{name}: {info['description']}")
        print(f"  {info['num_nodes']:,} nodes, {info['num_edges']:,} edges, "
              f"{info['num_target_edges']:,} target edges")
        print(f"  Top-5 component sizes: {info['top5_component_sizes']}")
