"""Load PrimeKG with node-type and relation-type metadata, cache to disk.

The archived loader built a plain NetworkX graph with no type information.
This module extends it to surface the node and relation types we need for
schema partitioning (see ``src/data/schemas.py``).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "primekg_graph.pkl"

EXPECTED_NUM_NODES = 129_375
EXPECTED_NUM_EDGES = 4_049_642


@dataclass
class PrimeKG:
    graph: nx.Graph                                      # undirected, int-indexed nodes
    node_texts: dict[int, str]                           # node_id -> description string
    node_types: dict[int, str]                           # node_id -> type name (e.g. "drug")
    relations_by_pair: dict[tuple[int, int], frozenset[str]]  # sorted-pair -> all relations
    node_type_vocab: list[str]                           # canonical ordering of node types
    relation_type_vocab: list[str]                       # canonical ordering of relations
    node_info: dict[int, dict] = field(default_factory=dict)

    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def nodes_of_type(self, node_type: str) -> list[int]:
        return [n for n, t in self.node_types.items() if t == node_type]

    def relations_on(self, u: int, v: int) -> frozenset[str]:
        """All relations on the undirected edge (u,v). Empty frozenset if no edge."""
        key = (u, v) if u <= v else (v, u)
        return self.relations_by_pair.get(key, frozenset())


def _build_graph(skb) -> tuple[nx.Graph, dict[tuple[int, int], frozenset[str]]]:
    """Build NetworkX graph from an SKB, attaching the full relation set per edge.

    A single undirected pair can carry multiple relations in PrimeKG (e.g. a
    drug-disease pair that is both 'indication' and 'side effect'). We keep
    the full set on each edge as the 'relations' attribute, and also pick a
    canonical 'primary_relation' (the lex-smallest, for deterministic display).
    """
    G = nx.Graph()
    G.add_nodes_from(range(skb.num_nodes()))

    edge_index = skb.edge_index.numpy()
    edge_types = skb.edge_types.numpy()
    edge_type_dict = skb.edge_type_dict

    pair_relations: dict[tuple[int, int], set[str]] = {}
    for (u, v), t in zip(edge_index.T, edge_types):
        u_i, v_i = int(u), int(v)
        if u_i == v_i:
            continue  # drop self-loops; they carry no structural signal
        key = (u_i, v_i) if u_i <= v_i else (v_i, u_i)
        pair_relations.setdefault(key, set()).add(edge_type_dict[int(t)])

    relations_by_pair: dict[tuple[int, int], frozenset[str]] = {}
    edges_to_add: list[tuple[int, int, dict]] = []
    for (u, v), rels in pair_relations.items():
        frozen = frozenset(rels)
        relations_by_pair[(u, v)] = frozen
        primary = min(rels)
        edges_to_add.append((u, v, {"relations": frozen, "primary_relation": primary}))

    G.add_edges_from(edges_to_add)
    return G, relations_by_pair


def _extract_node_metadata(skb) -> tuple[dict[int, str], dict[int, str], dict[int, dict]]:
    """Pull (text, type, raw info) per node from the SKB."""
    node_types_tensor = skb.node_types.numpy()
    node_type_dict = skb.node_type_dict  # int -> type name

    texts: dict[int, str] = {}
    types: dict[int, str] = {}
    info: dict[int, dict] = {}
    for node_id in range(skb.num_nodes()):
        try:
            texts[node_id] = skb.get_doc_info(node_id, add_rel=False) or ""
        except (KeyError, AttributeError, TypeError):
            texts[node_id] = ""
        types[node_id] = node_type_dict[int(node_types_tensor[node_id])]
        try:
            info[node_id] = dict(skb.node_info[node_id])
        except (KeyError, AttributeError, TypeError):
            info[node_id] = {}
    return texts, types, info


def load_primekg(
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    verbose: bool = True,
) -> PrimeKG:
    """Load PrimeKG as a ``PrimeKG`` dataclass, caching the build to disk.

    The first call downloads the SKB via ``stark_qa.load_skb`` and builds the
    NetworkX graph + node/edge metadata (~5 minutes). Subsequent calls read
    the pickle (~5 seconds).
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_refresh:
        if verbose:
            print(f"Loading PrimeKG from cache: {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    from stark_qa import load_skb
    from stark_qa.skb.prime import PrimeSKB

    if verbose:
        print("Downloading/loading PrimeKG SKB (first run: ~5 min)...")
    skb = load_skb("prime", download_processed=True)

    if verbose:
        print(f"Building graph: {skb.num_nodes()} nodes...")
    graph, relations_by_pair = _build_graph(skb)

    if verbose:
        print("Extracting node texts and types...")
    node_texts, node_types, node_info = _extract_node_metadata(skb)

    primekg = PrimeKG(
        graph=graph,
        node_texts=node_texts,
        node_types=node_types,
        relations_by_pair=relations_by_pair,
        node_type_vocab=list(PrimeSKB.NODE_TYPES),
        relation_type_vocab=list(PrimeSKB.RELATION_TYPES),
        node_info=node_info,
    )

    # Sanity check
    n, e = primekg.num_nodes(), primekg.num_edges()
    if verbose:
        print(f"Built: {n} nodes, {e} edges")
    if n != EXPECTED_NUM_NODES:
        print(f"WARNING: got {n} nodes, expected {EXPECTED_NUM_NODES}")
    # edge count can differ slightly due to symmetrization / self-loops;
    # we only warn if the order of magnitude is off.
    if abs(e - EXPECTED_NUM_EDGES) > 100_000:
        print(f"WARNING: got {e} edges, expected ~{EXPECTED_NUM_EDGES}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(primekg, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"Cached to {cache_path}")

    return primekg


if __name__ == "__main__":
    kg = load_primekg()
    print(f"\nPrimeKG loaded: {kg.num_nodes()} nodes, {kg.num_edges()} edges")
    print(f"Node type vocab ({len(kg.node_type_vocab)}): {kg.node_type_vocab}")
    print(f"Relation vocab  ({len(kg.relation_type_vocab)}): {kg.relation_type_vocab}")
    from collections import Counter
    print("\nNode type counts:")
    for t, c in Counter(kg.node_types.values()).most_common():
        print(f"  {t:25s} {c:>8d}")
    print("\nRelation counts (pair-level, any occurrence, top 10):")
    rel_counter: Counter[str] = Counter()
    for rels in kg.relations_by_pair.values():
        rel_counter.update(rels)
    for r, c in rel_counter.most_common(10):
        print(f"  {r:25s} {c:>8d}")
