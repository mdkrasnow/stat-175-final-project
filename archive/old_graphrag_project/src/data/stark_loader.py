"""Load STaRK benchmark datasets and wrap them for our retrieval experiments."""

import networkx as nx
import numpy as np
from stark_qa import load_skb, load_qa


class StarkGraphWrapper:
    """Wraps a STaRK semi-structured knowledge base as a NetworkX graph
    with text attributes accessible for retrieval."""

    def __init__(self, skb):
        self.skb = skb
        self.graph = self._build_networkx_graph()
        self.node_texts = self._extract_node_texts()

    def _build_networkx_graph(self) -> nx.Graph:
        """Convert the STaRK SKB into a NetworkX graph."""
        G = nx.Graph()
        num_nodes = self.skb.num_nodes()

        # Add all nodes at once
        G.add_nodes_from(range(num_nodes))

        # Add edges in bulk from edge_index tensor
        edge_index = self.skb.edge_index
        if edge_index is not None:
            # Convert to numpy and transpose for edge list: [(src, dst), ...]
            edges = edge_index.numpy().T
            G.add_edges_from(edges.tolist())

        print(f"  Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _extract_node_texts(self) -> dict[int, str]:
        """Extract text descriptions for each node (lazy — loads on demand)."""
        print(f"  Extracting node texts for {self.graph.number_of_nodes()} nodes...")
        texts = {}
        for node_id in range(self.skb.num_nodes()):
            try:
                texts[node_id] = self.skb.get_doc_info(node_id, add_rel=False)
            except Exception:
                texts[node_id] = ""
            if (node_id + 1) % 25000 == 0:
                print(f"    ...processed {node_id + 1} nodes")
        return texts

    def get_neighbors(self, node_id: int, k_hops: int = 1) -> set[int]:
        """Get all neighbors within k hops of a node."""
        visited = {node_id}
        frontier = {node_id}
        for _ in range(k_hops):
            next_frontier = set()
            for n in frontier:
                for neighbor in self.graph.neighbors(n):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
        return visited - {node_id}

    def get_subgraph(self, node_ids: set[int]) -> nx.Graph:
        """Extract the induced subgraph over a set of node IDs."""
        return self.graph.subgraph(node_ids).copy()

    def get_paths(self, source: int, target: int, max_length: int = 4) -> list[list[int]]:
        """Find all simple paths between source and target up to max_length."""
        try:
            return list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
        except nx.NetworkXError:
            return []

    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def num_edges(self) -> int:
        return self.graph.number_of_edges()


def load_stark_dataset(dataset_name: str = "prime") -> tuple[StarkGraphWrapper, list]:
    """Load a STaRK dataset and return the graph wrapper + QA pairs.

    Args:
        dataset_name: One of 'prime', 'amazon', 'mag'

    Returns:
        (graph_wrapper, qa_dataset)
    """
    assert dataset_name in ("prime", "amazon", "mag"), \
        f"Unknown dataset: {dataset_name}. Choose from 'prime', 'amazon', 'mag'."

    print(f"Loading STaRK-{dataset_name} knowledge base...")
    skb = load_skb(dataset_name, download_processed=True)

    print(f"Loading STaRK-{dataset_name} QA pairs...")
    qa_dataset = load_qa(dataset_name)

    print(f"Building graph wrapper...")
    graph = StarkGraphWrapper(skb)

    print(f"Loaded: {graph.num_nodes()} nodes, {graph.num_edges()} edges, {len(qa_dataset)} QA pairs")
    return graph, qa_dataset
