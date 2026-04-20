"""Subgraph-centric retrieval: extract local subgraphs around query-relevant entities.

Reference: GRAG (2024) subgraph retrieval + MoR (2025) structural retrieval.
"""

import numpy as np

from src.retrievers.base import BaseRetriever
from src.retrievers.shared_index import SharedIndex
from src.data.stark_loader import StarkGraphWrapper


class SubgraphRetriever(BaseRetriever):
    """Extract k-hop subgraphs around query-relevant seed nodes.

    Strategy:
    1. Find seed nodes relevant to the query (via embedding similarity).
    2. Expand each seed by k hops to get a local neighborhood.
    3. If the neighborhood is too large, keep only the most relevant nodes.
    4. Serialize the subgraph (nodes + edges) as context.
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        shared_index: SharedIndex,
        num_seeds: int = 3,
        k_hops: int = 1,
        max_subgraph_nodes: int = 200,
    ):
        super().__init__(graph)
        self.shared = shared_index
        self.num_seeds = num_seeds
        self.k_hops = k_hops
        self.max_subgraph_nodes = max_subgraph_nodes

    def _get_seed_nodes(self, query_emb: np.ndarray) -> list[int]:
        _, node_ids = self.shared.search(query_emb, self.num_seeds)
        return node_ids

    def _expand_subgraph(self, seeds: list[int], query_emb: np.ndarray) -> list[int]:
        """Expand seed nodes by k hops, cap size by re-ranking on relevance."""
        all_nodes = set(seeds)
        for seed in seeds:
            neighbors = self.graph.get_neighbors(seed, self.k_hops)
            all_nodes.update(neighbors)

        if len(all_nodes) <= self.max_subgraph_nodes:
            return list(all_nodes)

        # Too many nodes — re-rank by embedding similarity and keep top
        node_list = list(all_nodes)
        embs = self.shared.get_node_embeddings(node_list)
        scores = (embs @ query_emb.T).flatten()
        ranked_indices = np.argsort(-scores)[:self.max_subgraph_nodes]
        # Always include seeds
        result = set(seeds)
        for idx in ranked_indices:
            result.add(node_list[idx])
            if len(result) >= self.max_subgraph_nodes:
                break
        return list(result)

    def _format_subgraph_context(self, node_ids: list[int]) -> str:
        node_set = set(node_ids)
        subgraph = self.graph.graph.subgraph(node_set)

        parts = []
        parts.append("=== Entities ===")
        for nid in sorted(node_ids)[:50]:  # Cap displayed nodes
            text = self.graph.node_texts.get(nid, "")
            if text:
                if len(text) > 300:
                    text = text[:300] + "..."
                parts.append(f"[Node {nid}] {text}")

        parts.append("\n=== Relationships ===")
        edge_count = 0
        for src, dst in subgraph.edges():
            parts.append(f"Node {src} -- Node {dst}")
            edge_count += 1
            if edge_count >= 100:  # Cap displayed edges
                parts.append(f"... and {subgraph.number_of_edges() - 100} more edges")
                break

        return "\n".join(parts)

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        query_emb = self.shared.encode_query(query)
        seeds = self._get_seed_nodes(query_emb)
        expanded = self._expand_subgraph(seeds, query_emb)

        # Re-rank expanded nodes by relevance, seeds first
        embs = self.shared.get_node_embeddings(expanded)
        scores = (embs @ query_emb.T).flatten()
        ranked_indices = np.argsort(-scores)

        result = []
        seen = set()
        # Seeds first
        for s in seeds:
            if s not in seen:
                result.append(s)
                seen.add(s)
        # Then by score
        for idx in ranked_indices:
            nid = expanded[idx]
            if nid not in seen:
                result.append(nid)
                seen.add(nid)
            if len(result) >= top_k:
                break
        return result[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> str:
        query_emb = self.shared.encode_query(query)
        seeds = self._get_seed_nodes(query_emb)
        expanded = self._expand_subgraph(seeds, query_emb)
        return self._format_subgraph_context(expanded)
