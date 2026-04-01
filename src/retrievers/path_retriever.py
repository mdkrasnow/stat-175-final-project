"""Path-centric retrieval: retrieve reasoning paths connecting query-relevant entities.

Reference: Inspired by GraphFlow (2025) — flow-based retrieval that traces
reasoning paths through the knowledge graph.
"""

import numpy as np
import networkx as nx

from src.retrievers.base import BaseRetriever
from src.retrievers.shared_index import SharedIndex
from src.data.stark_loader import StarkGraphWrapper


class PathRetriever(BaseRetriever):
    """Retrieve reasoning paths that connect query-relevant seed nodes.

    Strategy:
    1. Find seed nodes relevant to the query (via embedding similarity).
    2. Find shortest paths between seed node pairs.
    3. Score and rank paths by aggregate node relevance to query.
    4. Collect unique nodes from top paths as retrieved set.
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        shared_index: SharedIndex,
        num_seeds: int = 5,
        max_path_length: int = 4,
    ):
        super().__init__(graph)
        self.shared = shared_index
        self.num_seeds = num_seeds
        self.max_path_length = max_path_length

    def _get_seed_nodes(self, query_emb: np.ndarray) -> list[int]:
        _, node_ids = self.shared.search(query_emb, self.num_seeds)
        return node_ids

    def _find_paths(self, seeds: list[int]) -> list[list[int]]:
        """Find shortest paths between all pairs of seed nodes."""
        all_paths = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                try:
                    path = nx.shortest_path(self.graph.graph, seeds[i], seeds[j])
                    if len(path) <= self.max_path_length + 1:
                        all_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        return all_paths

    def _score_path(self, path: list[int], query_emb: np.ndarray) -> float:
        """Score a path by the average similarity of its nodes to the query."""
        path_embs = self.shared.get_node_embeddings(path)
        if len(path_embs) == 0:
            return 0.0
        similarities = path_embs @ query_emb.T
        return float(similarities.mean())

    def _format_path_context(self, paths: list[list[int]]) -> str:
        parts = []
        for i, path in enumerate(paths):
            node_texts = []
            for nid in path:
                text = self.graph.node_texts.get(nid, f"Node {nid}")
                if len(text) > 200:
                    text = text[:200] + "..."
                node_texts.append(text)
            path_str = " -> ".join(node_texts)
            parts.append(f"[Path {i+1}] {path_str}")
        return "\n\n".join(parts)

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        query_emb = self.shared.encode_query(query)
        seeds = self._get_seed_nodes(query_emb)
        paths = self._find_paths(seeds)

        if not paths:
            return seeds

        scored = [(p, self._score_path(p, query_emb)) for p in paths]
        scored.sort(key=lambda x: x[1], reverse=True)

        seen = set()
        result = []
        for path, _ in scored:
            for nid in path:
                if nid not in seen:
                    seen.add(nid)
                    result.append(nid)
                if len(result) >= top_k:
                    return result
        for s in seeds:
            if s not in seen:
                result.append(s)
                seen.add(s)
            if len(result) >= top_k:
                break
        return result

    def retrieve(self, query: str, top_k: int = 10) -> str:
        query_emb = self.shared.encode_query(query)
        seeds = self._get_seed_nodes(query_emb)
        paths = self._find_paths(seeds)

        if not paths:
            return self._format_node_context(seeds)

        scored = [(p, self._score_path(p, query_emb)) for p in paths]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_paths = [p for p, _ in scored[:top_k]]
        return self._format_path_context(top_paths)
