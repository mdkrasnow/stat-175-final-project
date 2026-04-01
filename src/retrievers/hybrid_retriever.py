"""Hybrid retrieval: combine structural and textual signals.

Reference: MoR (2025) — Mixture of Structural-and-Textual Retrieval.
"""

import numpy as np

from src.retrievers.base import BaseRetriever
from src.retrievers.shared_index import SharedIndex
from src.data.stark_loader import StarkGraphWrapper


class HybridRetriever(BaseRetriever):
    """Combine node-level textual retrieval with subgraph structural retrieval.

    Inspired by MoR's mixture approach: scores from text similarity and
    graph-structural relevance are combined with a tunable weight.
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        shared_index: SharedIndex,
        text_weight: float = 0.5,
        num_seeds: int = 3,
        k_hops: int = 1,
    ):
        super().__init__(graph)
        self.shared = shared_index
        self.text_weight = text_weight
        self.struct_weight = 1.0 - text_weight
        self.num_seeds = num_seeds
        self.k_hops = k_hops

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        query_emb = self.shared.encode_query(query)

        # Textual scores: top candidates by embedding similarity
        text_scores, text_indices = self.shared.index.search(query_emb, top_k * 5)
        text_score_map = {}
        for score, idx in zip(text_scores[0], text_indices[0]):
            nid = self.shared.node_ids[idx]
            text_score_map[nid] = float(score)

        # Structural: expand seeds by k hops
        _, seed_ids = self.shared.search(query_emb, self.num_seeds)
        seed_set = set(seed_ids)
        expanded = set(seed_ids)
        for seed in seed_ids:
            neighbors = self.graph.get_neighbors(seed, self.k_hops)
            expanded.update(neighbors)

        # Structural score: 1.0 for seeds, 0.5 for 1-hop, 0.25 otherwise
        struct_score_map = {}
        for nid in expanded:
            if nid in seed_set:
                struct_score_map[nid] = 1.0
            else:
                for seed in seed_ids:
                    if self.graph.graph.has_edge(nid, seed):
                        struct_score_map[nid] = max(struct_score_map.get(nid, 0), 0.5)
                        break
                else:
                    struct_score_map[nid] = 0.25

        # Combine scores
        all_candidates = set(text_score_map.keys()) | expanded
        combined = []
        for nid in all_candidates:
            t_score = text_score_map.get(nid, 0.0)
            s_score = struct_score_map.get(nid, 0.0)
            combined_score = self.text_weight * t_score + self.struct_weight * s_score
            combined.append((nid, combined_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in combined[:top_k]]

    def retrieve(self, query: str, top_k: int = 10) -> str:
        node_ids = self.retrieve_ids(query, top_k)
        return self._format_node_context(node_ids)
