"""Hybrid retrieval: combine structural and textual signals.

Reference: MoR (2025) — Mixture of Structural-and-Textual Retrieval.
"""

import numpy as np

from src.retrievers.base import BaseRetriever
from src.retrievers.node_retriever import NodeRetriever
from src.retrievers.subgraph_retriever import SubgraphRetriever
from src.data.stark_loader import StarkGraphWrapper


class HybridRetriever(BaseRetriever):
    """Combine node-level textual retrieval with subgraph structural retrieval.

    Inspired by MoR's mixture approach: scores from text similarity and
    graph-structural relevance are combined with a tunable weight.
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        embedding_model: str = "all-MiniLM-L6-v2",
        text_weight: float = 0.5,
        num_seeds: int = 3,
        k_hops: int = 2,
    ):
        super().__init__(graph)
        self.text_weight = text_weight
        self.struct_weight = 1.0 - text_weight

        # Reuse the node retriever for textual scores
        self.node_retriever = NodeRetriever(graph, embedding_model)
        # Reuse the subgraph retriever for structural expansion
        self.subgraph_retriever = SubgraphRetriever(
            graph, embedding_model, num_seeds, k_hops
        )

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        # Get textual scores for all nodes
        query_emb = self.node_retriever.encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        text_scores, text_indices = self.node_retriever.index.search(query_emb, top_k * 5)

        text_score_map = {}
        for score, idx in zip(text_scores[0], text_indices[0]):
            nid = self.node_retriever.node_ids[idx]
            text_score_map[nid] = float(score)

        # Get structurally relevant nodes via subgraph expansion
        seeds = self.subgraph_retriever._get_seed_nodes(query)
        expanded = self.subgraph_retriever._expand_subgraph(seeds)

        # Structural score: 1.0 for seeds, decays by distance
        struct_score_map = {}
        seed_set = set(seeds)
        for nid in expanded:
            if nid in seed_set:
                struct_score_map[nid] = 1.0
            else:
                # Simple decay: 1-hop neighbors get 0.5, further gets 0.25
                for seed in seeds:
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
