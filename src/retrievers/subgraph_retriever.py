"""Subgraph-centric retrieval: extract local subgraphs around query-relevant entities.

Reference: GRAG (2024) subgraph retrieval + MoR (2025) structural retrieval.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.retrievers.base import BaseRetriever
from src.data.stark_loader import StarkGraphWrapper


class SubgraphRetriever(BaseRetriever):
    """Extract k-hop subgraphs around query-relevant seed nodes.

    Strategy:
    1. Find seed nodes relevant to the query (via embedding similarity).
    2. Expand each seed by k hops to get a local neighborhood.
    3. Merge neighborhoods into a single subgraph.
    4. Serialize the subgraph (nodes + edges) as context.
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        embedding_model: str = "all-MiniLM-L6-v2",
        num_seeds: int = 3,
        k_hops: int = 2,
    ):
        super().__init__(graph)
        self.encoder = SentenceTransformer(embedding_model)
        self.num_seeds = num_seeds
        self.k_hops = k_hops
        self.node_ids: list[int] = []
        self.index: faiss.IndexFlatIP | None = None
        self._build_index()

    def _build_index(self):
        """Encode all node texts and build a FAISS index for seed selection."""
        self.node_ids = sorted(self.graph.node_texts.keys())
        texts = [self.graph.node_texts[nid] for nid in self.node_ids]

        print(f"[SubgraphRetriever] Encoding {len(texts)} node texts...")
        embeddings = self.encoder.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _get_seed_nodes(self, query: str) -> list[int]:
        """Find the most query-relevant nodes as subgraph seeds."""
        query_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_emb, self.num_seeds)
        return [self.node_ids[idx] for idx in indices[0]]

    def _expand_subgraph(self, seeds: list[int]) -> set[int]:
        """Expand seed nodes by k hops and merge into one node set."""
        all_nodes = set(seeds)
        for seed in seeds:
            neighbors = self.graph.get_neighbors(seed, self.k_hops)
            all_nodes.update(neighbors)
        return all_nodes

    def _format_subgraph_context(self, node_ids: set[int]) -> str:
        """Format a subgraph as context: nodes + their connections."""
        subgraph = self.graph.get_subgraph(node_ids)

        parts = []
        # Node descriptions
        parts.append("=== Entities ===")
        for nid in sorted(node_ids):
            text = self.graph.node_texts.get(nid, "")
            if text:
                # Truncate for context window management
                if len(text) > 300:
                    text = text[:300] + "..."
                parts.append(f"[Node {nid}] {text}")

        # Edge list
        parts.append("\n=== Relationships ===")
        for src, dst in subgraph.edges():
            parts.append(f"Node {src} -- Node {dst}")

        return "\n".join(parts)

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        seeds = self._get_seed_nodes(query)
        expanded = self._expand_subgraph(seeds)
        # Return seeds first, then expanded neighbors
        result = list(seeds)
        for nid in expanded:
            if nid not in seeds:
                result.append(nid)
            if len(result) >= top_k:
                break
        return result[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> str:
        seeds = self._get_seed_nodes(query)
        expanded = self._expand_subgraph(seeds)

        # Cap the subgraph size to avoid overwhelming the LLM context
        if len(expanded) > top_k * 10:
            # Keep seeds + closest neighbors by re-expanding with fewer hops
            expanded = set(seeds)
            for seed in seeds:
                neighbors = self.graph.get_neighbors(seed, k_hops=1)
                expanded.update(neighbors)

        return self._format_subgraph_context(expanded)
