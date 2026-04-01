"""Path-centric retrieval: retrieve reasoning paths connecting query-relevant entities.

Reference: Inspired by GraphFlow (2025) — flow-based retrieval that traces
reasoning paths through the knowledge graph.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.retrievers.base import BaseRetriever
from src.data.stark_loader import StarkGraphWrapper


class PathRetriever(BaseRetriever):
    """Retrieve reasoning paths that connect query-relevant seed nodes.

    Strategy:
    1. Find seed nodes relevant to the query (via embedding similarity).
    2. Find paths between seed node pairs in the graph.
    3. Score and rank paths by aggregate relevance.
    4. Serialize the top paths as context.
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        embedding_model: str = "all-MiniLM-L6-v2",
        num_seeds: int = 5,
        max_path_length: int = 3,
    ):
        super().__init__(graph)
        self.encoder = SentenceTransformer(embedding_model)
        self.num_seeds = num_seeds
        self.max_path_length = max_path_length
        self.node_ids: list[int] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.IndexFlatIP | None = None
        self._build_index()

    def _build_index(self):
        """Encode all node texts and build a FAISS index for seed selection."""
        self.node_ids = sorted(self.graph.node_texts.keys())
        texts = [self.graph.node_texts[nid] for nid in self.node_ids]

        print(f"[PathRetriever] Encoding {len(texts)} node texts...")
        self.embeddings = self.encoder.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def _get_seed_nodes(self, query: str) -> list[int]:
        """Find the most query-relevant nodes as path seeds."""
        query_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_emb, self.num_seeds)
        return [self.node_ids[idx] for idx in indices[0]]

    def _find_paths(self, seeds: list[int]) -> list[list[int]]:
        """Find paths between all pairs of seed nodes."""
        all_paths = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                paths = self.graph.get_paths(seeds[i], seeds[j], self.max_path_length)
                all_paths.extend(paths)
        return all_paths

    def _score_path(self, path: list[int], query_emb: np.ndarray) -> float:
        """Score a path by the average similarity of its nodes to the query."""
        node_indices = []
        for nid in path:
            try:
                idx = self.node_ids.index(nid)
                node_indices.append(idx)
            except ValueError:
                continue
        if not node_indices:
            return 0.0
        path_embs = self.embeddings[node_indices]
        similarities = path_embs @ query_emb.T
        return float(similarities.mean())

    def _format_path_context(self, paths: list[list[int]]) -> str:
        """Format paths as readable context."""
        parts = []
        for i, path in enumerate(paths):
            node_texts = []
            for nid in path:
                text = self.graph.node_texts.get(nid, f"Node {nid}")
                # Truncate long node texts for path display
                if len(text) > 200:
                    text = text[:200] + "..."
                node_texts.append(text)
            path_str = " -> ".join(node_texts)
            parts.append(f"[Path {i+1}] {path_str}")
        return "\n\n".join(parts)

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        seeds = self._get_seed_nodes(query)
        paths = self._find_paths(seeds)

        if not paths:
            return seeds  # Fallback to seeds if no paths found

        query_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scored = [(p, self._score_path(p, query_emb)) for p in paths]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Collect unique node IDs from top paths
        seen = set()
        result = []
        for path, _ in scored:
            for nid in path:
                if nid not in seen:
                    seen.add(nid)
                    result.append(nid)
                if len(result) >= top_k:
                    return result
        return result

    def retrieve(self, query: str, top_k: int = 10) -> str:
        seeds = self._get_seed_nodes(query)
        paths = self._find_paths(seeds)

        if not paths:
            return self._format_node_context(seeds)

        query_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scored = [(p, self._score_path(p, query_emb)) for p in paths]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_paths = [p for p, _ in scored[:top_k]]
        return self._format_path_context(top_paths)
