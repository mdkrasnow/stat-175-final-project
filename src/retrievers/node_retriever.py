"""Node-centric retrieval: retrieve top-k nodes by embedding similarity.

Reference: Baseline approach from GRAG (2024).
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.retrievers.base import BaseRetriever
from src.data.stark_loader import StarkGraphWrapper


class NodeRetriever(BaseRetriever):
    """Retrieve top-k nodes whose text is most similar to the query.

    This is the simplest retrieval strategy — purely text-based,
    ignoring graph structure entirely. Serves as the baseline.
    """

    def __init__(self, graph: StarkGraphWrapper, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__(graph)
        self.encoder = SentenceTransformer(embedding_model)
        self.node_ids: list[int] = []
        self.index: faiss.IndexFlatIP | None = None
        self._build_index()

    def _build_index(self):
        """Encode all node texts and build a FAISS index."""
        self.node_ids = sorted(self.graph.node_texts.keys())
        texts = [self.graph.node_texts[nid] for nid in self.node_ids]

        print(f"Encoding {len(texts)} node texts...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Inner product index (cosine similarity since embeddings are normalized)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        query_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_emb, top_k)
        return [self.node_ids[idx] for idx in indices[0]]

    def retrieve(self, query: str, top_k: int = 10) -> str:
        node_ids = self.retrieve_ids(query, top_k)
        return self._format_node_context(node_ids)
