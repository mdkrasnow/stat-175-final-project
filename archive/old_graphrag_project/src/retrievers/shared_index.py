"""Shared FAISS index for all retrievers — encode node texts only once."""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.data.stark_loader import StarkGraphWrapper


class SharedIndex:
    """Pre-computed node embeddings + FAISS index shared across all retrievers."""

    def __init__(self, graph: StarkGraphWrapper, embedding_model: str = "all-MiniLM-L6-v2"):
        self.graph = graph
        self.encoder = SentenceTransformer(embedding_model)
        self.node_ids: list[int] = sorted(graph.node_texts.keys())
        self.node_id_to_idx: dict[int, int] = {nid: i for i, nid in enumerate(self.node_ids)}

        print(f"[SharedIndex] Encoding {len(self.node_ids)} node texts...")
        texts = [graph.node_texts[nid] for nid in self.node_ids]
        self.embeddings = self.encoder.encode(
            texts, show_progress_bar=True, normalize_embeddings=True, batch_size=256
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        print(f"[SharedIndex] FAISS index built with {self.index.ntotal} vectors.")

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)

    def search(self, query_emb: np.ndarray, top_k: int) -> tuple[np.ndarray, list[int]]:
        """Return (scores, node_ids) for top-k most similar nodes."""
        scores, indices = self.index.search(query_emb, top_k)
        node_ids = [self.node_ids[idx] for idx in indices[0]]
        return scores[0], node_ids

    def get_node_embedding(self, node_id: int) -> np.ndarray:
        idx = self.node_id_to_idx[node_id]
        return self.embeddings[idx]

    def get_node_embeddings(self, node_ids: list[int]) -> np.ndarray:
        indices = [self.node_id_to_idx[nid] for nid in node_ids if nid in self.node_id_to_idx]
        return self.embeddings[indices]
