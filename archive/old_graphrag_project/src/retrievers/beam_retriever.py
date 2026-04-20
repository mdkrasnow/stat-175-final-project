"""Beam-search retrieval: embedding-guided hop-by-hop expansion with pruning.

Reference: Think-on-Graph (Sun et al., ICLR 2024) uses LLM-guided beam search
over knowledge graphs. We adapt this to embedding-guided beam search for
efficiency, scoring neighbor nodes by cosine similarity at each hop and keeping
only the top beam_width candidates. This converts exponential k-hop expansion
(60^k on PrimeKG) to linear cost (beam_width * max_hops).
"""

import numpy as np

from src.retrievers.base import BaseRetriever
from src.retrievers.shared_index import SharedIndex
from src.data.stark_loader import StarkGraphWrapper


class BeamRetriever(BaseRetriever):
    """Embedding-guided beam search over the knowledge graph.

    Algorithm:
    1. Find seed nodes by embedding similarity to the query.
    2. At each hop, expand the current beam by collecting neighbors.
    3. Score all new (unvisited) neighbors by embedding similarity.
    4. Prune to keep only the top beam_width nodes as the next beam.
    5. After all hops, re-rank ALL visited nodes and return top_k.

    Args:
        graph: StarkGraphWrapper with the underlying NetworkX graph.
        shared_index: SharedIndex with pre-computed node embeddings and FAISS index.
        num_seeds: Number of initial seed nodes selected by embedding similarity.
        beam_width: Maximum number of nodes kept at each hop (pruning threshold).
        max_hops: Maximum number of expansion hops from the seeds.
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        shared_index: SharedIndex,
        num_seeds: int = 3,
        beam_width: int = 10,
        max_hops: int = 3,
    ):
        super().__init__(graph)
        self.shared = shared_index
        self.num_seeds = num_seeds
        self.beam_width = beam_width
        self.max_hops = max_hops

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        """Retrieve node IDs via embedding-guided beam search.

        Args:
            query: Natural language query string.
            top_k: Number of node IDs to return.

        Returns:
            List of node IDs ranked by embedding similarity to the query.
        """
        # Step 1: Encode the query
        query_emb = self.shared.encode_query(query)  # shape (1, dim)

        # Step 2: Get initial seed nodes
        _, seed_ids = self.shared.search(query_emb, self.num_seeds)
        beam = list(seed_ids)
        visited = set(beam)

        # Steps 3-4: Hop-by-hop expansion with pruning
        for hop in range(self.max_hops):
            # Collect all neighbors of current beam nodes
            candidates = []
            for node in beam:
                for neighbor in self.graph.graph.neighbors(node):
                    if neighbor not in visited:
                        candidates.append(neighbor)

            # Deduplicate candidates within this hop
            candidates = list(set(candidates))

            # Early stopping: no new neighbors to explore
            if not candidates:
                break

            # Score candidates by embedding similarity to query
            # Filter to candidates that exist in the shared index
            valid_candidates = [
                c for c in candidates if c in self.shared.node_id_to_idx
            ]
            if not valid_candidates:
                break

            candidate_embs = self.shared.get_node_embeddings(valid_candidates)
            scores = (candidate_embs @ query_emb.T).flatten()

            # Keep top beam_width candidates as the new beam
            if len(valid_candidates) <= self.beam_width:
                beam = valid_candidates
            else:
                top_indices = np.argsort(-scores)[: self.beam_width]
                beam = [valid_candidates[i] for i in top_indices]

            # Mark new beam nodes as visited
            visited.update(beam)

        # Step 5: Re-rank ALL visited nodes by embedding similarity
        visited_list = [
            nid for nid in visited if nid in self.shared.node_id_to_idx
        ]
        if not visited_list:
            return []

        all_embs = self.shared.get_node_embeddings(visited_list)
        all_scores = (all_embs @ query_emb.T).flatten()
        ranked_indices = np.argsort(-all_scores)

        # Step 6: Return top_k
        result = [visited_list[i] for i in ranked_indices[:top_k]]
        return result

    def retrieve(self, query: str, top_k: int = 10) -> str:
        """Retrieve context string via beam search.

        Args:
            query: Natural language query string.
            top_k: Number of nodes to include in the context.

        Returns:
            Formatted context string with the top-k most relevant nodes.
        """
        node_ids = self.retrieve_ids(query, top_k)
        return self._format_node_context(node_ids)
