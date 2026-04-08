"""QCTR Retriever — beam search guided by a learned MLP transition scorer."""

import numpy as np
import torch

from src.models.qctr_model import TransitionScorer
from src.retrievers.base import BaseRetriever
from src.retrievers.shared_index import SharedIndex


class QCTRRetriever(BaseRetriever):
    """Retriever that uses beam search with a trained transition scorer
    to navigate the knowledge graph toward relevant answer nodes."""

    def __init__(
        self,
        graph,
        shared_index: SharedIndex,
        model_path: str,
        edge_type_lookup: dict,
        beam_width: int = 10,
        max_hops: int = 4,
        device: str = "cpu",
        scoring_mode: str = "learned",
        use_edge_types: bool = True,
    ):
        super().__init__(graph)
        self.shared = shared_index
        self.edge_type_lookup = edge_type_lookup
        self.beam_width = beam_width
        self.max_hops = max_hops
        self.device = torch.device(device)
        self.scoring_mode = scoring_mode  # "learned" or "cosine"
        self.use_edge_types = use_edge_types

        # Load trained transition scorer
        if scoring_mode == "learned":
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model = TransitionScorer()
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None

    def _get_query_entity(self, query_emb: np.ndarray) -> int:
        """Return the closest node to the query embedding."""
        _, node_ids = self.shared.search(query_emb, 1)
        return node_ids[0]

    @torch.no_grad()
    def _score_transitions(
        self, query_emb: np.ndarray, current_node: int, candidates: list[int]
    ) -> list[tuple[int, float]]:
        """Score candidate neighbor transitions from current_node in a single batch."""
        n = len(candidates)
        if n == 0:
            return []

        curr_emb = self.shared.get_node_embedding(current_node)  # [384]

        # Vectorised candidate embeddings
        cand_embs = np.stack(
            [self.shared.get_node_embedding(c) for c in candidates]
        )  # [N, 384]

        # Cosine similarities (embeddings assumed normalised by SharedIndex)
        cos_q_cand = cand_embs @ query_emb  # [N]

        # Cosine-only mode: just rank by query-candidate similarity
        if self.scoring_mode == "cosine":
            order = np.argsort(-cos_q_cand)
            return [(candidates[i], float(cos_q_cand[i])) for i in order]

        # Learned scorer mode
        cos_q_curr = np.dot(query_emb, curr_emb)  # scalar
        cos_curr_cand = cand_embs @ curr_emb  # [N]

        # Build feature matrix [N, 1155]
        q_tile = np.tile(query_emb, (n, 1))  # [N, 384]
        curr_tile = np.tile(curr_emb, (n, 1))  # [N, 384]
        cosines = np.stack(
            [np.full(n, cos_q_curr), cos_q_cand, cos_curr_cand], axis=1
        )  # [N, 3]
        features = np.concatenate([q_tile, curr_tile, cand_embs, cosines], axis=1)  # [N, 1155]

        # Edge types (zero out if ablating)
        if self.use_edge_types:
            edge_types = np.array(
                [self.edge_type_lookup.get((current_node, c), -1) for c in candidates],
                dtype=np.int64,
            )
        else:
            edge_types = np.full(n, -1, dtype=np.int64)

        # Forward pass
        feat_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        et_t = torch.tensor(edge_types, dtype=torch.long, device=self.device)
        assert self.model is not None
        logits = self.model(feat_t, et_t).cpu().numpy()  # [N]

        # Sort descending by score
        order = np.argsort(-logits)
        return [(candidates[i], float(logits[i])) for i in order]

    @torch.no_grad()
    def _beam_search(
        self, query_emb: np.ndarray, start_node: int
    ) -> list[tuple[int, float]]:
        """Guided beam search using learned transition scores."""
        beam: list[tuple[int, float]] = [(start_node, 0.0)]
        visited: set[int] = {start_node}
        all_visited: dict[int, float] = {start_node: 0.0}

        for _ in range(self.max_hops):
            candidates: list[tuple[int, float]] = []
            for node, path_score in beam:
                neighbors = list(self.graph.graph.neighbors(node))
                new_neighbors = [n for n in neighbors if n not in visited]
                if not new_neighbors:
                    continue
                scored = self._score_transitions(query_emb, node, new_neighbors)
                for cand_id, step_score in scored:
                    candidates.append((cand_id, path_score + step_score))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[: self.beam_width]

            # Track visited and best scores
            for node, score in beam:
                visited.add(node)
                if node not in all_visited or score > all_visited[node]:
                    all_visited[node] = score

        # Return all visited nodes ranked by best score
        ranked = sorted(all_visited.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        query_emb = self.shared.encode_query(query).flatten()  # [384]
        start = self._get_query_entity(query_emb.reshape(1, -1))
        ranked = self._beam_search(query_emb, start)
        return [node_id for node_id, _ in ranked[:top_k]]

    def retrieve(self, query: str, top_k: int = 10) -> str:
        ids = self.retrieve_ids(query, top_k)
        return self._format_node_context(ids)
