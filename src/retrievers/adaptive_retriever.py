"""Query-adaptive routing retriever based on retrieval score skewness.

Reference: SkewRoute (Wang et al., EMNLP 2025) — retrieval score skewness
correlates with query difficulty.  High skewness means a few nodes dominate
similarity (focused / easy query), so a tight 1-seed 1-hop config suffices.
Low skewness means scores are diffuse (hard / multi-hop query), so we need
broader expansion.
"""

import numpy as np
from scipy.stats import skew

from src.retrievers.base import BaseRetriever
from src.retrievers.shared_index import SharedIndex
from src.retrievers.subgraph_retriever import SubgraphRetriever
from src.data.stark_loader import StarkGraphWrapper


class AdaptiveRetriever(BaseRetriever):
    """Route each query to a focused or broad SubgraphRetriever based on
    the skewness of its top-50 retrieval scores.

    Routing logic:
        skewness > threshold  -->  focused (1 seed, 1-hop, 200 nodes)
        skewness <= threshold -->  broad   (5 seeds, 2-hop, 500 nodes)
    """

    def __init__(
        self,
        graph: StarkGraphWrapper,
        shared_index: SharedIndex,
        skew_threshold: float = 1.0,
    ):
        super().__init__(graph)
        self.shared = shared_index
        self.skew_threshold = skew_threshold

        # Pre-build both sub-retrievers so we pay config cost once.
        self.focused_retriever = SubgraphRetriever(
            graph, shared_index, num_seeds=1, k_hops=1, max_subgraph_nodes=200
        )
        self.broad_retriever = SubgraphRetriever(
            graph, shared_index, num_seeds=5, k_hops=2, max_subgraph_nodes=500
        )

        # Populated after each query for analysis / logging.
        self.last_route: str | None = None
        self.last_skewness: float | None = None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, query: str) -> SubgraphRetriever:
        """Decide which sub-retriever to use for *query*."""
        query_emb = self.shared.encode_query(query)
        scores, _ = self.shared.index.search(query_emb, 50)
        score_array = scores[0].astype(np.float64)

        self.last_skewness = float(skew(score_array))

        if self.last_skewness > self.skew_threshold:
            self.last_route = "focused"
            return self.focused_retriever
        else:
            self.last_route = "broad"
            return self.broad_retriever

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        retriever = self._route(query)
        return retriever.retrieve_ids(query, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 10) -> str:
        retriever = self._route(query)
        return retriever.retrieve(query, top_k=top_k)
