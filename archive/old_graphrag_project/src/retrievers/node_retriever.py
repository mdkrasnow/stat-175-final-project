"""Node-centric retrieval: retrieve top-k nodes by embedding similarity.

Reference: Baseline approach from GRAG (2024).
"""

from src.retrievers.base import BaseRetriever
from src.retrievers.shared_index import SharedIndex
from src.data.stark_loader import StarkGraphWrapper


class NodeRetriever(BaseRetriever):
    """Retrieve top-k nodes whose text is most similar to the query.

    This is the simplest retrieval strategy — purely text-based,
    ignoring graph structure entirely. Serves as the baseline.
    """

    def __init__(self, graph: StarkGraphWrapper, shared_index: SharedIndex):
        super().__init__(graph)
        self.shared = shared_index

    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        query_emb = self.shared.encode_query(query)
        _, node_ids = self.shared.search(query_emb, top_k)
        return node_ids

    def retrieve(self, query: str, top_k: int = 10) -> str:
        node_ids = self.retrieve_ids(query, top_k)
        return self._format_node_context(node_ids)
