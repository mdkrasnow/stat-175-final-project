"""Base class for all retrieval strategies."""

from abc import ABC, abstractmethod

from src.data.stark_loader import StarkGraphWrapper


class BaseRetriever(ABC):
    """Abstract base class for graph retrieval strategies.

    All retrievers take a query string and return a context string
    that will be passed to the LLM for answer generation.
    """

    def __init__(self, graph: StarkGraphWrapper):
        self.graph = graph

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> str:
        """Retrieve relevant context from the knowledge graph.

        Args:
            query: The natural language query.
            top_k: Number of items to retrieve.

        Returns:
            A context string to be passed to the LLM.
        """
        ...

    @abstractmethod
    def retrieve_ids(self, query: str, top_k: int = 10) -> list[int]:
        """Retrieve relevant node IDs from the knowledge graph.

        Args:
            query: The natural language query.
            top_k: Number of node IDs to retrieve.

        Returns:
            A list of node IDs ranked by relevance.
        """
        ...

    def _format_node_context(self, node_ids: list[int]) -> str:
        """Format a list of node IDs into a readable context string."""
        parts = []
        for nid in node_ids:
            text = self.graph.node_texts.get(nid, "")
            if text:
                parts.append(f"[Node {nid}] {text}")
        return "\n\n".join(parts)
