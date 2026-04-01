"""LLM generation wrapper — frozen model for answer generation."""

import os


class LLMGenerator:
    """Wrapper around an API-based LLM for answer generation.

    Uses the same frozen model for all retrieval strategies to ensure
    we're measuring retrieval quality, not generation differences.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        self.client = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, query: str, context: str, max_tokens: int = 256) -> str:
        """Generate an answer given a query and retrieved context.

        Args:
            query: The original question.
            context: Retrieved context from a retrieval strategy.
            max_tokens: Maximum tokens in the response.

        Returns:
            The generated answer string.
        """
        prompt = ANSWER_PROMPT.format(context=context, query=query)

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.content[0].text.strip()


ANSWER_PROMPT = """You are answering questions based on a knowledge graph. Use ONLY the provided context to answer. If the context does not contain enough information, say "I don't know."

=== Retrieved Context ===
{context}

=== Question ===
{query}

Answer concisely with just the entity name(s) or fact(s) requested."""
