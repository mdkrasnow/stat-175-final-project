"""Evaluation metrics for retrieval and QA quality."""

import numpy as np


def hit_at_k(predicted_ids: list[int], gold_ids: list[int], k: int = 1) -> float:
    """Hit@k: 1.0 if any gold ID appears in the top-k predictions, else 0.0."""
    top_k = set(predicted_ids[:k])
    return 1.0 if top_k & set(gold_ids) else 0.0


def mrr(predicted_ids: list[int], gold_ids: list[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first correct prediction."""
    gold_set = set(gold_ids)
    for i, pid in enumerate(predicted_ids):
        if pid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def exact_match(predicted: str, gold: str) -> float:
    """Exact match after normalization."""
    return 1.0 if _normalize(predicted) == _normalize(gold) else 0.0


def f1_score_tokens(predicted: str, gold: str) -> float:
    """Token-level F1 score between predicted and gold answers."""
    pred_tokens = set(_normalize(predicted).split())
    gold_tokens = set(_normalize(gold).split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_retrieval(
    retriever,
    qa_dataset,
    ks: list[int] = [1, 5, 10],
    max_samples: int | None = None,
) -> dict[str, float]:
    """Run retrieval evaluation over a full QA dataset.

    Args:
        retriever: A BaseRetriever instance.
        qa_dataset: STaRK QA dataset.
        ks: Values of k for Hit@k.
        max_samples: If set, only evaluate on the first N samples.

    Returns:
        Dictionary of metric_name -> score.
    """
    hit_scores = {k: [] for k in ks}
    mrr_scores = []

    n = len(qa_dataset) if max_samples is None else min(max_samples, len(qa_dataset))
    for i in range(n):
        query, query_id, gold_ids, meta = qa_dataset[i]
        predicted_ids = retriever.retrieve_ids(query, top_k=max(ks))

        for k in ks:
            hit_scores[k].append(hit_at_k(predicted_ids, gold_ids, k))
        mrr_scores.append(mrr(predicted_ids, gold_ids))

    results = {}
    for k in ks:
        results[f"Hit@{k}"] = float(np.mean(hit_scores[k]))
    results["MRR"] = float(np.mean(mrr_scores))

    return results


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().strip().split())
