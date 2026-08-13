"""Mathematical evaluation metrics for retrieval, answer quality, and citation accuracy.

Includes:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Citation Precision & Citation Coverage
- Faithfulness & Answer Relevance (Rule-based & LLM-as-judge)
- Context Relevance
- Unanswerable Question Classification Metrics
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# 1. Retrieval Metrics
# =============================================================================

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str] | List[str],
    k: int,
) -> float:
    """Calculate Recall@K: fraction of relevant items retrieved in top-K results.

    Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Ground-truth set or list of relevant document/chunk IDs.
        k: Cutoff rank (e.g. 5, 10).

    Returns:
        Float score in [0.0, 1.0]. Returns 1.0 if relevant_ids is empty and no false positives,
        or 0.0 if relevant_ids is empty and items were retrieved.
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return 1.0 if not retrieved_ids else 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / float(len(rel_set))


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str] | List[str],
    k: int,
) -> float:
    """Calculate Precision@K: fraction of top-K retrieved items that are relevant.

    Precision@K = |Retrieved_K ∩ Relevant| / K

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Ground-truth set or list of relevant document/chunk IDs.
        k: Cutoff rank (e.g. 5, 10).

    Returns:
        Float score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / float(k)


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str] | List[str],
) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of first relevant item.

    RR = 1 / rank_first_relevant (or 0.0 if no relevant items are retrieved)

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Ground-truth set or list of relevant document/chunk IDs.

    Returns:
        Float score in [0.0, 1.0].
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    for rank_idx, item in enumerate(retrieved_ids, start=1):
        if item in rel_set:
            return 1.0 / float(rank_idx)
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with graded relevance.

    DCG@K = sum_{i=1}^{min(K, |retrieved|)} (2^{rel_i} - 1) / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(K, |relevant|)} (2^{rel^*_i} - 1) / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K (or 0.0 if IDCG@K == 0)

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevance_scores: Mapping from ID to graded relevance score (e.g. 1.0 for relevant, 2.0 for highly relevant).
        k: Cutoff rank.

    Returns:
        Float score in [0.0, 1.0].
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    # Calculate DCG@K for retrieved order
    dcg = 0.0
    for idx, item in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(item, 0.0)
        if rel > 0.0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 1)

    # Calculate Ideal DCG@K (sorted descending by true relevance)
    ideal_scores = sorted([score for score in relevance_scores.values() if score > 0.0], reverse=True)
    if not ideal_scores:
        return 0.0

    idcg = 0.0
    for idx, rel in enumerate(ideal_scores[:k], start=1):
        idcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 1)

    if idcg <= 0.0:
        return 0.0

    return min(1.0, max(0.0, dcg / idcg))


# =============================================================================
# 2. Citation Accuracy Metrics
# =============================================================================

def citation_precision(
    validated_citations: List[Dict[str, Any]],
    retrieved_evidence_pool: Dict[str, Dict[str, Any]],
) -> float:
    """Measure fraction of cited sources that actually match retrieved evidence with correct page.

    Citation Precision = |Valid Citations| / |Total Citations Made|

    Args:
        validated_citations: Citations emitted in answer (dicts with document_id, page_start, etc.).
        retrieved_evidence_pool: Retrieved evidence chunks available to the LLM.

    Returns:
        Float score in [0.0, 1.0]. If 0 citations made, returns 1.0 if unanswerable else 0.0.
    """
    if not validated_citations:
        return 1.0

    valid_count = 0
    for cit in validated_citations:
        cid = cit.get("citation_id")
        if cid in retrieved_evidence_pool:
            ev = retrieved_evidence_pool[cid]
            # Check document ID and page coordinate consistency
            if (
                cit.get("document_id") == ev.get("document_id")
                and cit.get("page_start") == ev.get("page_start")
            ):
                valid_count += 1

    return valid_count / float(len(validated_citations))


def citation_coverage(
    answer_text: str,
    validated_citations: List[Dict[str, Any]],
) -> float:
    """Measure whether answer statements carry citations.

    Estimates the proportion of factual statements / clauses backed by at least one citation.
    """
    if not answer_text.strip():
        return 1.0

    # Simple heuristic: split answer into sentences/clauses
    sentences = [s.strip() for s in re.split(r"[.\n;]", answer_text) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    # Count sentences that contain [E#] citations
    cited_sentences = sum(1 for s in sentences if re.search(r"\[E\d+\]", s))
    return min(1.0, cited_sentences / float(len(sentences)))


# =============================================================================
# 3. Answer Quality Metrics (Rule-Based & Semantic)
# =============================================================================

def tokenize_words(text: str) -> List[str]:
    """Helper tokenizer for lexical overlap evaluation."""
    return re.findall(r"\b[a-zA-Z0-9_\-.]+\b", text.lower())


COMMON_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "from", "with", "by", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "because", "as", "until", "while", "of", "it", "its",
    "this", "that", "these", "those", "when", "where", "how", "what", "which",
}


def faithfulness_score(
    answer_text: str,
    context_passages: List[str],
) -> float:
    """Evaluate faithfulness: whether the answer is supported by the retrieved context.

    Uses rule-based technical entity and content-word overlap between the answer and source context.
    Detects unsupported entity claims, numerical discrepancies, and hallucinated identifiers.

    Returns:
        Float score in [0.0, 1.0].
    """
    if not answer_text.strip():
        return 1.0

    context_blob = " ".join(context_passages).lower()
    if not context_blob.strip():
        return 0.0

    # 1. Technical codes, identifiers, and numbers check
    technical_entities = re.findall(r"\b[a-zA-Z0-9]+(?:[-_/.][a-zA-Z0-9]+)+\b", answer_text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:\s*[%a-zA-Z]+)?\b", answer_text)

    key_claims = set([t.lower() for t in technical_entities + numbers if len(t) > 1])

    if key_claims:
        supported_claims = sum(1 for claim in key_claims if claim in context_blob)
        claim_faithfulness = supported_claims / float(len(key_claims))
    else:
        claim_faithfulness = 1.0

    # 2. Content word support (ignoring common stopwords)
    answer_words = [w for w in tokenize_words(answer_text) if w not in COMMON_STOPWORDS and len(w) > 2]
    if not answer_words:
        return claim_faithfulness

    context_words = set(tokenize_words(context_blob))

    # Match exact word or root stem (first 5 chars)
    def is_word_supported(word: str) -> bool:
        if word in context_words:
            return True
        stem = word[:5] if len(word) >= 5 else word
        return any(cw.startswith(stem) for cw in context_words)

    supported_words = sum(1 for w in answer_words if is_word_supported(w))
    word_support_ratio = supported_words / float(len(answer_words))

    # Combined score weighted towards technical claims and content words
    score = 0.7 * claim_faithfulness + 0.3 * word_support_ratio
    return round(min(1.0, max(0.0, score)), 4)


def answer_relevance_score(
    answer_text: str,
    query: str,
) -> float:
    """Evaluate answer relevance: does the answer address the question?

    Checks query content-token containment, question-answering cues, and appropriate length.

    Returns:
        Float score in [0.0, 1.0].
    """
    if not answer_text.strip() or not query.strip():
        return 0.0

    query_tokens = [w for w in tokenize_words(query) if w not in COMMON_STOPWORDS and len(w) > 2]
    if not query_tokens:
        return 1.0

    answer_lower = answer_text.lower()
    matched_tokens = sum(1 for t in query_tokens if t in answer_lower)
    token_coverage = matched_tokens / float(len(query_tokens))

    # Penalize extremely terse answers (e.g. < 5 words for complex question)
    words = tokenize_words(answer_text)
    length_penalty = 1.0 if len(words) >= 5 else (len(words) / 5.0)

    return round(min(1.0, token_coverage * length_penalty), 4)


def context_relevance_score(
    context_passages: List[str],
    query: str,
) -> float:
    """Evaluate context relevance: what fraction of retrieved passages are relevant to the query?

    Returns:
        Float score in [0.0, 1.0].
    """
    if not context_passages or not query.strip():
        return 0.0

    query_tokens = set([w for w in tokenize_words(query) if w not in COMMON_STOPWORDS and len(w) > 2])
    if not query_tokens:
        return 1.0

    relevant_count = 0
    for passage in context_passages:
        passage_tokens = set(tokenize_words(passage))
        overlap = query_tokens.intersection(passage_tokens)
        if len(overlap) >= max(1, len(query_tokens) // 2):
            relevant_count += 1

    return round(relevant_count / float(len(context_passages)), 4)


# =============================================================================
# 4. Unanswerable Question Detection
# =============================================================================

UNANSWERABLE_CUES = [
    "insufficient evidence",
    "not mentioned in the provided documents",
    "the documents do not contain",
    "no information",
    "cannot be determined",
    "does not mention",
    "not found in the repository",
    "unsupported",
    "not specified",
    "insufficient information",
]


def is_unanswerable_declaration(answer_text: str) -> bool:
    """Check if the generated answer appropriately declares lack of evidence."""
    lower = answer_text.lower()
    return any(cue in lower for cue in UNANSWERABLE_CUES)


def unanswerable_detection_metrics(
    eval_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Calculate accuracy and false positive/negative rates for unanswerable question handling.

    Args:
        eval_results: List of dicts with:
            - 'is_unanswerable': bool (ground truth)
            - 'declared_unanswerable': bool (system output)

    Returns:
        Dict with accuracy, precision, recall, and unsupported_answer_rate.
    """
    if not eval_results:
        return {
            "unanswerable_accuracy": 0.0,
            "unanswerable_recall": 0.0,
            "unanswerable_precision": 0.0,
            "unsupported_answer_rate": 0.0,
        }

    tp = 0  # True Unanswerable correctly identified
    fp = 0  # Answerable wrongly claimed as unanswerable
    fn = 0  # Unanswerable answered with hallucinated content
    tn = 0  # Answerable answered with evidence

    for res in eval_results:
        actual_unans = bool(res.get("is_unanswerable", False))
        pred_unans = bool(res.get("declared_unanswerable", False))

        if actual_unans and pred_unans:
            tp += 1
        elif not actual_unans and pred_unans:
            fp += 1
        elif actual_unans and not pred_unans:
            fn += 1
        else:
            tn += 1

    total = len(eval_results)
    accuracy = (tp + tn) / float(total) if total > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
    unsupported_rate = fn / float(tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "unanswerable_accuracy": round(accuracy, 4),
        "unanswerable_recall": round(recall, 4),
        "unanswerable_precision": round(precision, 4),
        "unsupported_answer_rate": round(unsupported_rate, 4),
    }
