"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import math
import os
import re
from typing import Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - dependency may be absent
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - dependency may be absent
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    import requests
except ImportError:  # pragma: no cover - dependency may be absent
    requests = None

JINA_API_KEY = os.getenv("JINA_API_KEY")


def _keyword_overlap_score(query: str, content: str) -> float:
    """Simple lexical overlap bonus for fallback reranking."""
    if not query or not content:
        return 0.0

    query_tokens = {
        token.lower()
        for token in re.findall(r"\w+", query)
        if token and token.isalnum()
    }
    content_tokens = {
        token.lower()
        for token in re.findall(r"\w+", content)
        if token and token.isalnum()
    }

    if not query_tokens:
        return 0.0

    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    return overlap


def _simple_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Fallback reranking that uses original scores plus lexical overlap."""
    if not candidates:
        return []

    ranked = []
    for candidate in candidates:
        item = candidate.copy()
        base_score = float(item.get("score", 0.0))
        overlap = _keyword_overlap_score(query, str(item.get("content", "")))
        item["score"] = base_score + overlap
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[: min(top_k, len(ranked))]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates using Jina Cross-Encoder.

    Args:
        query: User query.
        candidates: List of {
            "content": str,
            "score": float,
            "metadata": dict
        }
        top_k: Number of results to return.

    Returns:
        List of candidates sorted by rerank score.
    """
    if not candidates:
        return []

    if not JINA_API_KEY or requests is None:
        return _simple_rerank(query, candidates, top_k)

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=60,
    )

    response.raise_for_status()
    results = response.json()["results"]

    reranked = []
    for r in results:
        item = candidates[r["index"]].copy()
        item["score"] = r["relevance_score"]
        reranked.append(item)

    return reranked


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    top_k = min(top_k, len(candidates))

    selected_indices = []
    selected_results = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < top_k:
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = cosine_similarity(
                query_embedding,
                candidates[idx]["embedding"],
            )

            if selected_indices:
                max_sim = max(
                    cosine_similarity(
                        candidates[idx]["embedding"],
                        candidates[s]["embedding"],
                    )
                    for s in selected_indices
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        item = candidates[best_idx].copy()
        item["score"] = best_score          # cập nhật score thành MMR score
        item["mmr_score"] = best_score

        selected_results.append(item)
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return selected_results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0 or not ranked_lists:
        return []
    if k < 0:
        raise ValueError("k must be greater than or equal to 0")

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    seen_order = 0

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue

        # Một document chỉ được đóng góp một lần trong mỗi ranked list. Điều này
        # tránh việc dữ liệu trùng lặp trong cùng retriever làm tăng điểm giả tạo.
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if content in seen_in_list:
                continue
            seen_in_list.add(content)

            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)
            if content not in content_map:
                # Giữ phiên bản xuất hiện ở thứ hạng cao nhất đầu tiên; thường đây
                # là bản có metadata/score gốc đáng tin cậy nhất.
                content_map[content] = item
                first_seen[content] = seen_order
                seen_order += 1

    ordered_contents = sorted(
        rrf_scores,
        key=lambda content: (-rrf_scores[content], first_seen[content]),
    )

    results = []
    for content in ordered_contents[:top_k]:
        result = content_map[content].copy()
        result["score"] = rrf_scores[content]
        results.append(result)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if top_k <= 0 or not candidates:
        return []

    normalized_method = method.strip().lower()
    if normalized_method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif normalized_method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif normalized_method == "rrf":
        # Interface này nhận một ranked list. Trong hybrid pipeline, hãy gọi
        # rerank_rrf([dense_results, sparse_results]) trực tiếp để fuse nhiều list.
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
