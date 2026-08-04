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
from typing import Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if not candidates:
        return []

    pairs = [[query, c["content"]] for c in candidates]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        scores = model(**inputs).logits.squeeze(-1).cpu().tolist()

    reranked = []

    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked[:top_k]


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
    # TODO: Implement RRF
    #
    # rrf_scores = {}  # content -> score
    # content_map = {}  # content -> full dict
    #
    # for ranked_list in ranked_lists:
    #     for rank, item in enumerate(ranked_list, 1):
    #         key = item["content"]
    #         rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
    #         content_map[key] = item
    #
    # # Sort by RRF score
    # sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    #
    # results = []
    # for content, score in sorted_items[:top_k]:
    #     item = content_map[content].copy()
    #     item["score"] = score
    #     results.append(item)
    #
    # return results
    raise NotImplementedError("Implement rerank_rrf")


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
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
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
