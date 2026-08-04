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

from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    raise NotImplementedError("Implement rerank_cross_encoder")


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
    # TODO: Implement MMR (optional — dùng RRF thay thế)
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Công thức: RRF(d) = Σ 1 / (k + rank_r(d))
    Trong đó:
        - k = 60 (smoothing constant, từ paper Cormack et al. 2009)
        - rank_r(d) = thứ hạng của document d trong ranked list r (bắt đầu từ 1)

    Ví dụ: document xuất hiện ở rank 1 trong cả 2 list:
        RRF = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}    # content → accumulated RRF score
    content_map: dict[str, dict] = {}    # content → full dict (giữ metadata)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            # Giữ phiên bản đầu tiên gặp (ưu tiên ranker đầu = semantic)
            if key not in content_map:
                content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

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

    Khi method="rrf" và nhận 1 candidate list (từ test hoặc pipeline đơn giản):
    → Sort candidates theo score gốc để tạo ranked list, rồi áp dụng RRF.

    Khi cần merge nhiều nguồn (semantic + BM25), gọi trực tiếp rerank_rrf()
    với nhiều ranked lists — xem Task 9 pipeline.

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
        # Cần query_embedding — gọi rerank_mmr() trực tiếp
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # Sort candidates theo score gốc → tạo 1 ranked list → áp dụng RRF
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("score", 0), reverse=True
        )
        return rerank_rrf([sorted_candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Task 7: RRF Reranking")
    print("=" * 60)

    # ── Test 1: rerank() unified interface (single candidate list) ──
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]

    print("\n[INFO] Test 1: rerank() with single candidate list")
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']}")

    # ── Test 2: rerank_rrf() with multiple ranked lists (semantic + BM25) ──
    semantic_results = [
        {"content": "Tuition fee payment schedule", "score": 0.9, "metadata": {}},
        {"content": "Scholarship eligibility", "score": 0.7, "metadata": {}},
        {"content": "Library study room guide", "score": 0.5, "metadata": {}},
    ]
    bm25_results = [
        {"content": "Scholarship eligibility", "score": 5.2, "metadata": {}},
        {"content": "Tuition fee payment schedule", "score": 3.1, "metadata": {}},
        {"content": "Campus parking regulations", "score": 2.8, "metadata": {}},
    ]

    print("\n[INFO] Test 2: rerank_rrf() with 2 ranked lists (Semantic + BM25)")
    results = rerank_rrf([semantic_results, bm25_results], top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']}")
    print(f"\n  [NOTE] max RRF score ~= {2/(60+1):.4f} (2 lists, rank 1 ca 2)")
