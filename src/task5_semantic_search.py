"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (ChromaDB + Cosine similarity).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    from src.task4_chunking_indexing import get_collection, get_embedding_model

    model = get_embedding_model()
    collection = get_collection()

    if collection.count() == 0:
        try:
            from src.task4_chunking_indexing import run_pipeline
            run_pipeline()
        except Exception:
            pass

    if collection.count() == 0:
        return []

    count = collection.count()
    actual_top_k = min(top_k, count)

    if model:
        query_vector = model.encode(query)
        query_embeddings = [query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)]
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=actual_top_k,
            include=["documents", "metadatas", "distances"],
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=actual_top_k,
            include=["documents", "metadatas", "distances"],
        )

    output = []
    if results and results.get("documents") and len(results["documents"]) > 0:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Cosine distance range [0, 2], similarity score = max(0, 1 - dist)
            score = max(0.0, 1.0 - float(dist))
            output.append({
                "content": doc,
                "score": round(score, 4),
                "metadata": meta if meta else {}
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
