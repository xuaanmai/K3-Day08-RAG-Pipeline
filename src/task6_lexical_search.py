"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_NEWS_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# ---------------------------------------------------------------------------
# Module-level cache — lazy initialization, chỉ build 1 lần
# ---------------------------------------------------------------------------
_bm25_index: BM25Okapi | None = None
_corpus: list[dict] = []


# ===========================================================================
# Corpus loading
# ===========================================================================

def _load_corpus() -> list[dict]:
    """
    Load corpus từ data/standardized/ (primary), fallback sang data/landing/news/.

    Ưu tiên load từ standardized vì đó là output chuẩn hoá của Task 3.
    Nếu standardized rỗng (Task 3 chưa chạy), load từ landing/news JSON.
    """
    corpus: list[dict] = []

    # ── Primary: load từ standardized .md files (output của Task 3) ──
    if STANDARDIZED_DIR.exists():
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if len(content.strip()) > 50:
                    doc_type = "legal" if "legal" in str(md_file) else "news"
                    corpus.append({
                        "content": content,
                        "metadata": {"source": md_file.name, "type": doc_type},
                    })
            except (UnicodeDecodeError, OSError):
                continue

    # ── Fallback: load từ landing/news JSON nếu standardized rỗng ──
    if not corpus and LANDING_NEWS_DIR.exists():
        for json_file in sorted(LANDING_NEWS_DIR.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                content = data.get("content_markdown", "")
                if len(content.strip()) > 50:
                    corpus.append({
                        "content": content,
                        "metadata": {
                            "source": data.get("url", json_file.name),
                            "type": "news",
                            "title": data.get("title", ""),
                        },
                    })
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue

    return corpus


# ===========================================================================
# BM25 Index
# ===========================================================================

def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index object
    """
    # Tokenize — simple split() phù hợp cho tiếng Anh;
    # có thể thay bằng underthesea.word_tokenize cho tiếng Việt
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def _ensure_index():
    """Lazy init: tự động build BM25 index lần đầu khi gọi lexical_search()."""
    global _bm25_index, _corpus
    if _bm25_index is None:
        _corpus = _load_corpus()
        if _corpus:
            _bm25_index = build_bm25_index(_corpus)
            print(f"  [OK] BM25 index built: {len(_corpus)} documents")
        else:
            print("  [WARN] Khong tim thấy corpus cho BM25 index")


# ===========================================================================
# Search
# ===========================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_index()

    if not _corpus or _bm25_index is None:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Lấy top_k indices có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Chỉ giữ kết quả có keyword match
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": _corpus[idx]["metadata"],
            })

    return results


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Task 6: BM25 Lexical Search")
    print("=" * 60)

    test_queries = [
        "IELTS speaking test tips",
        "writing band score descriptors",
        "tuition fee payment methods",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 40)
        results = lexical_search(q, top_k=3)
        if results:
            for r in results:
                content_preview = r['content'][:100].replace('\n', ' ')
                print(f"  [{r['score']:.3f}] {content_preview}...")
        else:
            print("  (Khong co ket qua)")
