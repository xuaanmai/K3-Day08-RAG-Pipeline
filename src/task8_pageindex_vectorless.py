"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_ROOT = Path(__file__).parent.parent
DOC_IDS_FILE = PROJECT_ROOT / "pageindex_doc_ids.json"
API_BASE_URL = os.getenv("PAGEINDEX_API_URL", "https://api.pageindex.ai").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("PAGEINDEX_REQUEST_TIMEOUT", "30"))
POLL_TIMEOUT = float(os.getenv("PAGEINDEX_POLL_TIMEOUT", "90"))
POLL_INTERVAL = float(os.getenv("PAGEINDEX_POLL_INTERVAL", "2"))


def _headers() -> dict[str, str]:
    return {"api_key": PAGEINDEX_API_KEY}


def _require_api_key() -> None:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Thiếu PAGEINDEX_API_KEY. Hãy thêm key vào file .env trước khi dùng PageIndex."
        )


def _response_json(response: requests.Response) -> dict[str, Any]:
    """Raise an informative error and return a JSON object."""
    try:
        payload = response.json()
    except ValueError as exc:
        response.raise_for_status()
        raise RuntimeError("PageIndex trả về response không phải JSON") from exc

    if not response.ok:
        detail = payload.get("detail") or payload.get("message") or payload
        raise RuntimeError(f"PageIndex API error {response.status_code}: {detail}")
    if not isinstance(payload, dict):
        raise RuntimeError("PageIndex trả về JSON không đúng định dạng object")
    return payload


def _load_doc_ids() -> dict[str, str]:
    if not DOC_IDS_FILE.exists():
        return {}
    try:
        raw = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    # Hỗ trợ cả cache dạng {path: doc_id} và {path: {doc_id: ...}}.
    doc_ids: dict[str, str] = {}
    if isinstance(raw, dict):
        for path, value in raw.items():
            doc_id = value.get("doc_id") if isinstance(value, dict) else value
            if isinstance(doc_id, str) and doc_id:
                doc_ids[str(path)] = doc_id
    return doc_ids


def _save_doc_ids(doc_ids: dict[str, str]) -> None:
    DOC_IDS_FILE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def upload_documents() -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    _require_api_key()

    pdf_files = sorted(STANDARDIZED_DIR.rglob("*.pdf"))
    if not pdf_files:
        # Một số phiên bản starter lưu PDF gốc trong data/landing/.
        pdf_files = sorted((PROJECT_ROOT / "data" / "landing").rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            "Không tìm thấy PDF trong data/standardized hoặc data/landing để upload."
        )

    doc_ids = _load_doc_ids()
    for pdf_path in pdf_files:
        cache_key = str(pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if cache_key in doc_ids:
            print(f"  [cached] {pdf_path.name} -> {doc_ids[cache_key]}")
            continue

        print(f"  Uploading: {pdf_path.name}")
        with pdf_path.open("rb") as file_handle:
            response = requests.post(
                f"{API_BASE_URL}/doc/",
                headers=_headers(),
                files={"file": (pdf_path.name, file_handle, "application/pdf")},
                timeout=REQUEST_TIMEOUT,
            )
        payload = _response_json(response)
        doc_id = payload.get("doc_id") or payload.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"Upload {pdf_path.name} không trả về doc_id: {payload}")

        doc_ids[cache_key] = doc_id
        _save_doc_ids(doc_ids)  # Lưu ngay để không mất ID nếu upload sau bị lỗi.
        print(f"  [uploaded] {pdf_path.name} -> {doc_id}")

    return doc_ids


def _iter_relevant_items(value: Any):
    """Flatten relevant_contents across old and current response variants."""
    if isinstance(value, dict):
        if value.get("relevant_content") or value.get("content"):
            yield value
        else:
            for nested in value.values():
                yield from _iter_relevant_items(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_relevant_items(nested)


def _parse_retrieval(payload: dict[str, Any], doc_id: str) -> list[dict]:
    results = []
    for node in payload.get("retrieved_nodes") or []:
        if not isinstance(node, dict):
            continue
        node_title = node.get("title") or node.get("section_title") or "Unknown section"
        for item in _iter_relevant_items(node.get("relevant_contents") or []):
            content = item.get("relevant_content") or item.get("content") or ""
            if not isinstance(content, str) or not content.strip():
                continue
            results.append(
                {
                    "content": content.strip(),
                    "metadata": {
                        "source": node_title,
                        "section": item.get("section_title") or node_title,
                        "node_id": node.get("node_id"),
                        "page_index": item.get("page_index"),
                        "doc_id": doc_id,
                        "type": "pageindex",
                    },
                    "source": "pageindex",
                }
            )
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0 or not query or not query.strip():
        return []
    _require_api_key()

    doc_ids = list(dict.fromkeys(_load_doc_ids().values()))
    if not doc_ids:
        raise RuntimeError(
            "Chưa có document ID. Hãy chạy upload_documents() trước khi search."
        )

    # Submit retrieval cho tất cả tài liệu trước, rồi poll chung để tránh phải chờ
    # tuần tự 90 giây cho từng document.
    pending: dict[str, str] = {}
    for doc_id in doc_ids:
        response = requests.post(
            f"{API_BASE_URL}/retrieval/",
            headers=_headers(),
            json={"doc_id": doc_id, "query": query.strip(), "thinking": False},
            timeout=REQUEST_TIMEOUT,
        )
        payload = _response_json(response)
        retrieval_id = payload.get("retrieval_id") or payload.get("id")
        if isinstance(retrieval_id, str) and retrieval_id:
            pending[retrieval_id] = doc_id

    if not pending:
        return []

    collected: list[dict] = []
    deadline = time.monotonic() + POLL_TIMEOUT
    while pending and time.monotonic() < deadline:
        for retrieval_id, doc_id in list(pending.items()):
            response = requests.get(
                f"{API_BASE_URL}/retrieval/{retrieval_id}/",
                headers=_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            payload = _response_json(response)
            status = str(payload.get("status", "")).lower()
            if status == "completed":
                collected.extend(_parse_retrieval(payload, doc_id))
                pending.pop(retrieval_id, None)
            elif status in {"failed", "error", "cancelled", "canceled"}:
                pending.pop(retrieval_id, None)

        if pending and time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)

    # PageIndex legacy retrieval không cung cấp relevance score chuẩn hoá. Gán
    # score giảm theo rank để tương thích contract của Task 9, không dùng score
    # này làm ngưỡng fallback semantic.
    results = []
    seen_content: set[str] = set()
    for item in collected:
        content = item["content"]
        if content in seen_content:
            continue
        seen_content.add(content)
        ranked_item = item.copy()
        ranked_item["score"] = 1.0 / (len(results) + 1)
        results.append(ranked_item)
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[warning] Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search(
            "What are the IELTS Writing Band 8 lexical resource criteria?", top_k=3
        )
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
