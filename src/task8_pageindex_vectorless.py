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

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"


# ===========================================================================
# Helpers
# ===========================================================================

def _save_doc_ids(doc_ids: dict):
    """Lưu danh sách doc_id đã upload để dùng lại cho search."""
    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
    print(f"  [SAVE] Saved {len(doc_ids)} doc IDs -> {DOC_IDS_FILE}")


def _load_doc_ids() -> dict:
    """Load danh sách doc_id đã upload trước đó."""
    if DOC_IDS_FILE.exists():
        try:
            return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _md_to_pdf(md_path: Path, pdf_path: Path):
    """
    Convert markdown file sang PDF đơn giản bằng fpdf2.
    PageIndex chỉ nhận PDF, không nhận .md trực tiếp.
    """
    from fpdf import FPDF

    content = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=10)

    for line in content.split("\n"):
        # Encode sang latin-1 (fpdf2 default) — replace ký tự không hỗ trợ
        clean_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, clean_line)

    pdf.output(str(pdf_path))


# ===========================================================================
# Upload Documents
# ===========================================================================

def upload_documents():
    """
    Upload toàn bộ documents lên PageIndex.

    Hỗ trợ:
        - PDF files từ data/landing/legal/ (upload trực tiếp)
        - Markdown files từ data/standardized/ (convert sang PDF trước)

    Doc IDs được lưu vào data/pageindex_doc_ids.json để dùng lại khi search.
    """
    if not PAGEINDEX_API_KEY:
        print("[WARN] Hay set PAGEINDEX_API_KEY trong file .env")
        print("  Dang ky tai: https://pageindex.ai/")
        return

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids = _load_doc_ids()

    # ── 1. Upload PDFs từ landing/legal/ (trực tiếp) ──
    if LANDING_LEGAL_DIR.exists():
        pdf_files = list(LANDING_LEGAL_DIR.glob("*.pdf"))
        print(f"\n[INFO] Tim thay {len(pdf_files)} PDF files trong landing/legal/")

        for pdf_file in pdf_files:
            if pdf_file.name in doc_ids:
                print(f"  [SKIP] Da upload truoc: {pdf_file.name}")
                continue
            try:
                print(f"  [UPLOAD] Uploading: {pdf_file.name} ({pdf_file.stat().st_size // 1024}KB)...")
                resp = client.submit_document(str(pdf_file))
                doc_id = resp.get("doc_id") or resp.get("id")
                if doc_id:
                    doc_ids[pdf_file.name] = doc_id
                    print(f"  [OK] doc_id: {doc_id}")
                else:
                    print(f"  [FAIL] Khong lay duoc doc_id: {resp}")
            except Exception as e:
                print(f"  [FAIL] Loi upload {pdf_file.name}: {e}")

    # ── 2. Upload markdown từ standardized/ (convert → PDF trước) ──
    if STANDARDIZED_DIR.exists():
        md_files = list(STANDARDIZED_DIR.rglob("*.md"))
        if md_files:
            print(f"\n[INFO] Tim thay {len(md_files)} markdown files trong standardized/")
            tmp_dir = Path(__file__).parent.parent / "data" / ".tmp_pdf"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            for md_file in md_files:
                if md_file.name in doc_ids:
                    print(f"  [SKIP] Da upload truoc: {md_file.name}")
                    continue
                try:
                    pdf_path = tmp_dir / f"{md_file.stem}.pdf"
                    _md_to_pdf(md_file, pdf_path)

                    print(f"  [UPLOAD] Uploading: {md_file.name} (-> PDF)...")
                    resp = client.submit_document(str(pdf_path))
                    doc_id = resp.get("doc_id") or resp.get("id")
                    if doc_id:
                        doc_ids[md_file.name] = doc_id
                        print(f"  [OK] doc_id: {doc_id}")
                    else:
                        print(f"  [FAIL] Khong lay duoc doc_id: {resp}")
                except Exception as e:
                    print(f"  [FAIL] Loi voi {md_file.name}: {e}")

    _save_doc_ids(doc_ids)
    print(f"\n{'=' * 40}")
    print(f"[OK] Tong documents da upload: {len(doc_ids)}")


# ===========================================================================
# Search
# ===========================================================================

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
    if not PAGEINDEX_API_KEY:
        print("  [WARN] PAGEINDEX_API_KEY not set — skipping PageIndex search")
        return []

    doc_ids = _load_doc_ids()
    if not doc_ids:
        print("  [WARN] Chua upload documents len PageIndex — chay upload_documents() truoc")
        return []

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    all_results: list[dict] = []

    for doc_name, doc_id in doc_ids.items():
        try:
            # Submit query cho document này
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")

            if not retrieval_id:
                print(f"  [WARN] Khong lay duoc retrieval_id cho {doc_name}: {resp}")
                continue

            # Poll cho đến khi status == "completed" (max ~30 giây)
            retrieval = None
            for attempt in range(15):
                retrieval = client.get_retrieval(retrieval_id)
                status = retrieval.get("status", "")
                if status == "completed":
                    break
                if status == "failed":
                    print(f"  [FAIL] Retrieval failed cho {doc_name}")
                    break
                time.sleep(2)

            if not retrieval or retrieval.get("status") != "completed":
                continue

            # Parse retrieved_nodes → relevant_contents
            # Schema: retrieved_nodes[].relevant_contents = list[list[{section_title, relevant_content}]]
            for node in retrieval.get("retrieved_nodes", [])[:3]:
                for group in node.get("relevant_contents", []):
                    for item in group:
                        content = item.get("relevant_content", "")
                        if content.strip():
                            all_results.append({
                                "content": content.strip(),
                                "score": 0.0,  # Sẽ gán theo rank bên dưới
                                "metadata": {
                                    "section": item.get("section_title", ""),
                                    "source": doc_name,
                                },
                                "source": "pageindex",
                            })

        except Exception as e:
            print(f"  [WARN] PageIndex query error ({doc_name}): {e}")
            continue

    # Gán score theo rank (PageIndex không trả score trực tiếp)
    # Dùng 1/(1+rank) để tạo score giảm dần tự nhiên
    for i, r in enumerate(all_results):
        r["score"] = round(1.0 / (1 + i), 4)

    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[WARN] Hay set PAGEINDEX_API_KEY trong file .env")
        print("  Dang ky tai: https://pageindex.ai/")
    else:
        print("=" * 60)
        print("Task 8: PageIndex Vectorless RAG")
        print("=" * 60)

        print("\n[INFO] Step 1: Upload documents...")
        upload_documents()

        print("\n[INFO] Step 2: Test query...")
        results = pageindex_search("IELTS speaking test tips", top_k=3)
        if results:
            for r in results:
                print(f"  [{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
        else:
            print("  (Khong co ket qua - co the documents chua duoc xu ly xong)")
