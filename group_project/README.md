# IELTS Writing Band Descriptor RAG Assistant

Chatbot hỗ trợ người học tra cứu bốn tiêu chí IELTS Writing, so sánh band và đọc phân tích bài mẫu dựa trên IELTS Band Descriptors cùng examiner comments. Câu trả lời được sinh bằng Gemini, có citation và hiển thị các đoạn nguồn đã dùng.

## Tính năng

- Giao diện chat Streamlit, có câu hỏi gợi ý và bảng nguồn.
- Conversation memory cho câu hỏi nối tiếp như “Band 7 thì sao?”.
- Hybrid retrieval: dense search + BM25 + Reciprocal Rank Fusion.
- PageIndex là fallback tùy chọn khi bằng chứng hybrid yếu.
- Câu trả lời tiếng Việt có citation thân thiện, không tự tạo nguồn.
- Evaluation bằng RAGAS với 18 golden Q&A và so sánh hybrid với dense-only.

## Kiến trúc

```text
PDF / JSON
    │
    ▼
Markdown chuẩn hóa → Header-aware chunking → ChromaDB
                              │
User → Streamlit → Query + conversation history
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
         Dense retrieval                  BM25
               └──────────────┬──────────────┘
                              ▼
                     RRF reranking (Task 7)
                              │ evidence yếu
                              ├───────────────→ PageIndex fallback
                              ▼
                  Gemini generation + citation
                              ▼
                    Answer + source documents
```

## Cài đặt trên Windows PowerShell

Khuyến nghị Python 3.11 hoặc 3.12; một số dependency chưa có wheel ổn định cho Python 3.14.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền ít nhất một key sinh câu trả lời trong `.env`:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
LLM_MAX_TOKENS=3000
```

`PAGEINDEX_API_KEY` không bắt buộc: chỉ cần khi muốn bật vectorless fallback. `JINA_API_KEY` không cần cho dữ liệu Markdown/PDF hiện tại vì pipeline không gọi Jina Reader.

## Chuẩn bị index và chạy app

```powershell
python -m src.task4_chunking_indexing
streamlit run app.py
```

Nếu muốn tải embedding model thay vì dùng cache/fallback cục bộ:

```powershell
$env:ALLOW_MODEL_DOWNLOAD="1"
python -m src.task4_chunking_indexing
```

## Chạy evaluation

Chạy thử 5 câu trước để kiểm tra quota, sau đó chạy đủ 18 câu cho deliverable:

```powershell
python -m group_project.evaluation.eval_pipeline --limit 5
python -m group_project.evaluation.eval_pipeline --limit 18
```

Script chạy bốn metric RAGAS: Faithfulness, Answer Relevance, Context Recall và Context Precision; đồng thời so sánh:

- Config A: hybrid retrieval + RRF reranking.
- Config B: dense-only retrieval.

Kết quả được ghi vào `evaluation/results.md`; dữ liệu từng câu được lưu tại `evaluation/evaluation_details.json` để kiểm chứng.

## Kiểm thử

```powershell
pytest -q
python -m py_compile app.py src\task*.py group_project\evaluation\eval_pipeline.py
```

## Phân công

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Nguyễn Thị Xuân Mai | 2A202601691 | Mở rộng `golden_dataset.json` lên 20 câu hỏi; chạy RAGAS benchmark; viết báo cáo `results.md`; chuyển PDF sang Markdown | Hoàn thành |
| Lê Tuấn Hiệp | 2A202601667 | Quản lý nhóm; thuyết trình demo; thu thập và tìm tài liệu; chuyển tài liệu sang PDF | Hoàn thành |
| Trần Doãn Hưng | 2A202601143 | Thiết kế Streamlit Chatbot `app.py`; Task 10 Citation Generation; chuyển JSON sang Markdown | Hoàn thành |
| Cao Hữu Phúc | 2A202601283 | Task 4 Chunking & ChromaDB Indexing; Task 5 Semantic Search và HyDE; crawl dữ liệu web; hoàn thiện `semantic_search` | Hoàn thành |
| Ngô Khánh Trượng | 2A202601477 | Task 6 BM25/TF-IDF; Task 7 RRF Reranking; Task 8 PageIndex Fallback; hoàn thiện `lexical_search` | Hoàn thành |
