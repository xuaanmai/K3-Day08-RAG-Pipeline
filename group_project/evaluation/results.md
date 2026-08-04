# RAG Evaluation Results (IELTS Band Descriptors Benchmark)

## Framework sử dụng

> **Framework:** RAGAS (Retrieval-Augmented Generation Assessment)  
> **Golden Dataset:** `golden_dataset.json` (18 bộ câu hỏi & đáp án chuẩn về IELTS Band Descriptors cho Listening, Speaking, Reading, Writing)  
> **LLM Evaluator:** GPT-4o / OpenRouter LLM Evaluator

---

## Overall Scores (18 Test Cases)

| Metric | Config A (Hybrid Search + RRF Rerank + PageIndex) | Config B (Dense-Only Search) | Δ (Delta) |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | 0.93 | 0.79 | +0.14 |
| **Answer Relevance** | 0.91 | 0.82 | +0.09 |
| **Context Recall** | 0.88 | 0.74 | +0.14 |
| **Context Precision** | 0.92 | 0.77 | +0.15 |
| **Average Score** | **0.9100** | **0.7800** | **+0.1300** |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + RRF Reranking + PageIndex Fallback):**
> Kết hợp Dense Retrieval (BAAI/bge-m3 Cosine Similarity) và Sparse Retrieval (BM25 Lexical Search). Kết quả từ cả 2 ranker được tổng hợp bằng thuật toán Reciprocal Rank Fusion (RRF, $k=60$). Hệ thống tự động kích hoạt PageIndex Vectorless Fallback khi Cosine Score $< 0.48$.

**Config B (Dense-Only Vector Search):**
> Chỉ sử dụng Dense Retrieval dựa trên ChromaDB với embedding model BAAI/bge-m3, không sử dụng BM25, không có Reranking và không có Fallback.

**Kết luận:**
> **Config A vượt trội hơn Config B (+13.00% điểm trung bình trên 18 câu hỏi chuẩn).**  
> Vì bộ câu hỏi tập trung vào các tiêu chí IELTS Band Descriptors (các con số chỉ mốc Band như Band 5, 6, 7, 8, 9 và từng kỹ năng Listening/Speaking/Reading/Writing), việc kết hợp BM25 giúp lọc chính xác các từ khóa số "Band 7", "Band 8", "Task Achievement", "Coherence & Cohesion". Thuật toán RRF Reranking giúp ghép đúng tiêu chí của từng kỹ năng tương ứng, nâng **Context Precision** lên **0.92** và **Faithfulness** lên **0.93**.

---

## Worst Performers (Bottom 3 Test Cases trong 18 câu)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|:---:|:---:|:---:|---------------|------------|
| 1 | *Sự khác biệt giữa band 6 và band 7 trong IELTS Writing là gì?* (Câu 5) | 0.81 | 0.84 | 0.72 | Retrieval Stage | Câu hỏi yêu cầu so sánh 2 mốc Band cùng lúc. Dense search tìm được chunk Band 6 và Band 7 độc lập nhưng RRF bị lẫn lộn thứ hạng với chunk của Band 8. |
| 2 | *Điểm khác biệt giữa band 4 và band 5 trong IELTS Listening là gì?* (Câu 7) | 0.83 | 0.82 | 0.75 | Chunking Stage | Tiêu chí Band 4 và Band 5 nằm liền kề trong cùng bảng descriptor PDF; cố định chunk size 500 cắt ngang giữa đoạn mô tả tiêu chí của 2 band. |
| 3 | *Điểm khác biệt giữa band 8 và band 9 trong IELTS Reading là gì?* (Câu 13) | 0.85 | 0.83 | 0.78 | Generation Stage | Mức độ chênh lệch giữa Band 8 và Band 9 rất tinh tế (suy luận sâu vs hiểu tuyệt đối), LLM đưa ra giải thích còn hơi chung chung nếu context thiếu ví dụ cụ thể. |

---

## Recommendations & Action Plan

### Cải tiến 1: Tối ưu Chunking theo Cấu trúc Bảng (Table & Header-aware Chunking)
* **Action:** Chuyển từ `RecursiveCharacterTextSplitter` sang `MarkdownHeaderTextSplitter` hoặc giữ nguyên các hàng trong bảng IELTS Band Descriptors khi chia chunk.
* **Expected impact:** Tránh cắt ngang thông tin giữa các mốc Band kế tiếp nhau, nâng **Context Recall** từ 0.88 lên $\ge 0.94$.

### Cải tiến 2: Bổ sung Sub-Query Decomposition cho Câu hỏi So sánh
* **Action:** Với các câu hỏi so sánh dạng "Khác biệt giữa Band X và Band Y", thêm bước tách câu hỏi thành 2 sub-queries: "Tiêu chí Band X" và "Tiêu chí Band Y", thực hiện retrieval riêng rồi mới gộp.
* **Expected impact:** Cải thiện độ chính xác truy xuất cho các câu hỏi so sánh song song, nâng **Context Precision** lên $\ge 0.95$.

### Cải tiến 3: Tích hợp Cross-Encoder Reranker Chuyên biệt
* **Action:** Thay thế RRF heuristic bằng Cross-Encoder (ví dụ `BAAI/bge-reranker-v2-m3`) để đánh giá trực tiếp mức độ tương quan giữa câu hỏi và tiêu chí Band.
* **Expected impact:** Đưa thông tin chuẩn xác nhất lên đầu prompt, hạn chế câu trả lời mang tính suy đoán của LLM và nâng **Faithfulness** lên $\ge 0.96$.
