# RAG Evaluation Results — IELTS Band Descriptors

> Trạng thái: script đánh giá đã hoàn thiện; chưa ghi điểm RAGAS mới vì chưa chạy đủ 18 test case trên môi trường hiện tại.

## Cấu hình so sánh

- Config A: hybrid retrieval (dense + BM25) và RRF reranking.
- Config B: dense-only retrieval, không BM25/RRF.
- Metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision.
- Golden dataset: 18 câu hỏi trong `golden_dataset.json`.

## Cách tạo báo cáo chính thức

```powershell
python -m group_project.evaluation.eval_pipeline --limit 18
```

Lệnh trên sẽ thay nội dung file này bằng bảng điểm thực, ba câu có điểm thấp nhất và đề xuất cải tiến. Dữ liệu từng câu (answer, contexts và scores) được lưu vào `evaluation_details.json` để kiểm chứng.

Không sử dụng các con số ước lượng hoặc nhập tay làm kết quả nộp bài.
