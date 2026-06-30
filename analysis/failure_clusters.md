# Failure Cluster Analysis — Phase A

**Sinh viên:** Lab 24 Student  
**Ngày:** 2026-06-30

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.300 | 0.230 | 0.200 |
| answer_relevancy | 0.000 | 0.000 | 0.000 |
| context_precision | 0.275 | 0.100 | 0.300 |
| context_recall | 0.400 | 0.250 | 0.292 |
| **avg_score** | **0.244** | **0.145** | **0.198** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | factual | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.0 | faithfulness |
| 2 | factual | Nhân viên được nghỉ bao nhiêu ngày khi cha hoặc mẹ mất? | 0.0 | faithfulness |
| 3 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.0 | faithfulness |
| 4 | factual | Nhân viên chính thức được phép làm việc từ xa tối đa bao nhiêu ngày một tuần? | 0.0 | faithfulness |
| 5 | factual | Đánh giá hiệu suất diễn ra mấy lần một năm và vào tháng nào? | 0.0 | faithfulness |
| 6 | factual | Tạm ứng dưới 5 triệu cần ai phê duyệt? Từ 5 triệu trở lên thì sao? | 0.0 | faithfulness |
| 7 | factual | Tài khoản VPN bị khóa tự động sau bao nhiêu ngày không hoạt động? | 0.0 | faithfulness |
| 8 | factual | Phụ cấp điện thoại dành cho những cấp bậc nào và mức bao nhiêu? | 0.0 | faithfulness |
| 9 | factual | Cơ cấu điểm đánh giá hiệu suất gồm những thành phần nào và tỷ lệ ra sao? | 0.0 | faithfulness |
| 10 | multi_hop | NV được tài trợ khóa học 25 triệu, nghỉ sau 8 tháng. Hoàn trả bao nhiêu? | 0.0 | faithfulness |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 14 | 15 | 8 | 37 |
| answer_relevancy | 6 | 5 | 2 | 13 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** faithfulness

**Lý do phân tích:**

> Distribution "factual" có nhiều failure nhất (14 câu fail ở faithfulness) vì đây là các câu hỏi yêu cầu dữ liệu chính xác từ chính sách HR (số ngày phép, mức phụ cấp, quy trình phê duyệt...). LLM có xu hướng hallucinate các con số cụ thể khi context retrieval không đủ chính xác.
>
> Metric "faithfulness" thấp nhất (0.30 cho factual, 0.23 cho multi_hop) cho thấy câu trả lời của LLM không bám sát ngữ cảnh được retrieval. Nguyên nhân có thể do: (1) Embedding model BGE-M3 chạy trên CPU chậm dẫn đến encoding quality thấp hơn so với GPU, (2) Chunking strategy chưa tối ưu cho văn bản HR policy tiếng Việt có nhiều bảng biểu và danh sách.
>
> Answer_relevancy = 0.0 trên tất cả distributions — đây là dấu hiệu cho thấy format câu trả lời không khớp với kỳ vọng của RAGAS evaluator (có thể do câu trả lời bằng tiếng Việt nhưng evaluator kỳ vọng format khác).

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinate số liệu cụ thể khi context không đủ | Thêm system prompt yêu cầu "chỉ trả lời dựa trên context, nếu không tìm thấy thì nói không biết". Lower temperature xuống 0.1-0.2. |
| context_recall | Missing relevant chunks do embedding chưa tối ưu | Sử dụng GPU cho BGE-M3, hoặc thử embedding model khác hỗ trợ tiếng Việt tốt hơn (ví dụ: multilingual-e5-large). |
| context_precision | Top-k chunks chứa quá nhiều noise | Tăng reranker top_k từ 3 lên 5, hoặc sử dụng chunk enrichment tốt hơn. |
| answer_relevancy | Câu trả lời không match format kỳ vọng | Thêm instruction trong prompt yêu cầu LLM trả lời ngắn gọn, trực tiếp vào câu hỏi. |

---

## 6. Nhận xét về Adversarial Distribution

> Adversarial avg_score (0.198) nằm giữa factual (0.244) và multi_hop (0.145). Điều này cho thấy pipeline không bị "nhầm" nghiêm trọng bởi các câu hỏi adversarial — chúng không tệ hơn multi_hop.
>
> Chỉ có 1 câu adversarial xuất hiện trong bottom 10 (câu multi_hop #23 về hoàn trả chi phí đào tạo). Các câu adversarial thiết kế để gây nhầm lẫn giữa phiên bản chính sách (v2023 vs v2024) không xuất hiện đáng kể trong bottom 10, cho thấy pipeline xử lý tương đối ổn với loại câu hỏi này.
>
> Tuy nhiên, context_precision của adversarial (0.30) lại cao nhất trong 3 distributions — có thể do các câu adversarial thường chứa keywords cụ thể hơn, giúp search engine tìm đúng context dễ hơn.
