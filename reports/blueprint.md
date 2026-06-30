# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Tống Anh Huy
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~25.58ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL_ADDRESS detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~6.58ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → Gemini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Kết quả từ Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 21.00 | 25.58 | 25.58 | <10ms |
| NeMo Input Rail | 3.87 | 6.58 | 6.58 | <300ms |
| RAG Pipeline | N/A | N/A | N/A | <2000ms |
| NeMo Output Rail | N/A | N/A | N/A | <300ms |
| **Total Guard** | 23.44 | **31.19** | 31.19 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Guard stack tổng P95 = 31.19ms, rất thoải mái so với budget 500ms. Presidio PII scan P95 (25.58ms) cao hơn budget mong muốn (<10ms) nhưng vẫn nằm trong ngưỡng chấp nhận được cho production. NeMo input rail cực kỳ nhanh (<7ms P95) nhờ sử dụng keyword pre-filter trước khi gọi semantic matching.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | factual: 0.244, multi_hop: 0.145, adversarial: 0.198 |
| Worst metric | answer_relevancy (0.0 trên cả 3 distributions) |
| Dominant failure distribution | factual (14 failures ở faithfulness) |
| Cohen's κ | 0.0 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 31.19 ms |

---

## Nhận xét & Cải tiến

> **Điều hoạt động tốt:** Guard stack (Presidio + NeMo + keyword pre-filter) đạt 20/20 adversarial pass rate với P95 latency cực thấp (31ms), cho thấy việc kết hợp nhiều tầng bảo vệ (defense-in-depth) rất hiệu quả. Presidio phát hiện chính xác PII tiếng Việt (CCCD, SĐT) nhờ custom PatternRecognizer.
>
> **Điều cần cải thiện:** RAGAS scores rất thấp — đặc biệt answer_relevancy = 0.0 và faithfulness trung bình chỉ đạt 0.24. Nguyên nhân chính là RAG pipeline chưa tối ưu cho tiếng Việt (embedding model BGE-M3 encode chậm trên CPU, context retrieval chưa chính xác). Cohen's κ = 0.0 cho thấy LLM Judge chưa đủ phân biệt giữa các câu trả lời.
>
> **Nếu deploy production:** (1) Sử dụng GPU để tăng tốc encoding BGE-M3 và reranking, (2) Fine-tune hoặc thay thế embedding model phù hợp hơn cho tiếng Việt, (3) Tăng cường prompt engineering cho LLM Judge để cải thiện Cohen's κ, (4) Thêm rate limiting và logging chi tiết cho guard stack, (5) Triển khai Qdrant server riêng (Docker) thay vì in-memory để dữ liệu persistent qua các lần restart.
