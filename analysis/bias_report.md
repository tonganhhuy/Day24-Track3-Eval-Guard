# LLM Judge Bias Report — Phase B

**Sinh viên:** Lab 24 Student  
**Ngày:** 2026-06-30  
**Judge model:** gemini-3.1-flash-lite

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên demo question)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | tie | Cả hai câu trả lời đều đưa ra số liệu cụ thể nhưng khác nhau (15 vs 12 ngày), judge không phân biệt được đáp án nào chính xác hơn |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng demo)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | tie | tie | tie | Yes |

**Position bias rate:** 0% (= 0 case NOT consistent / 1 tổng)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** Kết quả từ judge demo — tất cả đều tie (score 0.0)

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 0 | No |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 0 | No |
| 21 | 0 | 0 | Yes |
| 23 | 1 | 0 | No |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 0 | No |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 0 | No |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 0.0  
**Interpretation:** poor — Judge không đủ tin cậy, cho tất cả các cặp đều là tie.

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 0 / 0 cases
- B thắng + B dài hơn A: 0 / 0 cases  
- **Verbosity bias rate:** 0.0%

**Kết luận:** Không có verbosity bias vì judge trả về "tie" cho tất cả các cases. Tuy nhiên, điều này cũng có nghĩa judge không có khả năng phân biệt chất lượng câu trả lời, nên kết luận "không có verbosity bias" không có ý nghĩa thống kê.

---

## 5. Nhận xét chung

> **Cohen's κ = 0.0 — Judge chưa đáng tin cậy.** Judge model (gemini-3.1-flash-lite) trả về "tie" cho hầu hết mọi cặp, cho thấy model không có đủ khả năng phân biệt chất lượng câu trả lời trong domain HR policy tiếng Việt. Điều này có thể do model quá nhẹ (flash-lite) để thực hiện reasoning so sánh phức tạp.
>
> **Position bias = 0% — nhưng vô nghĩa.** Vì tất cả kết quả đều là "tie", position bias rate thấp không phản ánh chất lượng judge thực sự. Swap-and-average technique hoạt động đúng đắn nhưng không giúp ích khi judge luôn trả về tie.
>
> **Khuyến nghị cho production:** (1) Sử dụng model mạnh hơn làm judge (GPT-4o hoặc Claude Sonnet) để có reasoning quality tốt hơn, (2) Cung cấp rubric chấm điểm chi tiết hơn trong prompt, (3) Chạy judge trên nhiều cặp answers hơn (tối thiểu 50 cặp) để có thống kê κ có ý nghĩa, (4) Kết hợp LLM judge với human evaluation trong giai đoạn đầu để calibrate.
