# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PREDICTIONS) — LAB 03

---

## Prediction 1 — Về các cặp từ tương đồng nhất

### Câu hỏi
Trong tập các từ: `doctor`, `physician`, `hospital`, `banana`, `car`, những từ nào sẽ có độ tương đồng (similarity) gần nhau nhất trong không gian embedding?

### Dự đoán (Prediction)
- **Cặp gần nhau nhất:** Cặp từ `(doctor, physician)` sẽ có độ tương đồng cao nhất trong không gian vector (cosine similarity cao nhất, dự kiến $> 0.75$).

---

## Prediction 2 — Về ảnh hưởng của kích thước Context Window (2 - 5)

### Câu hỏi
Nếu context window tăng từ 2 - 5, độ tương đồng (similarity) giữa các từ có thay đổi không? Sự thay đổi diễn ra theo chiều hướng nào?

### Dự đoán (Prediction)
- **Độ tương đồng CHẮC CHẮN SẼ THAY ĐỔI rõ rệt.**
- Cụ thể:
  - **Với Window nhỏ (2):** Mô hình thiên về học **quan hệ hoán vị cú pháp / đồng nghĩa chức năng (Paradigmatic / Syntactic similarity)**. Cặp từ đồng nghĩa chặt chẽ cùng từ loại như `doctor - physician` hoặc các cặp cùng lớp như `cat - dog` sẽ đạt điểm similarity tương đối cao hơn so với quan hệ chủ đề.
  - **Với Window lớn (5):** Mô hình thiên về học **quan hệ liên tưởng ngữ nghĩa / chủ đề toàn cục (Syntagmatic / Topical / Associative relatedness)**. Similarity giữa các từ liên quan theo chủ đề như `doctor - hospital`, `doctor - patient`, `doctor - disease` sẽ tăng vọt và có xu hướng tiệm cận hoặc thậm chí vượt qua cặp từ đồng nghĩa hẹp.

---

## Prediction 3 — Về ảnh hưởng của kích thước Vector Dimension (50 - 100 - 300)

### Câu hỏi
Nếu embedding dimension tăng dần từ 50 - 100 - 300, chất lượng của mô hình (độ chính xác của similarity và analogy) có chắc chắn luôn luôn tăng theo không?

### Dự đoán (Prediction)
- **KHÔNG CHẮC CHẮN LUÔN TĂNG.**
- Chất lượng mô hình sẽ tuân theo dạng **đường cong chữ U ngược (Inverted U-curve)**:
  - Tăng từ 50 - 100: Hiệu năng mô hình (biểu diễn ngữ nghĩa và analogy) có thể cải thiện rõ rệt do không gian vector có đủ dung lượng (capacity) để phân tách các chiều ngữ nghĩa độc lập.
  - Tăng từ 100 - 300: Nếu tập dữ liệu huấn luyện có quy mô vừa hoặc nhỏ (như tập C4 con từ 10.000 - 30.000 câu), chất lượng có thể **chững lại (plateau) hoặc thậm chí suy giảm (degrade)** do hiện tượng quá khớp (overfitting) và thưa thớt tham số (parameter sparsity).

---

## Prediction 4 — Về chất lượng Similarity khi Corpus chỉ có 100 câu

### Câu hỏi
Từ `doctor` và `physician` có chắc chắn gần nhau trong không gian vector không nếu corpus huấn luyện chỉ bao gồm 100 câu văn bản?

### Dự đoán (Prediction)
- **HOÀN TOÀN KHÔNG CHẮC CHẮN**, thậm chí **gần như chắc chắn chúng SẼ KHÔNG GẦN NHAU** (hoặc bị loại khỏi từ vựng do không đủ tần suất tối thiểu `min_count`).