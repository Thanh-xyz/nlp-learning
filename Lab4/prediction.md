# LAB 04 — Dự đoán trước thực nghiệm (Pre-experiment Predictions)
---

## Các dự đoán chi tiết trước thực nghiệm 

---

### Prediction 1: TF-IDF + Logistic Regression có khả năng hoạt động tốt không?

* **Prediction:**
  **Có, hoạt động rất tốt.** Mô hình Baseline này dự kiến sẽ đạt độ chính xác (Accuracy) và F1-score rất cao, dao động trong khoảng **88% – 90%** trên tập test của IMDb.
---

### Prediction 2: Average Word2Vec có chắc chắn tốt hơn TF-IDF không?

* **Prediction:**
  **Hoàn toàn KHÔNG chắc chắn tốt hơn, thậm chí nhiều khả năng sẽ KÉM HƠN TF-IDF.**
---

### Prediction 3: Nếu tăng vocabulary size thì accuracy chắc chắn tăng không?

* **Prediction:**
  **KHÔNG chắc chắn tăng.** Khi kích thước từ điển vượt qua một ngưỡng tối ưu, accuracy sẽ đạt trạng thái bão hòa (plateau) hoặc thậm chí bị **suy giảm nhẹ** do overfitting.

---

### Prediction 4: Nếu bỏ stopwords thì performance chắc chắn tăng không?

* **Prediction:**
  **KHÔNG chắc chắn tăng, thậm chí có thể làm GIẢM performance nếu loại bỏ danh sách stopword tiêu chuẩn một cách máy móc.**

---

### Prediction 5: Accuracy và F1 có thể cho kết luận khác nhau không?

* **Prediction:**
  **HOÀN TOÀN CÓ THỂ cho kết luận trái ngược nhau**, đặc biệt rõ rệt khi tập dữ liệu bị mất cân bằng lớp (Class Imbalance) hoặc khi các mô hình có sự chênh lệch lớn giữa Precision và Recall.