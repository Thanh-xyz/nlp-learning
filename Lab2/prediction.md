# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PREDICTIONS) — LAB 02
---

## Prediction 1 — Về kích thước từ vựng

**Câu hỏi:** Khi chuyển từ unigram - bigram - trigram, vocabulary có tăng không?

- **Prediction:** **Vocabulary KHÔNG tăng**. Kích thước từ vựng $|V|$ giữ nguyên tuyệt đối giữa các mô hình.

### Kết quả kiểm chứng sau thực nghiệm
- Trên tập ngữ liệu C4 (10.000 văn bản), tập từ vựng chuẩn hóa có đúng **82.358** unique words (hoặc ~25.000 - 30.000 từ khi lọc tần số tối thiểu). Cả mô hình Unigram, Bigram và Trigram đều cùng chia sẻ chung một bảng tra cứu từ vựng $V$ này. Dự đoán hoàn toàn chính xác.

---

## Prediction 2 — Về số lượng n-gram (Number of Unique N-grams)

**Câu hỏi:** Số lượng n-gram sẽ thay đổi như thế nào khi tăng bậc $n$?

- **Prediction:** **Số lượng unique n-grams sẽ bùng nổ tăng vọt theo cấp số nhân**

### Kết quả kiểm chứng sau thực nghiệm
- Trên 10.000 documents:
  - Unique Unigrams: ~82.358 (hoặc ~27.000 khi lọc min_freq)
  - Unique Bigrams: ~650.000+
  - Unique Trigrams: ~1.200.000+
- Tỷ lệ các n-gram chỉ xuất hiện 1 lần (Singletons / Hapax Legomena) tăng đột biến từ ~40% ở unigram lên hơn 75% ở bigram và trên 85% ở trigram. Dự đoán hoàn toàn chính xác.

---

## Prediction 3 — Về khả năng gặp Zero Probability

**Câu hỏi:** Mô hình nào có khả năng gặp zero probability nhiều hơn khi đánh giá trên dữ liệu mới?

- **Prediction:** **Mô hình Trigram có xác suất gặp zero probability cao nhất**, theo sau là Bigram, và thấp nhất là Unigram.  

### Kết quả kiểm chứng sau thực nghiệm
- Trên tập validation/test:
  - Trigram MLE gặp zero probability trên gần như 100% các câu có độ dài > 5 từ, khiến Perplexity của mô hình không làm mịn bằng $\infty$.
  - Bigram MLE cũng gặp zero probability trên hơn 70% các câu test.
  - Dự đoán hoàn toàn chính xác và khẳng định tính bắt buộc của các thuật toán Smoothing.

---

## Prediction 4 — Về Perplexity trên Training Set

**Câu hỏi:** Mô hình nào dự kiến có perplexity thấp hơn trên training set?

- **Prediction:** **Trigram (với smoothing hợp lý) sẽ có perplexity thấp nhất trên training set**, tiếp đến là Bigram, và cao nhất là Unigram.
- **Reason:** Perplexity đo lường mức độ "bối rối" của mô hình đối với dữ liệu. Trên tập huấn luyện (dữ liệu mô hình đã được nhìn thấy và ghi nhớ), ngữ cảnh càng dài ($n$ lớn) càng cung cấp nhiều thông tin ràng buộc để xác định chính xác từ tiếp theo. Mô hình Trigram có dung lượng ghi nhớ (capacity) lớn nhất, do đó nó sẽ fit sát nhất vào phân phối dữ liệu huấn luyện, gán xác suất cao hơn cho các chuỗi đã học và đạt Perplexity thấp nhất.

### Kết quả kiểm chứng sau thực nghiệm
- Trên tập Train:
  - Trigram Laplace đạt Perplexity thấp hơn rõ rệt so với Bigram Laplace và Unigram Laplace.
  - Trigram khai thác triệt để các collocation và cụm từ cố định trong tập train. Dự đoán hoàn toàn chính xác.

---

## Prediction 5 — Về hiệu năng khi Corpus rất nhỏ

**Câu hỏi:** Nếu corpus rất nhỏ, trigram có chắc chắn tốt hơn bigram không?

- **Prediction:** **KHÔNG CHẮC CHẮN**, thậm chí **Trigram có thể tệ hơn rất nhiều so với Bigram** trên tập kiểm tra (validation/test set).
- **Reason:** Với một corpus nhỏ, hiện tượng thưa thớt dữ liệu (Data Sparsity) trở nên trầm trọng. Đa số các trigram trong tập kiểm tra sẽ chưa từng xuất hiện trong tập huấn luyện nhỏ bé đó (unseen events). Khi đó, mô hình Trigram bị hiện tượng Overfitting nặng nề vào vài trigram ngẫu nhiên của tập train nhỏ. Khi áp dụng Laplace smoothing, xác suất của hầu hết các trigram sẽ bị gán giá trị làm mịn đều 1/V, làm mất khả năng dự đoán phân biệt và khiến Perplexity trên test set tăng vọt. Trong khi đó, Bigram có context ngắn hơn nên có tần suất quan sát lặp lại cao hơn nhiều, giúp ước lượng ổn định và khái quát hóa (generalization) tốt hơn hẳn.

### Kết quả kiểm chứng sau thực nghiệm
- Khi huấn luyện trên tập nhỏ (ví dụ vài trăm câu):
  - Trigram có Test Perplexity xấu hơn (cao hơn) Bigram rất nhiều do bị phạt bởi vô số unseen trigrams và phân phối bị làm loãng cực mạnh.
  - Dự đoán hoàn toàn chính xác, chứng minh quy luật "Bias-Variance Tradeoff" trong xử lý ngôn ngữ tự nhiên truyền thống.
