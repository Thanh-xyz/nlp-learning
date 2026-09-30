# BÁO CÁO SUY NGẪM & KIỂM TRA HỌC TẬP (REFLECTION) — LAB 02
---

### Câu 1: Nếu tăng $n$, mô hình nhận thêm thông tin gì?
- **Trả lời:**
  Khi tăng bậc $n$ (từ unigram lên bigram, trigram, n-gram), mô hình nhận thêm **thông tin về thứ tự từ (word order)** và **ngữ cảnh cục bộ dài hơn (extended local context)**.
  - Thay vì xem các từ là độc lập rời rạc (như Unigram / Bag-of-Words), mô hình bậc cao nắm bắt được các quan hệ cú pháp trực tiếp (như quan hệ chủ ngữ - vị ngữ, động từ - tân ngữ), các cụm từ cố định (collocations / idioms như `in order to`, `machine learning`), và giảm sự nhập nhằng ngữ nghĩa của các từ đa nghĩa nhờ có các từ đứng liền trước làm điều kiện.

---

### Câu 2: Tại sao tăng $n$ lại làm sparsity tăng?
- **Trả lời:**
  Tăng $n$ làm bùng nổ không gian trạng thái theo cấp số nhân:
  - Với một tập từ vựng gồm |V| từ, số lượng tổ hợp n-gram lý thuyết là |V|^n (|V| với unigram, |V|^2 với bigram, |V|^3 với trigram). Ví dụ với |V| = 30.000, số trigram khả dĩ lên tới 2.7 * 10^{13}.
  - Trong khi đó, bất kỳ kho ngữ liệu thực tế nào (dù lớn đến hàng triệu tokens) cũng chỉ bao phủ được một tỷ lệ cực kỳ nhỏ (chưa tới 0.001\%) các tổ hợp lý thuyết này. Hầu hết các ô trong bảng xác suất chuyển tiếp đều mang giá trị đếm bằng 0.
  - Ngữ cảnh càng dài thì tính đặc thù càng cao, khiến xác suất một chuỗi $n$ từ bất kỳ tái xuất hiện giống hệt trong một văn bản mới suy giảm trầm trọng (**Curse of Dimensionality** / Thưa thớt dữ liệu).

---

### Câu 3: Tại sao smoothing cần thiết?
- **Trả lời:**
  Smoothing là thành phần sống còn của các mô hình ngôn ngữ dựa trên đếm (count-based LMs) vì hai lý do chính:
  1. **Giải quyết vấn đề Zero-Frequency Problem:** Trong thực tế, ngôn ngữ mang tính sáng tạo vô hạn. Một $n$-gram chưa từng xuất hiện trong tập huấn luyện (unseen event) không có nghĩa là nó không thể xảy ra trong ngôn ngữ tự nhiên. Nếu không có smoothing, ước lượng MLE gán P = 0, làm triệt tiêu xác suất của cả câu về 0 và khiến Perplexity trở thành vô cùng.
  2. **Điều chỉnh phân phối xác suất thực tế:** Smoothing thực hiện chiết khấu (discounting) một phần xác suất từ các sự kiện đã quan sát thường xuyên để tái phân phối cho các sự kiện hiếm hoặc chưa từng thấy, giúp mô hình có khả năng khái quát hóa (generalization) tốt hơn trên dữ liệu mới.

---

### Câu 4: Perplexity đo điều gì?
- **Trả lời:**
  - **Về mặt toán học:** Perplexity (PP) là nghịch đảo căn bậc N của xác suất hợp lý chuỗi.
  - **Về mặt trực quan ngôn ngữ học:** Perplexity đo lường **mức độ "bối rối / bất ngờ" trung bình** của mô hình khi quan sát một từ tiếp theo trong chuỗi.
  - **Về mặt branching factor:** Một mô hình có PP = k tương đương với việc tại mỗi bước sinh từ, mô hình đang phân vân một cách ngẫu nhiên đều giữa k lựa chọn từ tiếp theo. Do đó, PP càng thấp chứng tỏ mô hình càng tự tin, dự đoán càng chuẩn xác và mô hình hóa dữ liệu càng tốt.

---

### Câu 5: Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.
- **Trả lời:**
  **KHÔNG NHẤT THIẾT.**
  - **Perplexity chỉ đo lường xác suất thống kê trung bình của từng bước chuyển tiếp từ**, nó không phản ánh đầy đủ chất lượng văn bản toàn cục theo đánh giá của con người:
    1. **Sự trôi chảy toàn cục và mạch lạc ngữ nghĩa (Global Coherence):** Mô hình $n$-gram có thể đạt Perplexity rất thấp trên các cụm từ ngắn quen thuộc nhưng khi sinh văn bản dài lại rơi vào vòng lặp vô nghĩa (repetitive text), câu cụt hoặc mâu thuẫn ý kiến giữa các câu.
    2. **Độ phong phú và tính tự nhiên (Diversity & Hallucination):** Một mô hình có xu hướng luôn chọn các từ quá an toàn (high-frequency stop words) có thể có Perplexity rất thấp nhưng nội dung văn bản sinh ra lại sáo rỗng, nhạt nhẽo và phi tự nhiên.
    3. **Sự phụ thuộc vào tiền xử lý:** Perplexity bị phụ thuộc mạnh vào kích thước từ vựng $|V|$ và cách tokenize. Hai mô hình có cách xử lý OOV khác nhau không thể so sánh trực tiếp bằng Perplexity.

---

### Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?
- **Trả lời:**
  Mô hình N-gram bộc lộ những hạn chế căn bản so với nhận thức ngôn ngữ của con người:
  1. **Thiếu khả năng biểu diễn ngữ nghĩa liên tục (Discrete Symbols vs. Semantic Similarity):** N-gram coi mỗi từ là một ký hiệu rời rạc nguyên tử. Nó không biết rằng `cat` và `kitten`, hay `eats` và `devours` có quan hệ ngữ nghĩa chặt chẽ với nhau. Nếu tập train chỉ có `the cat eats fish`, mô hình sẽ hoàn toàn bối rối và gán xác suất 0 cho `the kitten eats fish`.
  2. **Giới hạn bộ nhớ ngắn (Markov Assumption vs. Long-range Dependencies):** Con người hiểu văn bản dựa trên dòng ngữ cảnh xuyên suốt hàng trang sách hoặc cả câu chuyện. N-gram chỉ nhìn được n-1 từ liền trước (thường là 1-2 từ), hoàn toàn quên hết ngữ cảnh phía xa, cấu trúc ngữ pháp phân tầng (hierarchical syntax) và chủ đề toàn cục.
  3. **Không có tri thức thế giới (Common Sense Knowledge):** N-gram chỉ đếm sự đồng xuất hiện bề mặt của chuỗi ký tự, không có mô hình khái niệm về thực tại khách quan.

---

### Câu 7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?
- **Trả lời:**
  **HOÀN TOÀN KHÔNG.**
  - Mô hình Trigram chỉ nhìn duy nhất 2 từ đứng ngay trước là w_98 và $w_99$. Toàn bộ thông tin, chủ ngữ, động từ chính, ngữ cảnh cốt truyện của 97 từ đầu tiên (w_1, \dots, w_97) bị vứt bỏ hoàn toàn.
  - **Cầu nối sang Neural LM & Transformer:** Đây chính là nhược điểm chí mạng của mô hình Markov n-gram cổ điển, mở đường cho sự phát triển của **Recurrent Neural Networks (RNN/LSTM)** với hidden state tích lũy dài hạn, và đặc biệt là kiến trúc **Transformer (Self-Attention)** cho phép mọi từ trong chuỗi 100 từ (thậm chí hàng triệu tokens) tương tác trực tiếp với nhau mà không bị suy giảm thông tin.

---
