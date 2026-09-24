## I. Suy ngẫm sau thực nghiệm (Part 16 — Reflection)

### 1. Prediction nào của em sai (hoặc chưa hoàn toàn đúng)?
- Lúc đầu em nghĩ rằng việc lọc bỏ stopwords (ở Pipeline B) sẽ luôn giúp kết quả tìm kiếm chính xác hơn và làm từ điển thu nhỏ đi đáng kể.
- Nhưng khi chạy thực tế em thấy:
  - Kích thước từ điển của Pipeline B hầu như không giảm mấy, chỉ bớt được khoảng hơn 300 từ so với Pipeline A (từ 193.837 từ xuống còn 193.521 từ). Lý do là vì danh sách stopwords tuy xuất hiện rất nhiều lần trong văn bản, nhưng số lượng từ loại của chúng lại rất ít.
  - Ngoài ra, với những câu truy vấn có các từ nối quan trọng quyết định ngữ cảnh (ví dụ *"flight to London"* hay *"Mac OS X"*), nếu máy xóa mất chữ "to" hoặc chữ "X" thì câu bị mất nghĩa, khiến kết quả tìm kiếm bị lệch đi.

### 2. Kết quả nào làm em bất ngờ nhất?
- Điều làm em bất ngờ nhất là **độ thưa (Sparsity) của ma trận lớn khủng khiếp**: đạt tới **99.91%** (Pipeline A) và **99.94%** (Pipeline B).
- Trước khi làm em cũng đoán là trên 99.9%, nhưng đến lúc in kết quả ra thấy trong hơn 5.7 tỷ ô của ma trận mà chỉ có vỏn vẹn từ 3.7 đến 5.1 triệu ô là có số, em mới thấy trực quan. Nếu lúc làm mà lỡ tay đổi sang mảng numpy thông thường (dense) thì máy tính chắc chắn bị sập vì tràn RAM ngay. Điều này giúp em hiểu vì sao trong NLP bắt buộc phải dùng thư viện ma trận thưa (`scipy.sparse`).

### 3. Thực nghiệm nào cung cấp bằng chứng rõ ràng nhất?
- Thực nghiệm 2 (so sánh 3 pipeline A, B, C) là phần cho thấy rõ ràng nhất.
- Nó chứng minh cho em thấy: tiền xử lý văn bản không chỉ đơn giản là bước "dọn rác cho sạch", mà nó ảnh hưởng trực tiếp đến kết quả của mô hình:
- Chỉ cần bỏ bớt stopwords (Pipeline B) là ma trận đã giảm đi hơn 1.35 triệu ô dữ liệu không cần thiết, giúp máy tính chạy nhẹ hơn và các từ khóa chính nổi bật hơn hẳn so với Pipeline A.

### 4. Failure case (trường hợp tìm kiếm sai) quan trọng nhất là gì?
- Điểm yếu lớn nhất em quan sát được là hệ thống chỉ biết so khớp từng chữ cái chứ hoàn toàn không hiểu nghĩa:
- Khi em thử tìm *"medical image classification"* (phân loại ảnh y tế), kết quả trả về lại có bài về phân loại bắp ngô và bài hướng dẫn chọn kích thước ảnh trên WordPress. Lý do là vì các bài này vô tình chứa chữ "image" và chữ "classification", máy cứ thấy trùng chữ là cộng điểm vào.
- Ngược lại, nếu tìm *"heart attack treatment"* mà tài liệu lại viết là *"myocardial infarction therapy"* thì hệ thống bỏ sót hoàn toàn vì không trùng chữ nào.

### 5. Nếu được làm lại hoặc cải tiến search engine, em sẽ thay đổi điều gì?
- Nếu được làm lại, em sẽ không dùng mỗi TF-IDF đơn thuần nữa mà sẽ cải tiến theo hướng:
  1. Sử dụng biểu diễn ngữ nghĩa dạng vector dày (Dense Retrieval như Sentence-BERT hoặc các mô hình Embedding) để máy hiểu được các từ đồng nghĩa và hiểu ngữ cảnh của cả câu.
  2. Kết hợp cả hai (Hybrid Search): dùng TF-IDF/BM25 để lọc nhanh các từ khóa hiếm hoặc tên riêng, sau đó dùng mô hình ngữ nghĩa để sắp xếp lại (rerank) các kết quả đứng đầu cho chính xác hơn.

### 6. Khai báo sử dụng AI (AI Usage):
- Trong bài lab này, em có sử dụng AI để hỏi cú pháp Python (như cách đọc file nén `.json.gz`, cách dùng hàm trong `scipy.sparse`) và kiểm tra lại công thức và các độ đo P@5, MRR.