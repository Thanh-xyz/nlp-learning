# QUY CHUẨN BỐN TÀI LIỆU KHOA HỌC & LATEX

Trong các bài tập phân tích, Machine Learning và NLP, bốn tài liệu sau đóng vai trò cốt lõi trong việc đánh giá mức độ thấu hiểu của người học và tính khoa học của bài làm:

1. `calculations.md` (Tính toán chi tiết bằng tay)
2. `prediction.md` (Dự đoán trước thực nghiệm)
3. `error_analysis.md` (Phân tích lỗi thực nghiệm)
4. `reflection.md` (Chiêm nghiệm và bài học)

Mọi tài liệu phải được viết bằng định dạng **Markdown có công thức LaTeX hợp lệ** và sau đó được chuyển đổi sang định dạng **PDF** tương ứng (`.pdf`) mà **không được xóa hoặc thay thế file `.md` gốc**.

---

## 1. `calculations.md` — Bài Tập Tính Toán Bằng Tay

### 1.1. Mục đích
Chứng minh năng lực hiểu sâu cơ chế hoạt động của thuật toán ở cấp độ đại số tuyến tính và giải tích, không phụ thuộc vào việc "chạy hàm đen" (black-box API) của thư viện.

### 1.2. Cấu trúc bắt buộc cho từng bài tính toán
Mỗi bài tính toán phải trình bày đầy đủ 6 phần:
1. **Dữ kiện đầu vào / Giả thiết**: Liệt kê rõ các vector, ma trận, kích thước, bias và điều kiện ban đầu.
2. **Công thức toán học**: Sử dụng ký hiệu chuẩn LaTeX.
3. **Thay số cụ thể**: Hiển thị chính xác các giá trị số được đưa vào công thức.
4. **Các bước biến đổi trung gian**: Trình bày từng phép nhân ma trận - vector, phép cộng bias và giá trị trước khi qua hàm phi tuyến (pre-activation). Tuyệt đối **không được nhảy bước** hoặc chỉ ghi đáp số.
5. **Kết quả cuối cùng**: Giá trị số được làm tròn hợp lý (thường từ 4 đến 5 chữ số thập phân) kèm đơn vị nếu có.
6. **Kiểm tra tính hợp lý (Sanity Check)**: Kiểm tra miền giá trị (ví dụ: hàm $\tanh \in (-1, 1)$, hàm $\sigma \in (0, 1)$, xác suất $\sum = 1$) và đối chiếu với code Python độc lập (tuyệt đối không dùng code để che giấu việc thiếu lời giải trung gian).

### 1.3. Chuẩn ký hiệu LaTeX
- Vector cột: $x_t = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$
- Ma trận: $W_{xh} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}$
- Phép nhân từng phần tử (Hadamard product): $\odot$ (`\odot`)
- Hàm phi tuyến: $\tanh$, $\sigma$, $\text{softmax}$.

---

## 2. `prediction.md` — Dự Đoán Trước Thực Nghiệm

### 2.1. Mục đích
Rèn luyện tư duy khoa học: Đưa ra giả thuyết và dự đoán dựa trên nguyên lý kiến trúc mô hình trước khi có kết quả thực nghiệm.

### 2.2. Quy tắc thời gian (Timeline Rule)
- **Bắt buộc hoàn thành và khóa nội dung TRƯỚC KHI chạy mô hình huấn luyện**.
- Tuyệt đối không được xem kết quả chạy thực nghiệm rồi mới quay lại viết prediction để làm cho dự đoán "luôn luôn đúng".
- Sau khi có kết quả thực nghiệm, phần đối chiếu (Post-experiment Comparison) chỉ được viết bổ sung ở một mục riêng biệt, phân định rõ ràng giữa "Dự đoán ban đầu" và "Thực tế quan sát".

### 2.3. Cấu trúc chuẩn của mỗi mục dự đoán
```markdown
### Dự đoán [Số thứ tự]: [Tiêu đề câu hỏi / Giả thuyết]

- **Câu hỏi / Bài toán so sánh**: ...
- **Prediction**: Đưa ra nhận định cụ thể (Model A tốt hơn Model B ở khía cạnh nào, xu hướng metric thay đổi ra sao).
- **Reason (Lý giải khoa học)**: Phải dựa trên cơ chế cấu tạo của mô hình (ví dụ: cơ chế cổng forget/input gate của LSTM, gradient highway, số lượng tham số, mức độ nén context). Tuyệt đối không chấp nhận lý do mơ hồ như "vì mô hình này hiện đại hơn".
- **Confidence**: Đánh giá độ tự tin (Cao / Trung bình / Thấp) kèm lý do.
- **Tiêu chí kiểm chứng (Verification Criteria)**: Metric hoặc biểu đồ nào trong thực nghiệm sẽ được dùng để xác nhận hoặc bác bỏ dự đoán này.
```

### 2.4. Lưu ý về chính sách AI và suy luận cá nhân
- Đối với các câu hỏi đòi hỏi suy luận và trực giác của sinh viên, agent đóng vai trò gợi ý khung phân tích khoa học (draft có căn cứ), sau đó đánh dấu rõ ràng `[Cần sinh viên xác nhận/chỉnh sửa]` để đảm bảo tính liêm chính học thuật.

---

## 3. `error_analysis.md` — Phân Tích Lỗi Thực Nghiệm

### 3.1. Mục đích
Hiểu rõ biên giới năng lực của mô hình, xác định mô hình thất bại ở đâu, tại sao thất bại và các nguyên nhân đến từ dữ liệu, biểu diễn hay kiến trúc.

### 3.2. Yêu cầu về mẫu phân tích
- Phải trích xuất từ **kết quả dự đoán thực tế của mô hình trên tập Test/Validation thật**.
- Tuyệt đối không tự bịa ra các câu văn mẫu giả định rồi gán nhãn là lỗi của mô hình.
- Phải đảm bảo số lượng tối thiểu theo yêu cầu của đề bài (ví dụ: tối thiểu 3 mẫu dự đoán đúng và 3 mẫu dự đoán sai).

### 3.3. Bảng tổng hợp mẫu kiểm tra
```markdown
| STT | Câu văn (Text snippet) | True Label | Predicted Label | Confidence / Prob | Phân loại lỗi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | "The movie was not bad at all..." | Positive | Negative | 0.78 | Negation scope |
...
```

### 3.4. Hệ thống phân loại nguyên nhân lỗi (Error Taxonomy)
Khi phân tích một mẫu dự đoán sai, bắt buộc phải xem xét các góc độ:
1. **Phủ định (Negation)**: Phủ định kép, từ phủ định đứng xa tính từ ("not entirely good").
2. **Phụ thuộc tầm xa (Long-range Dependency)**: Thông tin chốt nằm ở đầu hoặc cuối câu dài bị trôi mất qua các bước recurrent.
3. **Mỉa mai / Châm biếm (Sarcasm & Irony)**: Dùng từ ngữ tích cực nhưng ngữ cảnh mang tính tiêu cực.
4. **Cảm xúc mơ hồ / Tương phản (Ambiguous / Contrastive)**: Câu có mệnh đề "nhưng", "mặc dù" ("I thought it would be bad, but it was great").
5. **Từ hiếm / Ngoài từ điển (Rare words / OOV)**: Bị ánh xạ vào token `<UNK>`.
6. **Cắt cụt sequence (Truncation)**: Thông tin quyết định nằm ngoài ngưỡng `max_length`.
7. **Nhiễu nhãn (Label Noise)**: Nhãn gốc trong tập dữ liệu bị đánh sai hoặc chủ quan.

**Câu hỏi bắt buộc phải trả lời**: Lỗi này có thực sự do cơ chế sequence modeling gây ra hay do hạn chế của dữ liệu / preprocessing?

---

## 4. `reflection.md` — Chiêm Nghiệm & Bài Học

### 4.1. Mục đích
Tổng kết toàn bộ chu trình học tập, đối chiếu lý thuyết với thực tế, nhận diện các bài toán đánh đổi (trade-offs) và định vị kiến thức trong lộ trình môn học.

### 4.2. Nội dung bắt buộc
1. **Kiến thức đã được kiểm chứng bằng thực nghiệm**: Những định lý, cơ chế lý thuyết nào đã được xác nhận qua số liệu thực tế?
2. **Những điểm bất ngờ hoặc mâu thuẫn với dự đoán ban đầu**: Tại sao thực nghiệm lại cho kết quả khác với trực giác ban đầu?
3. **Các bài toán đánh đổi (Trade-offs)**:
   - Đánh đổi giữa Context length và Chi phí tính toán / Bộ nhớ / Tốc độ.
   - Đánh đổi giữa Độ phức tạp của mô hình (RNN vs LSTM vs GRU) và Nguy cơ overfitting.
   - So sánh giữa mô hình chuỗi (Sequential) và mô hình tĩnh (TF-IDF / Word2Vec).
4. **Giới hạn của phương pháp thực nghiệm**: Những giả thiết nào chưa được kiểm chứng đầy đủ do giới hạn tài nguyên (GPU, thời gian, kích thước tập dữ liệu)?
5. **Mối liên hệ tới bước phát triển tiếp theo**: Tại sao các mô hình hồi quy (RNN/LSTM/GRU) bộc lộ giới hạn xử lý tuần tự tuần tự, từ đó tạo tiền đề cho sự ra đời của cơ chế **Attention** và **Transformer**?
