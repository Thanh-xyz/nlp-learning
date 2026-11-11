# LAB 08 — DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PREDICTION BEFORE EXPERIMENT)
## Pretrained Transformers and BERT Fine-Tuning

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Sinh viên thực hiện**: Nguyễn Văn Thành  
**Mã bài tập**: LAB 08  
**Thời điểm lập và khóa dự đoán**: Trước khi thực thi bất kỳ pipeline huấn luyện mô hình nào  

---

## 1. Dự Đoán 1: Pretrained Transformer vs Transformer Train From Scratch

### Đặt vấn đề
Một mô hình Pretrained Transformer (như BERT) được fine-tune trên tập dữ liệu IMDb có khả năng đạt kết quả tốt hơn một Transformer nhỏ được huấn luyện từ đầu (train from scratch) trên cùng tập dữ liệu hay không?

### Bản dự đoán (Prediction)
**Có, Pretrained Transformer sẽ vượt trội hoàn toàn so với Transformer Train From Scratch cả về Accuracy, F1-Score lẫn tốc độ hội tụ.**

### Cơ sở lý luận khoa học (Reason)
1. **Kiến thức ngữ cảnh thu nhận từ Pretraining (Prior Knowledge)**:
   - Mô hình Pretrained Transformer đã được huấn luyện tự giám sát (self-supervised) trên hàng tỷ từ vựng (Wikipedia tiếng Anh và BookCorpus) thông qua mục tiêu Masked Language Modeling (MLM). Nhờ đó, các tầng Self-Attention và Feed-Forward đã nắm vững cú pháp tiếng Anh, cấu trúc phân cấp, đa nghĩa của từ vựng và sự phụ thuộc tầm xa.
   - Ngược lại, Transformer train from scratch xuất phát từ trọng số ngẫu nhiên. Mô hình này phải đồng thời học cách trích xuất đặc trưng ngữ nghĩa cơ bản của ngôn ngữ và ranh giới phân loại cảm xúc chỉ từ tập dữ liệu downstream nhỏ.
2. **Độ sâu biểu diễn và hiện tượng Overfitting**:
   - Transformer train from scratch trên tập dữ liệu nhỏ rất dễ rơi vào hiện tượng quá khớp (overfitting) với các hư từ hoặc mẫu câu cục bộ, không thể khái quát hóa tốt cho tập Test. Trong khi đó, các trọng số đã được tiền huấn luyện đóng vai trò như một điểm khởi đầu tối ưu (inductive bias mạnh mẽ), giúp mô hình downstream nhanh chóng hội tụ về nghiệm tối ưu toàn cục tốt.

### Tiêu chí kiểm chứng kỳ vọng (Expected Evidence)
- **Test Accuracy & F1-Score**: Pretrained Transformer đạt F1 cao hơn rõ rệt (dự kiến chênh lệch từ $+10\%$ đến $+20\%$ so với Transformer scratch).
- **Độ ổn định huấn luyện**: Pretrained Transformer đạt hiệu năng cao ngay từ epoch đầu tiên, trong khi Transformer scratch cần nhiều epoch và có độ biến thiên loss cao.

---

## 2. Dự Đoán 2: Feature Extraction (Frozen Encoder) vs Full Fine-Tuning

### Đặt vấn đề
Nếu đóng băng toàn bộ trọng số của Pretrained Encoder (`freeze_encoder = True`) và chỉ huấn luyện tầng phân loại (Linear Classification Head):
$$\text{BERT encoder (frozen)} \to \text{Linear Head (trainable)} \to \text{Predictions}$$
Hiệu năng phân loại (Accuracy / F1) sẽ:
- **A. Tốt hơn fine-tuning toàn bộ**
- **B. Tương đương**
- **C. Kém hơn**
- **D. Không thể dự đoán**

### Bản dự đoán (Prediction)
**Lựa chọn: C. Kém hơn (Worse than full fine-tuning)**

### Cơ sở lý luận khoa học (Reason)
1. **Sự khác biệt giữa biểu diễn tổng quát và biểu diễn chuyên biệt cho tác vụ (Domain/Task Adaptation)**:
   - Trọng số ban đầu của BERT được tối ưu hóa cho tác vụ Masked Language Modeling (khôi phục từ bị che) và Next Sentence Prediction trên văn bản bách khoa toàn thư. Biểu diễn của token `[CLS]` ở trạng thái frozen là biểu diễn trung tính, mang tính khái quát ngôn ngữ chung.
   - Khi đóng băng encoder, chỉ có một ma trận tuyến tính đơn giản $W \in \mathbb{R}^{d \times C}$ được tối ưu. Tầng tuyến tính này bị giới hạn trong việc phân chia siêu phẳng tuyến tính trên không gian biểu diễn cố định, không thể tái cấu trúc không gian đặc trưng để làm nổi bật các tín hiệu cảm xúc đặc thù (sentiment features) như tính từ phân cực mạnh, cấu trúc đảo ngữ hoặc phủ định tinh vi.
2. **Cơ chế tái cấu trúc biểu diễn qua Backpropagation**:
   - Trong chế độ Full Fine-Tuning, gradient lỗi từ hàm loss phân loại được truyền ngược qua toàn bộ các tầng Multi-Head Attention. Điều này cho phép các attention heads tinh chỉnh lại trọng số chú ý để tập trung đặc biệt vào các token mang sắc thái cảm xúc, giúp biểu diễn token `[CLS]` tại tầng đỉnh tách biệt rõ rệt hai lớp Positive và Negative.

### Tiêu chí kiểm chứng kỳ vọng (Expected Evidence)
- Test F1-Score của chế độ Fine-Tuning cao hơn chế độ Frozen khoảng $3\% - 8\%$.
- Tuy nhiên, chế độ Frozen Encoder sẽ có thời gian huấn luyện nhanh hơn đáng kể vì không cần tính và lưu trữ gradient cho hàng triệu tham số của Encoder.

---

## 3. Dự Đoán 3: Tác Động Của Khối Lượng Dữ Liệu Huấn Luyện (Data Efficiency)

### Đặt vấn đề
Khảo sát hiệu năng khi cắt giảm dữ liệu huấn luyện xuống các tỷ lệ:
$$10\% \to 25\% \to 50\% \to 100\%$$
(Giữ nguyên tập Validation và Test cố định).
Mô hình Pretrained hay mô hình Train From Scratch sẽ bị giảm hiệu năng mạnh hơn khi dữ liệu huấn luyện bị thu hẹp?

### Bản dự đoán (Prediction)
**Mô hình Train From Scratch sẽ bị sụt giảm hiệu năng nghiêm trọng và khốc liệt hơn rất nhiều so với Pretrained Transformer.**

### Cơ sở lý luận khoa học (Reason)
1. **Độ phức tạp mẫu (Sample Complexity) của kiến trúc Transformer**:
   - Transformer là kiến trúc không có sẵn inductive bias mạnh mẽ về tính cục bộ như CNN hay tính liên tục thời gian như RNN. Do đó, để học được ngữ pháp cơ bản và sự liên kết ngữ nghĩa từ đầu, Transformer đòi hỏi một lượng mẫu cực kỳ lớn.
   - Khi chỉ còn $10\%$ dữ liệu (vài trăm mẫu), số lượng mẫu huấn luyện quá ít so với số lượng tham số tự do, dẫn đến hiện tượng mô hình ghi nhớ máy móc (memorization) thay vì khái quát hóa. Hiệu năng của From-Scratch sẽ rơi thẳng về mức tương đương đoán ngẫu nhiên ($\approx 50\%$).
2. **Khả năng chuyển giao tri thức của Pretraining (Transfer Learning)**:
   - Pretrained model đã sở hữu sẵn nền tảng biểu diễn ngôn ngữ vững chắc. Việc học tác vụ downstream trên $10\%$ dữ liệu thực chất chỉ là quá trình hiệu chỉnh nhẹ góc quay của siêu phẳng phân loại (classification boundary) và tinh chỉnh các attention weights ở tầng cao.
   - Vì vậy, ngay cả ở mức $10\%$ dữ liệu, Pretrained Transformer vẫn giữ được mức hiệu năng rất cao, minh chứng cho tính chất Data-Efficiency vượt trội của mô hình tiền huấn luyện.

### Tiêu chí kiểm chứng kỳ vọng (Expected Evidence)
- Đường cong học tập Data-Efficiency ($10\% \to 100\%$):
  - Pretrained Transformer: Đường cong có độ dốc phẳng (gentle slope), đạt hiệu năng tốt ngay từ $10\%$ dữ liệu.
  - From-Scratch Transformer: Đường cong dốc đứng (steep drop), sụt giảm thê thảm tại mốc $10\%$ và $25\%$.

