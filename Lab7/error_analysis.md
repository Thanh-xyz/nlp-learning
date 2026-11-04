# BÁO CÁO PHÂN TÍCH LỖI THỰC NGHIỆM — TRANSFORMER ARCHITECTURE

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Chủ đề**: From Attention to Transformer  
**Sinh viên thực hiện**: Nguyễn Văn Thành  
**Mô hình phân tích**: Transformer Classifier ($d_{model}=128, n_{heads}=4, d_{ff}=512$, Mean Pooling)  

---

## 1. Phương Pháp và Bối Cảnh Phân Tích
Phân tích lỗi (Error Analysis) được thực hiện trên $1,500$ mẫu văn bản của tập kiểm thử độc lập (Test Set) thuộc bộ dữ liệu chuẩn IMDb Movie Reviews. Mô hình Transformer đạt độ chính xác $72.40\%$ và điểm F1 $0.7409$.

Để xác định biên giới năng lực của kiến trúc Self-Attention, chúng tôi trích xuất $6$ trường hợp tiêu biểu thực tế từ kết quả dự đoán của mô hình (bao gồm $3$ ca dự đoán đúng và $3$ ca dự đoán sai), phân loại theo hệ thống nguyên nhân (Error Taxonomy), đối chiếu với phân bố trọng số Attention và thảo luận về bản chất diễn giải của mô hình.

---

## 2. Bảng Tổng Hợp 6 Mẫu Kiểm Tra Thực Tế

| STT | Trích đoạn văn bản (Text snippet) | True Label | Predicted Label | Confidence / Prob | Phân loại lỗi / Hiện tượng |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **C1** | *"let me set the scene. it is the school holidays and there is absolutely nothing at the movies... we look for a good fun film and found it..."* | **Positive** | **Positive** | 0.8252 | Thành công: Nhận diện mạch ngữ cảnh tích cực tường minh. |
| **C2** | *"i can watch b,b and c and feel all the emotions i felt when i first saw it at aged 18... miss k.novak has lost none of her charm and beauty..."* | **Positive** | **Positive** | 0.5576 | Thành công: Tích hợp từ ngữ cảm xúc mang tính hoài niệm. |
| **C3** | *"the tortuous emotional impact is degrading, whether adult or adolescent the personal values shown in this movie belong in a bad psychodrama..."* | **Negative** | **Negative** | 0.0925 | Thành công: Từ ngữ tiêu cực mạnh mẽ ("tortuous", "degrading", "bad"). |
| **E1** | *"i've seen my share of woody allen's movies, and while they're not always great, you can usually be sure you're going to be entertained. probably the lowest point..."* | **Negative** | **Positive** | 0.8769 | Mệnh đề tương phản (Contrast) và Cắt cụt (Truncation). |
| **E2** | *"paul muni and bette davis overact monstrously while lacklustre studio hack archie mayo seems distracted and oblivious in this racially provocative film..."* | **Negative** | **Positive** | 0.5072 | Từ vựng hiếm (Rare words) và Tên riêng diễn viên lớn. |
| **E3** | *"rebecca de mornay can be a fascinating beautiful actress but as for the parts she's given to play, if you cannot say something nice... de mornay portrays..."* | **Negative** | **Positive** | 0.8907 | Mỉa mai ngầm (Polite Sarcasm) và Ngữ cảnh văn hóa. |

---

## 3. Phân Tích Chi Tiết Từng Ca Dự Đoán Sai

### Ca 1 (Mẫu E1): Thất bại do Mệnh đề tương phản (Contrast) và Cắt cụt (Truncation)
- **Văn bản**: Người viết mở đầu bằng việc khen ngợi đạo diễn Woody Allen: *"while they're not always great, you can usually be sure you're going to be entertained"*. Phần chê bai kịch liệt thực sự nằm ở các câu tiếp theo: *"probably the lowest point of his career..."*.
- **Cơ chế lỗi**: 
  1. Do giới hạn chiều dài chuỗi $max\_length = 100$, phần lớn các câu chê gay gắt ở cuối bài đánh giá đã bị cắt bỏ (`truncated`).
  2. Trong $100$ token đầu tiên, các từ mang cực tính dương như *"great"*, *"entertained"*, *"sure"* chiếm ưu thế áp đảo.
  3. Mô hình Transformer mặc dù cho phép các token tương tác trực tiếp nhưng khi Mean Pooling được áp dụng trên chuỗi bị cắt cụt, biểu diễn tổng thể bị thiên lệch hoàn toàn về phía tích cực ($P(\text{Positive}) = 0.8769$).

### Ca 2 (Mẫu E2): Thất bại do Từ vựng hiếm (Rare words) và Trọng số thực thể
- **Văn bản**: Bài đánh giá sử dụng các từ ngữ chỉ trích cao cấp: *"overact monstrously"*, *"lacklustre"*, *"studio hack"*, *"distracted and oblivious"*.
- **Cơ chế lỗi**:
  1. Các cụm từ phê bình sâu sắc như *"lacklustre"*, *"monstrously"* có tần suất thấp trong tập huấn luyện $10,000$ mẫu, do đó bị ánh xạ vào token `<UNK>` hoặc có vector embedding chưa hội tụ tối ưu.
  2. Ngược lại, tên của các diễn viên kinh điển danh tiếng (*"paul muni"*, *"bette davis"*) xuất hiện nhiều trong các bộ phim kinh điển được đánh giá cao, mang embedding có xu hướng dương tính nhẹ.
  3. Mô hình rơi vào trạng thái lưỡng lự ở ngưỡng ranh giới với xác suất $0.5072$ và gán nhãn sai thành Positive.

### Ca 3 (Mẫu E3): Thất bại do Mỉa mai ngầm (Irony / Polite Sarcasm)
- **Văn bản**: *"rebecca de mornay can be a fascinating beautiful actress but as for the parts she's given to play, if you cannot say something nice..."*. Người viết khéo léo dùng ngạn ngữ: *"Nếu không thể nói điều gì tốt đẹp thì đừng nói gì cả"*, ám chỉ bộ phim quá tệ.
- **Cơ chế lỗi**:
  1. Mô hình không có tri thức thế giới (commonsense knowledge) để hiểu hàm ý của câu ngạn ngữ bị bỏ lửng.
  2. Các từ có trọng số attention cao nhất lại là *"fascinating"*, *"beautiful"*, *"nice"*. Sự kết hợp giữa các tính từ khen ngợi bề mặt khiến toàn bộ các head tập trung vào các từ này, dẫn tới kết luận sai lệch nghiêm trọng ($P(\text{Positive}) = 0.8907$).

---

## 4. Phân Tích Động Học Trọng Số Chú Ý (Attention Behavior)
Khi quan sát ma trận trọng số $A \in \mathbb{R}^{h \times T \times T}$ qua $4$ heads ở Experiment 3 (`attention_heads.png`), chúng tôi ghi nhận:
1. **Head 1 (Local Syntax)**: Tập trung chủ yếu vào đường chéo phụ ($|i - j| \le 2$), tức là liên kết chặt chẽ giữa các từ đứng cạnh nhau (ví dụ *"this" $\rightarrow$ "movie"*, *"would" $\rightarrow$ "be"*).
2. **Head 2 (Sentiment Focus)**: Tập trung phân bổ trọng số cao vào các tính từ và trạng từ biểu cảm (*"bad"*, *"excellent"*, *"mediocre"*).
3. **Head 3 (Structural Delimiters)**: Dồn trọng số vào các dấu câu và liên từ tương phản (*","*, *"but"*), đóng vai trò làm điểm tựa định tuyến thông tin (information routing hub).
4. **Head 4 (Dispersed / Diffuse)**: Trọng số phân tán đều trên toàn bộ câu, tương tự như một bộ lọc làm mịn (smoothing filter).

---

## 5. Thảo Luận Bắt Buộc: Phân Biệt Giữa Model Behavior và Human Interpretation ("Attention is NOT Explanation")
Một trong những cạm bẫy phổ biến nhất khi nghiên cứu Transformer là vội vã đồng nhất:
$$\text{Attention Weight Cao} \equiv \text{Lý do mô hình đưa ra quyết định}$$

Dựa trên các nghiên cứu nền tảng (Serrano & Smith 2019, Jain & Wallace 2019, Wiegreffe & Pinter 2019), chúng tôi phân biệt rạch ròi hai khái niệm:
1. **Model Behavior (Hành vi nội tại của mô hình)**:
   - Trọng số $A_{ij}$ chỉ đơn thuần biểu thị hệ số tổ hợp tuyến tính của phép nhân vô hướng giữa Query và Key trong không gian con trung gian: $z_i = \sum_j A_{ij} v_j$.
   - Nó phản ánh mức độ tương tác thông tin giữa hai vị trí token trong mạng nơ-ron.
2. **Human Interpretation (Diễn giải của con người)**:
   - Là nỗ lực của con người gán ý nghĩa ngữ nghĩa học (chủ ngữ, vị ngữ, quan hệ nhân quả) cho các trọng số đó.
   - **Vì sao Attention không phải là Explanation đầy đủ?**
     - Thứ nhất, sau lớp Attention còn có kết nối tắt Residual ($x + \text{Attention}(x)$), tầng chuẩn hóa LayerNorm, và tầng biến đổi phi tuyến hai lớp FFN. Đầu ra của Attention bị bẻ cong và tái cấu trúc mạnh mẽ trước khi đến được bộ phân loại cuối cùng.
     - Thứ hai, các nghiên cứu thực nghiệm đã chứng minh ta hoàn toàn có thể tráo đổi hoặc thay đổi đáng kể phân phối trọng số attention mà xác suất dự đoán đầu ra của mô hình hầu như không đổi (Adversarial Attention).
     - Do đó, ma trận Attention chỉ cung cấp một góc nhìn trực quan về luồng dẫn truyền dữ liệu (information flow), chứ không phải là bằng chứng nhân quả (causal explanation) cho quyết định của mô hình.
