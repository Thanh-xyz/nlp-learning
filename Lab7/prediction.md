# BÁO CÁO DỰ ĐOÁN TRƯỚC THỰC NGHIỆM — TRANSFORMER ARCHITECTURE

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Chủ đề**: From Attention to Transformer  
**Sinh viên thực hiện**: Nguyễn Văn Thành  
**Thời điểm lập và khóa dự đoán**: Trước khi thực thi bất kỳ pipeline thí nghiệm nào  

---

## 1. Nguyên Tắc Khoa Học Về Dự Đoán
Theo quy chuẩn khoa học của môn học và hướng dẫn tại Mục 8 của đề bài `W7.pdf`, toàn bộ 6 dự đoán dưới đây được lập luận dựa trên nguyên lý giải tích, đại số tuyến tính và kiến trúc của mạng nơ-ron trước khi chạy mã nguồn thực nghiệm. Sau khi đã quan sát kết quả đo đạc từ mô hình, nội dung dự đoán ban đầu tuyệt đối không bị thay đổi; việc đối chiếu và lý giải các trường hợp thực tế sẽ được trình bày ở phần tổng kết độc lập.

---

## 2. Danh Sách 6 Dự Đoán Chi Tiết

### Dự đoán 1: Khả Năng Xử Lý Song Song Giữa Transformer và RNN
- **Câu hỏi / Bài toán so sánh**: *Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?*
- **Prediction**: **CÓ**, Transformer có khả năng song song hóa hoàn toàn quá trình tính toán trên toàn bộ chuỗi ở giai đoạn huấn luyện, vượt trội rõ rệt so với RNN/LSTM.
- **Reason (Lý giải khoa học)**: 
  - Về mặt bản chất tính toán, kiến trúc Recurrent (RNN/LSTM/GRU) có sự phụ thuộc tuần tự nghiêm ngặt theo thời gian: Trạng thái ẩn $h_t$ tại bước $t$ bắt buộc phải chờ trạng thái $h_{t-1}$ của bước trước đó hoàn thành ($h_t = f(h_{t-1}, x_t)$). Do đó, RNN không thể tính toán đồng thời các vị trí khác nhau trên phần cứng GPU/CPU song song.
  - Ngược lại, Transformer thực hiện phép nhân ma trận toàn cục $Q = X W^Q, K = X W^K, V = X W^V$ và tính tích vô hướng $Q K^T$ đồng thời cho toàn bộ ma trận $X \in \mathbb{R}^{B \times T \times d}$. Toàn bộ các tương tác giữa mọi cặp token $(i, j)$ được thực thi qua một phép nhân ma trận song song duy nhất (Batch Matrix Multiplication - `bmm`), tận dụng tối đa hàng nghìn nhân tính toán của phần cứng hiện đại.
- **Confidence**: **RẤT CAO** (99%), do đây là tính chất toán học cơ bản và là động lực tiên quyết mà bài báo gốc *"Attention Is All You Need"* đề xuất để thay thế các mô hình hồi quy.
- **Tiêu chí kiểm chứng**: So sánh thời gian huấn luyện mỗi epoch (`training_time`) giữa mô hình LSTM và mô hình Transformer trên cùng tập dữ liệu IMDb, cùng độ dài chuỗi $T=100$ và cùng batch size trong Experiment 4.

---

### Dự đoán 2: Nhận Thức Về Thứ Tự Token Khi Không Có Positional Encoding
- **Câu hỏi / Bài toán so sánh**: *Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?*
- **Prediction**: **KHÔNG**, nếu không có Positional Encoding, Self-Attention thuần túy hoàn toàn không có khái niệm về trật tự trước - sau của các token trong chuỗi.
- **Reason (Lý giải khoa học)**: 
  - Phép tính Attention cốt lõi là $A_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_m \exp(q_i \cdot k_m / \sqrt{d_k})}$ và $z_i = \sum_j A_{ij} v_j$. Nếu ta hoán vị bất kỳ các hàng của ma trận đầu vào $X$ theo một ma trận hoán vị $P$, thì $Q' = P Q$, $K' = P K$, $V' = P V$. Khi đó ma trận scores trở thành $S' = P S P^T$ và đầu ra là $Z' = P Z$.
  - Tính chất này được gọi là **tính hoán vị đồng biến (Permutation Equivariance)**: Các vector đầu ra bị tráo đổi đúng theo vị trí tráo đổi của đầu vào mà không hề nhận biết được ngữ cảnh thứ tự cấu trúc ngữ pháp thay đổi. Sau phép gộp toàn cục (Mean Pooling), biểu diễn câu hoàn toàn không đổi: $\frac{1}{T}\sum z'_i = \frac{1}{T}\sum z_i$. Mô hình bị suy biến thành một dạng "Bag-of-Words" tương tác.
- **Confidence**: **RẤT CAO** (99%), đã được chứng minh chặt chẽ về mặt toán học.
- **Tiêu chí kiểm chứng**: Phép đo khoảng cách cosine hoặc chuẩn sai phân $L_2$ giữa biểu diễn của hai câu hoán vị $S_1 = \text{"dog bites man"}$ và $S_2 = \text{"man bites dog"}$ trong Experiment 1 khi không có Positional Encoding (sai số xấp xỉ 0).

---

### Dự đoán 3: Khả Năng Phân Biệt Hai Câu Cùng Tập Token Nhưng Khác Thứ Tự
- **Câu hỏi / Bài toán so sánh**: *Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?*
- **Prediction**: **KHÔNG THỂ PHÂN BIỆT ĐƯỢC**, biểu diễn toàn câu (sequence representation) của hai câu hoán vị sẽ giống nhau 100%.
- **Reason (Lý giải khoa học)**: 
  - Xét $S_1 = \text{"dog bites man"}$ và $S_2 = \text{"man bites dog"}$. Tập hợp các vector embedding đầu vào là đa tập hợp (multiset) gồm ba vector: $\{e_{\text{dog}}, e_{\text{bites}}, e_{\text{man}}\}$.
  - Do Self-Attention không có positional encoding là phép biến đổi bất biến theo tập hợp đối với tổng thể và phép Pooling cuối cùng (Mean Pooling) là phép cộng giao hoán $\frac{1}{3}(z_1 + z_2 + z_3)$, nên vector đặc trưng đại diện cho câu $S_1$ và $S_2$ trước khi vào phân loại nhị phân sẽ hoàn toàn đồng nhất.
  - Do đó, mọi bộ phân loại tuyến tính đặt phía sau đều sẽ cho ra cùng một xác suất dự đoán $P(y=1|S_1) = P(y=1|S_2)$, khiến mô hình không thể phân biệt được chủ thể hành động và đối tượng chịu tác động.
- **Confidence**: **RẤT CAO** (99%).
- **Tiêu chí kiểm chứng**: Giá trị sai lệch tuyệt đối $\| \mathbf{h}_{S_1} - \mathbf{h}_{S_2} \|$ bằng 0 trên mô hình không có PE, và chuyển thành giá trị dương khác biệt rõ rệt khi bật Positional Encoding trong Experiment 1 (Permutation test).

---

### Dự đoán 4: Mức Độ Tương Đồng Giữa Các Attention Heads
- **Câu hỏi / Bài toán so sánh**: *Hai attention heads có nhất thiết học cùng một attention pattern không?*
- **Prediction**: **KHÔNG NHẤT THIẾT**, các attention heads sẽ có xu hướng học các mẫu phân bổ trọng số khác nhau, nhưng vẫn có khả năng xuất hiện một số head dư thừa (redundant) hoặc có phân bố gần tương tự nhau.
- **Reason (Lý giải khoa học)**: 
  - Mỗi head $i$ được khởi tạo độc lập với các ma trận chiếu ngẫu nhiên $W_i^Q, W_i^K, W_i^V$. Dưới tác động của hàm mất mát và quá trình lan truyền ngược, gradient sẽ kéo các ma trận trọng số theo các hướng tối ưu hóa khác nhau trong không gian đặc trưng.
  - Một head có thể tập trung vào các token cục bộ lân cận (local syntax), trong khi head khác có thể chú ý mạnh vào các từ mang cực tính cảm xúc chủ chốt (sentiment tokens) hoặc liên từ tương phản ("but", "however"). Tuy nhiên, nếu không có cơ chế phạt trực giao (orthogonal regularization), hiện tượng trùng lặp (head redundancy) vẫn hoàn toàn có thể xảy ra trong các mô hình nhỏ.
- **Confidence**: **CAO** (85%).
- **Tiêu chí kiểm chứng**: Trực quan hóa bản đồ nhiệt của 4 heads trong Experiment 3 (`attention_heads.png`), quan sát độ lệch chuẩn và sự phân bố tập trung/phân tán của từng ma trận trọng số.

---

### Dự đoán 5: Tác Động Của Việc Tăng Số Lượng Attention Heads Lên Độ Chính Xác
- **Câu hỏi / Bài toán so sánh**: *Tăng số attention heads có chắc chắn làm accuracy tăng không?*
- **Prediction**: **KHÔNG CHẮC CHẮN**, hiệu năng (Accuracy và F1) sẽ có dạng đường cong bão hòa hoặc suy giảm nhẹ nếu số heads tăng quá mức trong điều kiện kích thước $d_{model}$ giữ nguyên cố định.
- **Reason (Lý giải khoa học)**: 
  - Khi cố định $d_{model}$ (ví dụ $d_{model} = 128$), việc tăng số heads $h$ từ 1 lên 2, 4, 8 sẽ làm giảm kích thước của từng head tương ứng: $d_k = \frac{d_{model}}{h}$. Khi $h = 8$, số chiều của mỗi head bị co hẹp xuống chỉ còn $d_k = 16$.
  - Kích thước $d_k$ quá nhỏ làm suy giảm năng lực biểu diễn của tích vô hướng $Q K^T$, giới hạn không gian hình học để phân tách các mẫu ngữ nghĩa tinh vi. Đồng thời, việc phân chia quá nhiều head nhỏ trên tập dữ liệu kích thước khiêm tốn có thể dẫn đến hiện tượng khó hội tụ hoặc phân tán gradient. Thường cấu hình $h = 2$ hoặc $h = 4$ sẽ đạt điểm cân bằng tối ưu.
- **Confidence**: **CAO** (80%).
- **Tiêu chí kiểm chứng**: Bảng số liệu đo lường Accuracy và F1 trên tập kiểm thử IMDb tương ứng với $n_{heads} \in \{1, 2, 4, 8\}$ trong Experiment 2.

---

### Dự đoán 6: Tốc Độ Thực Thi Giữa Transformer Và RNN Theo Chiều Dài Chuỗi
- **Câu hỏi / Bài toán so sánh**: *Transformer có luôn nhanh hơn RNN trên mọi sequence length không?*
- **Prediction**: **KHÔNG**, Transformer chỉ vượt trội về tốc độ ở các chuỗi ngắn và trung bình ($T \le 128$); khi độ dài chuỗi $T$ tăng lên rất lớn ($T \ge 512, 1000$), chi phí tính toán và bộ nhớ bậc hai $O(T^2)$ của ma trận Attention sẽ tăng vọt, khiến Transformer chậm hơn hoặc thậm chí tràn bộ nhớ (Out of Memory) so với độ phức tạp tuyến tính $O(T)$ của RNN.
- **Reason (Lý giải khoa học)**: 
  - Chi phí tính toán của một bước RNN trên chuỗi dài $T$ là $O(T \cdot d^2)$. Mặc dù không thể song song hóa, nhưng độ phức tạp theo chiều dài chuỗi là **tuyến tính** $O(T)$.
  - Trong khi đó, phép tính Attention Scores của Transformer đòi hỏi nhân ma trận $Q K^T$ có kích thước $T \times T$, dẫn tới chi phí tính toán $O(T^2 \cdot d)$ và chi phí lưu trữ bộ nhớ kích hoạt là $O(B \cdot h \cdot T^2)$. Khi $T$ tăng gấp 4 lần (từ 64 lên 256), bộ nhớ attention tăng gấp $4^2 = 16$ lần. Ở các chuỗi cực dài, điểm nghẽn bộ nhớ $O(T^2)$ và chi phí giao tiếp bộ nhớ sẽ lấn át lợi thế song song hóa.
- **Confidence**: **RẤT CAO** (95%).
- **Tiêu chí kiểm chứng**: Thực nghiệm đo bộ nhớ lý thuyết và thời gian xử lý khi thay đổi $T \in \{32, 64, 128, 256\}$ trong Experiment 5.

