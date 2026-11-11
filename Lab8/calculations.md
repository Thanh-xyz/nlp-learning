# LAB 08 — TÍNH TOÁN BẰNG TAY (HAND CALCULATIONS)
## Pretrained Transformers and BERT Fine-Tuning

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Sinh viên thực hiện**: Nguyễn Văn Thành  
**Mã bài tập**: LAB 08  

---

## 1. Bài Toán 1: Tính Masked Language Modeling (MLM) Loss

### 1.1. Dữ kiện đầu vào và Giả thiết
Trong mô hình Masked Language Modeling (điển hình như BERT), câu đầu vào được che (mask) một số token ngẫu nhiên.
Cho câu gốc:
$$\text{"The cat eats fish."}$$

Sau khi áp dụng cơ chế masking tại 1 vị trí:
$$\text{"The cat [MASK] fish."}$$

- Tập vị trí bị mask: $M = \{t\}$, với $t$ là vị trí của từ `"eats"`.
- Nhãn token thực tế tại vị trí mask: $y_t = \text{"eats"}$.
- Giả sử không gian từ vựng rút gọn gồm 4 từ với phân phối xác suất dự đoán sau hàm Softmax của mô hình:
  - $P(\text{"cat"} \mid x) = 0.10$
  - $P(\text{"dog"} \mid x) = 0.05$
  - $P(\text{"eats"} \mid x) = 0.70$
  - $P(\text{"fish"} \mid x) = 0.15$

Tổng xác suất được chuẩn hóa:
$$\sum_{w} P(w \mid x) = 0.10 + 0.05 + 0.70 + 0.15 = 1.00$$

### 1.2. Công thức toán học
Hàm mất mát Cross-Entropy tổng quát cho tác vụ MLM trên tập các vị trí bị mask $M$ được định nghĩa:
$$L_{\text{MLM}} = - \sum_{i \in M} \log P(y_i \mid x)$$

Trong đó logarit thông thường trong Machine Learning và PyTorch là logarit tự nhiên (cơ số $e$, ký hiệu $\ln$ hoặc $\log$).
Khi chỉ có một token duy nhất bị che tại vị trí $t$, công thức rút gọn thành:
$$L = - \ln P(y_t \mid x)$$

### 1.3. Thay số cụ thể
Thay xác suất dự đoán cho token thật $y_t = \text{"eats"}$ vào công thức:
$$P(\text{"eats"} \mid x) = 0.70$$
$$L = - \ln(0.70)$$

### 1.4. Các bước biến đổi trung gian
Sử dụng khai triển Taylor hoặc tính toán giải tích trực tiếp của logarit tự nhiên:
$$\ln(0.70) = \ln\left(\frac{7}{10}\right) = \ln(7) - \ln(10)$$

Với giá trị chuẩn:
$$\ln(7) \approx 1.94591015$$
$$\ln(10) \approx 2.30258509$$
$$\ln(0.70) = 1.94591015 - 2.30258509 = -0.35667494$$

Do đó, giá trị hàm mất mát là:
$$L = - (-0.35667494) = 0.35667494$$

### 1.5. Kết quả cuối cùng
$$L \approx 0.35667$$

### 1.6. Kiểm tra tính hợp lý (Sanity Check)
- Miền giá trị của hàm mất mát Cross-Entropy: Với $P \in (0, 1]$, ta luôn có $-\ln(P) \ge 0$.
- Do $P = 0.70 > 0.5$ (xác suất khá cao), giá trị mất mát $0.35667$ là nhỏ hơn $-\ln(0.5) \approx 0.69315$, phản ánh đúng độ tự tin cao của mô hình.

---

### 1.7. Trả lời các câu hỏi mở rộng của Bài 1

#### Câu hỏi 1: Nếu mô hình dự đoán $P(\text{"eats"}) = 0.95$, loss thay đổi như thế nào?
- **Tính toán**:
  $$L_{0.95} = - \ln(0.95) \approx - (-0.051293) = 0.05129$$
- **Độ biến thiên**:
  $$\Delta L = 0.05129 - 0.35667 = -0.30538 \quad (\text{giảm } 85.62\%)$$
- **Ý nghĩa**: Khi mô hình cực kỳ tự tin vào token đích ($95\%$), hàm mất mát tiến rất gần về 0. Gradient cập nhật sẽ nhỏ, mô hình chỉ cần tinh chỉnh nhẹ trọng số.

#### Câu hỏi 2: Nếu mô hình dự đoán $P(\text{"eats"}) = 0.01$, loss thay đổi như thế nào?
- **Tính toán**:
  $$L_{0.01} = - \ln(0.01) = \ln(100) \approx 4.60517$$
- **Độ biến thiên**:
  $$\Delta L = 4.60517 - 0.35667 = +4.24850 \quad (\text{tăng } 1191.15\%)$$
- **Ý nghĩa**: Khi mô hình gán xác suất rất thấp ($1\%$) cho token đúng (dự đoán sai lệch nghiêm trọng), hàm mất mát bùng nổ rất mạnh. Điều này tạo ra tín hiệu phạt cực lớn để đẩy lùi trọng số gây ra lỗi.

#### Câu hỏi 3: Vì sao Cross-Entropy phù hợp với bài toán Masked Language Modeling?
1. **Bản chất Maximum Likelihood Estimation (MLE)**: Việc cực tiểu hóa tổng Cross-Entropy trên toàn bộ kho ngữ liệu tương đương trực tiếp với việc cực đại hóa hàm hợp lý logarit (Log-Likelihood) của phân phối ngôn ngữ $P(w_t \mid w_{\ne t})$.
2. **Hình phạt bất đối xứng phi tuyến**: Hàm $-\ln(p)$ phạt theo hàm số mũ khi $p \to 0$ ($\lim_{p \to 0^+} -\ln(p) = +\infty$), ngăn chặn mô hình đưa ra các dự đoán hoàn toàn tự tin nhưng sai lệch.
3. **Đạo hàm tuyến tính tối ưu khi kết hợp Softmax**: Đặt $z_k$ là logit trước Softmax, ta có:
   $$\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$$
   Gradient có dạng sai số tuyến tính trực tiếp giữa xác suất dự đoán $\hat{y}_k$ và nhãn one-hot $y_k$, triệt tiêu hoàn toàn hiện tượng bão hòa đạo hàm (vanishing gradient) gặp phải ở hàm Mean Squared Error (MSE).

---

## 2. Bài Toán 2: Phân Tích Tokenization Trên Pretrained BERT

### 2.1. Dữ kiện đầu vào và Giả thiết
Cho câu văn bản thô:
$$\text{"The cats are playing."}$$

Mô hình sử dụng: `bert-base-uncased` (sử dụng thuật toán **WordPiece Tokenizer** với kích thước từ điển $V = 30,522$).

### 2.2. Trình tự Tokenization và Kết quả chi tiết

#### 1. Tokenize bằng Tokenizer được cung cấp (`BertTokenizer`)
Chuỗi sau khi được tiền xử lý chuyển thành chữ thường (uncased), tách dấu câu và bổ sung các token đặc biệt:
$$\text{Tokens} = [\text{"[CLS]"}, \text{"the"}, \text{"cats"}, \text{"are"}, \text{"playing"}, \text{"."}, \text{"[SEP]" rigidity}]$$

Cụ thể từng token:
1. `[CLS]` (token đại diện đầu chuỗi)
2. `the` (mạo từ xác định)
3. `cats` (danh từ số nhiều - xuất hiện nguyên khối trong từ điển BERT)
4. `are` (trợ động từ)
5. `playing` (động từ đuôi -ing - xuất hiện nguyên khối trong từ điển BERT)
6. `.` (dấu chấm câu)
7. `[SEP]` (token phân tách kết thúc chuỗi)

#### 2. Ghi Token Sequence
$$\text{Sequence} = \big[ \text{"[CLS]"}, \text{"the"}, \text{"cats"}, \text{"are"}, \text{"playing"}, \text{"."}, \text{"[SEP]"} \big]$$

#### 3. Ghi Token IDs
Tra cứu theo bảng từ vựng `vocab.txt` chính thức của `bert-base-uncased`:
- `[CLS]` $\to 101$
- `the` $\to 1996$
- `cats` $\to 8870$
- `are` $\to 2024$
- `playing` $\to 2652$
- `.` $\to 1012$
- `[SEP]` $\to 102$

$$\text{Token IDs} = [101, 1996, 8870, 2024, 2652, 1012, 102]$$

#### 4. Xác định Special Tokens
- **`[CLS]` (Classification Token)**: Mã ID là **101**. Được đặt ở vị trí chỉ số 0 của mọi sequence. Trạng thái ẩn tại vị trí này ở tầng cuối cùng sẽ được trích xuất làm vector đại diện toàn cục cho cả câu phục vụ bài toán Text Classification.
- **`[SEP]` (Separator Token)**: Mã ID là **102**. Đánh dấu ranh giới kết thúc của một phân đoạn văn bản hoặc phân tách Sentence A và Sentence B trong các bài toán cặp câu (Next Sentence Prediction, NLI, QA).

#### 5. Xác định Attention Mask
Do câu có độ dài $T = 7$ và không có token padding `[PAD]` nào cần chèn thêm:
$$\text{Attention Mask} = [1, 1, 1, 1, 1, 1, 1]$$
- Giá trị $1$: Chỉ định cho cơ chế Self-Attention rằng các vị trí này là token hợp lệ và được phép tham gia tính toán ma trận tương tác chú ý $QK^T / \sqrt{d_k}$.
- Nếu chuỗi được pad đến độ dài tối đa (ví dụ $T_{\max} = 10$), các vị trí pad mang giá trị $0$ để bị mặt nạ hóa với giá trị âm vô cùng ($-\infty$) trước Softmax.

#### 6. So sánh số lượng Words vs Tokens vs Subwords

| Đại lượng đo lường | Số lượng | Danh sách chi tiết |
| :--- | :---: | :--- |
| **Số từ thực tế (Words)** | 4 (hoặc 5) | {"The", "cats", "are", "playing"} (loại dấu câu) hoặc kèm {"."} |
| **Số Tokens (Tokens)** | 7 | `[CLS]`, `the`, `cats`, `are`, `playing`, `.`, `[SEP]` |
| **Số Subword Tokens** | 5 | `the`, `cats`, `are`, `playing`, `.` |

**Nhận xét sâu về cơ chế Subword Tokenization**:
- Với các từ phổ biến, WordPiece giữ nguyên từ hoàn chỉnh (cả `"cats"` ID 8870 và `"playing"` ID 2652 đều có mặt trong từ điển BERT uncased).
- Tuy nhiên, nếu gặp từ hiếm hoặc biến thể hình thái học không nằm trong từ điển, ví dụ `"unbelievable"`, tokenizer sẽ phân rã thành các subword: `un`, `##belie`, `##vable`. Ký tự tiền tố `##` đánh dấu subword là phần tiếp nối của từ phía trước chứ không phải bắt đầu một từ mới.
- Cơ chế này giải quyết triệt để vấn đề Out-Of-Vocabulary (OOV) vốn làm tê liệt các mô hình từ điển tĩnh cổ điển.

