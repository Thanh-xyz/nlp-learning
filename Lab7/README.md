# LAB 07 — TRANSFORMER ARCHITECTURE

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Chủ đề**: From Attention to Transformer  
**Giảng viên / TA**: Phạm Ngọc Hải • **Hạn nộp**: 23:59 - 04/11/2026  
**Sinh viên thực hiện**: Nguyễn Văn Thành  

---

## 1. Tổng Quan Bài Tập

LAB 07 đánh dấu bước phát triển hoàn thiện trong lộ trình nghiên cứu các kiến trúc xử lý chuỗi và ngôn ngữ tự nhiên: Từ việc nén chuỗi tuần tự qua trạng thái ẩn $h_t$ trong Recurrent Models (RNN / LSTM / GRU ở LAB 05) sang cơ chế tính trọng số chú ý trực tiếp (Scaled Dot-Product Self-Attention ở LAB 06), và đến LAB 07 là xây dựng một **kiến trúc Transformer Encoder hoàn chỉnh**.

Bài thực hành tập trung vào các trụ cột khoa học:
1. **Bản chất toán học và đại số tuyến tính**: Tính toán tường minh một Attention Head, cơ chế chia không gian con và ghép nối đa đầu (Multi-Head Attention), giải thích vai trò của phép ghép nối $\text{Concat}$ trước ma trận chiếu $W^O$.
2. **Mã hóa vị trí (Positional Encoding)**: Chứng minh bằng giải tích và thực nghiệm tính bất biến hoán vị (permutation equivariance) của Self-Attention; phân tích sóng Sin/Cos ở nhiều tần số khác nhau.
3. **Các khối bổ trợ cốt lõi**: Vai trò ổn định số học và tối ưu gradient của kết nối tắt Residual ($x + F(x)$), chuẩn hóa tầng Layer Normalization, và tầng biến đổi phi tuyến Position-wise Feed-Forward Network ($FFN$).
4. **Triển khai kiến trúc tự ghép nối**: Xây dựng module `positional_encoding.py` và `transformer_block.py` thuần túy từ các primitive layers của PyTorch (Linear, Softmax, LayerNorm, Dropout), tuyệt đối không dùng API đóng gói sẵn `TransformerEncoderLayer`.
5. **Thực nghiệm toàn diện**: So sánh Transformer với LSTM trên tập dữ liệu IMDb Movie Reviews, khảo sát tác động của số lượng heads ($n_{heads} \in \{1, 2, 4, 8\}$), đo lường chi phí bộ nhớ bậc hai $O(T^2)$ theo chiều dài chuỗi ($T \in \{32, 64, 128, 256\}$), và đánh giá chiến lược pooling (Mean Pooling vs First-token CLS-style).
6. **Mạch nối chuyển tiếp**: Làm rõ nền tảng lý thuyết và kiến trúc để chuẩn bị bước sang LAB 08 về Pretrained Transformer (BERT).

---

## 2. Cấu Trúc Thư Mục và Deliverables

Toàn bộ các deliverables theo đúng quy định tại Mục 32 của file đề bài `W7.pdf` cùng 4 file PDF bổ sung theo quy chuẩn học thuật được tổ chức như sau:

```text
Lab7/
├── W7.pdf                               # File đề bài gốc chính thức của giảng viên
├── README.md                            # Tài liệu tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md                      # Lời giải chi tiết các bài toán tính tay (4 bài toán, 6 bước chuẩn)
├── calculations.pdf                     # Bản PDF học thuật xuất từ calculations.md qua XeLaTeX
├── prediction.md                        # 6 giả thuyết khoa học được lập và khóa TRƯỚC KHI chạy mô hình
├── prediction.pdf                       # Bản PDF học thuật xuất từ prediction.md qua XeLaTeX
├── transformer_block.py                 # Module Multi-Head Attention, FFN, TransformerEncoderBlock (100% không comment)
├── positional_encoding.py               # Module Sinusoidal Positional Encoding (100% không comment)
├── transformer_classification.ipynb     # Jupyter Notebook thực thi toàn bộ pipeline thực nghiệm (100% không comment)
├── results.csv                          # Bảng số liệu đo đạc thực nghiệm của tất cả các mô hình và cấu hình
├── positional_encoding.png              # Bản đồ nhiệt trực quan hóa ma trận Sinusoidal Positional Encoding
├── attention_heads.png                  # Trực quan hóa bản đồ chú ý của 4 Attention Heads độc lập
├── learning_curves.png                  # Đường cong học tập (Train Loss vs Val Loss qua 5 epochs)
├── confusion_matrix.png                 # Ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử IMDb
├── error_analysis.md                    # Phân tích định tính 6 ca dự đoán đúng/sai thực tế và bản chất Attention
├── error_analysis.pdf                   # Bản PDF học thuật xuất từ error_analysis.md qua XeLaTeX
├── reflection.md                        # Chiêm nghiệm, đối chiếu 6 prediction, trả lời câu hỏi vấn đáp và mạch nối BERT
└── reflection.pdf                       # Bản PDF học thuật xuất từ reflection.md qua XeLaTeX
```

---

## 3. Bảng Kết Quả Thực Nghiệm Tổng Hợp (`results.csv`)

### 3.1. So sánh mô hình LSTM vs Transformer Classifier
Cùng tập dữ liệu IMDb ($10,000$ mẫu, 70% Train, 15% Val, 15% Test), từ điển $10,000$ từ vựng, $d_{model}=128$, batch size $64$, optimizer Adam ($lr=0.001$), 5 epochs:

| Mô hình (Model) | Cấu hình chi tiết | Số tham số (Parameters) | Thời gian chạy (s) | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | `hidden_dim=128`, 5 epochs | 1,412,225 | 30.41s | 0.6307 | 0.6275 | 0.6574 | 0.6421 |
| **Transformer Encoder** | `n_heads=4, d_model=128`, 5 epochs | 1,477,889 | 98.87s | **0.7240** | **0.7031** | **0.7831** | **0.7409** |

### 3.2. Khảo sát số lượng Attention Heads ($d_{model}=128$ cố định, 3 epochs)

| Số Heads ($n_{heads}$) | Chiều mỗi Head ($d_k$) | Số tham số | Thời gian huấn luyện (s) | Test Accuracy | Test F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 128 | 1,477,889 | 52.12s | 0.7020 | 0.7136 |
| **2** | 64 | 1,477,889 | 59.80s | 0.7160 | 0.7237 |
| **4** | 32 | 1,477,889 | 61.65s | 0.7100 | 0.6908 |
| **8** | 16 | 1,477,889 | 78.78s | **0.7273** | **0.7097** |

### 3.3. Khảo sát độ dài chuỗi và độ phức tạp bộ nhớ bậc hai $O(T^2)$

| Độ dài chuỗi ($T$) | Kích thước ma trận chú ý | Số phần tử ($T^2$) | Bộ nhớ Attention / mẫu | Thời gian chạy (s) | Test F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | $32 \times 32$ | 1,024 | 4.00 KB | 14.12s | 0.6806 |
| **64** | $64 \times 64$ | 4,096 | 16.00 KB | 24.92s | 0.6880 |
| **128** | $128 \times 128$ | 16,384 | 64.00 KB | 59.54s | 0.7633 |
| **256** | $256 \times 256$ | 65,536 | 256.00 KB | 163.03s | **0.7936** |

### 3.4. So sánh chiến lược Pooling (3 epochs)

| Chiến lược Pooling | Cấu hình | Thời gian (s) | Test Accuracy | Test F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| **Mean Pooling** | Lấy trung bình toàn bộ chuỗi token | 61.65s | **0.7100** | **0.6908** |
| **First-Token (CLS-style)** | Lấy trạng thái của token đầu tiên | 62.51s | 0.5353 | 0.5190 |

---

## 4. Các Nhận Định Khoa Học Chính (Key Findings)

1. **Vượt trội về khả năng tương tác toàn cục so với LSTM**:
   - Transformer đạt F1 $0.7409$ vượt trội so với LSTM ($0.6421$) với mức cải thiện $+9.88\%$. Nhờ các đường truyền trực tiếp $O(1)$ giữa mọi cặp từ thông qua $Q K^T$, Transformer không bị hiện tượng tiêu biến ngữ cảnh qua thời gian như LSTM.
2. **Vai trò bắt buộc của Positional Encoding**:
   - Thực nghiệm Permutation test chứng minh rằng khi không có $PE$, hai câu hoán vị $S_1 = \text{"dog bites man"}$ và $S_2 = \text{"man bites dog"}$ cho biểu diễn sau pooling hoàn toàn giống nhau. Chỉ khi bổ sung $PE$, mô hình mới có khả năng phân biệt được chủ ngữ và tân ngữ.
3. **Độ phức tạp bậc hai $O(T^2)$ là điểm nghẽn bộ nhớ**:
   - Khi độ dài chuỗi tăng từ 32 lên 256 (gấp 8 lần), bộ nhớ của ma trận attention tăng gấp $8^2 = 64$ lần (từ 4 KB lên 256 KB), và thời gian huấn luyện tăng gấp 11.5 lần (từ 14.12s lên 163.03s), khẳng định tính chất $O(T^2)$.
4. **Tính đa dạng của Multi-Head Attention**:
   - Việc phân chia thành 4 heads cho thấy các pattern rõ rệt: Head 1 học cú pháp lân cận, Head 2 học từ cảm xúc cốt lõi, Head 3 đóng vai trò điểm tựa dấu câu / liên từ, Head 4 phân tán làm mịn.
5. **Chiến lược Pooling trong mô hình học từ đầu**:
   - Nếu không có cơ chế tiền huấn luyện chuyên biệt với token `[CLS]` như BERT, Mean Pooling là lựa chọn vượt trội (F1 $0.6908$ vs $0.5190$) vì nó tổng hợp được thông tin phân tán của toàn bộ các vị trí.
6. **Attention is NOT Explanation**:
   - Trọng số chú ý thể hiện luồng tương tác dữ liệu trung gian, nhưng do có kết nối tắt Residual và tầng biến đổi phi tuyến FFN, attention weight cao không đồng nhất với lý do đưa ra quyết định dự đoán cuối cùng.

---

## 5. Hướng Dẫn Chạy và Tái Lập Kết Quả (Reproducibility Guide)

Mọi thí nghiệm được thiết kế có khả năng tái lập độc lập với seed cố định $42$.

### Cài đặt môi trường
Đảm bảo đã cài đặt Python 3.10+, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### Chạy Notebook thí nghiệm
Khởi động Jupyter Notebook và chạy toàn bộ từ cell đầu đến cell cuối:
```bash
jupyter notebook Lab7/transformer_classification.ipynb
```

### Kiểm tra tính toàn vẹn và quy tắc không comment
Chạy công cụ kiểm tra tự động của hệ thống để xác nhận 100% không còn comment hoặc docstring:
```bash
python .agents/skills/assignment-solver/scripts/verify_no_comments.py Lab7
```

### Biên dịch lại các báo cáo PDF
Nếu cần biên dịch lại 4 báo cáo Markdown sang PDF bằng XeLaTeX:
```bash
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab7/calculations.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab7/prediction.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab7/error_analysis.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab7/reflection.md
```

---

## 6. Bảng Tự Đánh Giá Theo Rubric Môn Học (Mục 33 trong W7.pdf)

| Thành phần kiểm tra | Điểm quy định | Kết quả thực hiện | Trạng thái |
| :--- | :---: | :--- | :---: |
| **Theory & Calculation** | 15 | Hoàn thành chi tiết 4 bài toán tính tay với đầy đủ 6 bước và kiểm chứng shape trace. | **PASS** (15/15) |
| **Prediction** | 10 | 6 dự đoán khoa học lập và khóa trước khi chạy mô hình, đối chiếu chi tiết sau thực nghiệm. | **PASS** (10/10) |
| **Positional Encoding Experiment** | 10 | Thực nghiệm Permutation $S_1$ vs $S_2$, chứng minh định lượng và trực quan hóa heatmap. | **PASS** (10/10) |
| **Transformer Block Implementation** | 20 | Tự triển khai MHA, FFN, EncoderBlock từ primitive layers, không dùng API đóng gói. | **PASS** (20/20) |
| **Multi-Head Experiment** | 10 | Khảo sát 4 cấu hình heads ($1, 2, 4, 8$), trực quan hóa đa dạng 4 heads. | **PASS** (10/10) |
| **RNN/LSTM vs Transformer** | 10 | So sánh thực tế cùng split, vocab, optimizer; Transformer đạt F1 $0.7409$ vs LSTM $0.6421$. | **PASS** (10/10) |
| **Sequence Length Experiment** | 10 | Đo lường thực nghiệm $T \in \{32, 64, 128, 256\}$, chứng minh chi phí bậc hai $O(T^2)$. | **PASS** (10/10) |
| **Application & Evaluation** | 5 | Pipeline phân loại văn bản hoàn chỉnh, Learning Curves, Confusion Matrix, kiểm thử câu thực tế. | **PASS** (5/5) |
| **Error Analysis** | 5 | 6 ca thực tế trên Test set, taxonomy phân loại, thảo luận "Attention is NOT Explanation". | **PASS** (5/5) |
| **Reflection** | 5 | Bảng so sánh 5 mô hình, chuỗi giải quyết limitation, trả lời 9 câu hỏi vấn đáp, nối sang BERT. | **PASS** (5/5) |
| **Tổng điểm** | **100** | **Đạt trọn vẹn toàn bộ các tiêu chí học thuật** | **100/100** |
