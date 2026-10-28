# LAB 06 — ATTENTION MECHANISM

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Chủ đề**: From Recurrent State to Context-Aware Representation  
**Giảng viên / TA**: Phạm Ngọc Hải • **Hạn nộp**: 23:59 - 28/10/2026  

---

## 1. Tổng Quan Bài Tập

LAB 06 đánh dấu bước chuyển mình mang tính cách mạng trong quá trình học tập môn Xử lý ngôn ngữ tự nhiên: Từ việc nén toàn bộ chuỗi vào một trạng thái ẩn tái quy duy nhất ($h_T$ trong RNN/LSTM ở LAB 05) sang **cơ chế Attention** cho phép mô hình linh hoạt chọn lọc và tổng hợp thông tin liên quan trực tiếp giữa các vị trí trong chuỗi.

Cốt lõi của bài thực hành tập trung vào:
1. Bản chất đại số tuyến tính của bộ ba **Query ($Q$), Key ($K$), Value ($V$)**.
2. Cơ chế chuẩn hóa và tính chất giải tích của hàm **Softmax ổn định số học** (Numerically Stable Softmax).
3. Tác động của hệ số chia tỉ lệ **Scaling Factor** ($\sqrt{d_k}$) và hệ số nhiệt độ **Temperature** ($T$) lên động học gradient.
4. Triển khai thuật toán từ đầu bằng **NumPy thuần** (`attention_numpy.py`) và ứng dụng mô hình **Self-Attention Classifier** trên tập dữ liệu chuẩn **IMDb Movie Reviews** ($10,000$ mẫu).
5. Nhận thức sâu sắc về sự đánh đổi giữa năng lực biểu diễn ngữ cảnh và **chi phí tính toán bậc hai $O(T^2)$** theo độ dài chuỗi, làm tiền đề trực tiếp dẫn tới kiến trúc Transformer trong LAB 07.

---

## 2. Cấu Trúc Thư Mục và Deliverables

Toàn bộ các deliverables theo đúng yêu cầu tại Mục 32 của file đề bài `W6.pdf` cùng 4 file PDF bổ sung theo quy chuẩn học thuật được tổ chức như sau:

```text
Lab6/
├── W6.pdf                          # File đề bài gốc chính thức của giảng viên
├── README.md                       # Tài liệu tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md                 # Lời giải chi tiết các bài toán tính tay (6 bước chuẩn)
├── calculations.pdf                # Bản PDF học thuật xuất từ calculations.md qua XeLaTeX
├── prediction.md                   # 5 giả thuyết khoa học được lập và khóa TRƯỚC KHI chạy mô hình
├── prediction.pdf                  # Bản PDF học thuật xuất từ prediction.md qua XeLaTeX
├── attention_numpy.py              # Module Scaled Dot-Product Attention từ đầu bằng NumPy (100% không comment)
├── self_attention.ipynb            # Jupyter Notebook thực thi toàn bộ pipeline thực nghiệm (100% không comment)
├── results.csv                     # Bảng số liệu đo đạc thực nghiệm của tất cả các mô hình
├── attention_heatmap.png           # Bản đồ nhiệt trực quan hóa ma trận Attention tương tác giữa các token
├── confusion_matrix.png            # Ma trận nhầm lẫn (Confusion Matrix) của mô hình trên tập kiểm thử
├── error_analysis.md               # Phân tích định tính 6 ca dự đoán đúng/sai thực tế và taxonomy lỗi
├── error_analysis.pdf              # Bản PDF học thuật xuất từ error_analysis.md qua XeLaTeX
├── reflection.md                   # Chiêm nghiệm, trả lời bộ câu hỏi phỏng vấn và mạch nối sang Transformer
└── reflection.pdf                  # Bản PDF học thuật xuất từ reflection.md qua XeLaTeX
```

---

## 3. Bảng Kết Quả Thực Nghiệm Tổng Hợp (`results.csv`)

| Mô hình (Model) | Biểu diễn (Representation) | Độ dài chuỗi (Seq Len) | Số tham số (Parameters) | Thời gian chạy (s) | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | TF-IDF (10k features) | Toàn bộ | 10,001 | 1.33 | **0.8700** | **0.8537** | **0.8955** | **0.8741** |
| **Simple RNN** | Embedding ($d=64$) | 100 | 648,450 | 13.47 | 0.4920 | 0.4956 | 0.4458 | 0.4694 |
| **LSTM** | Embedding ($d=64$) | 100 | 673,410 | 11.00 | 0.5220 | 0.5271 | 0.5013 | 0.5139 |
| **GRU** | Embedding ($d=64$) | 100 | 665,090 | 32.75 | 0.6667 | 0.6711 | 0.6640 | 0.6676 |
| **Self-Attention** | Embedding ($d=64$) | 100 | 652,353 | **8.10** | **0.7633** | **0.7614** | **0.7725** | **0.7669** |

---

## 4. Các Nhận Định Khoa Học Chính (Key Findings)

1. **Hiệu năng vượt trội của Self-Attention so với Recurrent Models**:
   - Khi cùng huấn luyện từ đầu (train from scratch) trên $7,000$ mẫu huấn luyện IMDb trong $5$ epoch, mô hình Self-Attention đạt $F_1 = 0.7669$, cao hơn hẳn LSTM ($0.5139$) và GRU ($0.6676$).
   - Nguyên nhân chính là Self-Attention cho phép các token biểu đạt cảm xúc (như *great*, *terrible*, *succeeded*) tương tác trực tiếp với bất kỳ vị trí nào trong chuỗi mà không bị suy hao tín hiệu qua các bước thời gian tuần tự.
2. **Khả năng song song hóa và tốc độ huấn luyện**:
   - Do loại bỏ vòng lặp tuần tự `for t in range(seq_len)`, Self-Attention thực hiện toàn bộ phép tính chuỗi qua phép nhân ma trận $Q K^T$. Thời gian huấn luyện chỉ mất $8.10$ giây, nhanh hơn đáng kể so với GRU ($32.75$s) và Simple RNN ($13.47$s).
3. **Cơ chế Scaling Factor ($\sqrt{d_k}$) và Temperature ($T$)**:
   - Chia cho $\sqrt{d_k}$ giúp khống chế phương sai của tích vô hướng về mức $1$, ngăn ngừa việc các score quá lớn đẩy Softmax vào vùng bão hòa triệt tiêu gradient.
   - Temperature $T < 1$ tạo ra phân phối tập trung cao độ (Hard Attention), trong khi $T > 1$ làm phẳng hóa phân phối (Soft/Diffused Attention).
4. **Cảnh báo về giới hạn chi phí tính toán $O(T^2)$**:
   - Ma trận Attention có kích thước $T \times T$. Khi chiều dài chuỗi $T$ tăng từ $100$ lên $10,000$, số phần tử tăng gấp $10,000$ lần (từ $10^4$ lên $10^8$), tiêu tốn hàng trăm Megabytes bộ nhớ GPU cho mỗi tầng. Đây là giới hạn lớn nhất của Attention chuẩn cần các kỹ thuật tối ưu sau này giải quyết.
5. **Cảnh báo về giải thích bản đồ nhiệt Attention**:
   - Trọng số Attention cao phản ánh mức độ định tuyến thông tin số học, không tự động là bằng chứng về lập luận ngôn ngữ học (Reasoning). Không được ngộ nhận tương quan số học thành quan hệ nhân quả.

---

## 5. Hướng Dẫn Tái Lập Thực Nghiệm (Reproducibility Guide)

### 5.1. Yêu cầu môi trường
- Python $\ge 3.10$ (đã kiểm thử và chạy ổn định trên Python 3.14).
- Các thư viện cần thiết: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `nbformat`, `nbconvert`.
- Trình biên dịch LaTeX: MiKTeX `xelatex` (hỗ trợ font Times New Roman tiếng Việt).

### 5.2. Chạy kiểm tra module `attention_numpy.py`
Mã nguồn được thiết kế hoàn toàn tự tài liệu hóa (self-documenting, 0 comment). Chạy kiểm tra các hàm cơ bản:
```bash
python -c "import sys; sys.path.append('Lab6'); from attention_numpy import stable_softmax, scaled_dot_product_attention; import numpy as np; print('Softmax stable:', stable_softmax(np.array([1000., 1001., 1002.])))"
```

### 5.3. Chạy Notebook
Khởi động Jupyter Lab hoặc VSCode Notebook:
```bash
jupyter notebook Lab6/self_attention.ipynb
```
Hoặc thực thi lại toàn bộ notebook tự động từ dòng lệnh:
```bash
jupyter nbconvert --to notebook --execute Lab6/self_attention.ipynb --output self_attention.ipynb
```

### 5.4. Kiểm tra quy tắc không comment (Zero-Comment Audit)
Chạy script kiểm tra tự động đã tích hợp sẵn trong skill:
```bash
python .agents/skills/assignment-solver/scripts/verify_no_comments.py Lab6
```
Kết quả mong đợi: `RESULT: ALL CLEAN - 0 comments or docstrings detected.`

### 5.5. Biên dịch lại các file PDF học thuật
```bash
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab6/calculations.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab6/prediction.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab6/error_analysis.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab6/reflection.md
```

---

## 6. Tuyên Bố Sử Dụng AI (AI Assistance Statement)

- **Tool**: Google Antigravity AI Agent.
- **Purpose**: Hỗ trợ xây dựng pipeline thực nghiệm khoa học, chuẩn hóa các công thức toán học LaTeX, xuất bản các file PDF học thuật và rà soát lỗi cú pháp.
- **Generated**: Khung cấu trúc thực nghiệm, mã nguồn NumPy tự tài liệu hóa `attention_numpy.py`, notebook `self_attention.ipynb`, các bảng số liệu đo đạc và đồ thị biểu diễn.
- **Modified**: Tinh chỉnh các siêu tham số huấn luyện khách quan, cố định random seed 42 để đảm bảo tính tái lập 100%.
- **Verified**: Mọi kết quả số liệu, công thức biến đổi toán học và ảnh chụp biểu đồ đều đã được kiểm tra chéo độc lập qua dòng lệnh terminal và kiểm tra trực quan.
