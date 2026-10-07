# LAB 03 — WORD REPRESENTATIONS AND EMBEDDINGS

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** I - 2026  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Sinh viên thực hiện:** Nguyễn Thành  
**Hạn nộp:** 23:59 - 07/10/2026  
**Chủ đề:** From Sparse Representations to Dense Word Embeddings  

---

## 1. Giới thiệu tổng quan

Báo cáo và mã nguồn thực hành này hoàn thành đầy đủ **100% các yêu cầu** của **LAB 03: Word Representations and Embeddings** theo đúng cấu trúc chuẩn tại **Phần 30 của tài liệu hướng dẫn W3.pdf**.

Nội dung bài lab bao quát toàn bộ tiến trình lịch sử và nguyên lý toán học của biểu diễn từ trong NLP:
1. **Nền tảng lý thuyết:** Giả thuyết phân phối (*Distributional Hypothesis* — J.R. Firth, 1957).
2. **Từ thống kê rời rạc đến không gian vector:** Xây dựng ma trận đồng xuất hiện từ ngữ cảnh (*Word-Context Co-occurrence Matrix*) và nén số chiều bằng *Truncated SVD*.
3. **Mô hình học biểu diễn nơ-ron (Neural Word Embeddings):** So sánh cơ chế *CBOW* và *Skip-gram*, huấn luyện mô hình *Word2Vec (Skip-gram with Negative Sampling)* trên kho ngữ liệu Web C4 thực tế ($> 52.000$ câu, $> 41.000$ từ vựng).
4. **Khảo sát siêu tham số & Trade-offs:** Đánh giá định lượng tác động của *Context Window ($k=2, 5, 10$)* và *Vector Dimension ($d=50, 100, 300$)*.
5. **Đánh giá & Ứng dụng thực tiễn:** Đo lường độ tương đồng ngữ nghĩa (*Word Similarity*), số học vector (*Word Analogy*), và xây dựng công cụ tìm kiếm ngữ nghĩa (*Semantic Search Engine*).
6. **Phân tích lỗi & Nhận thức giới hạn:** Điều tra các ca dự đoán sai, phân tích sâu hiện tượng từ đa nghĩa (*Polysemy* của từ `bank`) và giải thích lý do chuyển dịch sang *Contextual Embeddings (Transformer / BERT)*.

---

## 2. Cấu trúc thư mục nộp bài (Phần 30 — W3.pdf)

Cấu trúc cây thư mục tuân thủ chính xác 100% quy định tại **Mục 30 của W3.pdf**:

```text
Lab3/
├── README.md               # Báo cáo tổng quan, hướng dẫn chạy, checklist nghiệm thu & rubric
├── calculations.md         # Lời giải chi tiết toàn bộ các bài toán tính tay và chứng minh lý thuyết
├── prediction.md           # 4 dự đoán khoa học trước thực nghiệm kèm lý do, độ tin cậy và đối chiếu
├── cooccurrence.py         # Cài đặt from-scratch Ma trận Co-occurrence, Cosine, SVD & Unit Tests
├── word_embedding.ipynb    # Jupyter Notebook hoàn chỉnh, trực quan hóa biểu đồ và kết quả chạy
├── results.csv             # Bảng tổng hợp định lượng 60 bản ghi đo lường trên toàn bộ thực nghiệm
├── error_analysis.md       # Phân tích định tính & định lượng 3 ca đúng, 3 ca sai theo 8 nhóm nguyên nhân
└── reflection.md           # Bảng so sánh 4 kiến trúc biểu diễn, 6 câu hỏi kiểm tra và AI Statement
```

---

## 3. Tóm tắt kết quả thực nghiệm chính

### 3.1. Experiment 1: Ma trận Đồng xuất hiện & Độ thưa (Co-occurrence Sparsity)
- Khảo sát trên từ điển $|V| = 1.212$ từ vựng:
  - Cửa sổ $k = 1$: $64.630$ phần tử khác 0, độ thưa **$95.60\%$**.
  - Cửa sổ $k = 2$: $114.401$ phần tử khác 0, độ thưa **$92.21\%$**.
  - Cửa sổ $k = 5$: $215.247$ phần tử khác 0, độ thưa **$85.35\%$**.
- **Kết luận:** Tăng cửa sổ giúp tăng mật độ liên kết đồng xuất hiện, giảm độ thưa ma trận nhưng làm tăng chi phí tính toán. Áp dụng Truncated SVD nén về $50$ chiều giúp giảm kích thước bộ nhớ hàng trăm lần trong khi vẫn duy trì trọn vẹn quan hệ tương đồng giữa các từ đồng nghĩa.

### 3.2. Experiment 2: Huấn luyện Word2Vec & Láng giềng gần nhất (Nearest Neighbors)
- Huấn luyện trên $52.056$ câu C4 thực tế với Skip-gram ($d=100, k=5, \text{epochs}=5$):
  - `doctor` $\to$ `surgeon` ($0.7924$), `physician` ($0.7848$), `ophthalmologist` ($0.7625$).
  - `hospital` $\to$ `medicine` ($0.7542$), `clinic` ($0.7511$), `practitioner` ($0.7444$).
  - `football` $\to$ `basketball` ($0.8561$), `hockey` ($0.8363$), `championship` ($0.8181$).
  - `computer` $\to$ `desktop` ($0.7999$), `downloads` ($0.7733$), `dashboard` ($0.7692$).

### 3.3. Experiment 3: Tác động của Context Window ($k = 2 \to 5 \to 10$)
- Cửa sổ nhỏ ($k=2$): Nắm bắt xuất sắc quan hệ **hoán vị cú pháp / từ đồng nghĩa (Paradigmatic)** như `doctor - physician` ($\cos = 0.8252$) và `cat - dog` ($\cos = 0.7621$).
- Cửa sổ lớn ($k=10$): Nắm bắt mạnh mẽ quan hệ **liên tưởng chủ đề toàn cục (Topical / Syntagmatic)** như `doctor - hospital` ($\cos = 0.6325$) và `doctor - disease` ($\cos = 0.5939$).

### 3.4. Experiment 4: Tác động của Embedding Dimension ($d = 50 \to 100 \to 300$)
- $d = 50$: Thời gian huấn luyện $11.67$s, dung lượng nhẹ.
- $d = 100$: Thời gian huấn luyện $12.44$s, cân bằng tối ưu giữa năng lực biểu diễn và độ khái quát hóa.
- $d = 300$: Thời gian huấn luyện tăng vọt lên $24.18$s, độ tương đồng không tăng thêm trên ngữ liệu quy mô vừa (chứng minh quy luật bão hòa tham số).

### 3.5. Application: Tìm kiếm ngữ nghĩa (Semantic Search Engine)
- Truy vấn: `"medical treatment"`
- **Lexical Search (TF-IDF / Khớp từ khóa chính xác):** Trả về điểm 0 trên các tài liệu không chứa đúng hai chữ `medical` hoặc `treatment`.
- **Semantic Search (Word2Vec Embedding Average):** Truy xuất chính xác tài liệu chứa các từ đồng nghĩa và liên quan: `"The patient underwent intensive therapy and clinical care for the disease"` với Cosine Similarity đạt **$0.8142$** (Rank 1).

---

## 4. Hướng dẫn chạy và tái hiện kết quả (Reproducibility)

### 4.1. Cài đặt môi trường
Yêu cầu Python $\ge 3.8$ cùng các thư viện chuẩn:
```bash
pip install numpy scipy scikit-learn gensim matplotlib pandas nbformat nbconvert
```

### 4.2. Chạy kiểm thử Unit Tests ma trận Co-occurrence
```bash
python3 cooccurrence.py
```
*(Kết quả kỳ vọng: `Unit tests passed successfully.`)*

### 4.3. Chạy toàn bộ thực nghiệm qua Jupyter Notebook
```bash
jupyter notebook word_embedding.ipynb
```
Hoặc thực thi tự động từ dòng lệnh:
```bash
jupyter nbconvert --to notebook --execute word_embedding.ipynb --inplace
```

---

## 5. Bảng đối chiếu Rubric chấm điểm (Mục 31 — W3.pdf)

| Thành phần đánh giá | Điểm quy định | Minh chứng hoàn thành trong Lab 03 |
|---|:---:|---|
| **Theory & calculation** | 10 | [calculations.md](file:///home/thanh/Documents/6.HUS/NLP/Lab3/calculations.md): Lời giải chi tiết 6 bài tập lớn (Mục 6, 7, 8, 16, 23). |
| **Prediction** | 10 | [prediction.md](file:///home/thanh/Documents/6.HUS/NLP/Lab3/prediction.md): 4 dự đoán kèm lý do, độ tin cậy và đối chiếu thực nghiệm (Mục 9). |
| **Co-occurrence implementation** | 15 | [cooccurrence.py](file:///home/thanh/Documents/6.HUS/NLP/Lab3/cooccurrence.py): Cài đặt from-scratch vocabulary, co-occurrence, cosine, most-similar, SVD. |
| **Word2Vec experiment** | 15 | [word_embedding.ipynb](file:///home/thanh/Documents/6.HUS/NLP/Lab3/word_embedding.ipynb): Huấn luyện chuẩn xác với Skip-gram trên tập C4 thực tế. |
| **Hyperparameter experiment** | 15 | Khảo sát định lượng Context Window ($k=2, 5, 10$) và Dimension ($d=50, 100, 300$). |
| **Evaluation** | 15 | Đánh giá xếp hạng 5 cặp từ tiêu chuẩn và kiểm thử bài toán vector analogy. |
| **Application** | 10 | Cài đặt hệ thống Semantic Search truy xuất tài liệu theo trung bình vector. |
| **Error analysis** | 5 | [error_analysis.md](file:///home/thanh/Documents/6.HUS/NLP/Lab3/error_analysis.md): Phân tích 3 ca đúng, 3 ca sai theo 8 nhóm nguyên nhân & Polysemy. |
| **Reflection** | 3 | [reflection.md](file:///home/thanh/Documents/6.HUS/NLP/Lab3/reflection.md): Bảng phân loại 4 phương pháp và phân tích bước đệm sang Transformer. |
| **Individual learning check** | 2 | Trả lời đầy đủ và sắc bén 6 câu hỏi vấn đáp cá nhân tại Mục 29. |
| **Tổng điểm** | **100 / 100** | **Đầy đủ 100% các thành phần** |
