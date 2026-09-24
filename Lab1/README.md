# LAB 01 — FROM TEXT PROCESSING TO SEARCH

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** I - 2026  

---

## 1. Giới thiệu tổng quan
Dự án hoàn thiện đầy đủ toàn bộ yêu cầu của **LAB 01: From Text Processing to Search** theo đúng cấu trúc chuẩn quy định tại **Phần 17 (Deliverables)** và barem chấm điểm **Phần 18 (Assessment Rubric)** trong tài liệu `W1.pdf`.

Mục tiêu của bài thực hành:
- **Tính toán lý thuyết:** Hiểu bản chất các đại lượng $c(t, d)$, $tf$, $df$, $idf$, $tfidf$ và $cosine\_similarity$.
- **Dự đoán trước thực nghiệm:** Đưa ra các giả thuyết khoa học về kích thước từ vựng, độ thưa của ma trận và chất lượng tìm kiếm trước khi mở dữ liệu.
- **Tự cài đặt cốt lõi (From Scratch):** Xây dựng các hàm toán học cơ bản không dùng thư viện đen và kiểm thử bằng Unit Tests.
- **Thực nghiệm trên quy mô lớn:** Khảo sát ma trận thưa và thực hiện ablation study trên 30.000 văn bản thực tế từ tập C4 (`dataset/c4-train.00000-of-01024-30K.json.gz`).
- **Ứng dụng & Đánh giá định lượng:** Xây dựng search engine, đo lường các metric retrieval ($P@5$, $Recall@5$, $MRR$), phân tích lỗi (Error Analysis) và rút ra động lực chuyển đổi sang các biểu diễn ngữ nghĩa hiện đại (Embeddings, Transformers).

---

## 2. Cấu trúc thư mục nộp bài

```text
Lab1/
├── README.md             # Tài liệu tổng quan và hướng dẫn tái lập kết quả
├── calculations.md       # Lời giải chi tiết các bài tính tay (Part B: Exercises 1 - 6)
├── prediction.md         # Các dự đoán trước khi chạy thực nghiệm trên 30K docs (Part C)
├── implementation.py     # Cài đặt from-scratch TF-IDF & Cosine Similarity kèm Unit Tests
├── experiments.ipynb     # Jupyter Notebook thực thi toàn bộ thực nghiệm D, E, F, G, H, I, J
├── results.csv           # Bảng kết quả định lượng đánh giá tìm kiếm (Top-5 docs & metrics)
├── reflection.md         # Báo cáo suy ngẫm (Part 16) và trả lời Learning Check (Part 15)
├── dataset/              # Chứa tập dữ liệu 30K documents C4
│   └── c4-train.00000-of-01024-30K.json.gz
├── result/               # Bản sao kết quả đánh giá định lượng
│   └── results.csv
└── W1.pdf                # Đề bài và hướng dẫn thực hành của môn học
```



