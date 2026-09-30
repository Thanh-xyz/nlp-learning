# LAB 02 — N-GRAM LANGUAGE MODELS, SMOOTHING AND PERPLEXITY

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** I - 2026  
**Giảng viên / TA:** Phạm Ngọc Hải  

---

## 1. Giới thiệu tổng quan

Báo cáo và mã nguồn thực hành này hoàn thành đầy đủ 100% các yêu cầu của **LAB 02: N-gram Language Models, Smoothing and Perplexity**.

---

## 2. Cấu trúc thư mục nộp bài

```text
Lab2/
├── README.md               # Báo cáo tổng quan, hướng dẫn chạy và checklist nghiệm thu
├── calculations.pdf       # Lời giải chi tiết toàn bộ các bài toán tính tay
├── prediction.md           # 5 dự đoán trước thực nghiệm kèm lý do, độ tin cậy và đối chiếu sau chạy
├── ngram_lm.py             # Cài đặt from-scratch N-gram LM & Unit Tests (KHÔNG CHỨA COMMENT)
├── experiments.ipynb       # Jupyter Notebook thực thi toàn bộ thực nghiệm (KHÔNG CHỨA COMMENT TRONG CODE)
├── results.csv             # Bảng tổng hợp định lượng kết quả Perplexity, Next-Word và Sentence Ranking
├── error_analysis.md       # Phân tích định tính & định lượng 2 ca đúng, 2 ca sai theo 8 nhóm nguyên nhân
├── reflection.md           # Trả lời 7 câu hỏi suy ngẫm (Mục 24) và 5 câu hỏi kiểm tra cá nhân (Mục 26)
├── ngram_freq_distribution.png # Biểu đồ phân phối tần số Zipf của Unigram, Bigram, Trigram
└── dataset/                # Đường dẫn liên kết tới tập ngữ liệu C4 (10K - 30K documents)
    └── c4-train.00000-of-01024-30K.json.gz
```