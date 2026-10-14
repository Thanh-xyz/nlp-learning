# LAB 04 — Text Classification: From Text Representation to Supervised NLP


---

## 1. Giới thiệu tổng quan (Overview)

Trong **LAB 04**, các phương pháp biểu diễn trên được đưa vào một bài toán NLP có giám sát thực tế: **Phân loại văn bản (Text Classification)**, cụ thể là phân tích cảm xúc các bài đánh giá phim trên tập dữ liệu chuẩn **IMDb Movie Reviews (50,000 mẫu)**.

Mục tiêu cốt lõi của lab không chỉ là xây dựng một bộ phân loại có độ chính xác cao, mà quan trọng hơn là trả lời được hai câu hỏi khoa học bản chất:
> **"Representation nào phù hợp với bài toán nào, tại sao, và model đang sai ở đâu?"**

---

## 2. Cấu trúc thư mục nộp bài (Deliverables Structure)

Toàn bộ các thành phần bài làm được tổ chức chuẩn hóa và đầy đủ theo đúng yêu cầu trong **Mục 26. Deliverables** của `W4.pdf`:

```text
lab04/
├── README.md               # Báo cáo tổng quan, hướng dẫn chạy, checklist Rubric & AI Statement
├── calculations.md         # Lời giải chi tiết 4 bài tập tính toán (Logits, Sigmoid, Threshold, CM, Trap)
├── prediction.md           # 5 dự đoán trước thực nghiệm theo cấu trúc Prediction/Reason/Confidence
├── classification.ipynb    # Jupyter Notebook hoàn chỉnh, đã chạy qua toàn bộ 50,000 mẫu (14/14 cells executed)
├── preprocessing.py        # Module Python tiền xử lý, tokenization, average Word2Vec & split chống leakage
├── results.csv             # Bảng tổng hợp số liệu định lượng (Accuracy, Precision, Recall, F1, Time)
├── confusion_matrix.png    # Đồ thị Heatmap ma trận nhầm lẫn độ phân giải cao trên tập test 10,000 mẫu
├── error_analysis.md       # Phân tích chuyên sâu các ca dự đoán đúng/sai, phân loại lỗi & trọng số mô hình
└── reflection.md           # Tổng kết tri thức, trả lời 6 câu hỏi kiểm tra và cầu nối sang LAB 05 (Sequence Models)
```

---

## 3. Tóm tắt kết quả thực nghiệm (Executive Summary of Results)

Tất cả các mô hình được huấn luyện trên tập **Train ($35,000$ mẫu - 70%)**, tinh chỉnh trên **Validation ($5,000$ mẫu - 10%)** và đánh giá khách quan trên **Test ($10,000$ mẫu - 20%)** nhằm loại bỏ hoàn toàn nguy cơ **Data Leakage**.

Dữ liệu được trích xuất trực tiếp từ [results.csv](results.csv):

| STT | Mô hình | Phương pháp biểu diễn (Representation) | Thuật toán phân loại | Kích thước Vocabulary | Accuracy | Precision | Recall | F1-Score | Thời gian huấn luyện |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Baseline** | TF-IDF (Unigram) | Logistic Regression | $25,000$ | $89.60\%$ | $88.67\%$ | $90.80\%$ | $89.72\%$ | $10.12\text{s}$ |
| **2** | **Model B** | Average Word2Vec ($100\text{d}$) | Logistic Regression | $53,858$ | $84.50\%$ | $84.27\%$ | $84.84\%$ | $84.55\%$ | $31.74\text{s}$ |
| **3** | **Model C** | TF-IDF (Unigram) | Linear SVM | $25,000$ | $89.17\%$ | $88.88\%$ | $89.54\%$ | $89.21\%$ | $7.58\text{s}$ |
| **4** | **Model D** | TF-IDF (Uni + Bigram) | Logistic Regression | $50,000$ | $90.39\%$ | $89.87\%$ | $91.04\%$ | $90.45\%$ | $22.79\text{s}$ |
| **5** | **Model E** | TF-IDF (Uni + Bigram) | Linear SVM | $50,000$ | **$91.21\%$** | **$90.87\%$** | **$91.62\%$** | **$91.25\%$** | $21.58\text{s}$ |

### Những phát hiện then chốt:
1. **TF-IDF đánh bại Average Word2Vec (~$5.1\%$ chênh lệch F1):**
   * Trong các bài văn bản dài chứa nhiều thông tin nhiễu, các từ biểu cảm then chốt quyết định nhãn (ví dụ: *"masterpiece"*, *"terrible"*) được TF-IDF giữ nguyên và gán trọng số lớn. Trong khi đó, phép trung bình cộng vector của Word2Vec làm "pha loãng" tín hiệu cảm xúc vào hàng trăm từ trung tính.
2. **Thí nghiệm về trật tự từ (Word Order Experiment):**
   * Giữa hai câu $S_1 = \text{"this movie is good"}$ và $S_2 = \text{"this movie is not good"}$, độ tương đồng cosine trong không gian Average Word2Vec lên tới **$0.9570$** ($95.7\%$). Cả hai câu đều bị đoán là `positive`. Phép cộng vector hoàn toàn bất lực trước từ phủ định *"not"*.
   * Ngược lại, khi mở rộng sang **TF-IDF Uni + Bigram**, cụm từ *"not good"* được trích xuất thành một đặc trưng độc lập với trọng số âm mạnh, giúp mô hình dự đoán chính xác $S_2$ là `negative`.
3. **Tính giải thích cao của mô hình tuyến tính (Model Interpretability):**
   * Trọng số $w_i$ của Logistic Regression phản ánh chính xác ngôn ngữ học: Top Positive gồm `great` ($+7.71$), `excellent` ($+5.77$), `perfect` ($+4.98$), `the best` ($+4.54$); Top Negative gồm `bad` ($-8.99$), `worst` ($-7.93$), `awful` ($-6.96$), `the worst` ($-6.30$).

---


## 4. Tuyên bố sử dụng trợ lý AI (AI Assistance Statement)

Tuân thủ quy định **AI Policy (Section 24)** trong đề bài, dưới đây là biên bản minh bạch việc sử dụng công cụ AI:

* **Tool:** Google Antigravity Coding Assistant (Gemini 3.8 Flash High)
* **Purpose:** Hỗ trợ kiểm thử mã nguồn, tính toán ma trận nhầm lẫn tự động, vẽ biểu đồ heatmap độ phân giải cao, hỗ trợ sinh khung script thực nghiệm và định dạng tài liệu báo cáo Markdown.
* **Generated:** Mã nguồn các script thực nghiệm tự động (`run_experiments.py`, `build_notebook.py`), khung đồ thị trực quan hóa `matplotlib`.
* **Modified:** Tinh chỉnh cơ chế trích xuất n-gram, phân tích chất lượng từng ca lỗi thực tế trên các trích đoạn văn bản IMDb, tính toán toán học từng bước giải bài tập lý thuyết.
* **Verified:** Toàn bộ công thức toán học được tính toán và kiểm chứng độc lập bằng tay; toàn bộ 14 cell của notebook đã được thực thi kiểm tra thành công trên 50,000 mẫu dữ liệu thực tế; mọi kết luận khoa học đều được đối chiếu chặt chẽ với nguyên lý xử lý ngôn ngữ tự nhiên.
