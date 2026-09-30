# PHÂN TÍCH LỖI (ERROR ANALYSIS) — LAB 02

---

## 1. Phương pháp thực nghiệm và thiết lập

- **Mô hình sử dụng:** Bigram và Trigram Language Model với Laplace Smoothing ($Add-1$).
- **Dữ liệu huấn luyện:** 16.000 câu văn bản trích xuất từ tập ngữ liệu web C4 (`c4-train.00000-of-01024-30K.json.gz`).
- **Nhiệm vụ:** Với một ngữ cảnh, mô hình tính phân phối xác suất trên toàn bộ từ vựng $V$ và chọn ra từ có xác suất cao nhất
---

## 2. Các trường hợp dự đoán đúng (Correct Predictions)

### Trường hợp đúng 1 (Correct Case 1)
- **Context:** `in order`
- **Model prediction:** `to`
- **Expected:** `to`
- **Xác định nguyên nhân thành công:**
  - **Collocation / Thành ngữ cố định (Idiomatic Phrasing):** Cụm từ `in order to` là một liên từ mục đích cực kỳ phổ biến trong tiếng Anh. Tần suất xuất hiện đồng thời của cặp `(order, to)` trong tập huấn luyện là rất lớn, vượt trội hoàn toàn so với các từ khác (như `for`, `that`).
  - **Context phù hợp:** Context độ dài 2 từ (`in order`) đủ hẹp và chặt chẽ để loại bỏ sự đa nghĩa.
  - **Smoothing bảo toàn thứ tự:** Laplace smoothing không làm xáo trộn thứ tự của các n-gram có tần số quan sát cao vượt trội.

---

### Trường hợp đúng 2 (Correct Case 2)
- **Context:** `united`
- **Model prediction:** `states`
- **Expected:** `states`
- **Xác định nguyên nhân thành công:**
  - **Tên riêng / Thực thể định danh (Named Entity):** Cụm từ `United States` xuất hiện với mật độ dày đặc trong văn bản web tiếng Anh (C4).
  - **Quan hệ chuyển tiếp mạnh mẽ:** Trong hầu hết các tài liệu tin tức và mô tả địa lý, sau tính từ `united` thì danh từ `states` chiếm tỷ trọng áp đảo so với `nations`, `kingdom`, hoặc `airlines`.
  - **Đại diện thống kê tốt trong dữ liệu:** Dữ liệu huấn luyện 16.000 câu đủ lớn để bao quát các thực thể toàn cầu phổ biến này.

---

## 3. Các trường hợp dự đoán sai (Incorrect Predictions)

### Trường hợp sai 1 (Incorrect Case 1)
- **Context:** `natural language`
- **Model prediction:** `</s>` (kết thúc câu) hoặc `of` ($P = 0.000815$)
- **Expected:** `processing` ($P = 0.000091$)
- **Probability:** Xác suất từ kỳ vọng `processing` bị rơi xuống nhóm xác suất nền của unseen tokens ($P = 0.000091$).
- **Xác định nguyên nhân lỗi (Root Causes):**
  1. **Unseen n-gram & Data Sparsity:**
     Trong tập huấn luyện 16.000 câu C4 tổng quát từ web, cụm chuyên ngành `natural language processing` không xuất hiện (hoặc chỉ xuất hiện tách rời). Cụm `(language, processing)` có tần số quan sát $C = 0$.
  2. **Insufficient training data & Domain Mismatch:**
     Tập ngữ liệu C4 là văn bản web đại chúng (tin tức, diễn đàn, thương mại, du lịch), không phải là tập dữ liệu học thuật máy tính (arXiv, NLP papers). Do đó các thuật ngữ chuyên ngành NLP bị thiếu hụt dữ liệu huấn luyện nghiêm trọng.
  3. **Tác động tiêu cực của Laplace Smoothing:**
     Khi một bigram chưa từng xuất hiện, Laplace smoothing gán tử số là $0 + 1 = 1$, khiến xác suất của `processing` chỉ ngang bằng với hàng nghìn từ vựng ngẫu nhiên khác trong $V$, và bị lấn át bởi token kết thúc câu `</s>` hoặc giới từ `of` (vốn có unigram frequency rất cao).

---

### Trường hợp sai 2 (Incorrect Case 2)
- **Context:** `the cat`
- **Model prediction:** `had` ($P = 0.000182$)
- **Expected:** `was` ($P = 0.000091$)
- **Probability:** Xác suất của `was` sau `cat` thấp hơn `had`.
- **Xác định nguyên nhân lỗi (Root Causes):**
  1. **Context quá ngắn (Short Context Window):**
     Mô hình n-gram chỉ nhìn được 1 hoặc 2 từ phía trước. Ngữ cảnh `the cat` quá chung chung và thiếu hoàn toàn thông tin chủ đề toàn cục (Topic / Semantic context) của đoạn văn. Đứng sau chủ ngữ `the cat` có thể là bất kỳ động từ tiếng Anh nào (`is`, `was`, `had`, `ran`, `sat`, `eats`,...).
  2. **Không có khả năng tổng quát hóa ngữ nghĩa (No Semantic Generalization):**
     Mô hình n-gram biểu diễn từ dưới dạng các ký hiệu rời rạc (discrete discrete symbols / one-hot). Nó không nhận biết được rằng `cat` là một loài động vật sống thường đi kèm với các hành vi như `sleep`, `meow`, `eat`.
  3. **Nhiễu từ vựng và tần suất ngẫu nhiên (Sample Variance):**
     Trong một mẫu dữ liệu con cụ thể, một câu chứa `the cat had a collar` xuất hiện làm tăng count của `had`, trong khi `the cat was sleeping` lại không xuất hiện trong tập train con đó. N-gram hoàn toàn bị lệ thuộc vào sự may rủi của tần số đếm trên mẫu hữu hạn.

---