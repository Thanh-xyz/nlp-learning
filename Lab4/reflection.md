# LAB 04 — Báo cáo Tổng kết & Suy ngẫm (Reflection & Learning Synthesis)
---

## 1. Phân tích thực nghiệm cốt lõi: Tại sao hai Representation tạo ra kết quả khác nhau? 

Trong thực nghiệm LAB 04, câu hỏi quan trọng nhất không dừng lại ở việc *"Mô hình nào đạt điểm accuracy cao hơn?"*, mà là:  
**Tại sao hai phương pháp biểu diễn văn bản (TF-IDF vs. Average Word2Vec) lại tạo ra kết quả và hành vi phân loại khác nhau rõ rệt khi kết hợp cùng một bộ phân loại Logistic Regression?**

### 1.1. So sánh đặc tính biểu diễn (Representation Characteristics)

| Tiêu chí | TF-IDF (Term Frequency - Inverse Document Frequency) | Average Word2Vec (Dense Static Embeddings) |
| :--- | :--- | :--- |
| **Bản chất không gian** | **Thưa (Sparse), số chiều rất lớn** (V = 25,000 - 50,000 chiều). Hầu hết các phần tử trong vector bằng 0. | **Dày đặc (Dense), số chiều thấp cố định** (d = 100 - 300 chiều). Mọi phần tử đều là số thực liên tục. |
| **Cơ chế biểu diễn** | **Từ vựng rời rạc (Lexical / Exact match):** Mỗi từ là một tọa độ trực giao độc lập. Không hiểu từ đồng nghĩa. | **Hình học ngữ nghĩa (Continuous Semantic):** Các từ có ngữ cảnh phân phối tương đồng sẽ nằm gần nhau trong không gian vector. |
| **Khả năng bảo toàn thông tin đặc trưng** | **Rất mạnh:** Nhờ trọng số IDF, các từ hiếm nhưng mang tính quyết định cảm xúc (như *"masterpiece", "terrible", "boring"*) được giữ nguyên giá trị riêng biệt và có sức nặng rất lớn. | **Bị triệt tiêu / Pha loãng (Vector wash-out):** Việc lấy trung bình cộng làm tín hiệu của các từ then chốt bị hòa tan vào hàng trăm từ trung tính. |
| **Khả năng phân tách tuyến tính** | Dữ liệu được đưa lên không gian chiều cực lớn giúp siêu phẳng tuyến tính dễ dàng tìm được ranh giới phân tách lớp. | Không gian 100 chiều bị nén quá mức, các văn bản tích cực và tiêu cực bị co cụm lại gần trọng tâm chung. |

---

### 1.2. Giải thích kết quả thực nghiệm

1. **Hiệu năng định lượng:**
   * `TF-IDF + Logistic Regression` đạt **89.60%** Accuracy (F1 = 89.72%).
   * `Average Word2Vec + Logistic Regression` chỉ đạt **84.50%** Accuracy (F1 = 84.55%).
   * TF-IDF vượt trội hơn Word2Vec trung bình gần **5.1%** trên tập kiểm thử IMDb.
2. **Nguyên nhân cốt lõi:**
   * Trong bài toán Sentiment Analysis với văn bản dài (độ dài trung bình 231 từ), cảm xúc tổng thể được quyết định bởi **sự xuất hiện của một vài từ khóa cảm xúc đặc trưng**. TF-IDF được thiết kế hoàn hảo để phát hiện và khuếch đại những từ khóa này thông qua cơ chế trọng số TF * IDF.
   * Ngược lại, Word2Vec là công cụ biểu diễn cấp độ từ rất mạnh nhưng phương pháp tổng hợp tài liệu bằng **trung bình cộng số học đơn giản** là một phép xấp xỉ quá thô sơ. Nó giả định rằng ý nghĩa của văn bản bằng tổng ngữ nghĩa của các từ chia đều cho độ dài, điều này hoàn toàn mâu thuẫn với cấu trúc cú pháp của ngôn ngữ tự nhiên.

---

## 2. Thí nghiệm về trật tự từ & Từ phủ định (Experiment 4 — Word Order: Section 17)

Xét hai câu văn kinh điển:
* S1: `"this movie is good"` (Tích cực)
* S2: `"this movie is not good"` (Tiêu cực)

### 2.1. Kết quả thực nghiệm đo đạc thực tế

Khi đưa vào mô hình Average Word2Vec (100 chiều) đã huấn luyện trên tập dữ liệu IMDb:
1. **Độ tương đồng Cosine (Cosine Similarity):**
   $CosineSim(v_{S_1}, v_{S_2}) = 0.9570$
   Hai vector biểu diễn của hai câu có ý nghĩa trái ngược nhau gần như **trùng khít lên nhau** (tương đồng tới 95.7\%).
2. **Kết quả dự đoán của bộ phân loại:**
   * Dự đoán trên $S_1$: **`positive`** (Đúng)
   * Dự đoán trên $S_2$: **`positive`** (SAI HOÀN TOÀN!)

### 2.2. Câu hỏi: Representation này có hiểu được vai trò của "not" không?

**Khẳng định:** **Average Word2Vec HOÀN TOÀN KHÔNG HIỂU được vai trò của từ "not"**:
* Phép tính vector tài liệu: $x = \frac{1}{n} \sum_{i=1}^{n} w_i$.
* Trong câu $S_2$, từ `"not"` chỉ đơn thuần là một vector thành phần $w_{not}$ được cộng dồn vào cùng với $w_{this} + w_{movie} + w_{is} + w_{good}$.
* Vì phép cộng vector có tính chất giao hoán ($A + B = B + A$), vector của `"not"` không hề có toán tử đảo ngược dấu (negation operator) đối với vector của `"good"`. Hơn nữa, với 3 trong 4 từ giống hệt nhau, vector trung bình của $S_2$ vẫn bị chi phối bởi cụm `"this movie is good"`.
* **Giới hạn bộc lộ:** Đây là bằng chứng thực nghiệm rõ ràng nhất cho thấy cả Bag-of-Words và Average Word2Vec đều bị "mù" trước trật tự từ (word order) và cấu trúc ngữ pháp.

---

## 3. Thí nghiệm mở rộng đặc trưng n-gram (Experiment 5 — n-gram features: Section 18)

Mở rộng từ điển TF-IDF từ **Unigram** sang **Unigram + Bigram**:

| Cấu hình đặc trưng | Kích thước Vocabulary | F1-Score | Thời gian huấn luyện (sec) |
| :--- | :---: | :---: | :---: |
| **Unigram (1, 1)** | 25 000 | 89.72% | 10.12 s |
| **Unigram + Bigram (1, 2)** | 50 000 | 90.45% | 22.79 s |
| **Unigram + Bigram (1, 2) + Linear SVM** | 50 000 | 91.25% | 21.58 s |

### Câu hỏi: Vì sao Bigram giúp phân loại tốt hơn nhưng đồng thời làm feature space lớn hơn?

1. **Vì sao Bigram giúp phân loại tốt hơn?**
   * Bigram giải quyết trực tiếp bài toán trật tự từ cục bộ (local word order). Nó tạo ra các đặc trưng cặp từ có ý nghĩa ngữ nghĩa hoàn chỉnh như *"not good"*, *"very bad"*, *"waste time"*, *"the best"*, *"must see"*.
   * Trong thực nghiệm trên, khi gặp câu *"this movie is not good"*, mô hình Unigram + Bigram đã dự đoán chính xác là **`negative`** vì nó nhận diện được cụm đặc trưng `"not good"` là một thuộc tính độc lập mang trọng số âm mạnh.
2. **Vì sao Bigram làm feature space bùng nổ?**
   * Về mặt toán học tổ hợp, nếu văn bản có kích thước từ vựng unigram là $|V|$, số lượng cặp từ bigram tiềm năng có thể lên tới $|V|^2$. Dù trong thực tế ngôn ngữ có các quy tắc cú pháp hạn chế, số lượng bigram xuất hiện trong tập văn bản vẫn gấp từ 5 đến 10 lần số lượng unigram.
   * Điều này dẫn tới ma trận đặc trưng khổng lồ, tiêu tốn nhiều RAM, đòi hỏi phải áp dụng các kỹ thuật cắt tỉa như `min_df` (tần suất tối thiểu) hoặc `max_features` để tránh bùng nổ tài nguyên tính toán và hạn chế overfitting.

---

## 4. Trả lời bộ câu hỏi kiểm tra cá nhân (Individual Learning Check — Section 25)

Dưới đây là câu trả lời ngắn gọn, chuẩn xác và đúng bản chất cho toàn bộ 6 câu hỏi kiểm tra:

### Câu 1: Tại sao accuracy 95% chưa chắc tốt?
* **Trả lời:** Khi dữ liệu bị mất cân bằng lớp nghiêm trọng (ví dụ: 95% mẫu là Negative, chỉ 5% là Positive), một mô hình "ngớ ngẩn" đoán tất cả các mẫu đều là Negative vẫn đạt Accuracy 95%. Tuy nhiên, mô hình này hoàn toàn tê liệt trên lớp Positive cần tìm (Recall = 0%). Do đó, Accuracy cao trong trường hợp này chỉ là một ảo ảnh thống kê ("bẫy độ chính xác").

### Câu 2: TF-IDF khác average Word2Vec ở điểm nào?
* **Trả lời:** 
  1. **TF-IDF:** Biểu diễn thưa (sparse, nhiều chiều), dựa trên từ vựng rời rạc (lexical), bảo toàn các từ khóa đặc thù có tính phân loại cao nhờ trọng số nghịch đảo tần suất văn bản (IDF).
  2. **Average Word2Vec:** Biểu diễn dày đặc (dense, ít chiều), dựa trên khoảng cách ngữ nghĩa liên tục (semantic), nhưng phép tính trung bình cộng làm triệt tiêu các từ khóa quan trọng và xóa nhòa ranh giới giữa các văn bản dài.

### Câu 3: Tại sao "not good" có thể gây lỗi?
* **Trả lời:** Trong các mô hình Bag-of-Words (Unigram) hoặc Average Word2Vec, từ `"not"` và `"good"` bị xử lý độc lập hoặc cộng gộp số học. Mô hình không hiểu được `"not"` là từ đảo ngữ có chức năng phủ định tính chất của `"good"`, dẫn tới việc nhận diện sai câu mang nghĩa tiêu cực thành tích cực.

### Câu 4: Data leakage xảy ra ở đâu nếu fit TF-IDF trước khi split?
* **Trả lời:** Data leakage xảy ra vì hàm `fit` của TF-IDF tính Document Frequency (DF) và nghịch đảo IDF trên toàn bộ tập dữ liệu (bao gồm cả tập test). Khi đó, phân phối từ vựng và thông tin của tập test đã bị "rò rỉ" vào tập huấn luyện, vi phạm nguyên tắc tập test phải hoàn toàn vô hình trong quá trình học, dẫn tới việc đánh giá hiệu năng bị lạc quan giả tạo.

### Câu 5: Precision và recall khác nhau thế nào?
* **Trả lời:**
  * **Precision (TP / (TP + FP)):** Trả lời câu hỏi *"Trong số những mẫu mô hình khẳng định là Positive, có bao nhiêu phần trăm thực sự đúng?"* (Đo lường độ tin cậy, tránh báo động giả).
  * **Recall (TP / (TP + FN)):** Trả lời câu hỏi *"Trong số tất cả các mẫu Positive thực tế tồn tại, mô hình đã tìm ra được bao nhiêu phần trăm?"* (Đo lường độ bao phủ, tránh bỏ sót).

### Câu 6: Tại sao thêm bigram có thể cải thiện sentiment classification?
* **Trả lời:** Thêm bigram cho phép mô hình chụp lại các cụm hai từ liền kề có ý nghĩa ngữ nghĩa hoàn chỉnh (như các cụm phủ định *"not good"*, *"never like"*, hoặc cụm nhấn mạnh cảm xúc *"very bad"*, *"the best"*), giúp khắc phục một phần nhược điểm mất trật tự từ của Unigram Bag-of-Words.

---

## 5. Mạch nối tri thức từ LAB 01 đến LAB 04 và Bước nhảy vọt sang LAB 05 (Section 29)

---

### 5.2. Hai giới hạn chí tử được bộc lộ tại LAB 04

Sau khi hoàn thành tất cả các thực nghiệm của LAB 04, chúng ta đi đến một kết luận mang tính bản lề của xử lý ngôn ngữ tự nhiên cổ điển:

**TF-IDF ≠ Semantic Understanding**
*TF-IDF chỉ là sự trùng khớp từ vựng bề mặt (lexical matching). Nó hoàn toàn bất lực trước từ đồng nghĩa (synonyms), nghĩa bóng, hoặc ngữ cảnh biến đổi.*

**Word2Vec ≠ Contextual Understanding**
*Word2Vec là static embedding: mỗi từ chỉ có duy nhất một vector cố định trong toàn bộ không gian từ điển, bất kể nó xuất hiện trong ngữ cảnh nào.*

#### Minh chứng cụ thể:
1. **Hiện tượng từ đa nghĩa (Polysemy):**
   * `"bank"` trong *"river bank"* (bờ sông)
   * `"bank"` trong *"financial bank"* (ngân hàng tài chính)
   * Trong Word2Vec, cả hai nghĩa trái ngược này bị ép buộc chia sẻ chung một vector duy nhất!
2. **Hiện tượng phụ thuộc trật tự từ (Word Order & Context Dependency):**
   * `"I like this movie"` vs `"I do not like this movie"`
   * Phép cộng vector đơn giản đánh mất vị trí của từ và cấu trúc ngữ pháp của câu.

---

### 5.3. Tiền đề vững chắc dẫn tới Sequence Models (LAB 05)

Những bế tắc trên của Bag-of-Words và Static Embeddings đặt ra một yêu cầu tất yếu:  
**Chúng ta cần một kiến trúc mô hình có khả năng duyệt qua từng từ theo trình tự thời gian (sequence), liên tục cập nhật trạng thái ẩn (hidden state) dựa trên từ hiện tại và ngữ cảnh của các từ đi trước.**

Đó chính là cánh cửa mở ra **LAB 05: Sequence Models (RNN, LSTM, GRU)**:
* Lần đầu tiên, trật tự từ và khoảng cách ngữ cảnh được mô hình hóa một cách tự nhiên và liên tục.
* Khắc phục hoàn toàn bài toán phủ định và trật tự cú pháp.
* Đặt nền móng tư duy cốt lõi trước khi tiếp cận cơ chế Chú ý (**Self-Attention**) và kiến trúc **Transformer (BERT, GPT)** ở các lab tiếp theo của chương trình học!
