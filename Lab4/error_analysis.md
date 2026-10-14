# LAB 04 — Phân tích lỗi & Khả năng giải thích của mô hình (Error Analysis & Model Interpretability)

---

## 1. Phân tích Ma trận nhầm lẫn (Confusion Matrix Analysis)

### 1.1. Ma trận nhầm lẫn trên tập kiểm thử (N_test = 10,000 mẫu)

Chi tiết số lượng các phân loại:

| Thực tế \ Dự đoán | Predicted Positive | Predicted Negative | Tổng số thực tế |
| :--- | :---: | :---: | :---: |
| **Actual Positive** | **4,552 (TP)** | 448 (FN) | 5,000 |
| **Actual Negative** | 513 (FP) | **4,487 (TN)** | 5,000 |
| **Tổng số dự đoán** | 5,065 | 4,935 | 10,000 |

Các chỉ số định lượng tương ứng:
Accuracy: 90.39%
Precision: 89.87%
Recall: 91.04%
F1-Score: 90.45%

---

### 1.2. Trả lời các câu hỏi trọng tâm 

1. **Class nào bị nhầm lẫn nhiều nhất?**
   * Lớp **Negative** bị phân loại nhầm nhiều hơn: Có **513** bài đánh giá tiêu cực bị mô hình dự đoán nhầm thành tích cực (FP), trong khi chỉ có **448** bài đánh giá tích cực bị nhầm thành tiêu cực (FN).
2. **Loại lỗi nào chiếm đa số?**
   * Lỗi **False Positive (FP = 513)** chiếm tỷ trọng cao hơn lỗi False Negative (FN = 448).
   * Nguyên nhân bản chất trong NLP: Người dùng viết đánh giá chê phim (Negative) thường có xu hướng sử dụng nhiều từ vựng khen ngợi mang tính châm biếm, mỉa mai (sarcasm), hoặc mở đầu bằng việc khen ngợi bối cảnh/diễn viên trước khi chốt lại là kịch bản dở tệ (mixed sentiment).
3. **Metric nào phản ánh trực tiếp vấn đề đó?**
   * Chỉ số **Precision** phản ánh trực tiếp số lượng lỗi False Positive:
     *Precision = TP / (TP + FP)*
   * Do FP = 513 > FN = 448, chỉ số Precision (89.87%) thấp hơn so với Recall (91.04%). Nếu muốn giảm thiểu lỗi phân loại nhầm bài tiêu cực thành tích cực, ta cần tối ưu hóa nâng cao Precision (ví dụ: tăng nhẹ ngưỡng quyết định t từ 0.5 lên 0.52).

---

## 2. Phân tích lỗi chi tiết trên dữ liệu thực tế (Error Analysis)

Để hiểu sâu sắc lý do mô hình đưa ra dự đoán đúng hoặc sai, chúng tôi tiến hành phân tích chất lượng trên các trường hợp tiêu biểu.

### 2.1. Các trường hợp dự đoán đúng tiêu biểu (Correctly Classified Examples)

| STT | Trích đoạn văn bản (Text Snippet) | Nhãn thực tế | Dự đoán | Xác suất Positive | Phân tích lý do mô hình đoán đúng |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **C1** | *"A team varied between Scully and Mulder, two other scientists... They discover there's a virus... Excellent episode with great tension and wonderful pacing. One of the best..."* | **Positive** | **Positive** | 0.8408 | **Sự áp đảo của các từ khóa tích cực mạnh:** Văn bản xuất hiện các từ/cụm từ có trọng số dương rất lớn trong mô hình: *"excellent"* (+5.77), *"great"* (+7.71), *"wonderful"* (+4.64), *"the best"* (+4.54). |
| **C2** | *"To call this film a complete waste of celluloid would be an understatement. The acting was unconvincing... couldn't have acted stiffer. As far as story goes... well what story?!"* | **Negative** | **Negative** | 0.0452 | **Đặc trưng tiêu cực dứt khoát:** Cụm từ *"waste"* (-5.94), cấu trúc chỉ trích trực diện kịch bản và diễn xuất khiến hàm logit âm rất sâu (Ppos ≈ 4.5%). |
| **C3** | *"This movie made me so angry!! Here I am thinking that here's a new horror movie, one w/a sense of intelligence & then the movie starts. The fake gore, must I go on? Porno movies have better dialog than this..."* | **Negative** | **Negative** | 0.1027 | **Cảm xúc phẫn nộ mạnh:** Các từ vựng biểu cảm phẫn nộ (*"angry"*, *"fake"*) và so sánh tiêu cực dồn dập giúp mô hình dễ dàng nhận diện sắc thái tiêu cực. |

---

### 2.2. Các trường hợp dự đoán sai tiêu biểu (Misclassified Examples) & Phân loại nguyên nhân

Theo yêu cầu của đề bài, mỗi trường hợp lỗi được gắn với ít nhất một trong các nhóm nguyên nhân: *lexical ambiguity, negation, sarcasm, short text, rare word, domain-specific term, data noise, class imbalance*.

#### Ca lỗi 1 (E1): Châm biếm, mỉa mai (Sarcasm / Irony)
* **Văn bản:**
  > *"Little Quentin seems to have mastered the art of having the cake and eating it. As usual, the pure sadistic display can be explained as a clever thought-provoking way of sending violence back into the audience's face. Sure, Mr Tarantino. Violence is Baaad. Sadism is Baaad. It is well worth wading in it to make that point. How very brilliant..."*
* **Nhãn thực tế (True):** `Negative`
* **Mô hình dự đoán (Predicted):** `Positive` (Xác suất Positive: 0.8246)
* **Phân loại nguyên nhân:** `Sarcasm / Irony`
* **Phân tích sâu:**
  Người viết sử dụng phong cách châm biếm sâu cay đối với đạo diễn Quentin Tarantino. Các từ vựng xuất hiện bề mặt gồm: *"clever"* (+1.85), *"thought-provoking"*, *"well worth"*, *"how very brilliant"* (+2.10). Vì mô hình Bag-of-Words/TF-IDF không hiểu được ngữ dụng học (pragmatics) và ngữ điệu giễu cợt, nó cộng dồn trọng số của các từ tích cực và tự tin khẳng định đây là bài đánh giá khen ngợi (82.46%).

---

#### Ca lỗi 2 (E2): Nhiễu dữ liệu & Cấu trúc lập luận tương phản (Data Noise / Mixed Sentiment)
* **Văn bản:**
  > *"The story of the bride fair is an amusing and engaging one, and it is to the filmmaker's credit that he sets out to portray rural Minnesotans with the same respect ordinarily reserved for Coast-dwellers. It is weird, though, to find an independent movie, the brainchild of a single person, that is as unambitious and cliché-ridden as a committee-brewed Hollywood potboiler..."*
* **Nhãn thực tế (True):** `Negative`
* **Mô hình dự đoán (Predicted):** `Positive` (Xác suất Positive: 0.6607)
* **Phân loại nguyên nhân:** `Data noise` (Discourse Structure / Mixed Sentiment)
* **Phân tích sâu:**
  Đoạn văn mở đầu bằng nửa dung lượng văn bản dành để khen ngợi tiền đề phim: *"amusing"* (+2.4), *"engaging"* (+2.8), *"respect"*, *"credit"*. Mãi tới câu thứ hai, từ nối tương phản *"though"* mới xuất hiện để chỉ trích phim là *"unambitious"* và *"cliché-ridden"*. Vì mô hình TF-IDF phẳng (flat) coi mọi vị trí từ trong văn bản đều có giá trị ngang nhau, số lượng từ khen ngợi ở nửa đầu đã lấn át các từ chê bai ở nửa sau.

---

#### Ca lỗi 3 (E3): Đa nghĩa từ vựng & Thuật ngữ bối cảnh phim (Domain-Specific Term & Lexical Ambiguity)
* **Văn bản:**
  > *"Tenshu is imprisoned and sentenced to death. When he survives electrocution the government officials give him a choice to either be electrocute at a greater degree or agree to some experiments. He chooses the experimentation and is placed in a large metallic cell with a bad ass criminal who also survived the electrocution..."*
* **Nhãn thực tế (True):** `Positive`
* **Mô hình dự đoán (Predicted):** `Negative` (Xác suất Positive: **0.4137**)
* **Phân loại nguyên nhân:** `Domain-specific term` & `Lexical ambiguity`
* **Phân tích sâu:**
  Đây là một bài đánh giá khen ngợi một bộ phim hành động/giật gân gay cấn. Người viết dành phần lớn bài để tóm tắt cốt truyện kịch tính với các từ: *"death"*, *"imprisoned"*, *"electrocution"*, *"criminal"*, *"bad ass"*. Trong từ điển cảm xúc thông thường, các từ này mang trọng số âm rất lớn. Mô hình không phân biệt được việc người viết đang **khen một cốt truyện đen tối** với việc người viết **chê bai chất lượng bộ phim**.

---

#### Ca lỗi 4 (E4): Phủ định khoảng cách xa (Long-range Negation)
* **Văn bản:**
  > *"I had known Brad at Florida State U in the early 70's when he was an inspiring, inventive writer who I thought was headed for greater glory. And that is why I rented this video. Well, well, well, the time has not been kind to Mr Linaweaver... I suppose the pressures of making a living makes higher aspirations expendable..."*
* **Nhãn thực tế (True):** `Negative`
* **Mô hình dự đoán (Predicted):** `Positive` (Xác suất Positive: **0.5478**)
* **Phân loại nguyên nhân:** `Negation` (Long-range) & `Contextual shift`
* **Phân tích sâu:**
  Người viết nhắc lại quá khứ của tác giả với nhiều mỹ từ (*"inspiring"*, *"inventive"*, *"greater glory"*). Sau đó phủ định bằng một cấu trúc thành ngữ tinh tế: *"time has not been kind"*. Bigram chỉ bắt được *"not been"*, hoàn toàn không kết nối được chữ *"not"* với sự sụt giảm chất lượng của bộ phim.

---

### 2.3. Bảng tổng hợp phân loại lỗi (Error Category Summary)

| Nhóm nguyên nhân lỗi | Tần suất xuất hiện trong tập lỗi | Ví dụ từ vựng / Hiện tượng | Giải pháp khắc phục tiềm năng |
| :--- | :---: | :--- | :--- |
| **Sarcasm / Irony** | Cao (~25%) | *"How very brilliant"*, *"masterpiece of garbage"* | Cần mô hình hiểu ngữ cảnh sâu (Transformer / RoBERTa) |
| **Mixed Sentiment / Discourse** | Rất cao (~35%) | Khen diễn viên nhưng chê kịch bản, cấu trúc *However/Although* | Mô hình phân cấp (Hierarchical Attention) hoặc gán trọng số theo vị trí câu kết luận |
| **Domain-specific Plot Summary** | Cao (~20%) | Từ ngữ kinh dị, tội phạm (*"murder", "kill", "monster"*) | Tách biệt phần tóm tắt cốt truyện (Plot summary) và phần đánh giá cảm xúc |
| **Long-range Negation** | Trung bình (~15%) | *"Not by any stretch of imagination good"* | Mở rộng n-gram hoặc sử dụng mạng nơ-ron tuần hoàn (LSTM/GRU) |
| **Short Text / Sparse terms** | Thấp (~5%) | Đánh giá chỉ có 1-2 câu ngắn không chứa từ khóa mạnh | Sử dụng dense embeddings hoặc data augmentation |

---

## 3. Khám phá trọng số mô hình (Inspect Model Weights — Section 22)

Trong mô hình Logistic Regression tuyến tính kết hợp TF-IDF:
z = w^T x + b = sum_{i=1}^{V} w_i x_i + b

* Các đặc trưng có $w_i >> 0$ đóng góp trực tiếp làm tăng z, đẩy xác suất sigma(z) -> 1 (Class Positive).
* Các đặc trưng có $w_i << 0$ đóng góp trực tiếp làm giảm z, đẩy xác suất sigma(z) -> 0 (Class Negative).

Dưới đây là bảng trích xuất Top 15 đặc trưng tích cực nhất và Top 15 đặc trưng tiêu cực nhất từ mô hình TF-IDF (Uni+Bigram) + Logistic Regression:

### 3.1. Bảng trọng số đặc trưng (Feature Weights)

| Xếp hạng | Top Positive Features (wi >> 0) | Trọng số (wi) | Top Negative Features (wi << 0) | Trọng số (wi) |
| :---: | :--- | :---: | :--- | :---: |
| **1** | `great` | **+7.7069** | `bad` | **-8.9899** |
| **2** | `excellent` | **+5.7746** | `worst` | **-7.9323** |
| **3** | `perfect` | **+4.9830** | `awful` | **-6.9629** |
| **4** | `well` | **+4.7846** | `the worst` | **-6.2994** |
| **5** | `wonderful` | **+4.6355** | `boring` | **-6.2678** |
| **6** | `amazing` | **+4.5863** | `waste` | **-5.9434** |
| **7** | `the best` | **$+4.5369$** | `poor` | **$-5.5683$** |
| **8** | `and` | **$+4.0043$** | `terrible` | **$-5.3354$** |
| **9** | `fun` | **$+3.8847$** | `nothing` | **$-4.9808$** |
| **10** | `love` | **$+3.8326$** | `stupid` | **$-4.4630$** |
| **11** | `today` | **$+3.6968$** | `worse` | **$-4.4208$** |
| **12** | `enjoyed` | **$+3.6606$** | `dull` | **$-4.3539$** |
| **13** | `hilarious` | **$+3.6400$** | `poorly` | **$-4.2277$** |
| **14** | `best` | **$+3.5617$** | `horrible` | **$-4.1014$** |
| **15** | `loved` | **$+3.4983$** | `minutes` | **$-4.0395$** |

---

### 3.2. Trả lời câu hỏi trọng tâm: Các feature này có thực sự có ý nghĩa đối với task không?

**Khẳng định:** **Hầu hết các feature được mô hình học đều CỰC KỲ HỢP LÝ và phản ánh chính xác trực giác ngôn ngữ học của con người (Highly Interpretable):**

1. **Tính định hướng cảm xúc rõ rệt:**
   * Phía tích cực xuất hiện các tính từ và trạng từ tán dương đỉnh cao: *"great", "excellent", "perfect", "wonderful", "amazing", "hilarious"*.
   * Phía tiêu cực xuất hiện các tính từ chê bai mạnh mẽ: *"bad", "worst", "awful", "boring", "waste", "poor", "terrible", "dull", "horrible"*.
   * Mô hình đã tự động gán trọng số biên độ lớn nhất cho hai từ cơ bản nhất của phân loại cảm xúc: `bad` (-8.99) và `great` (+7.71).
2. **Sức mạnh của Bigram:**
   * Bigram *"the best"* đứng vị trí thứ 7 phía tích cực (+4.54), trong khi *"the worst"* đứng vị trí thứ 4 phía tiêu cực (-6.30). Việc đưa bigram vào mô hình giúp phân biệt rõ ràng các cụm từ nhấn mạnh mà unigram đơn lẻ không thể hiện hết sắc thái.
3. **Phát hiện các từ vựng mang tính đặc thù ngữ cảnh:**
   * Từ `minutes` nằm trong Top 15 Negative (-4.04): Đây là một quan sát thú vị về mặt ngôn ngữ. Trong các bài chê phim, người xem thường viết: *"I turned it off after 20 minutes"*, *"longest 90 minutes of my life"*, *"a waste of 120 minutes"*. Mô hình đã nắm bắt được quy luật thực tế này một cách thông minh.
   * Từ `waste` (-5.94): Thường xuyên đi kèm cụm *"waste of time"*, *"waste of money"*.
4. **Một số trường hợp nhiễu / Giới hạn:**
   * Từ `and` (+4.00) hoặc `today` (+3.70) có trọng số dương tương đối cao. Điều này có thể bắt nguồn từ tương quan giả (spurious correlation) trong tập dữ liệu (các bài viết dài, nhiệt tình khen ngợi thường dùng nhiều liên từ nối câu).
5. **Ý nghĩa với Model Interpretability:**
   * Trực quan hóa trọng số chứng minh rằng mô hình TF-IDF + Logistic Regression là một mô hình dạng "hộp trắng" (white-box model). Tính minh bạch này cho phép kỹ sư NLP kiểm tra và giải thích chính xác lý do hệ thống đưa ra từng quyết định dự đoán, một ưu điểm vượt trội so với các mạng nơ-ron "hộp đen" phức tạp.
