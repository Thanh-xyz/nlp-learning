
## Prediction 1 — Về kích thước từ vựng (Vocabulary Size)

**Câu hỏi:** Nếu corpus có 30.000 documents từ văn bản web thực tế, vocabulary (tập các từ vựng phân biệt) sẽ có khoảng bao nhiêu unique terms?

**Dự đoán của em:**
- Em đoán kích thước từ vựng sẽ rơi vào khoảng **150.000 đến 200.000 từ khác nhau**.


## Prediction 2 — Về độ thưa của ma trận TF-IDF (Sparsity)

**Câu hỏi:** Ma trận TF-IDF kích thước $N \times V$ sẽ là ma trận dày (dense) hay ma trận thưa (sparse)? Tỷ lệ các phần tử mang giá trị 0 (zero entries) có thể lớn đến mức nào?

**Dự đoán của em:**
- Ma trận TF-IDF chắc chắn sẽ **cực kỳ thưa (sparse)**. Tỷ lệ số 0 dự kiến phải **trên 99.9%**.

## Prediction 3 — Về chất lượng tìm kiếm (Document Search Quality)

**Câu hỏi:** Với một query bất kỳ của người dùng, các documents đứng đầu kết quả tìm kiếm theo TF-IDF và Cosine Similarity có nhất thiết là các documents gần nghĩa nhất không?

**Dự đoán của em:**
- Các documents đứng đầu chưa chắc đã là các văn bản đúng nghĩa nhất. Chắc chắn sẽ có nhiều trường hợp tìm kiếm bị sai lệch.

