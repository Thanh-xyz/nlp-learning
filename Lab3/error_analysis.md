# PHÂN TÍCH LỖI & GIỚI HẠN CỦA EMBEDDING (ERROR ANALYSIS) — LAB 03


## 1. Phân tích các trường hợp tương đồng thành công (3 Successful Cases)

### Trường hợp đúng 1 (Successful Case 1): `doctor` - `physician` / `surgeon`
- **Observed:**
  - `doctor` - `surgeon` (cos = 0.7924)
  - `doctor` - `physician` (cos = 0.7848)
  - `doctor` - `ophthalmologist` (cos = 0.7625)
- **Expected:** Các từ chỉ bác sĩ y khoa, phẫu thuật viên hoặc chuyên gia y tế phải nằm trong Top-5 láng giềng gần nhất của `doctor`.
- **Possible Explanation (Giải thích nguyên nhân thành công):**
  - **Paradigmatic Relation (Quan hệ hoán vị / thay thế):** `doctor`, `physician`, và `surgeon` đều là các danh từ chỉ người làm việc trong ngành y, có thể hoán đổi vị trí trực tiếp cho nhau trong các cấu trúc cú pháp điển hình: `The [doctor/physician/surgeon] performed the examination`, `consult with a [doctor/physician]`.
  - **Phân phối ngữ cảnh tương đồng cao:** Cả hai từ đều chia sẻ các từ ngữ cảnh tần suất cao như `patient`, `hospital`, `prescribed`, `medical`, `treatment`, `clinic`.
- **Evidence from corpus (Bằng chứng trích xuất từ dữ liệu C4):**
  - Cụm từ `consult your doctor or physician` và `licensed physician or surgeon` xuất hiện nhiều lần trong các trang tư vấn sức khỏe của C4.
  - Tần suất quan sát lớn giúp Skip-gram tối ưu hóa các vector này hội tụ về cùng một vùng không gian dày đặc (cluster y khoa).

---

### Trường hợp đúng 2 (Successful Case 2): `car` - `truck` / `vehicle`
- **Observed:**
  - `car` - `truck` (cos = 0.7831)
  - `car` - `vehicle` (cos = 0.7329)
  - `car` - `bike` (cos = 0.7104)
- **Expected:** Các phương tiện giao thông đường bộ phải có độ tương đồng vượt trội so với các thực thể khác.
- **Possible Explanation:**
  - **Quan hệ cùng lớp khái niệm (Co-hyponymy):** `car` và `truck` là hai loài trực thuộc cùng một chi khái niệm rộng hơn là `vehicle`.
  - **Môi trường cú pháp và động từ hành động dùng chung:** Chúng cùng xuất hiện sau các động từ như `drive`, `park`, `repair`, `buy`, `rent`, `wash` và các giới từ như `in the car`, `into the truck`.
- **Evidence from corpus:**
  - Trong các bài viết thương mại và đánh giá xe cộ trên web C4, các câu văn bản dạng `buying a new car or truck`, `parking vehicles and cars` xuất hiện với mật độ cao, củng cố vector gradient theo đúng mô hình phân phối.

---

### Trường hợp đúng 3 (Successful Case 3): `football` - `basketball` / `hockey`
- **Observed:**
  - `football` - `basketball` (cos = 0.8561)
  - `football` - `hockey` (cos = 0.8363)
  - `football` - `championship` (cos = 0.8181)
- **Expected:** Các môn thể thao đồng đội thi đấu giải đấu phải có similarity cao nhất với `football`.
- **Possible Explanation:**
  - **Môi trường chủ đề chặt chẽ (Strong Topical Clustering):** Các từ chỉ môn thể thao này thường xuyên xuất hiện trong các bài báo thể thao, chia sẻ danh sách ngữ cảnh đặc thù như: `team`, `coach`, `league`, `tournament`, `game`, `season`, `scored`, `champions`.
  - **Độ tin cậy của Skip-gram với cửa sổ $k=5$:** Cửa sổ 5 từ đủ rộng để bắt trọn cả tên môn thể thao lẫn các từ vựng ngữ cảnh thể thao xung quanh.
- **Evidence from corpus:**
  - Trong các trang tin thể thao của C4, các câu văn thường liệt kê: `high school football and basketball teams`, `college hockey and football schedules`.

---

## 2. Phân tích các trường hợp thất bại / kết quả bất ngờ (3 Failure & Surprising Cases)

### Trường hợp lỗi 1 (Failure Case 1): `doctor` - `disease`
- **Observed:** `doctor` và `disease` có cosine similarity tương đối cao.
- **Expected:** Về mặt quan hệ ngữ nghĩa học thuần túy, `doctor` (người chữa bệnh) và `disease` (căn bệnh) là hai khái niệm hoàn toàn khác biệt về mặt bản thể (entity person vs condition/pathology), không phải từ đồng nghĩa. Ta kỳ vọng similarity giữa chúng phải thấp hơn nhiều so với cặp từ đồng nghĩa.
- **Root Cause & Possible Explanation:**
  - **Nguyên nhân chính: Topical Association (Quan hệ liên tưởng chủ đề / Syntagmatic relatedness) lấn át Functional Similarity (Quan hệ đồng nghĩa hoán vị / Paradigmatic similarity).**
  - **Tác động của Context Window:** Khi cửa sổ mở rộng từ 2 lên 5 và 10, mô hình đếm mọi từ xuất hiện chung trong cùng một đoạn văn. `doctor` và `disease` liên tục đồng xuất hiện trong các câu mô tả y tế (`Doctors diagnose chronic diseases`, `treating patients with rare diseases`).
  - Word2Vec không phân biệt được bản chất mối quan hệ ngữ nghĩa (không phân biệt được "là đồng nghĩa của nhau" với "thường xuyên nói cùng nhau trong một chủ đề").
- **Evidence from corpus:**
  - Hàng trăm câu trong C4 có cấu trúc: `a doctor who specializes in heart disease`.

---

### Trường hợp lỗi 2 (Failure Case 2): Từ láng giềng của `banana` bị nhiễu bởi các từ nấu ăn ngẫu nhiên
- **Observed:**
  - `banana -> [cornstarch (0.8828), minced (0.8825), tortilla (0.8815), shaker (0.8813), elastic (0.8783)]`
- **Expected:** Các loại hoa quả cùng nhóm như `apple`, `orange`, `fruit` phải có độ tương đồng cao nhất.
- **Root Cause & Possible Explanation:**
  - **Nguyên nhân 1: Domain Bias & Data Imbalance trong tập C4:** Các tài liệu chứa từ `banana` trong mẫu con C4 chủ yếu trích xuất từ các trang web chia sẻ công thức làm bánh ngọt / nấu ăn (cooking/baking recipe blogs).
  - **Nguyên nhân 2: Noisy Data & Recipe Listing Format:** Các trang web công thức nấu ăn thường liệt kê nguyên liệu dưới dạng danh sách ngắn cách nhau bởi dấu phẩy: `banana, cornstarch, minced garlic, flour, tortilla`. Mô hình Skip-gram coi các nguyên liệu đứng cạnh nhau trong danh sách này là ngữ cảnh trực tiếp.
  - **Nguyên nhân 3: Tần suất từ vựng (Frequency Effect):** Tần suất xuất hiện của `banana` trong mẫu nhỏ hơn nhiều so với `doctor` hay `car`, khiến vector của nó bị chi phối bởi một vài công thức làm bánh cụ thể thay vì khái niệm hoa quả tổng quát.
---

### Trường hợp lỗi 3 (Failure Case 3): Thất bại của phép toán Analogy `king - man + woman` trên tập C4
- **Observed:**
  - Truy vấn `most_similar(positive=['king', 'woman'], negative=['man'])` trả về: `['jamie', 'russell', 'drew', 'jack', 'barbara']` (các tên riêng) thay vì `queen`.
- **Expected:** Kết quả kinh điển trong tài liệu lý thuyết là `queen`.
- **Root Cause & Possible Explanation:**
  - **Nguyên nhân 1: Domain Mismatch & Tên riêng (Proper Names):** Trong văn bản web hiện đại đại chúng (C4), từ `King` xuất hiện phần lớn dưới dạng họ của người nổi tiếng (ví dụ: *Martin Luther King*, *Stephen King*, *Larry King*), hoặc tên đường phố, tên trường học. Nó không được sử dụng chủ yếu theo nghĩa chế độ quân chủ hoàng gia (Monarchy) như trong sách lịch sử hoặc bách khoa toàn thư Wikipedia.
  - **Nguyên nhân 2: Insufficient Training Data:** Để các quan hệ hình học tuyến tính tinh tế như quan hệ giới tính hoàng gia ($gender-offset$) tự động căn chỉnh hoàn hảo, Word2Vec đòi hỏi hàng tỷ từ (như mô hình Google News 100B words). Với tập dữ liệu con 52.000 câu, các chiều đặc trưng của vector bị chi phối bởi các phân phối tên riêng phổ biến trên web.

---
