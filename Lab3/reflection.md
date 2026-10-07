# BÁO CÁO SUY NGẪM & KIỂM TRA HỌC TẬP (REFLECTION) — LAB 03

---

### 1. Bảng so sánh các kiến trúc biểu diễn văn bản

Dưới đây là bảng phân loại và so sánh toàn diện 4 trường phái biểu diễn văn bản kinh điển trong lịch sử NLP:

| Representation | Context-dependent? | Sparse / Dense | Một từ có nhiều vector? | Ưu điểm cốt lõi | Nhược điểm chí mạng |
|---|---|---|---|---|---|
| **TF-IDF** (Lab 01) | **Không** (Tĩnh) | **Sparse** ($V$ chiều, hầu hết bằng 0) | **Không** (Mỗi từ chỉ là 1 chỉ số cột trong từ điển) | Đơn giản, giải thích tốt, truy xuất từ khóa chính xác | Orthogonal/Không hiểu ngữ nghĩa đồng nghĩa, chiều khổng lồ |
| **Co-occurrence Matrix** (Lab 03 - Core) | **Không** (Tĩnh) | **Sparse** ($V \times V$ chiều, $> 90\%$ là số 0) | **Không** (Mỗi từ có 1 vector hàng cố định) | Nắm bắt phân phối ngữ cảnh cục bộ trực tiếp từ thống kê | Bùng nổ chiều ($V \times V$), cực kỳ thưa, tốn RAM |
| **Word2Vec (CBOW / Skip-gram)** (Lab 03 - Neural) | **Không** (Tĩnh / Static lookup) | **Dense** (Vector thực dày đặc $50 - 300$ chiều) | **Không** (Mỗi token tương ứng đúng 1 vector duy nhất) | Vector kích thước nhỏ, cosine đo tốt ngữ nghĩa, giải được analogy | Bất lực trước từ đa nghĩa (Polysemy), Out-of-Vocabulary (OOV) |
| **Contextual Embedding (BERT / Transformer)** | **Có** (Động theo từng câu) | **Dense** (Vector dày đặc $768 - 1024$ chiều) | **CÓ** (Mỗi lần xuất hiện trong ngữ cảnh mới có vector khác nhau) | Giải quyết trọn vẹn từ đa nghĩa, nắm bắt ngữ cảnh toàn cục sâu | Chi phí tính toán cực lớn (GPU), kiến trúc phức tạp, khó giải thích |

---

### 2. Tại sao từ `bank` bắt buộc cần Contextual Representation?

- **Bản chất của từ vựng tự nhiên:** Trong ngôn ngữ loài người, một ký hiệu hình thái duy nhất (chuỗi ký tự `b-a-n-k`) có thể mang các ý niệm bản thể học (ontological concepts) hoàn toàn tách biệt:
  - *Nghĩa A:* Tổ chức tín dụng tài chính (`financial institution`).
  - *Nghĩa B:* Vùng đất dốc tiếp giáp bờ nước (`river bank`).
- **Thất bại của biểu diễn tĩnh (Static Representation):**
  - Trong Word2Vec, `bank` chỉ được cấp phát một vị trí hàng duy nhất trong ma trận trọng số. Mô hình buộc phải tối ưu một vector duy nhất để đồng thời giải thích cả hai ngữ cảnh trái ngược nhau. Kết quả là vector bị "lai tạp" (polysemy blending), biến thành trung bình cộng trọng số không chuẩn xác cho bất kỳ ngữ cảnh nào.
- **Lời giải từ Contextual Representation (Self-Attention & Transformer):**
  - Mô hình Transformer không sử dụng bảng tra cứu tĩnh để xuất vector cuối cùng. Nó bắt đầu từ vector biểu diễn khởi tạo, sau đó truyền qua nhiều tầng **Multi-Head Self-Attention**.
  - Trong câu *"I deposited money in the bank"*, cơ chế Attention tính trọng số tương tác giữa token `bank` với các token `money` và `deposit`, kích hoạt các neuron biểu diễn tài chính trong không gian ẩn.
  - Trong câu *"We sat on the river bank"*, token `bank` tương tác mạnh với `river` và `sat`, định hướng vector biểu diễn của nó hội tụ hoàn toàn về miền ngữ nghĩa địa lý.
  - **Kết luận:** Contextual representation là bước chuyển tất yếu từ việc xem từ như *các thực thể độc lập đóng băng* sang *các phần tử động biến đổi theo ngữ cảnh tương tác*.

---

## AI ASSISTANCE STATEMENT

- **Tool:** Antigravity AI Coding Assistant (Gemini 3.8 Flash Engine).
- **Purpose:** 
  1. Hỗ trợ gõ và định dạng công thức toán học LaTeX, cấu trúc tài liệu Markdown chuẩn mực.
  2. Hỗ trợ sinh mã khung chuẩn PEP-8 cho Jupyter Notebook và kịch bản thực nghiệm tối ưu tốc độ đọc file gzip/json.
  3. Kiểm tra tính toàn vẹn và độ chính xác của các đoạn code Python.
