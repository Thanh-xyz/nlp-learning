# QUY TẮC TUYỆT ĐỐI: KHÔNG COMMENT TRONG SOURCE CODE

## 1. Nguyên Tắc Cốt Lõi

Trong mọi bài tập được thực hiện bởi agent, **tuyệt đối không được thêm bất kỳ comment hay chú thích nào vào file mã nguồn**.

Mã nguồn phải đạt chuẩn tự tài liệu hóa (self-documenting code):
- Đặt tên biến, hàm, class rõ ràng, phản ánh chính xác ngữ nghĩa và vai trò toán học/kỹ thuật.
- Cấu trúc module và luồng điều khiển mạch lạc, trực quan.
- Mọi diễn giải, chứng minh công thức, giải thích thuật toán, phân tích tham số hoặc hướng dẫn sử dụng **bắt buộc phải được trình bày trong các file tài liệu Markdown (`README.md`, `calculations.md`, v.v.)**, không được đặt trong file code.

---

## 2. Phạm Vi Áp Dụng

Quy tắc này áp dụng cho toàn bộ các file mã nguồn và cấu hình có thể thực thi, bao gồm nhưng không giới hạn:

| Ngôn ngữ / Định dạng | Phần mở rộng file |
| :--- | :--- |
| **Python** | `.py` |
| **Jupyter Notebook** | `.ipynb` (áp dụng cho toàn bộ **code cells**) |
| **C / C++** | `.c`, `.cpp`, `.h`, `.hpp` |
| **Java** | `.java` |
| **JavaScript / TypeScript** | `.js`, `.jsx`, `.ts`, `.tsx` |
| **Shell Script** | `.sh`, `.bash`, `.zsh` |
| **SQL** | `.sql` |
| **Khác** | Bất kỳ file script thực thi nào phát sinh trong quá trình làm bài |

---

## 3. Danh Sách Các Thành Phần BỊ CẤM HOÀN TOÀN

Tuyệt đối không được xuất hiện bất kỳ thành phần nào sau đây trong file code:

1. **Comment một dòng**:
   - Python / Shell: `# comment...`
   - C / C++ / Java / JS: `// comment...`
   - SQL: `-- comment...`
2. **Comment nhiều dòng / khối**:
   - C / C++ / Java / JS: `/* ... */`
   - SQL: `/* ... */`
3. **Docstring hàm, class, module**:
   - Python: Các chuỗi `""" ... """` hoặc `''' ... '''` đứng đầu module, ngay dưới khai báo `class` hoặc `def`.
   - Ngay cả docstring mô tả tham số kiểu Google style / Sphinx / NumPy style cũng **không được phép**.
4. **TODO / FIXME / NOTE**:
   - Các dòng ghi chú dạng `# TODO: implement this`, `# NOTE: temporary fix`.
5. **Comment phân tách khối / tiêu đề section**:
   - `# =========================================`
   - `# Step 1: Preprocessing data`
   - `# -----------------------------------------`
6. **Comment giải thích công thức toán học**:
   - Thay vì `# compute h_t = tanh(W_xh * x_t + W_hh * h_prev + b_h)`, chỉ viết trực tiếp biểu thức tính toán.
7. **Comment ghi chú kết quả chạy thử**:
   - `# Output should be (batch_size, hidden_dim)`

---

## 4. Các Trường Hợp Hợp Lệ (Ngoại Lệ Duy Nhất)

Chỉ có những trường hợp sau đây được coi là cú pháp hợp lệ của chương trình và không bị coi là comment:

1. **Shebang bắt buộc ở dòng đầu tiên của script thực thi**:
   - Ví dụ: `#!/usr/bin/env python3` hoặc `#!/bin/bash` ở dòng 1 của script.
2. **String literals phục vụ runtime thực sự**:
   - Chuỗi định dạng in ấn: `print("Accuracy:", acc)`
   - Biểu thức chính quy: `re.compile(r'^[a-zA-Z0-9]+$')`
   - Dữ liệu text trong dictionary, list, prompt template hoặc schema:
     ```python
     CONFIG = {
         "model_name": "SimpleRNN",
         "loss_fn": "cross_entropy"
     }
     ```
   - Câu lệnh SQL thực thi dạng string: `cursor.execute("SELECT * FROM table")`

---

## 5. Quy Chuẩn Riêng Đối Với Jupyter Notebooks (`.ipynb`)

Notebook thường bao gồm cả **Markdown cells** và **Code cells**:
- **Code cells**: Tuân thủ nghiêm ngặt 100% quy tắc không comment (không `#`, không docstrings).
- **Markdown cells**: Được phép dùng để trình bày cấu trúc bài tập, đề mục thí nghiệm, bảng thông số hoặc giải thích quy trình. Tuy nhiên, nội dung trong Markdown cells phải phục vụ việc trình bày thí nghiệm một cách khoa học, không biến tướng thành nơi copy comment từ code ra.

---

## 6. Công Cụ Tự Động Kiểm Tra (Automated Audit)

Trước khi nghiệm thu hoặc nộp bài, bắt buộc phải chạy script kiểm tra tự động đã tích hợp sẵn trong skill:

```bash
python .agents/skills/assignment-solver/scripts/verify_no_comments.py <đường_dẫn_thư_mục_hoặc_file>
```

**Tiêu chí vượt qua (PASS)**:
- Kết quả trả về `RESULT: ALL CLEAN - 0 comments or docstrings detected.`
- Exit code = 0.
- Nếu exit code = 1, agent phải rà soát và loại bỏ toàn bộ các comment được chỉ rõ theo số dòng trước khi bàn giao.
