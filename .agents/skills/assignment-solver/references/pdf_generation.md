# HƯỚNG DẪN XUẤT BÁO CÁO LATEX & CHUYỂN ĐỔI SANG PDF

## 1. Mục Tiêu & Tiêu Chuẩn Chất Lượng PDF

Mọi tài liệu kỹ thuật, phân tích toán học và báo cáo thí nghiệm (`calculations`, `prediction`, `error_analysis`, `reflection`) đều phải được xuất ra định dạng PDF đạt chuẩn xuất bản học thuật:

1. **Hỗ trợ tiếng Việt có dấu hoàn hảo**: Không bị lỗi font, mất dấu hoặc xuất hiện các ký tự vô nghĩa (mojibake).
2. **Công thức toán học chuẩn LaTeX**: Render chính xác các biểu thức ma trận, vector, tích chập, đạo hàm, chỉ số trên/dưới.
3. **Bảng biểu căn chỉnh tự động (No Overflow)**: Bảng dữ liệu không bị tràn ra ngoài lề trang giấy (margin), các cột có độ rộng co giãn tự động và tự ngắt dòng.
4. **Bố cục phân cấp rõ ràng**: Có tiêu đề chính, thông tin môn học, ngày tạo, các đề mục section/subsection và số trang ở chân trang.
5. **Tính nhất quán tuyệt đối**: Nội dung trong PDF phải hoàn toàn trùng khớp với file Markdown nguồn.

---

## 2. Công Cụ & Pipeline Chuyển Đổi

Hệ thống hỗ trợ 2 pipeline chuyển đổi theo thứ tự ưu tiên:

### Pipeline 1: Script Tích Hợp Sẵn Của Skill (Khuyên Dùng)
Skill đã cung cấp sẵn công cụ chuyển đổi độc lập tại `scripts/md_to_pdf.py`. Script này tự động chuẩn hóa cú pháp Markdown, bọc cấu trúc tài liệu LaTeX chuẩn tiếng Việt và gọi trực tiếp trình biên dịch `xelatex`:

```bash
python .agents/skills/assignment-solver/scripts/md_to_pdf.py <file_nguon.md> [-o <file_dich.pdf>] [-t "Tiêu đề tùy chọn"]
```

**Ví dụ thực tế**:
```bash
python .agents/skills/assignment-solver/scripts/md_to_pdf.py lab05/calculations.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py lab05/prediction.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py lab05/error_analysis.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py lab05/reflection.md
```

### Pipeline 2: Pandoc + XeLaTeX (Khi Môi Trường Đã Cài Pandoc)
Nếu môi trường có sẵn Pandoc, có thể sử dụng câu lệnh:
```bash
pandoc input.md -o output.pdf --pdf-engine=xelatex -V mainfont="Times New Roman" -V geometry:margin=0.8in
```

---

## 3. Quy Tắc Định Dạng Markdown Để Xuất PDF Không Lỗi

Để bộ chuyển đổi hoạt động tối ưu và tạo ra văn bản đẹp:

1. **Công thức nội dòng (Inline Math)**:
   - Dùng `$ ... $`: Ví dụ `$h_t \in \mathbb{R}^h$` hoặc `$W_{xh} x_t$`.
   - Tránh để khoảng trắng thừa ngay sau hoặc trước dấu `$`: `$ a + b $` nên viết thành `$a + b$`.
2. **Công thức khối độc lập (Display Math)**:
   - Dùng `$$ ... $$` hoặc `\[ ... \]` đặt trên dòng riêng:
     ```markdown
     $$
     h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
     $$
     ```
3. **Ma trận và Vector**:
   - Dùng môi trường `bmatrix` hoặc `pmatrix`:
     ```markdown
     $$
     W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}
     $$
     ```
4. **Bảng số liệu (Tables)**:
   - Viết bảng chuẩn Markdown với hàng phân cách:
     ```markdown
     | Model | Parameters | Accuracy | F1 Score |
     | :--- | :--- | :--- | :--- |
     | RNN | 128,450 | 0.825 | 0.821 |
     | LSTM | 385,210 | 0.893 | 0.890 |
     ```
5. **Đoạn mã (Code Blocks)**:
   - Sử dụng fenced code blocks ba dấu backtick kèm tên ngôn ngữ (` ```python ` hoặc ` ```text `).

---

## 4. Quy Trình Kiểm Thử & Nghiệm Thu PDF (Verification Protocol)

Sau khi tạo file PDF, agent **bắt buộc phải thực hiện các bước xác minh thực tế**:

1. **Kiểm tra sự tồn tại và kích thước file**:
   - File `.pdf` phải tồn tại tại đúng đường dẫn yêu cầu và có dung lượng hợp lệ (> 10 KB).
2. **Kiểm tra trực quan bằng công cụ `view_file`**:
   - Gọi `view_file` trên file PDF vừa sinh ra để xem ảnh chụp trang (screenshot) và trích xuất OCR.
   - Kiểm tra các trang có bảng biểu hoặc công thức phức tạp để đảm bảo không bị cắt chữ hoặc tràn lề.
3. **Xác nhận tính toàn vẹn ký tự**:
   - Đảm bảo tiếng Việt hiển thị rõ ràng, không bị lỗi font hoặc mất dấu.
4. **Dọn dẹp file tạm**:
   - Tự động xóa các file phụ trợ do TeX sinh ra (`.aux`, `.log`, `.out`), chỉ giữ lại file `.tex` nguồn (nếu cần) và file `.pdf` chính thức.
