# BẢNG KIỂM TRA NGHIỆM THU & BÁO CÁO CUỐI CÙNG

Trước khi bàn giao kết quả bài tập cho người dùng, agent bắt buộc phải rà soát và đánh dấu qua từng tiêu chí nghiệm thu dưới đây.

---

## 1. Bảng Kiểm Tra Nghiệm Thu (Pre-Submission Checklist)

### 1.1. Deliverables & Cấu Trúc File
- [ ] Mọi file được yêu cầu trong đề bài đều đã được tạo tại đúng đường dẫn và tên gọi.
- [ ] Không có file nào bị đổi tên, gộp chung hoặc thay đổi phần mở rộng trái với đề bài.
- [ ] Bốn file PDF tương ứng (`calculations.pdf`, `prediction.pdf`, `error_analysis.pdf`, `reflection.pdf`) đã được sinh ra song hành cùng bốn file `.md` gốc.
- [ ] Các file ảnh biểu đồ (`learning_curves.png`, `confusion_matrix.png`, v.v.) mở được và hiển thị dữ liệu rõ ràng.

### 1.2. Quy Tắc Không Comment (Zero-Comment Audit)
- [ ] Đã chạy script kiểm tra tự động:
  ```bash
  python .agents/skills/assignment-solver/scripts/verify_no_comments.py <thu_muc_bai_lam>
  ```
- [ ] Kết quả kiểm tra đạt 100% `ALL CLEAN` (Exit code = 0).
- [ ] Không có comment một dòng (`#`, `//`, `--`).
- [ ] Không có comment nhiều dòng hoặc docstrings dùng làm chú thích trong code.
- [ ] Mọi code cell trong file notebook (`.ipynb`) đều sạch comment.

### 1.3. Tính Đúng Đắn & Tính Nhất Quán Thực Nghiệm
- [ ] Đã chạy code huấn luyện và đánh giá thật (không fake số liệu).
- [ ] Số liệu trong `results.csv` trùng khớp với các giá trị được phân tích trong `README.md` và các file báo cáo.
- [ ] Biểu đồ phản ánh chính xác lịch sử huấn luyện ghi nhận được.
- [ ] Dự đoán trong `prediction.md` được ghi nhận trước khi train model.
- [ ] Mẫu phân tích lỗi trong `error_analysis.md` được trích xuất từ dữ liệu thực nghiệm thật, có tối thiểu số lượng đúng/sai theo đề bài.

### 1.4. Chất Lượng Tài Liệu & PDF
- [ ] Tiếng Việt có dấu hiển thị chuẩn, không lỗi font.
- [ ] Công thức toán học (ma trận, vector, hàm phi tuyến) hiển thị đúng chuẩn LaTeX.
- [ ] Các bảng biểu không tràn khỏi lề trang giấy (margin).
- [ ] Đã kiểm tra trực quan từng trang PDF bằng công cụ `view_file`.

---

## 2. Định Dạng Báo Cáo Cuối Cùng (Final Report Structure)

Báo cáo kết thúc công việc phải tuân theo cấu trúc chuẩn 5 phần:

```markdown
### A. Tổng Quan
- **Bài tập đã thực hiện**: [Tên bài tập, ví dụ LAB 05 — Sequence Models]
- **Cấu trúc đầu ra áp dụng**: [Thư mục đích, ví dụ lab05/]
- **Những yêu cầu chính đã hoàn thành**: [Tóm tắt 3-5 gạch đầu dòng]

### B. Danh Sách Deliverables
| Đường dẫn file | Mục đích / Vai trò | Trạng thái | Bằng chứng kiểm chứng |
| :--- | :--- | :--- | :--- |
| `lab05/calculations.md` | Lời giải tính toán tay | Hoàn thành | File tồn tại, đầy đủ 6 bước |
| `lab05/calculations.pdf` | Bản PDF in ấn của tính toán | Hoàn thành | Đã biên dịch XeLaTeX (XX KB), kiểm tra trực quan |
| `lab05/rnn_cell.py` | Implementation RNN cell | Hoàn thành | Đã import thử, kiểm tra 0 comment |
...

### C. Kết Quả Kiểm Thử (Verification Status)
Phân loại rõ ràng từng hạng mục:
- **PASS**: [Các bài test, script audit, import check đã chạy thành công và có output xác nhận]
- **FAIL**: [Các bài test bị lỗi hoặc không đạt tiêu chuẩn]
- **BLOCKED**: [Các bài test bị nghẽn do yếu tố môi trường/tài nguyên]
- **NOT RUN**: [Các bài test không thực hiện cùng lý do cụ thể]

### D. Kiểm Tra Bốn File PDF
Xác nhận trạng thái của 4 file PDF:
1. `calculations.pdf`: [Đường dẫn, số trang, dung lượng, xác nhận trực quan]
2. `prediction.pdf`: [Đường dẫn, số trang, dung lượng, xác nhận trực quan]
3. `error_analysis.pdf`: [Đường dẫn, số trang, dung lượng, xác nhận trực quan]
4. `reflection.pdf`: [Đường dẫn, số trang, dung lượng, xác nhận trực quan]

### E. Phần Còn Thiếu & Giới Hạn (Limitations & Student Confirmation)
- Liệt kê các nội dung chưa thực hiện hoặc bị hạn chế (ví dụ: GPU memory, giới hạn số epoch).
- Các phần suy luận cá nhân cần người học duyệt hoặc bổ sung trực giác riêng theo AI policy của môn học.
```
