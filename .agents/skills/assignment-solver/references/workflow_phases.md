# CHI TIẾT WORKFLOW CÁC GIAI ĐOẠN THỰC HIỆN BÀI TẬP

Để đảm bảo tính khoa học, khả năng tái lập và tuân thủ tuyệt đối đề bài, skill `assignment-solver` yêu cầu agent thực thi theo quy trình 6 giai đoạn tuần tự nghiêm ngặt (Phases A → F).

---

## Phase A — Đọc và Phân Tích Đề Bài

1. **Khảo sát toàn diện tài liệu đề bài**:
   - Nếu là file PDF, đọc toàn bộ tất cả các trang bằng công cụ `view_file` (bao gồm hình ảnh kiến trúc, bảng biểu, công thức toán và các trang ghi chú đặc biệt).
   - Kiểm tra xem trích xuất văn bản có đầy đủ không; nếu có trang chứa bảng hoặc công thức phức tạp, xem trực tiếp ảnh chụp trang tương ứng.
2. **Trích xuất Checklist Yêu Cầu (Tối thiểu 13 mục)**:
   - Mục tiêu bài tập (Learning Objectives).
   - Các câu hỏi lý thuyết cần giải quyết.
   - Các bài tính toán bằng tay (Hand calculations).
   - Những dự đoán phải thực hiện trước experiment (Pre-experiment predictions).
   - Các bước triển khai code và kiến trúc mô hình.
   - Dataset, nguồn tải và yêu cầu tiền xử lý (Preprocessing / Tokenization).
   - Danh sách các thí nghiệm phải thực hiện.
   - Các metrics đánh giá và biểu đồ bắt buộc (Learning curves, Confusion matrix, v.v.).
   - Yêu cầu phân tích lỗi (Error analysis).
   - Yêu cầu tổng kết và chiêm nghiệm (Reflection).
   - Quy định sử dụng AI (AI policy & assistance statement).
   - Danh sách deliverables (Tên file, định dạng, vị trí).
   - Rubric và tiêu chí chấm điểm của giảng viên / môn học.
3. **Phân loại yêu cầu**:
   - `[BẮT BUỘC]`: Các mục có trong rubric hoặc deliverables list.
   - `[TÙY CHỌN]`: Các mở rộng hoặc gợi ý thêm.
   - `[CẦN XÁC NHẬN]`: Các câu hỏi suy luận cá nhân cần người học duyệt.
   - `[BỊ NGHẼN / THIẾU TÀI NGUYÊN]`: Các mục chưa thể chạy ngay do thiếu dữ liệu, GPU hoặc thư viện.

---

## Phase B — Xác Định Cấu Trúc Đầu Ra (Deliverables Mapping)

1. **Khóa cứng danh sách Deliverables**:
   - Lấy danh mục deliverables trong đề bài làm nguồn chuẩn duy nhất.
   - Tuyệt đối không tự ý đổi tên file, không gộp các file lại với nhau, không thay đổi đuôi mở rộng (extension) hoặc chuyển vị trí thư mục nếu đề bài đã quy định.
2. **Quy tắc tạo PDF song hành**:
   - Đối với 4 tài liệu kỹ thuật (`calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md`), **bắt buộc tạo thêm file PDF tương ứng** (`calculations.pdf`, `prediction.pdf`, `error_analysis.pdf`, `reflection.pdf`).
   - Giữ nguyên file `.md` gốc; không xóa hoặc ghi đè file Markdown bằng file PDF.
3. **Trường hợp đề bài không quy định cấu trúc**:
   - Đề xuất một cấu trúc thư mục tối giản, sạch sẽ và nhất quán với quy ước chung của project. Không tự ý tạo các thư mục rác hoặc file phụ trợ thừa thãi.

---

## Phase C — Lập Kế Hoạch Theo Thứ Tự Phụ Thuộc (Dependency Plan)

Lập kế hoạch 11 bước theo đúng trật tự logic khoa học:
1. Chuẩn bị môi trường, kiểm tra tài nguyên (GPU, RAM) và dataset.
2. Hoàn thành phần lý thuyết, tính toán bằng tay (`calculations.md`) và dự đoán trước thực nghiệm (`prediction.md`).
3. Xây dựng pipeline xử lý dữ liệu và các thành phần code module.
4. Chạy các thí nghiệm huấn luyện mô hình.
5. Thu thập kết quả đo đạc số liệu và vẽ biểu đồ (`learning_curves.png`, `confusion_matrix.png`, v.v.).
6. Đánh giá mô hình theo các metrics quy định và lưu vào bảng kết quả (`results.csv`).
7. Trích xuất mẫu thực tế và thực hiện phân tích lỗi (`error_analysis.md`).
8. Viết tổng kết và bài học (`reflection.md`).
9. Hoàn thiện tài liệu tổng quan (`README.md`).
10. Chuyển đổi 4 tài liệu sang PDF (`scripts/md_to_pdf.py`).
11. Rà soát toàn bộ deliverables theo rubric và kiểm tra quy tắc không comment.

> [!CRITICAL]
> **Quy tắc bất di bất dịch**: Dự đoán trong `prediction.md` phải được hoàn thành và khóa lại **trước khi bắt đầu chạy thực nghiệm**. Tuyệt đối không sửa ngược lại nội dung dự đoán sau khi đã thấy kết quả của mô hình.

---

## Phase D — Thực Hiện Bài Tập (Code Implementation & Experiments)

1. **Tuân thủ công nghệ**: Sử dụng đúng ngôn ngữ, framework (PyTorch, TensorFlow, Scikit-learn, v.v.) theo đề bài.
2. **Tính toàn vẹn dữ liệu (Data Integrity)**:
   - Phân chia Train / Validation / Test đúng tỷ lệ quy định.
   - Tuyệt đối ngăn chặn rò rỉ dữ liệu (Data Leakage): Vocabulary, scaler hoặc thống kê chỉ được fit trên tập Train, sau đó áp dụng sang Val và Test.
3. **Khả năng tái lập (Reproducibility)**:
   - Cố định random seed (cho Python, NumPy, PyTorch/TensorFlow, CUDA).
   - Giữ nguyên các siêu tham số (hyperparameters: learning rate, batch size, embedding dim, epochs) khi so sánh giữa các mô hình (ví dụ: so sánh công bằng giữa RNN, LSTM và GRU).
4. **Thực nghiệm trung thực (No Hallucination)**:
   - Chạy code thật, ghi nhận số liệu thật vào `results.csv`.
   - Tuyệt đối không bịa số liệu hoặc giả định kết quả khi chưa thực thi chương trình.
   - Nếu bị giới hạn tài nguyên (không có GPU, dataset quá lớn), phải ghi rõ nguyên nhân và giới hạn số epoch hoặc mẫu thử một cách minh bạch.
5. **Jupyter Notebooks**:
   - Notebook phải có khả năng chạy thông suốt từ đầu đến cuối (`Restart Kernel & Run All Cells`) mà không gây lỗi.
   - Tuân thủ nghiêm ngặt quy tắc không comment trong code cells.

---

## Phase E — Hoàn Thiện Bốn Tài Liệu & Biên Dịch PDF

1. Viết nội dung Markdown cho 4 tài liệu theo đúng chuẩn tại `references/four_documents_spec.md`.
2. Kiểm tra cú pháp công thức LaTeX:
   - Ký hiệu toán học chính xác.
   - Bảng số liệu rõ ràng.
3. Gọi script chuyển đổi:
   ```bash
   python .agents/skills/assignment-solver/scripts/md_to_pdf.py <file.md>
   ```
4. Kiểm tra trực quan bằng `view_file` trên từng file PDF vừa sinh ra để đảm bảo không tràn lề và hiển thị tiếng Việt hoàn hảo.

---

## Phase F — Kiểm Thử Toàn Diện, Rà Soát & Báo Cáo

1. **Kiểm tra cú pháp và khả năng import**: Chạy kiểm thử các hàm và module code.
2. **Kiểm tra tính nhất quán liên tài liệu**:
   - Số liệu trong `results.csv` phải khớp 100% với số liệu trong biểu đồ và bảng so sánh trong báo cáo.
3. **Quét sạch comment trong code (Audit Code Cleanliness)**:
   - Chạy script kiểm tra:
     ```bash
     python .agents/skills/assignment-solver/scripts/verify_no_comments.py <thu_muc_bai_lam>
     ```
   - Xác nhận trạng thái `ALL CLEAN` (0 violations).
4. **Đối chiếu Rubric**: Tự đánh giá đối chiếu với từng mục trong rubric của giảng viên.
5. **Lập Báo Cáo Kết Thúc** theo định dạng chuẩn quy định.
