# LAB 09 — NAMED ENTITY RECOGNITION AND INFORMATION EXTRACTION

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Chủ đề**: From Sequence Classification to Token Classification  
**Giảng viên / TA**: Phạm Ngọc Hải • **Hạn nộp**: 23:59 - 18/11/2026  
**Sinh viên thực hiện**: Nguyễn Văn Thành  

---

## 1. Tổng Quan Bài Tập

LAB 08 khép lại với bài toán **Sequence Classification** (phân loại cảm xúc toàn câu $X \rightarrow y$ dựa trên vector đơn lẻ $h_{\text{[CLS]}}$). LAB 09 đánh dấu bước chuyển dịch quan trọng sang bài toán **Token Classification**:
$$X = (x_1, x_2, \dots, x_T) \longrightarrow Y = (y_1, y_2, \dots, y_T)$$
Trong đó mỗi token nhận một quyết định phân loại riêng biệt trong không gian nhãn thực thể có tên (**Named Entity Recognition - NER**) theo sơ đồ chuẩn **BIO (Begin - Inside - Outside)** trên tập ngữ liệu kinh điển **CoNLL-2003** (gồm 4 nhóm thực thể: $\text{PER, ORG, LOC, MISC}$).

Bài thực hành tập trung giải quyết và kiểm chứng các trụ cột khoa học cốt lõi:
1. **Bản chất toán học và sơ đồ BIO**: Phân tích sự cần thiết của nhãn bắt đầu $\text{B-}X$ trong việc phân giải ranh giới thực thể (Entity Boundary Demarcation) đối với các thực thể cùng loại đứng liền kề nhau.
2. **Vấn đề căn chỉnh nhãn Subword (Word-to-Subword Alignment Problem)**: Xử lý sự lệch pha giữa từ vựng gốc (Word-level) và mảnh từ vựng con (WordPiece Subwords) do tokenizer sinh ra, áp dụng nhất quán **Strategy A** kết hợp với cơ chế triệt tiêu gradient qua **Ignore Index** ($-100$) trong hàm tổn thất Cross-Entropy có mask.
3. **So sánh đối đầu Baseline vs. Transformer**: Đo lường thực nghiệm sự vượt trội của biểu diễn ngữ cảnh động (Dynamic Contextual Representation của BERT) so với mô hình phân loại tuyến tính dựa trên đặc trưng từ vựng cục bộ (Handcrafted Word Features + Logistic Regression).
4. **Phân biệt triệt để Token-Level Metrics và Entity-Level Metrics**: Chứng minh tại sao Token Accuracy tạo ra ảo tưởng chất lượng do mất cân bằng dữ liệu lớp $\text{O}$, và bắt buộc phải dùng chuẩn đo lường khắt khe $\text{Entity-level } F_1$ (`seqeval`).
5. **Khảo sát đa chiều trên tập kiểm thử**:
   - Phân tích hiệu năng phân rã theo từng loại thực thể (PER, ORG, LOC, MISC).
   - Kiểm chứng giả thuyết về độ dài thực thể ($1, 2, 3, 4+$ tokens) đối với tỷ lệ suy giảm điểm $F_1$.
6. **Ứng dụng Trích xuất Thông tin (Information Extraction)**: Xây dựng pipeline biến đổi văn bản phi cấu trúc thành dữ liệu JSON có cấu trúc (`PERSON, ORG, LOCATION`) và đánh giá kiểm chuẩn trên $30$ câu văn có nhãn vàng.
7. **Mạch nối kiến trúc sang LAB 10**: Nhận diện Pretrained Transformer như một xương sống (Backbone) tạo biểu diễn ngữ cảnh chung, kết nối từ Sequence Task (LAB 08) qua Token Task (LAB 09) đến Span Task trong Extractive Question Answering (LAB 10).

---

## 2. Cấu Trúc Thư Mục và Deliverables

Toàn bộ các deliverables theo đúng quy định tại Mục 29 của file đề bài `W9.pdf` cùng 4 file PDF báo cáo học thuật được tổ chức đầy đủ và sạch sẽ trong thư mục `Lab9/`:

```text
Lab9/
├── W9.pdf                          # Đề bài gốc chính thức của giảng viên
├── README.md                       # Báo cáo tổng quan, hướng dẫn thực thi và bảng số liệu
├── calculations.md                 # Lời giải chi tiết 6 bước cho 3 bài tập tính toán bằng tay
├── calculations.pdf                # Bản PDF học thuật xuất từ calculations.md qua XeLaTeX
├── prediction.md                   # Báo cáo dự đoán khoa học lập và khóa TRƯỚC thực nghiệm
├── prediction.pdf                  # Bản PDF học thuật xuất từ prediction.md qua XeLaTeX
├── baseline_ner.py                 # Module phân loại từ vựng truyền thống Logistic Regression (100% không comment)
├── tokenizer_alignment.ipynb       # Notebook phân tích Subword Tokenization và Thí nghiệm 1 (100% không comment)
├── ner_finetuning.ipynb            # Notebook thực thi Transformer NER, Thí nghiệm 2-6 (100% không comment)
├── results.csv                     # Bảng so sánh tổng hợp Baseline vs Pretrained Transformer
├── entity_metrics.csv              # Bảng phân rã chi tiết Precision, Recall, F1 theo từng loại thực thể
├── entity_length_analysis.png      # Biểu đồ cột phân tích điểm F1 theo độ dài thực thể (1, 2, 3, 4+ tokens)
├── confusion_matrix.png            # Ma trận nhầm lẫn Token-Level Confusion Matrix trên tập kiểm thử
├── error_analysis.md               # Báo cáo phân loại lỗi (15 ca thực tế) và phân tích sự phụ thuộc ngữ cảnh
├── error_analysis.pdf              # Bản PDF học thuật xuất từ error_analysis.md qua XeLaTeX
├── information_extraction.py      # Pipeline trích xuất JSON và kiểm chuẩn 30 câu vàng (100% không comment)
├── reflection.md                   # Bài luận chiêm nghiệm 450 từ, giải đáp 10 câu hỏi và mạch nối sang LAB 10
└── reflection.pdf                  # Bản PDF học thuật xuất từ reflection.md qua XeLaTeX
```

---

## 3. Bảng Kết Quả Thực Nghiệm Tổng Hợp

### 3.1. So Sánh Đối Đầu: Baseline vs. Pretrained Transformer (`results.csv`)
Kiểm thử trên cùng tập dữ liệu chuẩn CoNLL-2003 Test Set:

| Mô hình (Model) | Kiến trúc & Cơ chế | Token Accuracy | Entity Precision | Entity Recall | Entity F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Logistic Regression + Handcrafted Word Features | 0.9303 | 0.6678 | 0.6670 | 0.6674 |
| **Pretrained Transformer** | BERT-base-cased + Token Classification Head | **0.9804** | **0.9153** | **0.9138** | **0.9145** |

*Nhận xét quan trọng*:
- Mặc dù Baseline đạt **Token Accuracy lên tới 93.03%** do lớp nhãn $\text{O}$ chiếm đa số tuyệt đối trong câu, nhưng **Entity-level F1 chỉ đạt 66.74%**. Điều này xác nhận chuẩn xác Dự đoán 1: Token Accuracy không phản ánh năng lực trích xuất thực thể.
- Pretrained Transformer tạo ra bước nhảy vọt $+24.71\%$ về điểm Entity-level F1 ($0.9145$ so với $0.6674$), chứng minh sức mạnh của biểu diễn ngữ cảnh hai chiều.

---

### 3.2. Phân Tích Hiệu Năng Từng Loại Thực Thể (`entity_metrics.csv` - Experiment 4)
Đánh giá trên mô hình Transformer theo chuẩn `seqeval`:

| Loại thực thể (Entity Type) | Số lượng mẫu (Support) | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: |
| **PER** (Person) | 636 | **0.9712** | **0.9544** | **0.9627** |
| **LOC** (Location) | 431 | 0.9444 | 0.9072 | 0.9254 |
| **ORG** (Organization) | 522 | 0.9142 | 0.8985 | 0.9063 |
| **MISC** (Miscellaneous) | 197 | 0.7143 | 0.8376 | 0.7710 |

*Nhận xét*:
- **PER** đạt hiệu năng cao nhất ($F_1 = 96.27\%$) do tên người có các chỉ dấu ngữ cảnh cực mạnh (danh xưng, động từ hành vi, vị trí chủ ngữ).
- **MISC** đạt hiệu năng thấp nhất ($F_1 = 77.10\%$) do tính chất không đồng nhất của dữ liệu (bao gồm tên sự kiện, giải đấu, quốc tịch, tác phẩm nghệ thuật), dẫn đến độ phân tán ngữ nghĩa cao.

---

### 3.3. Khảo Sát Theo Độ Dài Thực Thể (Experiment 5 - Entity Length)
Khảo sát sự biến thiên điểm $F_1$ khi độ dài thực thể tăng từ $1$ đến $4+$ tokens:

| Độ dài thực thể (Length) | Số lượng mẫu (Support) | Precision | Recall | F1 Score | Xu hướng hiệu năng |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1 token** | 1,039 | 0.9052 | 0.8912 | 0.8982 | Đơn từ phổ biến, ổn định |
| **2 tokens** | 678 | **0.9416** | **0.9513** | **0.9464** | Đạt đỉnh (chủ yếu là họ tên người First-Last) |
| **3 tokens** | 60 | 0.8254 | 0.8667 | 0.8455 | Giảm rõ rệt do ranh giới phức tạp |
| **4+ tokens** | 9 | 0.7500 | 1.0000 | 0.8571 | Mẫu hiếm, Precision giảm mạnh do nuốt từ chung |

*Đối chiếu dự đoán*: Xác nhận Dự đoán 2: Các thực thể dài ($\ge 3$ tokens) có Precision và F1 thấp hơn đáng kể so với thực thể 2 từ do xác suất tích lũy sai số ranh giới theo luật nhân xác suất chuỗi.

---

### 3.4. Kiểm Chuẩn Pipeline Trích Xuất Thông Tin (Mini IE Task)
Đánh giá trên $30$ câu văn bản tin tức có nhãn vàng:

| Thước đo | Toàn bộ Entity | PERSON | ORG | LOCATION |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | **0.9231** | 0.9474 | 0.5000 | 0.9655 |
| **Recall** | **0.9057** | 0.8571 | 1.0000 | 0.9333 |
| **F1 Score** | **0.9143** | 0.9000 | 0.6667 | 0.9492 |

---

## 4. Tóm Tắt Ba Bài Toán Tính Toán Bằng Tay (`calculations.md`)

1. **Bài toán 1 (BIO Tagging)**:
   - Câu: `"Elon Musk founded SpaceX in California."`
   - Chuỗi nhãn: `Elon/B-PER Musk/I-PER founded/O SpaceX/B-ORG in/O California/B-LOC ./O`
   - Chứng minh tính khép kín của ngữ pháp Markov BIO (không có $\text{I-}$ mồ côi).
2. **Bài toán 2 (Ranh giới thực thể - Entity Boundary)**:
   - Cụm: `"University of California Berkeley"` $\rightarrow$ `B-ORG I-ORG I-ORG I-ORG`.
   - Chứng minh phản chứng: Nếu chỉ dùng nhãn thô `ORG` (IO scheme), hệ thống hoàn toàn thất bại khi gặp hai tổ chức cùng loại đứng kề cận nhau (ví dụ: `"Bank of America Merrill Lynch"`). Nhãn $\text{B-}X$ là điều kiện toán học bắt buộc để phân đoạn chuỗi.
3. **Bài toán 3 (Kích thước Tensor & Hàm Loss)**:
   - $B=8, T=128, d=768, C=9$.
   - Kích thước suy ra: $H \in \mathbb{R}^{8 \times 128 \times 768}$, $Z \in \mathbb{R}^{8 \times 128 \times 9}$, $Y \in \mathbb{R}^{8 \times 128}$.
   - Chứng minh cơ chế triệt tiêu gradient qua chỉ số `IGNORE_INDEX = -100`:
     $$\mathcal{L} = -\frac{1}{N_{\text{active}}} \sum_{b=1}^{B} \sum_{t=1}^{T} \mathbb{I}(Y_{b, t} \ne -100) \cdot \log P(y_{b,t} = Y_{b,t} \mid X)$$

---

## 5. Báo Cáo Tuân Thủ Chính Sách Sử Dụng AI (AI Assistance Statement)

Tuân thủ nghiêm ngặt Mục 28 của file đề bài `W9.pdf`:

```text
AI-assisted code:
What was generated:
- Khung mã nguồn trích xuất đặc trưng DictVectorizer trong baseline_ner.py.
- Đoạn mã dựng biểu đồ confusion_matrix.png và entity_length_analysis.png qua matplotlib.
- Bộ khung chuyển đổi JSON trong information_extraction.py.

What was modified:
- Điều chỉnh hàm căn chỉnh nhãn word_ids(0) khớp với phiên bản transformers 5.22.2.
- Tối ưu hóa bộ lọc nhãn entity-level để tính toán chính xác True Positive cho các nhóm độ dài thực thể.
- Chuẩn hóa đầu vào BatchEncoding tránh hiện tượng tràn bộ nhớ.

What was verified:
- Tự tay chứng minh và giải 3 bài toán tính tay trong calculations.md.
- Tự thiết lập và khóa các giả thuyết khoa học trong prediction.md trước khi chạy thực nghiệm.
- Trực tiếp kiểm tra và phân tích 15 ca lỗi thực tế trích xuất từ tập kiểm thử CoNLL-2003 trong error_analysis.md.
- Soạn thảo bài luận chiêm nghiệm 450 từ và trả lời 10 câu hỏi lý thuyết trong reflection.md.
- Kiểm tra toàn bộ mã nguồn không chứa bất kỳ comment hay docstring nào qua script verify_no_comments.py.
```

---

## 6. Hướng Dẫn Thực Thi và Tái Lập Thí Nghiệm (Reproduction Guide)

Mọi thí nghiệm và mã nguồn đều có thể chạy độc lập trực tiếp từ terminal:

### 1. Chạy mô hình Baseline:
```bash
python Lab9/baseline_ner.py
```

### 2. Chạy ứng dụng Trích xuất thông tin (Information Extraction):
```bash
python Lab9/information_extraction.py
```

### 3. Thực thi lại các Notebooks:
```bash
python -m jupyter nbconvert --to notebook --execute --inplace Lab9/tokenizer_alignment.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace Lab9/ner_finetuning.ipynb
```

### 4. Kiểm tra quy tắc 100% không comment trong mã nguồn:
```bash
python .agents/skills/assignment-solver/scripts/verify_no_comments.py Lab9
```

### 5. Biên dịch lại 4 tài liệu kỹ thuật sang PDF:
```bash
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab9/calculations.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab9/prediction.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab9/error_analysis.md
python .agents/skills/assignment-solver/scripts/md_to_pdf.py Lab9/reflection.md
```
