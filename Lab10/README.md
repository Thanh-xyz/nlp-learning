# LAB 10 — EXTRACTIVE QUESTION ANSWERING

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Chủ đề**: From Token Classification to Span Prediction  
**Giảng viên / TA**: Phạm Ngọc Hải • **Hạn nộp**: 23:59 - 25/11/2026  
**Sinh viên thực hiện**: Nguyễn Văn Thành  

---

## 1. Tổng Quan Bài Tập

Nối tiếp LAB 08 (**Sequence Classification** qua `[CLS]`) và LAB 09 (**Token Classification** cho NER qua từng token), LAB 10 hoàn thiện bộ ba tác vụ đọc hiểu kinh điển trên Pretrained Transformer với bài toán **Extractive Question Answering (Span Prediction)**:
$$(Q, C) \longrightarrow (P_{\text{start}}, P_{\text{end}}) \longrightarrow A = C_{i:j}, \quad i \le j$$

Bài thực hành tập trung giải quyết các trụ cột kỹ thuật cốt lõi trên bộ dữ liệu chuẩn **SQuAD v1.1**:
1. **Toán học & Định vị Span**: Xác định chỉ số `(start, end)`, tính hàm mất mát $L = \frac{1}{2}(L_{\text{start}} + L_{\text{end}})$, và phân biệt sự khác nhau giữa Character Index, Word Index và Token Index.
2. **Xử lý Offset Mapping và Sliding Window**: Chuyển đổi nhãn vị trí ký tự `answer_start` sang chỉ số token với `return_offsets_mapping=True`, xử lý ngữ cảnh dài bằng `truncation="only_second"`, `stride`, `return_overflowing_tokens=True` cùng 4 trường hợp biên.
3. **So sánh Baseline Heuristic vs. Transformer**: Đánh giá hiệu năng qua cả **Exact Match (EM)** và **Token-level F1**.
4. **Chuỗi 5 thí nghiệm có kiểm soát**: Khảo sát `max_length` ($128, 256, 384$), đường cong huấn luyện `learning_curves.png`, tương quan trùng lặp từ vựng (Question/Context Matching), xáo trộn thứ tự câu (Sentence Shuffling), và độ dài đáp án (`1`, `2-3`, `4-6`, `7+` tokens).
5. **Ứng dụng Document QA**: Xây dựng hệ thống mini hỏi đáp trực tiếp trên văn bản và đánh giá định lượng trên bộ câu hỏi kiểm chuẩn đa miền.

---

## 2. Cấu Trúc Thư Mục và Deliverables

Toàn bộ các deliverables theo đúng Mục 31 của `W10.pdf` cùng 4 file PDF báo cáo học thuật được lưu tại thư mục `Lab10/`:

```text
Lab10/
├── W10.pdf                         # Đề bài gốc chính thức của giảng viên
├── README.md                       # Tài liệu tổng hợp, thông số tái lập và hướng dẫn chạy
├── calculations.pdf                # Bản PDF học thuật biên dịch từ calculations.md qua XeLaTeX
├── prediction.pdf                  # Bản PDF học thuật biên dịch từ prediction.md qua XeLaTeX
├── tokenizer_alignment.ipynb       # Notebook Offset Mapping, Token Type IDs và 4 ca biên 
├── qa_model.py                     # Module Baseline, ExtractiveQAModel, EM/F1 và Span Decoding ├── qa_finetuning.ipynb             # Notebook chạy 5 thí nghiệm, Shuffling, Error Extraction
├── results.csv                     # Bảng tổng hợp 19 dòng số liệu thực nghiệm
├── learning_curves.png             # Biểu đồ Train/Val Loss và EM/F1 qua 4 epochs
├── em_f1_analysis.png              # Biểu đồ đối chiếu EM vs F1 giữa các mô hình
├── answer_length_analysis.png      # Biểu đồ hiệu năng EM & F1 theo độ dài đáp án
├── error_analysis.pdf              # Bản PDF học thuật biên dịch từ error_analysis.md qua XeLaTeX
├── document_qa.py                  # Hệ thống Mini Document QA và kiểm chuẩn 12 câu đa miền 
└── reflection.pdf                  # Bản PDF học thuật biên dịch từ reflection.md qua XeLaTeX
```

---

## 3. Thông Số Tái Lập Thực Nghiệm (Reproducibility — Mục 30)

```text
Model (Fresh Head Fine-tuning): distilbert-base-uncased
Model (SQuAD Benchmark)       : distilbert-base-cased-distilled-squad
Tokenizer                     : AutoTokenizer (WordPiece Fast Tokenizer)
Dataset                       : rajpurkar/squad (SQuAD v1.1: 87,599 train / 10,570 validation)
Train split (Fresh Head)      : 128 samples (4 epochs)
Validation split (Curves)     : 80 samples
Test/Eval split (Experiments) : 400 validation samples
Max sequence length           : 128 / 256 / 384
Stride                        : 32 / 64 / 128
Batch size                    : 16 (train) / 32 (eval)
Learning rate                 : 5e-5 (fresh head) / 2e-5 (adaptation)
Epochs                        : 4 (Exp 2) / 1 (Exp 1)
Optimizer                     : AdamW
Random seed                   : 42
Hardware                      : CPU (Python 3.14, PyTorch 2.14.1, Transformers 5.22.2)
Training time                 : 230.31s (Exp 2, 4 epochs) | 5.70s / 13.16s / 21.13s (Exp 1)
```

---

## 4. Bảng Kết Quả Thực Nghiệm Tổng Hợp (`results.csv`)

### 4.1. Thí nghiệm 1: Ảnh Hưởng Của `max_length` (Context Length Ablation)
Đánh giá trên $400$ mẫu validation (kèm thời gian 1 epoch thích nghi trên 32 mẫu):

| Max Length | Train Time (s) | Single-Pass EM | Single-Pass F1 | Sliding Window EM | Sliding Window F1 | Số Windows |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | **5.70** | 0.7550 | 0.7786 | **0.8400** | **0.8699** | 757 |
| **256** | 13.16 | 0.8375 | 0.8662 | 0.8375 | 0.8658 | 463 |
| **384** | 21.13 | **0.8400** | **0.8713** | 0.8375 | 0.8688 | 420 |

*Kết luận*: Tăng `max_length` không luôn cải thiện chất lượng vô điều kiện. Ở chế độ cắt đơn (Single-Pass), `128 -> 256` tăng mạnh EM (`+8.25%`) do tránh mất đáp án cuối đoạn, còn `256 -> 384` gần như bão hòa (`+0.25%`) dù tốn thêm $60\%$ thời gian. Khi dùng Sliding Window (`stride`), `max_length = 128` đạt EM tốt nhất (`0.8400`) nhờ cô lập ngữ cảnh gọn gàng, giảm pha loãng chú ý.

---

### 4.2. Thí nghiệm 2: Learning Curves Huấn Luyện Từ Head Mới (`distilbert-base-uncased`)
| Epoch | Train Loss | Val Loss | Val EM | Val F1 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 5.3048 | 5.0722 | 0.0500 | 0.0974 |
| 2 | 4.2923 | 4.1760 | **0.1000** | 0.1170 |
| 3 | 3.4204 | **3.7401** | **0.1000** | **0.1187** |
| 4 | **2.7445** | 3.9062 | 0.0750 | 0.0923 |

---

### 4.3. Thí nghiệm 3: So Sánh Đối Đầu Baseline vs. Transformer (`em_f1_analysis.png`)
| Mô hình (Model) | Exact Match (EM) | Token-level F1 | Thời gian (s) |
| :--- | :---: | :---: | :---: |
| **Baseline (Keyword Overlap)** | 0.0000 | 0.1393 | 0.04 |
| **DistilBERT (Fresh Head, 128 ex)** | 0.0925 | 0.1203 | 230.31 |
| **Fine-tuned Transformer (SQuAD)** | **0.8400** | **0.8682** | 55.01 |

---

### 4.4. Thí nghiệm 4 & Kiểm Thử Xáo Trộn Câu (Question/Context Matching & Sentence Shuffling)
| Nhóm kiểm thử | Mô hình | Exact Match (EM) | Token-level F1 |
| :--- | :---: | :---: | :---: |
| **High Lexical Overlap** ($n=106$) | Baseline | 0.0000 | 0.1884 |
| **High Lexical Overlap** ($n=106$) | Transformer | **0.8585** | **0.8928** |
| **Low Lexical Overlap** ($n=86$) | Baseline | 0.0000 | 0.0831 |
| **Low Lexical Overlap** ($n=86$) | Transformer | **0.8023** | **0.8392** |
| **Original Context** ($n=400$) | Transformer | **0.8400** | **0.8682** |
| **Shuffled Context** ($n=400$) | Transformer | 0.8350 | 0.8632 |

---

### 4.5. Thí nghiệm 5: Hiệu Năng Theo Độ Dài Đáp Án (`answer_length_analysis.png`)
| Độ dài đáp án (Tokens) | Số mẫu (Support) | EM | F1 | Chênh lệch ($\text{F1} - \text{EM}$) |
| :---: | :---: | :---: | :---: | :---: |
| **1 token** | 182 | **0.8681** | 0.8733 | **0.0051** |
| **2–3 tokens** | 200 | 0.8150 | 0.8606 | 0.0456 |
| **4–6 tokens** | 16 | 0.8125 | 0.8899 | **0.0774** |
| **7+ tokens** | 2 | 1.0000 | 1.0000 | 0.0000 |

---

### 4.6. Thống Kê Wrong Start vs. Wrong End & Ứng Dụng `document_qa.py`
- Trên $400$ mẫu SQuAD: `#wrong_start` = **$8$**, `#wrong_end` = **$7$**, `#wrong_span` = **$49$**.
- Trên tập kiểm chuẩn $12$ câu hỏi đa miền của `document_qa.py` (Transformer Architecture, SQuAD, HUS FAQ): **EM = 0.8333**, **F1 = 0.9402**.

---

## 5. Báo Cáo Sử Dụng AI (AI Assistance Statement — Mục 29)

```text
AI-assisted code:
What was generated:
- Khung mã nguồn lớp ExtractiveQAModel và thuật toán decode_best_span trong qa_model.py.
- Đoạn mã dựng 3 biểu đồ matplotlib (learning_curves.png, em_f1_analysis.png, answer_length_analysis.png).
- Bộ khung thực thi kiểm chuẩn JSON trong document_qa.py.

What was modified:
- Điều chỉnh logic kiểm tra token_type_ids tương thích với cả BERT và DistilBERT trên transformers 5.22.2.
- Bổ sung kiểm soát đồng thời hai chế độ Single-Pass Truncation và Sliding Window Stride trong Thí nghiệm 1.
- Tinh chỉnh kích thước batch và tập dữ liệu con phù hợp với thực thi thuần CPU.

```

---

## 6. Hướng Dẫn Thực Thi và Tái Lập (Reproduction Commands)

```bash
python -m jupyter nbconvert --to notebook --execute --inplace Lab10/tokenizer_alignment.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace Lab10/qa_finetuning.ipynb
python Lab10/document_qa.py
python .agents/skills/assignment-solver/scripts/verify_no_comments.py Lab10
```
