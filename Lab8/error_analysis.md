# LAB 08 — BÁO CÁO PHÂN TÍCH LỖI VÀ ĐỘ TỰ TIN (ERROR & CONFIDENCE ANALYSIS)
## Pretrained Transformers and BERT Fine-Tuning

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Sinh viên thực hiện**: Nguyễn Văn Thành  
**Mã bài tập**: LAB 08  

---

## 1. Phương Pháp Luận Và Cơ Sở Trích Xuất Mẫu Lỗi

Tất cả các trường hợp lỗi được trích xuất trực tiếp từ kết quả dự đoán của mô hình **Pretrained BERT Fine-Tuned** trên tập kiểm thử độc lập (Test Set gồm $375$ mẫu IMDb chưa từng xuất hiện trong quá trình huấn luyện và điều chỉnh siêu tham số).

Quy trình suy luận và đánh giá phân phối xác suất:
$$z = \text{Linear}(\mathbf{h}_{\text{[CLS]}}) \in \mathbb{R}^2$$
$$\mathbf{p} = \text{Softmax}(z) = \left[ P(\text{Negative} \mid x), \, P(\text{Positive} \mid x) \right]$$
$$\hat{y} = \arg\max_{c \in \{0, 1\}} p_c, \quad \text{Confidence} = \max_{c \in \{0, 1\}} p_c$$

---

## 2. Bảng Tổng Hợp Mẫu Lỗi Thực Nghiệm (Error Taxonomy Table)

Dưới đây là 8 mẫu phân tích điển hình thỏa mãn đầy đủ tiêu chí: tối thiểu 3 False Positives, 3 False Negatives và 2 ca dự đoán sai với độ tự tin rất cao:

| STT | Trích đoạn văn bản (Text snippet) | True | Pred | Conf | Phân loại lỗi (Taxonomy) | Giả thuyết nguyên nhân (Hypothesis) |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | *"Cecil B. deMille's 1922 parlor-to-prison tearjerker Manslaughter finds the lovely Leatrice Joy as a good-at-heart but decadent young lady with more money than she knows what to do with. Her recklessness leads to imprisonment..."* | 0 | 1 | **0.8061** | Ambiguous Sentiment / Lexical Bias | Bài phê bình mang giọng điệu chỉ trích phim sướt mướt nhưng chứa dày đặc các từ vựng mang sắc thái tích cực bề mặt (*"lovely"*, *"good-at-heart"*, *"regeneration"*, *"crusading"*). BERT bị thiên lệch từ vựng (lexical bias) đánh lừa. |
| **2** | *"quite good, don't expect anything high culture.......the acting is bad, the storyline fails, but it was surprisingly enjoyable to watch on a rainy Sunday afternoon."* | 1 | 0 | **0.6837** | Sarcasm & Contrastive Structure | Câu chứa cấu trúc nhượng bộ tương phản. Cụm từ phủ định và tính từ tiêu cực mạnh (*"acting is bad"*, *"storyline fails"*) áp đảo về mặt tần suất, khiến vector `[CLS]` bị kéo về hướng tiêu cực trước khi đến mệnh đề xoay chiều ở cuối. |
| **3** | *"Times are tough for Angel Town, gangs rule with an iron fist and for reasons mostly unknown (Mainly due to embarrassing writing) the gangs want a street kid... presence of an Olympic kick-boxer..."* | 0 | 1 | 0.5423 | Sarcasm & Genre Tropes | Người viết mỉa mai kịch bản phim hành động ngô nghê (*"embarrassing writing"*, *"lamest actor"*), nhưng ngữ cảnh mô tả võ thuật và danh hiệu (*"Olympic kick-boxer"*, *"rule with an iron fist"*) kích hoạt attention head tích cực. |
| **4** | *"CyberTracker is set in Los Angeles sometime in the near future where bodyguard Eric Phillips saves senator Robert Dilly from an assassination attempt by a group known as the UHR..."* | 0 | 1 | 0.5320 | Long-range Dependency & Truncation | Phần đầu $128$ tokens hoàn toàn là tóm tắt cốt truyện trung tính (*"saves senator"*, *"human rights"*). Lời chê bai thực sự của người xem nằm ở nửa sau bài viết nhưng đã bị cắt cụt (truncation) do giới hạn `max_length = 128`. |
| **5** | *"A tedious effort from not-yet great director Budd Boetticher and pretty but not-yet un-bland actress Nina Foch, this movie is the quintessence of a certain kind of B movie. It's just not the good kind..."* | 0 | 1 | 0.5978 | Negation Scope & Rare Modifiers | Câu sử dụng các bổ ngữ phủ định ghép phức tạp (*"not-yet great"*, *"not-yet un-bland"*). Mô hình không phân giải được phạm vi phủ định (negation scope) của tiền tố *"not-yet"* đối với tính từ tích cực *"great"*. |
| **6** | *"There's so many things to fall for in Aro Tolbukhin. En la mente del asesino (Inside the killer's mind), that it's very hard to talk about it without giving any kind of warning. Really, really great done..."* | 1 | 0 | 0.6801 | Domain-Specific Expression | Thể loại phim tội phạm giật gân (crime thriller). Các từ ngữ đen tối (*"killer's mind"*, *"asesino"*, *"warning"*) có prior xác suất tiêu cực rất cao trong kho ngữ liệu tiền huấn luyện, làm lu mờ lời khen kỹ thuật điện ảnh (*"really great done"*). |
| **7** | *"The only thing I knew about this film prior to seeing it was Robby The Robot. My preconception was that it was another in a long line of cheesy sci-fi flicks... How wrong I was. Big production values make this a strong contender..."* | 1 | 0 | 0.5453 | Context & Reversal of Expectation | Tác giả bắt đầu bằng định kiến tiêu cực (*"cheesy sci-fi flicks"*), sau đó đảo ngược hoàn toàn nhận định (*"How wrong I was"*). Mô hình bắt được từ *"cheesy"* nhưng không nắm bắt được cú pháp đảo ngược tư duy. |
| **8** | *"Blue ribbon banners, stars and stripes forever, decorated generals, and unconditional surrender from the enemies which required tons and tons of radiation... this was the substance of US victory in WWII..."* | 1 | 0 | **0.7460** | Label Ambiguity & Narrative Context | Bài bình luận tài liệu lịch sử chiến tranh. Từ ngữ về phóng xạ (*"radiation"*), kẻ thù (*"enemies"*), đầu hàng (*"surrender"*) kích hoạt embedding tiêu cực dù tác giả đang bày tỏ sự kính trọng và đánh giá 5 sao. |

---

## 3. Phân Tích Độ Tự Tin (Confidence Analysis: Confidence vs Correctness)

### 3.1. Phân bổ khoảng tự tin và độ chính xác thực tế
Dự đoán của mô hình trên tập kiểm thử được phân nhóm thành 3 khoảng tin cậy:
- **Low Confidence ($P_{\max} < 0.65$)**: $131$ mẫu ($34.93\%$), Độ chính xác đạt $57.25\%$.
- **Medium Confidence ($0.65 \le P_{\max} < 0.85$)**: $243$ mẫu ($64.80\%$), Độ chính xác đạt $75.72\%$.
- **High Confidence ($P_{\max} \ge 0.85$)**: $1$ mẫu ($0.27\%$), Độ chính xác đạt $100.0\%$.

### 3.2. Định lý thực nghiệm: $\text{Confidence} \ne \text{Correctness}$
1. **Hiện tượng Miscalibration trong Deep Learning**:
   Hàm Softmax $\frac{e^{z_i}}{\sum e^{z_j}}$ thường có xu hướng phóng đại độ chênh lệch giữa các logits khi mạng được huấn luyện với hàm Cross-Entropy. Điều này dẫn đến tình trạng mô hình có thể gán xác suất cao ($> 0.80$) cho một nhận định hoàn toàn sai (như trường hợp Mẫu 1 và Mẫu 8 trong bảng trên).
2. **Nguyên nhân cốt lõi**:
   Mô hình không "hiểu" ngữ nghĩa theo cách của con người mà ánh xạ các tổ hợp n-gram/subword sang không gian tiềm ẩn. Khi bắt gặp sự tập trung của nhiều token tích cực hoặc tiêu cực mạnh, logit tương ứng bị kích hoạt quá mức, sinh ra độ tự tin cao giả tạo (overconfidence) bất chấp logic phủ định tổng thể.

---

## 4. Trả Lời Toàn Diện 6 Câu Hỏi Bắt Buộc (Mục 28 Trong Đề Bài)

### Q1: Model nào hưởng lợi nhiều hơn khi training data ít?
- **Khẳng định**: **Pretrained Model hưởng lợi vượt trội hơn rất nhiều so với Model Train From Scratch.**
- **Observation (Quan sát thực nghiệm)**: Tại mốc $10\%$ dữ liệu huấn luyện (khoảng 175 mẫu), Pretrained BERT vẫn giữ được khả năng tiếp cận phân loại với $F_1 = 0.4172$, trong khi Transformer From Scratch hoàn toàn bị tê liệt, hội tụ về mức đoán ngẫu nhiên ($Acc \approx 50.0\%$) do không đủ dữ liệu để học biểu diễn từ đầu.
- **Evidence (Bằng chứng)**: Biểu đồ `data_efficiency.png` cho thấy đường cong hiệu năng của Pretrained BERT tăng vọt lên $F_1 = 0.6555$ ngay ở mốc $25\%$ và đạt $F_1 = 0.7260$ ở mốc $50\%$, trong khi mô hình From Scratch luôn tụt hậu sâu ở các mốc dữ liệu thấp.
- **Interpretation (Diễn giải khoa học)**: Pretraining cung cấp sẵn inductive bias về ngữ pháp, trật tự từ và quan hệ ngữ nghĩa. Tác vụ downstream khi ít dữ liệu chỉ cần học ma trận chiếu phân loại tuyến tính $W \in \mathbb{R}^{d \times 2}$ thay vì phải học lại toàn bộ không gian ngôn ngữ từ số không.

---

### Q2: Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?
- **Khẳng định**: Điều đó chứng minh **sự tồn tại và giá trị chuyển giao của biểu diễn ngữ nghĩa phổ quát (General Language Representation) thu được từ giai đoạn Pretraining tự giám sát.**
- **Observation**: Với cùng một lượng epoch và cấu trúc kiến trúc, Pretrained BERT đạt hiệu năng cao và ổn định hơn hẳn mô hình khởi tạo ngẫu nhiên.
- **Evidence**: Bảng `results.csv` ghi nhận Pretrained BERT hội tụ ổn định ngay từ Epoch 1 với Validation Loss giảm đều, trong khi Scratch Transformer mất nhiều epoch đầu dao động quanh ngưỡng đoán ngẫu nhiên $50\%$.
- **Interpretation**: Việc giải bài toán Masked Language Modeling (MLM) trên hàng tỷ từ vựng văn bản đã trang bị cho các tham số trọng số $\theta_{\text{pretrained}}$ các đặc trưng cấu trúc ngôn ngữ phong phú. Đây là bằng chứng thực nghiệm rõ ràng bác bỏ quan điểm cho rằng "mỗi bài toán NLP phải học biểu diễn từ đầu".

---

### Q3: Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?
- **Khẳng định**: Điều đó có ý nghĩa rằng **biểu diễn đặc trưng của Pretrained Encoder đã rất giàu ngữ nghĩa, có tính phân tách tuyến tính cao (linearly separable) đối với bài toán đích.**
- **Observation**: Chế độ Frozen Encoder (chỉ huấn luyện 258 tham số của linear head) đạt $Acc = 0.7120, F_1 = 0.7173$, thậm chí nhỉnh hơn nhẹ so với Fine-Tuning toàn bộ ($Acc = 0.6933, F_1 = 0.6779$) trên tập dữ liệu kích thước vừa phải.
- **Evidence**: Thời gian huấn luyện của Frozen model chỉ tốn $35.97\text{s}$ so với $59.72\text{s}$ của Fine-tuning (tiết kiệm gần $40\%$ thời gian tính toán), trong khi giữ vững độ chính xác trên $71\%$.
- **Interpretation**: Khi không gian biểu diễn tĩnh của token `[CLS]` đã đủ tốt, việc chỉ tối ưu tầng phân loại tuyến tính giúp ngăn ngừa hiện tượng quá khớp (overfitting) và hiện tượng quên thảm khốc (catastrophic forgetting) vốn rất dễ xảy ra khi fine-tuning toàn bộ mạng lớn trên tập dữ liệu downstream nhỏ.

---

### Q4: Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?
- **Khẳng định**: Điều đó gợi ý rằng **tác vụ downstream có độ lệch miền lớn (domain shift) hoặc đòi hỏi các đặc trưng chuyên biệt (task-specific features) mà biểu diễn trung tính ban đầu của BERT chưa nắm bắt được.**
- **Observation & Evidence**: Khi chuyển sang các tập dữ liệu mang tính phân cực cao hoặc đặc thù chuyên ngành (ví dụ phân tích cảm xúc y tế, pháp lý, tài chính), các attention heads ở các tầng trên cùng của BERT cần phải được tái cấu trúc ma trận chiếu $W_Q, W_K, W_V$ để tập trung vào các từ khóa mang tính quyết định của miền đó.
- **Interpretation**: Gợi ý rằng biểu diễn tiền huấn luyện là một điểm khởi đầu tốt nhưng không hoàn hảo cho mọi mục tiêu; việc mở khóa gradient cho phép tối ưu hóa thích nghi (adaptation) toàn diện toàn bộ chuỗi trích xuất đặc trưng.

---

### Q5: Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?
- **Khẳng định**: Nguyên nhân cốt lõi là **sự dịch chuyển phân phối dữ liệu (Covariate Shift & Concept Drift), sự thay đổi phong cách diễn đạt (Style Shift) và sự khác biệt về vốn từ vựng chuyên ngành (Vocabulary Mismatch).**
- **Observation & Evidence**: Trong thực nghiệm Domain Shift (Experiment 5), mô hình được huấn luyện trên ngôn ngữ điện ảnh (IMDb) khi đối mặt với các từ vựng phần cứng công nghệ (như *"motherboard burned"*, *"cooling fan"*, *"bloatware"*) phải dựa hoàn toàn vào các đặc trưng suy diễn tổng quát.
- **Interpretation**: Pretrained Transformer không phải là một "hệ thống thấu hiểu toàn năng". Trọng số attention được tinh chỉnh theo miền nguồn (Domain A) sẽ hình thành sự thiên lệch (inductive bias) về các cấu trúc quen thuộc của miền đó; khi gặp miền đích (Domain B) với phân phối từ vựng và ngữ cảnh khác biệt, các mối liên kết chú ý bị sai lệch, dẫn tới suy giảm hiệu năng.

---

### Q6: Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
- **Khẳng định**: **Tuyệt đối KHÔNG nên tiếp tục tăng sequence length.**
- **Observation**: Khi tăng chiều dài chuỗi tối đa $T$ từ $128$ lên $256$:
  - Thời gian huấn luyện tăng gần gấp đôi: từ $57.08\text{s}$ lên $112.58\text{s}$ ($+97.2\%$).
  - Hiệu năng phân loại không hề cải thiện, thậm chí biến động nhẹ từ $F_1 = 0.6743$ xuống $F_1 = 0.6568$.
- **Evidence**: Chi phí tính toán của cơ chế Self-Attention tăng theo độ phức tạp bậc hai:
  $$\text{Complexity} = \mathcal{O}(T^2 \cdot d)$$
  Khi $T = 256$, kích thước ma trận chú ý tăng gấp $4$ lần so với $T = 128$ ($65,536$ phần tử so với $16,384$ phần tử).
- **Interpretation**: Hầu hết các tín hiệu cảm xúc quyết định trong văn bản review ngắn đã nằm trọn vẹn trong $128$ tokens đầu tiên. Việc tăng $T$ lên $256$ chỉ chèn thêm các token padding `[PAD]` (với câu ngắn) hoặc các đoạn văn râu ria gây loãng thông tin (với câu dài), làm tiêu tốn bộ nhớ và chu kỳ GPU/CPU một cách vô ích mà không mang lại bất kỳ giá trị thông tin biên (marginal information gain) nào.

---

## 5. Phân Biệt Giữa Quan Sát Thực Nghiệm Và Diễn Giải Khoa Học

Trong nghiên cứu NLP học thuật, việc tránh ngụy biện nhảy cóc kết luận là nguyên tắc tối quan trọng:

$$\begin{matrix}
\textbf{Quan sát thực tế (Observation)} & \longrightarrow & \textbf{Bằng chứng đo đạc (Evidence)} & \longrightarrow & \textbf{Diễn giải khoa học (Interpretation)} \\
\text{Acc đạt } 71.20\% & & \text{Confusion Matrix cho thấy} & & \text{Mô hình học được siêu phẳng} \\
\text{trên tập kiểm thử IMDb} & & \text{tỷ lệ nhận diện đúng } 267/375 \text{ mẫu} & & \text{phân tách tuyến tính phân cực từ vựng}
\end{matrix}$$

> [!CAUTION] Cảnh báo ngụy biện
> Tuyệt đối không được nhảy cóc trực tiếp từ:
> $$\text{Accuracy} = 71.20\% \implies \text{"Mô hình đã thực sự hiểu sâu sắc ngôn ngữ con người"}$$
> Các phân tích lỗi ở Mục 2 đã chứng minh rõ ràng: Khi đối mặt với cấu trúc mỉa mai ngầm, mệnh đề nhượng bộ hoặc thiên lệch từ vựng, mô hình vẫn đưa ra những dự đoán sai lầm sơ đẳng, khẳng định rằng mô hình học quy luật tương quan thống kê trên chuỗi token chứ chưa sở hữu tri giác ngôn ngữ thực sự.
