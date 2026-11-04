# BÀI TẬP TÍNH TOÁN BẰNG TAY — TRANSFORMER ARCHITECTURE

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ**: Học kỳ I - 2026  
**Chủ đề**: From Attention to Transformer  
**Sinh viên thực hiện**: Nguyễn Văn Thành  
**Hạn nộp**: 23:59 - 04/11/2026  

---

## 1. Bài Toán 1: Tính Toán Chi Tiết Một Attention Head và Multi-Head Concatenation

### 1.1. Mục tiêu và Bối cảnh
Bài toán nhằm minh họa tường minh cơ chế biến đổi đại số tuyến tính bên trong một attention head (Section 6 & 7 trong đề bài `W7.pdf`), từ các ma trận chiếu đầu vào, tích vô hướng thu phóng (scaled dot-product), phân phối trọng số Softmax, đến việc tổng hợp vector ngữ cảnh và ghép nối đa đầu (multi-head concatenation).

### 1.2. Dữ kiện đầu vào và Giả thiết
Cho chuỗi đầu vào gồm $T = 3$ token, mỗi token được biểu diễn bằng vector embedding có số chiều $d_{model} = 4$:

$$
X = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 2 & 0 & 1 \\
1 & 1 & 0 & 0
\end{bmatrix} \in \mathbb{R}^{3 \times 4}
$$

Giả sử chiều không gian của một head là $d_k = 2$. Các ma trận trọng số chiếu tuyến tính của Head 1 gồm $W_1^Q, W_1^K, W_1^V \in \mathbb{R}^{4 \times 2}$ được cho như sau:

$$
W_1^Q = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
W_1^K = \begin{bmatrix}
0 & 1 \\
1 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
W_1^V = \begin{bmatrix}
1 & 1 \\
0 & 1 \\
1 & 0 \\
0 & 2
\end{bmatrix}
$$

Hệ số thu phóng (scaling factor): $\sqrt{d_k} = \sqrt{2} \approx 1.41421$.

---

### 1.3. Các Bước Biến Đổi Trung Gian Cho Head 1

#### Bước 1: Tính ma trận Query ($Q$), Key ($K$), Value ($V$)
Thực hiện phép nhân ma trận $Q = X W_1^Q$:

$$
Q = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 2 & 0 & 1 \\
1 & 1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}
= \begin{bmatrix}
1\cdot 1 + 0\cdot 0 + 1\cdot 0 + 0\cdot 1 & 1\cdot 0 + 0\cdot 1 + 1\cdot 1 + 0\cdot 0 \\
0\cdot 1 + 2\cdot 0 + 0\cdot 0 + 1\cdot 1 & 0\cdot 0 + 2\cdot 1 + 0\cdot 1 + 1\cdot 0 \\
1\cdot 1 + 1\cdot 0 + 0\cdot 0 + 0\cdot 1 & 1\cdot 0 + 1\cdot 1 + 0\cdot 1 + 0\cdot 0
\end{bmatrix}
= \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 1
\end{bmatrix}
$$

Thực hiện phép nhân ma trận $K = X W_1^K$:

$$
K = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 2 & 0 & 1 \\
1 & 1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}
= \begin{bmatrix}
1\cdot 0 + 0\cdot 1 + 1\cdot 1 + 0\cdot 0 & 1\cdot 1 + 0\cdot 0 + 1\cdot 0 + 0\cdot 1 \\
0\cdot 0 + 2\cdot 1 + 0\cdot 1 + 1\cdot 0 & 0\cdot 1 + 2\cdot 0 + 0\cdot 0 + 1\cdot 1 \\
1\cdot 0 + 1\cdot 1 + 0\cdot 1 + 0\cdot 0 & 1\cdot 1 + 1\cdot 0 + 0\cdot 0 + 0\cdot 1
\end{bmatrix}
= \begin{bmatrix}
1 & 1 \\
2 & 1 \\
1 & 1
\end{bmatrix}
$$

Thực hiện phép nhân ma trận $V = X W_1^V$:

$$
V = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 2 & 0 & 1 \\
1 & 1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 1 \\
0 & 1 \\
1 & 0 \\
0 & 2
\end{bmatrix}
= \begin{bmatrix}
1\cdot 1 + 0\cdot 0 + 1\cdot 1 + 0\cdot 0 & 1\cdot 1 + 0\cdot 1 + 1\cdot 0 + 0\cdot 2 \\
0\cdot 1 + 2\cdot 0 + 0\cdot 0 + 1\cdot 0 & 0\cdot 1 + 2\cdot 1 + 0\cdot 0 + 1\cdot 2 \\
1\cdot 1 + 1\cdot 0 + 0\cdot 1 + 0\cdot 0 & 1\cdot 1 + 1\cdot 1 + 0\cdot 0 + 0\cdot 2
\end{bmatrix}
= \begin{bmatrix}
2 & 1 \\
0 & 4 \\
1 & 2
\end{bmatrix}
$$

#### Bước 2: Tính tích vô hướng tương đồng $Q K^T$
Chuyển vị ma trận $K$:

$$
K^T = \begin{bmatrix}
1 & 2 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

Nhân ma trận $Q K^T \in \mathbb{R}^{3 \times 3}$:

$$
Q K^T = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 1 \\
1 & 1 & 1
\end{bmatrix}
= \begin{bmatrix}
1\cdot 1 + 1\cdot 1 & 1\cdot 2 + 1\cdot 1 & 1\cdot 1 + 1\cdot 1 \\
1\cdot 1 + 2\cdot 1 & 1\cdot 2 + 2\cdot 1 & 1\cdot 1 + 2\cdot 1 \\
1\cdot 1 + 1\cdot 1 & 1\cdot 2 + 1\cdot 1 & 1\cdot 1 + 1\cdot 1
\end{bmatrix}
= \begin{bmatrix}
2 & 3 & 2 \\
3 & 4 & 3 \\
2 & 3 & 2
\end{bmatrix}
$$

#### Bước 3: Thu phóng tỉ lệ bằng $\sqrt{d_k} = \sqrt{2} \approx 1.41421$

$$
S = \frac{Q K^T}{\sqrt{2}} = \begin{bmatrix}
\frac{2}{\sqrt{2}} & \frac{3}{\sqrt{2}} & \frac{2}{\sqrt{2}} \\
\frac{3}{\sqrt{2}} & \frac{4}{\sqrt{2}} & \frac{3}{\sqrt{2}} \\
\frac{2}{\sqrt{2}} & \frac{3}{\sqrt{2}} & \frac{2}{\sqrt{2}}
\end{bmatrix}
\approx \begin{bmatrix}
1.41421 & 2.12132 & 1.41421 \\
2.12132 & 2.82843 & 2.12132 \\
1.41421 & 2.12132 & 1.41421
\end{bmatrix}
$$

#### Bước 4: Chuẩn hóa phân phối xác suất qua hàm Softmax
Áp dụng Softmax ổn định số học theo từng hàng của ma trận $S$:
Xét hàng 1: $s_1 = [1.41421, 2.12132, 1.41421]$.  
Giá trị cực đại $m_1 = 2.12132$.  
Các số mũ:
- $\exp(1.41421 - 2.12132) = \exp(-0.70711) \approx 0.49307$
- $\exp(2.12132 - 2.12132) = \exp(0) = 1.00000$
- $\exp(1.41421 - 2.12132) = \exp(-0.70711) \approx 0.49307$

Tổng số mũ: $\sum = 0.49307 + 1.00000 + 0.49307 = 1.98614$.  
Trọng số Attention hàng 1:
- $a_{11} = \frac{0.49307}{1.98614} \approx 0.24826$
- $a_{12} = \frac{1.00000}{1.98614} \approx 0.50349$
- $a_{13} = \frac{0.49307}{1.98614} \approx 0.24826$

Xét hàng 2: $s_2 = [2.12132, 2.82843, 2.12132]$.  
Giá trị cực đại $m_2 = 2.82843$.  
Các hiệu số $s_{2j} - m_2$ chính xác bằng $[-0.70711, 0, -0.70711]$, hoàn toàn trùng khớp với hàng 1!  
Do đó phân phối xác suất hàng 2 và hàng 3 giống hệt hàng 1:

$$
A = \text{softmax}(S) = \begin{bmatrix}
0.24826 & 0.50349 & 0.24826 \\
0.24826 & 0.50349 & 0.24826 \\
0.24826 & 0.50349 & 0.24826
\end{bmatrix}
$$

#### Bước 5: Tính đầu ra của Head 1 ($Z_1 = A V$)

$$
Z_1 = \begin{bmatrix}
0.24826 & 0.50349 & 0.24826 \\
0.24826 & 0.50349 & 0.24826 \\
0.24826 & 0.50349 & 0.24826
\end{bmatrix}
\begin{bmatrix}
2 & 1 \\
0 & 4 \\
1 & 2
\end{bmatrix}
$$

Với mỗi hàng của $Z_1$:
- Cột 1: $0.24826\cdot 2 + 0.50349\cdot 0 + 0.24826\cdot 1 = 0.49652 + 0 + 0.24826 = 0.74478$
- Cột 2: $0.24826\cdot 1 + 0.50349\cdot 4 + 0.24826\cdot 2 = 0.24826 + 2.01396 + 0.49652 = 2.75874$

Kết quả đầu ra của Head 1:

$$
\text{head}_1 = Z_1 = \begin{bmatrix}
0.74478 & 2.75874 \\
0.74478 & 2.75874 \\
0.74478 & 2.75874
\end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

---

### 1.4. Ghép Nối Đa Đầu (Multi-Head Concatenation) và Chiếu Tuyến Tính $W^O$
Giả sử hệ thống có $h = 2$ attention heads ($d_k = 2$, tổng chiều là $h \cdot d_k = 4 = d_{model}$).  
Đầu ra của Head 2 sau các phép tính độc lập (với bộ trọng số $W_2^Q, W_2^K, W_2^V$) thu được ma trận:

$$
\text{head}_2 = Z_2 = \begin{bmatrix}
2.18210 & 1.32716 \\
1.33333 & 1.66667 \\
1.75872 & 1.49651
\end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

Thực hiện ghép nối (Concatenation) dọc theo chiều đặc trưng:

$$
\text{Concat}(\text{head}_1, \text{head}_2) = \begin{bmatrix}
0.74478 & 2.75874 & 2.18210 & 1.32716 \\
0.74478 & 2.75874 & 1.33333 & 1.66667 \\
0.74478 & 2.75874 & 1.75872 & 1.49651
\end{bmatrix} \in \mathbb{R}^{3 \times 4}
$$

Cho ma trận chiếu đầu ra $W^O \in \mathbb{R}^{4 \times 4}$:

$$
W^O = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 1.0 \\
0.5 & 0.0 & 0.5 & 0.0 \\
0.0 & 0.5 & 0.0 & 0.5
\end{bmatrix}
$$

Nhân ma trận kết thúc Multi-Head Attention:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2) \cdot W^O = \begin{bmatrix}
1.83583 & 3.42232 & 1.83583 & 3.42232 \\
1.41145 & 3.59208 & 1.41145 & 3.59208 \\
1.62414 & 3.50700 & 1.62414 & 3.50700
\end{bmatrix} \in \mathbb{R}^{3 \times 4}
$$

### 1.5. Trả Lời Câu Hỏi Cốt Lõi (Section 7)
**Câu hỏi**: *Vì sao cần concatenation trước khi áp dụng $W^O$?*  
**Trả lời**:
1. **Phân tách không gian biểu diễn (Subspace Separation)**: Mỗi attention head học chiếu câu văn vào một không gian con độc lập (ví dụ một head học cú pháp vị ngữ - chủ ngữ, head khác học quan hệ thực thể - đại từ hoặc phủ định). Các head này tạo ra các biểu diễn có ngữ nghĩa khác nhau.
2. **Tổng hợp đa thông tin không làm mất mát (Non-destructive aggregation)**: Nếu ta cộng trực tiếp các head lại với nhau ($\text{head}_1 + \text{head}_2$), các đặc trưng sẽ bị triệt tiêu hoặc hòa lẫn trước khi mô hình học được trọng số kết hợp. Phép ghép nối $\text{Concat}$ giữ nguyên vẹn toàn bộ thông tin độc lập của cả $h$ heads.
3. **Pha trộn tuyến tính linh hoạt qua $W^O$**: Khi nhân ma trận $W^O \in \mathbb{R}^{(h \cdot d_k) \times d_{model}}$, mạng nơ-ron có thể tự do học cách lấy tổ hợp tuyến tính kết nối giữa kênh $i$ của Head 1 với kênh $j$ của Head 2, khôi phục lại không gian biểu diễn chuẩn $d_{model}$ để truyền vào khối Residual tiếp theo.

---

## 2. Bài Toán 2: Tính Toán Chi Tiết Sinusoidal Positional Encoding

### 2.1. Mục tiêu và Công thức Chuẩn
Để bù đắp việc Self-Attention có tính chất hoán vị bất biến (permutation invariant), Transformer bổ sung thông tin vị trí thông qua các sóng sin và cos ở nhiều tần số khác nhau:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right), \quad PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
$$

Trong đó:
- $pos \in \{0, 1, 2, \dots, T-1\}$ là chỉ số vị trí của token trong chuỗi.
- $i \in \{0, 1, \dots, \frac{d}{2}-1\}$ là chỉ số cặp chiều đặc trưng.
- $d = d_{model}$ là số chiều embedding của mô hình.

### 2.2. Dữ kiện bài toán (Section 10 trong W7.pdf)
- Kích thước embedding: $d = 4$.
- Các vị trí cần tính: $pos = 0, 1, 2$.

Với $d = 4$, ta có $i \in \{0, 1\}$ ứng với 4 chiều:
1. Với $i = 0$ (chiều 0 và 1):
   $$\lambda_0 = 10000^{\frac{2\cdot 0}{4}} = 10000^0 = 1.0$$
   Chu kỳ bước sóng: $T_0 = 2\pi \cdot 1 \approx 6.28$ tokens.
2. Với $i = 1$ (chiều 2 và 3):
   $$\lambda_1 = 10000^{\frac{2\cdot 1}{4}} = 10000^{0.5} = \sqrt{10000} = 100.0$$
   Chu kỳ bước sóng: $T_1 = 2\pi \cdot 100 \approx 628.32$ tokens.

---

### 2.3. Các Bước Tính Toán Tường Minh

#### Vị trí $pos = 0$:
- Chiều 0 ($2i = 0$): $PE(0, 0) = \sin\left(\frac{0}{1}\right) = \sin(0) = 0.00000$
- Chiều 1 ($2i+1 = 1$): $PE(0, 1) = \cos\left(\frac{0}{1}\right) = \cos(0) = 1.00000$
- Chiều 2 ($2i = 2$): $PE(0, 2) = \sin\left(\frac{0}{100}\right) = \sin(0) = 0.00000$
- Chiều 3 ($2i+1 = 3$): $PE(0, 3) = \cos\left(\frac{0}{100}\right) = \cos(0) = 1.00000$

$$PE(0) = \begin{bmatrix} 0.00000 & 1.00000 & 0.00000 & 1.00000 \end{bmatrix}^T$$

#### Vị trí $pos = 1$:
- Chiều 0 ($2i = 0$): $PE(1, 0) = \sin\left(\frac{1}{1}\right) = \sin(1 \text{ rad}) \approx 0.84147$
- Chiều 1 ($2i+1 = 1$): $PE(1, 1) = \cos\left(\frac{1}{1}\right) = \cos(1 \text{ rad}) \approx 0.54030$
- Chiều 2 ($2i = 2$): $PE(1, 2) = \sin\left(\frac{1}{100}\right) = \sin(0.01 \text{ rad}) \approx 0.0099998 \approx 0.01000$
- Chiều 3 ($2i+1 = 3$): $PE(1, 3) = \cos\left(\frac{1}{100}\right) = \cos(0.01 \text{ rad}) \approx 0.99995$

$$PE(1) = \begin{bmatrix} 0.84147 & 0.54030 & 0.01000 & 0.99995 \end{bmatrix}^T$$

#### Vị trí $pos = 2$:
- Chiều 0 ($2i = 0$): $PE(2, 0) = \sin\left(\frac{2}{1}\right) = \sin(2 \text{ rad}) \approx 0.90930$
- Chiều 1 ($2i+1 = 1$): $PE(2, 1) = \cos\left(\frac{2}{1}\right) = \cos(2 \text{ rad}) \approx -0.41615$
- Chiều 2 ($2i = 2$): $PE(2, 2) = \sin\left(\frac{2}{100}\right) = \sin(0.02 \text{ rad}) \approx 0.019999 \approx 0.02000$
- Chiều 3 ($2i+1 = 3$): $PE(2, 3) = \cos\left(\frac{2}{100}\right) = \cos(0.02 \text{ rad}) \approx 0.99980$

$$PE(2) = \begin{bmatrix} 0.90930 & -0.41615 & 0.02000 & 0.99980 \end{bmatrix}^T$$

---

### 2.4. Bảng Ma Trận Positional Encoding Tổng Hợp

$$
PE = \begin{bmatrix}
0.00000 & 1.00000 & 0.00000 & 1.00000 \\
0.84147 & 0.54030 & 0.01000 & 0.99995 \\
0.90930 & -0.41615 & 0.02000 & 0.99980
\end{bmatrix} \in \mathbb{R}^{3 \times 4}
$$

### 2.5. Sanity Check và Ý Nghĩa Khoa Học
1. **Kiểm tra chuẩn biên độ**:
   - Ở cặp chiều thứ nhất ($i=0$): $\sin^2(pos) + \cos^2(pos) = 1$ với mọi $pos$. Cụ thể:  
     $pos=0: 0^2 + 1^2 = 1.0$;  
     $pos=1: 0.84147^2 + 0.54030^2 = 0.70807 + 0.29192 = 0.99999 \approx 1.0$;  
     $pos=2: 0.90930^2 + (-0.41615)^2 = 0.82683 + 0.17318 = 1.00001 \approx 1.0$.
   - Ở cặp chiều thứ hai ($i=1$): $\sin^2(pos/100) + \cos^2(pos/100) = 1$ với mọi $pos$.
2. **Khả năng phân biệt vị trí**: Vector $PE(pos)$ ở mỗi vị trí là độc nhất (unique). Tần số cao ở các chiều đầu thay đổi liên tục giúp nhận biết các vị trí gần nhau, trong khi tần số thấp ở các chiều sau thay đổi chậm rãi giúp mã hóa khoảng cách vị trí tầm xa.

---

## 3. Bài Toán 3: Tính Toán Chi Tiết Position-Wise Feed-Forward Network (FFN)

### 3.1. Mục tiêu và Công thức
Khối Feed-Forward Network áp dụng độc lập và đồng nhất lên từng vị trí của chuỗi (position-wise):

$$
FFN(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2
$$

Trong đó $\sigma(\cdot)$ là hàm kích hoạt phi tuyến (ReLU: $\text{ReLU}(z) = \max(0, z)$).

### 3.2. Dữ kiện bài toán (Section 14 trong W7.pdf)
Cho vector biểu diễn đầu vào tại một vị trí:

$$
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \in \mathbb{R}^{2 \times 1}
$$

Ma trận trọng số và bias tầng thứ nhất:

$$
W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0 \\ -1 \end{bmatrix}
$$

Cho ma trận chiếu và bias tầng thứ hai để hoàn chỉnh toàn bộ FFN:

$$
W_2 = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix}, \quad b_2 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

---

### 3.3. Các Bước Tính Toán Tường Minh

#### Bước 1: Tính giá trị trước hàm kích hoạt (Pre-activation $z_1$)

$$
z_1 = W_1 x + b_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix}
= \begin{bmatrix} 1\cdot 1 + 0\cdot 2 \\ 0\cdot 1 + 1\cdot 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix}
= \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix}
= \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

#### Bước 2: Kích hoạt phi tuyến ReLU ($h$)

$$
h = \text{ReLU}(z_1) = \text{ReLU}\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right) = \begin{bmatrix} \max(0, 1) \\ \max(0, 1) \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

#### Bước 3: Chiếu tuyến tính tầng thứ hai ($y$)

$$
y = W_2 h + b_2 = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix}
= \begin{bmatrix} 2\cdot 1 + (-1)\cdot 1 \\ 1\cdot 1 + 3\cdot 1 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix}
= \begin{bmatrix} 1 \\ 4 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix}
= \begin{bmatrix} 2 \\ 4 \end{bmatrix}
$$

### 3.4. Sanity Check và Ý Nghĩa Cốt Lõi
1. **Kiểm tra tính phi tuyến**: Nhờ có hàm ReLU, nếu $x_2 \le 1$ thì $z_{1, 2} \le 0 \Rightarrow h_2 = 0$, mạch nơ-ron thực hiện phép ngắt ngưỡng.
2. **Vai trò của FFN trong Transformer Block**: Attention là phép tương tác **ngang** giữa các vị trí trong chuỗi (token mixing). FFN là phép biến đổi phi tuyến **dọc** theo từng vị trí (channel mixing), cho phép trích xuất các tổ hợp đặc trưng trừu tượng mà phép tổ hợp tuyến tính trong attention không làm được.

---

## 4. Bài Toán 4: Theo Dõi Kích Thước Tensor (Shape Trace) Qua Toàn Bộ Transformer Encoder Block

### 4.1. Thiết lập tham số (Section 18 trong W7.pdf)
- Kích thước batch: $B = 8$
- Chiều dài chuỗi: $T = 100$
- Kích thước mô hình: $d_{model} = 128$
- Số attention heads: $n_{heads} = 4$
- Chiều không gian của mỗi head: $d_k = \frac{d_{model}}{n_{heads}} = \frac{128}{4} = 32$
- Chiều ẩn của FFN: $d_{ff} = 4 \times d_{model} = 512$

---

### 4.2. Bảng Theo Dõi Shape Từng Bước (Full Forward Shape Trace)

| Bước thực thi | Tensor đại diện | Ký hiệu toán học / Phép biến đổi | Kích thước Tensor (Shape) | Diễn giải kỹ thuật |
| :--- | :--- | :--- | :--- | :--- |
| **0. Đầu vào** | $X$ | Input Embeddings + $PE$ | `(8, 100, 128)` | $B=8$ câu, mỗi câu $T=100$ tokens, vector $d=128$. |
| **1. Chiếu Q, K, V** | $Q, K, V$ | $X \cdot W^Q, X \cdot W^K, X \cdot W^V$ | `(8, 100, 128)` | Từng ma trận có trọng số `(128, 128)`. |
| **2. Tách Multi-Head** | $Q_{split}, K_{split}, V_{split}$ | Reshape `(B, T, h, dk)` $\rightarrow$ Permute | `(8, 4, 100, 32)` | $h=4$ heads song song, mỗi head $d_k=32$. |
| **3. Attention Scores** | $Scores$ | $\frac{Q \cdot K^T}{\sqrt{d_k}}$ | `(8, 4, 100, 100)` | Ma trận tương tác giữa $T$ tokens với nhau: $100 \times 100$. |
| **4. Attention Weights**| $A$ | $\text{Softmax}(Scores, \text{dim}=-1)$ | `(8, 4, 100, 100)` | Tổng mỗi hàng bằng 1 (xác suất chú ý). |
| **5. Context Matrix** | $Context$ | $A \cdot V$ | `(8, 4, 100, 32)` | Tổng hợp thông tin từ $V$ theo phân phối $A$. |
| **6. Ghép Heads** | $Concat$ | Permute $\rightarrow$ Reshape `(B, T, h*dk)` | `(8, 100, 128)` | Ghép nối 4 heads thành vector chiều 128 ban đầu. |
| **7. Chiếu đầu ra MHA** | $MHA_{out}$ | $Concat \cdot W^O$ | `(8, 100, 128)` | Chiếu tuyến tính qua $W^O \in \mathbb{R}^{128 \times 128}$. |
| **8. Residual 1** | $X_1$ | $X + MHA_{out}$ | `(8, 100, 128)` | Kết nối tắt bảo tồn thông tin đầu vào. |
| **9. LayerNorm 1** | $\tilde{X}_1$ | $\text{LayerNorm}(X_1)$ | `(8, 100, 128)` | Chuẩn hóa theo chiều cuối ($d_{model}=128$). |
| **10. FFN Tầng 1** | $H_{ffn}$ | $\text{ReLU}(\tilde{X}_1 \cdot W_1 + b_1)$ | `(8, 100, 512)` | Mở rộng không gian đặc trưng lên $d_{ff}=512$. |
| **11. FFN Tầng 2** | $FFN_{out}$ | $H_{ffn} \cdot W_2 + b_2$ | `(8, 100, 128)` | Thu hẹp về kích thước chuẩn $d_{model}=128$. |
| **12. Residual 2** | $X_2$ | $\tilde{X}_1 + FFN_{out}$ | `(8, 100, 128)` | Kết nối tắt khối FFN. |
| **13. LayerNorm 2** | $Output$ | $\text{LayerNorm}(X_2)$ | `(8, 100, 128)` | Đầu ra hoàn chỉnh của Transformer Block. |

---

### 4.3. Các Lưu Ý Về Lỗi Shape Phổ Biến Cần Tránh
1. **Lỗi thứ tự trục khi tính Attention**: Nếu quên hoán vị trục (permute) mà reshape trực tiếp từ `(8, 100, 128)` sang `(8, 4, 100, 32)`, các token sẽ bị trộn lẫn vào các head thay vì phân tách theo chiều đặc trưng. Thứ tự đúng bắt buộc là: `reshape(8, 100, 4, 32).transpose(1, 2)`.
2. **Chi phí bộ nhớ bậc hai $O(T^2)$**: Tensor Attention Scores chiếm kích thước `(8, 4, 100, 100)` tức $8 \times 4 \times 100 \times 100 = 320,000$ phần tử float32. Nếu chuỗi tăng lên $T = 1000$, số phần tử ma trận scores tăng lên $32,000,000$ phần tử (gấp 100 lần), minh chứng cho điểm nghẽn bộ nhớ $O(T^2)$.

