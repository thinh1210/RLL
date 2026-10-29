# Week 1 – Giới thiệu Reinforcement Learning

> Tuần mở đầu xây dựng nền tảng lý thuyết cốt lõi cho toàn khóa học: bài toán RL, bài toán Multi-Armed Bandit (RL tối giản), cân bằng Khám phá–Khai thác (Exploration–Exploitation), và phép toán vi tích phân đằng sau *Pong from Pixels* của Karpathy.  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

**Lộ trình**: §1 RL là gì → §2 Các thành phần → §3 Phân loại thuật toán → §4 Bandit và khám phá–khai thác (kèm chứng minh toán học) → §5 Pong from Pixels bằng toán (kèm vi tích phân) → §6 Phân tích chuyên sâu các tham số → §7 Thử thách đặc trưng của RL → §8 Bài tập.

---

## 1. Reinforcement Learning là gì?

Một **agent** tương tác với **môi trường** theo chu kỳ:

```
        hành động a_t
  Agent ───────────────► Môi trường
    ▲                         │
    └──── s_{t+1}, r_{t+1} ◄──┘
```
Mục tiêu: chọn hành động để **tối đa hoá tổng phần thưởng kỳ vọng** (return):
$$G_t=\sum_{k\ge0}\gamma^kr_{t+k+1},\qquad\max_\pi\ \mathbb E_\pi[G_0]$$

* **Giải thích ký hiệu ngay sau phương trình**:
  * $t \in \{0, 1, 2, \dots\}$: Chỉ số bước thời gian rời rạc.
  * $r_{t+k+1} \in \mathbb R$: Phần thưởng vô hướng nhận được tại thời điểm $t+k+1$.
  * $\gamma \in [0, 1)$: Hệ số chiết khấu tương lai (Discount factor).
  * $G_t \in \mathbb R$: Tổng lợi nhuận chiết khấu tích luỹ (Discounted Return) tính từ thời điểm $t$.
  * $\pi(a \mid s)$: Chính sách (Policy) phân phối xác suất chọn hành động $a$ khi ở trạng thái $s$.
  * $\mathbb E_\pi[\cdot]$: Kỳ vọng toán học theo phân phối xác suất của toàn bộ quỹ đạo sinh ra bởi chính sách $\pi$.

**Giả thuyết phần thưởng (reward hypothesis)**: mọi mục tiêu có thể diễn đạt là cực đại hoá kỳ vọng của một tín hiệu vô hướng tích luỹ.

### So sánh ba mô hình học

| | Có giám sát | Không giám sát | Reinforcement |
|---|---|---|---|
| Dữ liệu | $(x,y)$ do con người gán | $x$ không nhãn | do **chính agent** sinh ra khi hành động |
| Phản hồi | nhãn đúng cho từng mẫu | không | **vô hướng, trễ**, chỉ nói "tốt/xấu" chứ không nói "đúng là gì" |
| Dữ liệu i.i.d.? | có (giả định) | có | **không**: hành động hiện tại ảnh hưởng dữ liệu tương lai |
| Cần khám phá? | không | không | **có** |

---

## 2. Các thành phần

| Thành phần | Ký hiệu | Ví dụ |
|---|---|---|
| Trạng thái | $s_t$ | ảnh Atari, vị trí/vận tốc robot |
| Hành động | $a_t$ | lên/xuống, mô-men xoắn |
| Phần thưởng | $r_t$ | +1 ghi điểm, −1 mất điểm |
| Policy | $\pi(a|s)$ | quy tắc chọn hành động |
| Hàm giá trị | $V^\pi(s),Q^\pi(s,a)$ | "trạng thái này tốt đến đâu" |
| Mô hình (tuỳ chọn) | $P(s'|s,a)$ | dự đoán tương lai |

Công thức nền tảng sẽ được suy ra đầy đủ ở [Week 2](../Week2/BAI_GIANG.md): $G_t=r_{t+1}+\gamma G_{t+1}$, phương trình Bellman.

---

## 3. Phân loại thuật toán RL

| Trục | Lựa chọn | Ví dụ trong repo |
|---|---|---|
| Có mô hình môi trường? | model-free / model-based | Week 2–6 / Week 7 |
| Học cái gì? | value-based / policy-based / actor-critic | Week 3 / Week 4 (PG) / Week 4–5 (A2C, PPO) |
| Dữ liệu từ đâu? | on-policy / off-policy | PPO, A2C / Q-learning, DQN |
| Hành động | rời rạc / liên tục | DQN / PPO, ES |
| Không dùng gradient | tiến hoá | Week 6 |

---

## 4. Bandit và cân bằng Khám phá–Khai thác

Bài toán RL đơn giản nhất: **bandit $k$ tay** — một trạng thái, $k$ hành động, mỗi hành động $a$ cho reward ngẫu nhiên có kỳ vọng $q_*(a)$ chưa biết. Không có chuỗi thời gian, nhưng chứa trọn xung đột khám phá–khai thác.

### 4.1 Ước lượng giá trị hành động & Chứng minh công thức cập nhật tăng dần
Gọi $R_1, R_2, \dots, R_{n-1}$ là các phần thưởng nhận được sau $n-1$ lần chọn hành động $a$. Ước lượng giá trị hành động tại bước thứ $n$ là trung bình mẫu:
$$Q_n = \frac{1}{n-1} \sum_{i=1}^{n-1} R_i$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế công thức tính trung bình số học lưu trữ mảng mẫu $Q_n = \frac{1}{n-1}\sum_{i=1}^{n-1} R_i$.
> * **Khuyết điểm của phương pháp cũ**:
>   1. *Bùng nổ bộ nhớ*: Đòi hỏi lưu trữ toàn bộ lịch sử phần thưởng ($O(n)$ bộ nhớ) hoặc phải quản lý bộ đếm mẫu.
>   2. *Đóng băng tri thức trong môi trường biến đổi (Non-stationary)*: Khi phân phối xác suất phần thưởng thay đổi theo thời gian, trọng số $\frac{1}{n} \to 0$ khiến ước lượng không thể thích ứng với các biến động mới.
> * **Logic thiết kế toán học**: Biến đổi đại số đưa về dạng đệ quy trực tuyến $\text{Ước lượng mới} = \text{Ước lượng cũ} + \text{Bước học} \times (\text{Phần thưởng thực tế} - \text{Ước lượng cũ})$. Khi thay $\frac{1}{n}$ bằng hằng số $\alpha \in (0, 1]$, ta thu được trung bình có trọng số suy giảm hàm mũ $\alpha(1-\alpha)^{n-i}$, đảm bảo phần thưởng mới nhận luôn giữ nguyên tỷ trọng $\alpha$, giải quyết triệt để tính dừng.

* **Chứng minh toán học (Cơ chế suy diễn công thức cập nhật trực tuyến)**:
  Khi nhận thêm phần thưởng $R_n$ ở lần thử thứ $n$, ước lượng mới $Q_{n+1}$ được tính bằng trung bình của $n$ phần thưởng:
  $$Q_{n+1} = \frac{1}{n} \sum_{i=1}^n R_i = \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right)$$
  Từ định nghĩa của $Q_n$, ta có $\sum_{i=1}^{n-1} R_i = (n-1) Q_n$. Thay vào biểu thức trên:
  $$\begin{aligned}
  Q_{n+1} &= \frac{1}{n} \Big( R_n + (n-1) Q_n \Big) \\
  &= \frac{1}{n} \Big( R_n + n Q_n - Q_n \Big) \\
  &= Q_n + \frac{1}{n} (R_n - Q_n)
  \end{aligned}$$
  Đẳng thức được chứng minh. $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Biến Trung bình Lịch sử thành Đệ quy 1 bước:**
> - **Bước 1 (Tách mẫu mới nhất khỏi tổng lịch sử trong hàm gốc $Q_{n+1}$):**
>   Ta tách tổng $n$ phần tử $\sum_{i=1}^n R_i$ thành phần tử mới nhất $R_n$ cộng với tổng của $n-1$ phần tử cũ:
>   $$Q_{n+1} = \frac{1}{n}\Big(R_n + \underbrace{\sum_{i=1}^{n-1} R_i}_{\text{Cụm cần thay thế}}\Big)$$
> - **Bước 2 (Rút biểu thức thế từ định nghĩa của hàm bước trước $Q_n$):**
>   Theo định nghĩa ở đầu mục 4.1, trung bình của $n-1$ lần trước là $Q_n = \frac{1}{n-1}\sum_{i=1}^{n-1} R_i$. Nhân cả hai vế với $(n-1)$, ta cô lập được tổng lịch sử theo $Q_n$:
>   $$\sum_{i=1}^{n-1} R_i = (n-1)Q_n$$
> - **Bước 3 (Thực hiện phép thế và rút gọn đại số):**
>   Đem cụm $(n-1)Q_n$ ở Bước 2 **thế thẳng vào vị trí của $\sum_{i=1}^{n-1} R_i$** trong phương trình ở Bước 1, sau đó khai triển $(n-1)Q_n = nQ_n - Q_n$:
>   $$Q_{n+1} = \frac{1}{n}\Big(R_n + (n-1)Q_n\Big) = \frac{1}{n}\Big(nQ_n + R_n - Q_n\Big) = \frac{nQ_n}{n} + \frac{R_n - Q_n}{n} = Q_n + \frac{1}{n}(R_n - Q_n)$$
>   *Ý nghĩa của phép thế:* Toàn bộ tổng $\sum_{i=1}^{n-1} R_i$ (tốn $n-1$ ô nhớ) đã được **thay thế hoàn toàn bằng duy nhất 1 con số $Q_n$**!

* **Giải thích ký hiệu**:
  * $Q_n$: Ước lượng giá trị hành động hiện tại trước khi nhận phần thưởng thứ $n$.
  * $R_n$: Phần thưởng thực tế vừa quan sát được.
  * $R_n - Q_n$: Sai số dự đoán (Prediction error / Innovation).
  * $\frac{1}{n}$: Hệ số bước học (Step-size).

* **Mở rộng sang trường hợp môi trường biến đổi (Non-stationary)**:
  Thay bước học $\frac{1}{n}$ bằng hằng số $\alpha \in (0, 1]$:
  $$\boxed{Q_{n+1} = Q_n + \alpha (R_n - Q_n) = (1-\alpha) Q_n + \alpha R_n}$$
  * **Chứng minh tính chất suy giảm mũ (Exponential Recency-Weighted Average)**:
    Khai triển truy hồi liên tục từ bước $n$ lùi về bước 1:
    $$\begin{aligned}
    Q_{n+1} &= (1-\alpha) Q_n + \alpha R_n \\
    &= (1-\alpha) \big[ (1-\alpha) Q_{n-1} + \alpha R_{n-1} \big] + \alpha R_n \\
    &= (1-\alpha)^2 Q_{n-1} + \alpha (1-\alpha) R_{n-1} + \alpha R_n \\
    &= (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha (1-\alpha)^{n-i} R_i
    \end{aligned}$$
    Tổng trọng số của tất cả các phần thưởng quá khứ và giá trị khởi tạo được bảo toàn đúng bằng 1:
    $$(1-\alpha)^n + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} = (1-\alpha)^n + \alpha \frac{1 - (1-\alpha)^n}{1 - (1-\alpha)} = (1-\alpha)^n + 1 - (1-\alpha)^n = 1$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Thế truy hồi lùi $Q_n \leftarrow Q_{n-1} \leftarrow \dots \leftarrow Q_1$:**
> - **Phương trình gốc tại bước $n+1$:** $Q_{n+1} = (1-\alpha)Q_n + \alpha R_n$.
> - **Hàm đem thế (cùng quy tắc đó nhưng viết tại bước $n$ và $n-1$):**
>   $$Q_n = (1-\alpha)Q_{n-1} + \alpha R_{n-1}, \qquad Q_{n-1} = (1-\alpha)Q_{n-2} + \alpha R_{n-2}$$
> - **Thế lần 1 (Thay $Q_n$ trong phương trình gốc bằng biểu thức của $Q_{n-1}$):**
>   $$Q_{n+1} = (1-\alpha)\underbrace{\big[(1-\alpha)Q_{n-1} + \alpha R_{n-1}\big]}_{\text{Thế cho } Q_n} + \alpha R_n = (1-\alpha)^2 Q_{n-1} + \alpha(1-\alpha)^1 R_{n-1} + \alpha(1-\alpha)^0 R_n$$
> - **Thế lần 2 (Thay tiếp $Q_{n-1}$ bằng biểu thức của $Q_{n-2}$):**
>   $$Q_{n+1} = (1-\alpha)^2\underbrace{\big[(1-\alpha)Q_{n-2} + \alpha R_{n-2}\big]}_{\text{Thế cho } Q_{n-1}} + \alpha(1-\alpha)R_{n-1} + \alpha R_n = (1-\alpha)^3 Q_{n-2} + \alpha(1-\alpha)^2 R_{n-2} + \alpha(1-\alpha)^1 R_{n-1} + \alpha R_n$$
> - **Quy luật sau $n$ lần thế liên tiếp:** Mỗi khi lùi về quá khứ $n-i$ bước, mẫu phần thưởng $R_i$ bị nhân thêm đúng $n-i$ lần thừa số $(1-\alpha) < 1$, tạo ra trọng số suy giảm hàm mũ $\alpha(1-\alpha)^{n-i}$.

---

### 4.2 Chiến lược chọn hành động & Chứng minh UCB1 từ Bất đẳng thức Hoeffding

#### (a) Thuần tham lam (Pure Greedy)
$$a_t = \arg\max_a Q_t(a)$$
* Ưu điểm: Khai thác triệt để tri thức hiện tại.
* Nhược điểm: Bị kẹt vĩnh viễn ở hành động dưới tối ưu nếu giá trị khởi tạo bị lệch hoặc mẫu ban đầu mang phương sai bất lợi.

#### (b) $\epsilon$-greedy
$$\pi(a) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal A|} & \text{nếu } a = \arg\max_b Q_t(b) \\ \frac{\epsilon}{|\mathcal A|} & \text{nếu } a \ne \arg\max_b Q_t(b) \end{cases}$$
* $a_t$ được chọn ngẫu nhiên đều với xác suất $\epsilon$, và tham lam với xác suất $1-\epsilon$. Khám phá "mù quáng" vì đối xử với hành động rất tệ ngang với hành động tiềm năng.

#### (c) Thuật toán UCB1 (Upper Confidence Bound)
$$a_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế chiến lược $\epsilon$-greedy khám phá ngẫu nhiên đều.
> * **Khuyết điểm của phương pháp cũ**: $\epsilon$-greedy chịu mức **Hối tiếc tích lũy tăng tuyến tính $\Omega(T)$**, liên tục lãng phí tài nguyên thử nghiệm các hành động đã biết chắc chắn là rất tồi tệ.
> * **Logic thiết kế toán học**: Triết lý **"Lạc quan khi đối mặt với sự bất định" (Optimism in the Face of Uncertainty)**. Biểu thức $c \sqrt{\frac{\ln t}{N_t(a)}}$ được suy diễn từ Bất đẳng thức Hoeffding: khi hành động ít được chọn ($N_t(a)$ nhỏ), bán kính tin cậy phình to, buộc agent phải thử nghiệm nó; khi đã thử nghiệm đủ nhiều ($N_t(a)$ lớn), bán kính thu hẹp, nhường chỗ cho việc khai thác $Q_t(a)$.
> * **Cơ chế giải quyết**: Đưa hối tiếc tích lũy về mức **cận dưới tối ưu lý thuyết $O(\ln T)$**, triệt tiêu hoàn toàn sự lãng phí tài nguyên của $\epsilon$-greedy.

* **Chứng minh toán học (Cơ chế suy diễn từ Bất đẳng thức Hoeffding)**:
  * **Định lý Hoeffding**: Cho $X_1, \dots, X_n$ là các biến ngẫu nhiên độc lập cùng phân phối (i.i.d) có giá trị nằm trong đoạn $[0, 1]$ với kỳ vọng thực $\mu = \mathbb E[X]$. Gọi $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$ là trung bình mẫu. Khi đó với mọi $\epsilon > 0$:
    $$\mathbb P\big(\bar{X}_n - \mu \le -\epsilon\big) \le e^{-2 n \epsilon^2} \iff \mathbb P\big(\mu \ge \bar{X}_n + \epsilon\big) \le e^{-2 n \epsilon^2}$$
  * Áp dụng vào cánh tay $a$: Gọi $q_*(a)$ là giá trị kỳ vọng thực của cánh tay $a$, $Q_t(a)$ là trung bình mẫu sau $N_t(a)$ lần kéo:
    $$\mathbb P\Big(q_*(a) \ge Q_t(a) + U_t(a)\Big) \le \exp\Big(-2 N_t(a) U_t(a)^2\Big)$$
  * Ta muốn xác suất đuôi (xác suất mà giá trị thực vượt quá cận trên tin cậy $Q_t(a) + U_t(a)$) suy giảm cực nhanh theo thời gian $t$. Chọn ngưỡng xác suất $p = t^{-4}$:
    $$\exp\Big(-2 N_t(a) U_t(a)^2\Big) = t^{-4}$$
  * Lấy logarit tự nhiên hai vế:
    $$-2 N_t(a) U_t(a)^2 = \ln(t^{-4}) = -4 \ln t \iff U_t(a)^2 = \frac{4 \ln t}{2 N_t(a)} = \frac{2 \ln t}{N_t(a)}$$
    $$\implies \boxed{U_t(a) = \sqrt{\frac{2 \ln t}{N_t(a)}} = \sqrt{2} \sqrt{\frac{\ln t}{N_t(a)}}}$$
    Từ đây giải thích chính xác tại sao hằng số lý thuyết là $c = \sqrt{2} \approx 1.414$. $\square$

* **Giải thích ký hiệu**:
  * $t$: Tổng số bước thời gian đã tương tác trên toàn bộ hệ thống.
  * $N_t(a)$: Số lần cánh tay $a$ đã được kích hoạt tính đến bước $t$.
  * $Q_t(a)$: Trung bình phần thưởng thu được từ cánh tay $a$ (thành phần Khai thác).
  * $U_t(a) = c \sqrt{\frac{\ln t}{N_t(a)}}$: Độ bất định / bán kính tin cậy (thành phần Khám phá).

---

### 4.3 Regret (Hối tiếc tích luỹ)
$$L_T = \sum_{t=1}^T \big( q_*(a^*) - q_*(A_t) \big)$$
* $a^* = \arg\max_a q_*(a)$: Hành động tối ưu toàn cục thực sự.
* $A_t$: Hành động agent chọn tại bước $t$.
* **Định lý Lai & Robbins (1985)**: Cận dưới hối tiếc của mọi thuật toán bandit hợp lý bị chặn dưới bởi $\Omega(\ln T)$. UCB1 đạt được cận tối ưu này $O(\ln T)$, trong khi $\epsilon$-greedy cố định chịu mức phạt hối tiếc tuyến tính $\Omega(T)$.

---

## 5. Pong from Pixels (Karpathy) — toán đằng sau

Bài viết cài đặt policy gradient bằng ~130 dòng NumPy, chơi Pong từ điểm ảnh.

### 5.1 Thiết lập
- Đầu vào: **hiệu hai khung hình** $x=\text{frame}_t-\text{frame}_{t-1}$, đã cắt/giảm còn $80\times80=6400$ chiều ⇒ mã hoá chuyển động (khôi phục tính Markov).
- Policy: mạng 2 lớp
$$h=\mathrm{ReLU}(W_1x),\qquad z=w_2^\top h,\qquad p=\sigma(z)=\pi(\text{UP}|x)$$
với $W_1\in\mathbb R^{200\times6400}$ (≈1,28 triệu trọng số). Hành động: UP với xác suất $p$, DOWN với $1-p$.
- Reward: +1 ghi điểm, −1 mất điểm, 0 còn lại ⇒ **reward cực thưa** (mỗi điểm cách nhau hàng chục bước).

* **Giải thích ký hiệu**:
  * $x \in \mathbb R^{6400}$: Vector đầu vào đại diện cho trạng thái chuyển động.
  * $W_1 \in \mathbb R^{200 \times 6400}$: Ma trận trọng số tầng ẩn thứ nhất.
  * $w_2 \in \mathbb R^{200}$: Vector trọng số của tầng đầu ra.
  * $z \in \mathbb R$: Logit vô hướng trước hàm kích hoạt phi tuyến.
  * $\sigma(z) = \frac{1}{1 + e^{-z}}$: Hàm sigmoid ánh xạ logit $z \to (0, 1)$.

### 5.2 Chứng minh đạo hàm log-prob cho phân phối Bernoulli
Mạng dự đoán xác suất hành động UP là $p = \sigma(z)$. Đặt nhãn hành động thực hiện là $a \in \{0, 1\}$ (với $a=1$ là UP, $a=0$ là DOWN). Xác suất chính sách là:
$$\pi(a \mid x) = p^a (1-p)^{1-a} = \sigma(z)^a \big(1 - \sigma(z)\big)^{1-a}$$

* **Chứng minh toán học chi tiết**:
  * **Bước 1**: Đạo hàm của hàm Sigmoid:
    $$\sigma'(z) = \frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \big(1 - \sigma(z)\big)$$
  * **Bước 2**: Lấy logarit tự nhiên hàm xác suất $\pi(a \mid x)$:
    $$\ln \pi(a \mid x) = a \ln \sigma(z) + (1-a) \ln\big(1 - \sigma(z)\big)$$
  * **Bước 3**: Lấy đạo hàm riêng theo biến logit $z$ bằng quy tắc chuỗi:
    $$\begin{aligned}
    \frac{\partial \ln \pi(a \mid x)}{\partial z} &= a \frac{\sigma'(z)}{\sigma(z)} + (1-a) \frac{-\sigma'(z)}{1 - \sigma(z)} \\
    &= a \frac{\sigma(z)(1 - \sigma(z))}{\sigma(z)} - (1-a) \frac{\sigma(z)(1 - \sigma(z))}{1 - \sigma(z)} \\
    &= a \big(1 - \sigma(z)\big) - (1-a) \sigma(z) \\
    &= a - a \sigma(z) - \sigma(z) + a \sigma(z) \\
    &= a - \sigma(z) = a - p
    \end{aligned}$$
  Đẳng thức được chứng minh hoàn tất. $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Tại sao đạo hàm phức tạp của Log-Bernoulli lại rút gọn sạch sẽ thành $(a - p)$?**
> - **Hàm gốc sau khi lấy đạo hàm hàm hợp $\frac{d}{dz}\ln u(z) = \frac{u'(z)}{u(z)}$:**
>   $$\frac{\partial \ln \pi(a \mid x)}{\partial z} = a \frac{\sigma'(z)}{\sigma(z)} - (1-a)\frac{\sigma'(z)}{1-\sigma(z)}$$
> - **Hàm đem thế vào tử số:** Ở Bước 1, ta đã chứng minh đạo hàm của hàm Sigmoid tự biểu diễn qua chính nó:
>   $$\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$
> - **Thực hiện phép thế $\sigma'(z)$ vào cả 2 phân thức và giản ước mẫu số:**
>   - Ở phân thức thứ nhất: $\frac{\sigma'(z)}{\sigma(z)} = \frac{\cancel{\sigma(z)}\big(1-\sigma(z)\big)}{\cancel{\sigma(z)}} = 1 - \sigma(z)$.
>   - Ở phân thức thứ hai: $\frac{\sigma'(z)}{1-\sigma(z)} = \frac{\sigma(z)\cancel{\big(1-\sigma(z)\big)}}{\cancel{1-\sigma(z)}} = \sigma(z)$.
> - **Thế kết quả đã giản ước vào biểu thức tổng và thay lại ký hiệu $p = \sigma(z)$:**
>   $$\frac{\partial \ln \pi(a \mid x)}{\partial z} = a(1 - p) - (1 - a)p = a - \cancel{ap} - p + \cancel{ap} = a - p$$
>   *Ý nghĩa thực tiễn:* Nhờ tính chất tự đạo hàm $\sigma' = \sigma(1-\sigma)$ vừa khít với mẫu số của phân phối Bernoulli $p^a(1-p)^{1-a}$, hai mẫu số $\sigma(z)$ và $1-\sigma(z)$ (vốn có thể tiến về $0$ gây tràn số `NaN` khi chia) đã bị **triệt tiêu hoàn toàn trên giấy trước khi đưa vào code**!

* **Ý nghĩa thực chiến & Logic thiết kế (Mục đích ra đời)**:
  > 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
  > * **Thay thế thành phần nào**: Thay thế việc định giá hành động gián tiếp thông qua bảng $Q$ hoặc mạng Q-network chọn $\arg\max_a Q(s, a)$.
  > * **Khuyết điểm của phương pháp cũ**: Với đầu vào ảnh điểm ảnh chiều cao ($6400$ pixels) và môi trường có reward cực thưa, việc xấp xỉ hàm giá trị $Q(s, a)$ chính xác cho mọi hành động là cực kỳ khó khăn và dễ gây bất ổn định.
  > * **Logic thiết kế toán học**: Tham số hóa trực tiếp phân phối xác suất hành vi $\pi_\theta(a \mid x) = \sigma(z)^a (1-\sigma(z))^{1-a}$. Khi lấy đạo hàm $\nabla_z \ln \pi$, kết quả thu được là $a - p$ (hiệu giữa hành động đã lấy $a \in \{0, 1\}$ và xác suất dự đoán $p$). Khi nhân với lợi nhuận $\hat G_t$, nếu ván thắng ($\hat G_t > 0$), gradient đẩy logit $z$ theo hướng của $a$; nếu ván thua ($\hat G_t < 0$), gradient đẩy logit theo hướng ngược lại.
  > * **Cơ chế giải quyết**: Biến bài toán tối ưu tăng cường thành bài toán cập nhật có giám sát với nhãn giả $a$ và hàm mất mát Binary Cross-Entropy quen thuộc, mở đường cho việc tối ưu mạng nơ-ron sâu chơi game từ điểm ảnh.

### 5.3 Credit assignment bằng chiết khấu
Mỗi ván (một điểm) có ~20–200 bước nhưng chỉ có một reward ở cuối. Tính $G_t$ lùi **và đặt lại $G=0$ khi có reward khác 0** (ranh giới điểm), $\gamma=0.99$:
$$G_t=r_t+\gamma G_{t+1},\qquad G\leftarrow0\ \text{khi gặp điểm}$$
⇒ các hành động gần thời điểm ghi/mất điểm nhận tín hiệu lớn hơn ($\gamma^{k}$ giảm theo khoảng cách).

### 5.4 Chuẩn hoá và cập nhật
Chuẩn hoá $G$ thành z-score trên cả batch: $\hat G=(G-\bar G)/\mathrm{std}(G)$ (**baseline + chia thang đo**). Tổng gradient
$$\nabla_\theta J\approx\sum_t\hat G_t\,\nabla_\theta\log\pi(a_t|x_t)$$
tích luỹ qua nhiều ván (batch ~10 ván) rồi cập nhật bằng **RMSprop** ($\text{cache}\leftarrow\rho\,\text{cache}+(1-\rho)g^2$; $\theta\leftarrow\theta+\alpha\,g/(\sqrt{\text{cache}}+\epsilon)$).

### 5.5 Backprop thủ công (để hiểu lại mạng nhỏ)
Với $\delta z=\hat G\,(a-p)$:
$$\nabla_{w_2}=h\cdot\delta z,\qquad\delta h=w_2\,\delta z,\ \ \delta h\leftarrow\delta h\odot\mathbb 1[h>0],\qquad\nabla_{W_1}=\delta h\,x^\top$$
Đây là bài tập nên **tự làm lại** để hiểu autograd thực hiện gì.

### 5.6 Bài học của bài viết
- PG học được dù reward thưa **nhưng cần nhiều ván** (hàng chục nghìn): kém hiệu quả mẫu so với con người.
- Con người dùng tri thức nền (vật lý, mục tiêu) nên học nhanh hơn: động lực cho transfer/model-based (Week 7–8).
- Reward thưa + credit assignment kém ⇒ khó mở rộng lên bài toán dài (động lực cho GAE, shaping).

---

## 6. Phân tích chuyên sâu các tham số trong phương trình (Parameter Deep Dive)

Trong bài toán RL ở tuần 1, có 5 tham số then chốt xuất hiện trực tiếp trong các phương trình toán học:

### 6.1 Hệ số chiết khấu $\gamma$ (Discount Factor)
* **Xuất hiện tại**: $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1} = r_{t+1} + \gamma G_{t+1}$.
* **Bản chất toán học**: Trọng số hình học làm suy giảm giá trị của các phần thưởng trong tương lai xa; bảo đảm tính hội tụ chuỗi tổng $\sum_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma} < \infty$.
* **Cơ chế tác động**:
  * **Tầm nhìn hiệu dụng (Effective Horizon)**: $H_{eff} \approx \frac{1}{1-\gamma}$.
    * Khi $\gamma = 0.9 \implies H_{eff} \approx 10$ bước.
    * Khi $\gamma = 0.99 \implies H_{eff} \approx 100$ bước (Pong from Pixels).
    * Khi $\gamma = 0.999 \implies H_{eff} \approx 1000$ bước.
  * **Khi $\gamma \to 0$ (Myopic - Cận thị)**: Agent chỉ tối đa hoá phần thưởng tức thì $r_{t+1}$. Trong Pong, nếu đỡ bóng không được thưởng ngay mà phải chờ đối thủ hỏng bóng sau 20 bước, agent với $\gamma=0$ hoàn toàn không học được hành động đỡ bóng.
  * **Khi $\gamma \to 1$ (Far-sighted - Viễn thị)**: Agent coi trọng phần thưởng sau 1000 bước ngang bước kế tiếp. Tuy nhiên, phương sai của $G_t$ tăng theo cấp số nhân vì cộng dồn hàng nghìn bước ngẫu nhiên; bài toán trở nên cực kỳ khó hội tụ.

---

### 6.2 Bước học $\alpha$ (Step-size / Learning Rate)
* **Xuất hiện tại**: $Q_{n+1} = Q_n + \alpha (R_n - Q_n)$.
* **Bản chất toán học**: Hệ số co trong phép nội suy lồi giữa ước lượng cũ và quan sát mới: $Q_{n+1} = (1-\alpha) Q_n + \alpha R_n$.
* **Cơ chế tác động**:
  * **Trường hợp $\alpha = \frac{1}{n}$**: Trọng số mọi mẫu trong lịch sử bằng nhau: $Q_{n+1} = \frac{1}{n} \sum_{i=1}^n R_i$. Đây là trung bình số học không chệch. Tuy nhiên, khi $n \to \infty$ thì $\alpha \to 0$, agent mất khả năng thích nghi nếu môi trường thay đổi phân phối (non-stationary).
  * **Trường hợp $\alpha = \text{const} \in (0, 1]$**: Trọng số suy giảm theo hàm mũ:
    $$Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha (1-\alpha)^{n-i} R_i$$
    Mẫu mới nhất có trọng số $\alpha$, mẫu cũ hơn suy giảm theo cơ số $(1-\alpha)$.
  * **Khi $\alpha$ quá lớn ($\alpha \to 1$)**: $Q_{n+1} \approx R_n$, agent bị "quên sạch quá khứ", giá trị $Q$ dao động dữ dội theo từng biến động ngẫu nhiên của mẫu đơn lẻ.
  * **Khi $\alpha$ quá nhỏ ($\alpha \to 0$)**: Tốc độ học tiến về 0, agent hầu như không nhúc nhích khỏi giá trị khởi tạo $Q_1$.

---

### 6.3 Tỉ lệ khám phá $\epsilon$ trong $\epsilon$-greedy
* **Xuất hiện tại**: $\pi(a) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal A|} & a = \arg\max_b Q(b) \\ \frac{\epsilon}{|\mathcal A|} & a \ne \arg\max_b Q(b) \end{cases}$.
* **Cơ chế tác động**:
  * **Khi $\epsilon = 1$**: Hoàn toàn ngẫu nhiên (Uniform exploration). Dữ liệu thu thập có độ bao phủ cao nhưng hoàn toàn không khai thác tri thức đã học để tích lũy phần thưởng.
  * **Khi $\epsilon = 0$**: Thuần tham lam (Pure exploitation). Nếu giá trị khởi tạo bị lệch hoặc một hành động kém tình cờ nhận reward dương ở lần thử đầu, agent sẽ vĩnh viễn chọn hành động đó và bỏ lỡ hành động tối ưu thực sự.
  * **Regret**: Với $\epsilon > 0$ cố định, regret tích lũy luôn tăng tuyến tính $O(\epsilon T)$ vì luôn có xác suất $\epsilon$ chọn nhầm nhánh xấu. Do đó trong thực tế cần lịch giảm dần $\epsilon(t) \to 0$ (GLIE).

---

### 6.4 Hệ số tự tin $c$ trong UCB1 (Upper Confidence Bound)
* **Xuất hiện tại**: $a_t = \arg\max_a \Big[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \Big]$.
* **Bản chất toán học**: Bán kính khoảng tin cậy suy ra từ bất đẳng thức Hoeffding: $\Pr\big(q_*(a) > Q_t(a) + U_t(a)\big) \le e^{-2 N_t(a) U_t(a)^2}$. Đặt vế phải bằng $t^{-4}$ suy ra $U_t(a) = \sqrt{\frac{2\ln t}{N_t(a)}}$, tức $c = \sqrt{2}$.
* **Cơ chế tác động**:
  * Số hạng $\sqrt{\frac{\ln t}{N_t(a)}}$ thể hiện **độ bất định (uncertainty)**. Khi tổng số bước $t$ tăng, độ bất định của mọi nhánh tăng nhẹ ($\ln t$); khi nhánh $a$ được chọn, $N_t(a)$ tăng làm độ bất định của nhánh đó giảm mạnh.
  * **Khi $c$ quá lớn**: Agent trở thành "kẻ nghiện khám phá", liên tục nhảy qua lại giữa các cánh tay chỉ vì chúng có $N$ nhỏ, ngay cả khi $Q$ của chúng rất thấp.
  * **Khi $c \to 0$**: Thuật toán thoái hoá về thuần tham lam $\arg\max Q$, mất khả năng thoát cực tiểu địa phương.
  * **Tại $c = \sqrt{2}$**: Đảm bảo cận regret tối ưu lý thuyết $O(\ln t)$ (Lai & Robbins bound).

---

### 6.5 Các siêu tham số trong Pong from Pixels (RMSprop & Batch Size)
* **Batch size (Số episode tích lũy)**:
  * Trong PG, mỗi ván đấu kết thúc bằng thắng (+1) hoặc thua (-1). Nếu cập nhật sau mỗi ván đơn lẻ ($B=1$), gradient cực kỳ ồn ào.
  * Gom batch $\sim 10$ ván đấu giúp $\nabla_\theta J \approx \frac{1}{B} \sum_{i=1}^B \sum_t \hat{G}_t \nabla \log \pi(a_t|s_t)$ triệt tiêu bớt nhiễu, hướng vector cập nhật ổn định hơn.
* **Tốc độ học $\alpha_{RMS}$ và hệ số phân rã $\rho$**:
  * $v_t = \rho v_{t-1} + (1-\rho) g_t^2$.
  * Với $\rho = 0.99$, bộ tích lũy duy trì trung bình bình phương gradient trong khoảng $\frac{1}{1-\rho} \approx 100$ bước gần nhất, giúp tự động chia nhỏ bước nhảy cho các tham số có gradient quá lớn và tăng tốc cho các tham số có gradient nhỏ.

---

## 7. Thử thách đặc trưng của RL

1. **Credit assignment** (phần thưởng trễ): hành động nào gây ra kết quả?
2. **Khám phá–khai thác**.
3. **Dữ liệu tương quan và không dừng** (policy đổi ⇒ phân phối dữ liệu đổi).
4. **Reward thưa** hoặc thiết kế sai (reward hacking): tối ưu "đúng" cái được đo nhưng sai ý định.
5. **Hiệu quả mẫu**: Atari cần $\sim10^7$ khung hình (~giờ chơi liên tục).
6. **Tái lập/độ nhạy hyperparameter**: kết quả dao động mạnh theo seed.
7. **Quan sát một phần** (POMDP): trạng thái thật không thấy được.

---

## 8. Bài tập và câu hỏi kiểm tra

**Lý thuyết**
1. Chứng minh $Q_{n+1}=Q_n+\frac1n(R_n-Q_n)$ cho trung bình mẫu.
2. Vì sao $\epsilon$-greedy cố định có regret tuyến tính?
3. Chứng minh $\partial\log\pi(a)/\partial z=a-\sigma(z)$ cho policy Bernoulli.
4. Vì sao Karpathy dùng **hiệu khung hình** thay vì một khung?
5. Giải thích vì sao gradient bandit là trường hợp đặc biệt của REINFORCE.

**Thực hành**
1. Cài bandit 10 tay (reward $\mathcal N(q_*(a),1)$); so sánh tham lam, $\epsilon=0.1$, UCB $c=2$ trên 2000 lần chạy × 1000 bước; vẽ reward trung bình và % chọn hành động tối ưu.
2. Chạy CartPole với agent ngẫu nhiên, agent luật cứng ($a=\mathbb 1[\text{góc cột}>0]$) và so sánh.
3. Đọc *Pong from Pixels*, chạy lại code (hoặc cài lại bằng PyTorch) và vẽ reward trung bình theo số ván.
4. Đọc Sutton & Barto chương 1–2 và làm bài tập Tic-Tac-Toe / bandit.

**Công thức cần thuộc**: $G_t=r+\gamma G_{t+1}$ · $Q\leftarrow Q+\alpha(R-Q)$ · UCB · $\partial\log\pi/\partial z=a-p$ · regret.
