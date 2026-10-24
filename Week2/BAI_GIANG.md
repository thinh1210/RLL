# Week 2 – Nền tảng RL: MDP, Quy hoạch động, Model-free, Q-learning

> Mục tiêu: sau khi đọc xong bạn phải **tự suy ra và chứng minh được** phương trình Bellman, hiểu bản chất giải tích tính co Banach giải thích vì sao Q-learning hội tụ, và đọc hiểu từng dòng của [`frozenlake_Qlearning.ipynb`](frozenlake_Qlearning.ipynb).  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

**Lộ trình đọc**: §1 Bài toán MDP → §2 Hàm giá trị và Chứng minh phương trình Bellman → §3 Tính tối ưu & Chứng minh tính co Banach → §4 Quy hoạch động & Chứng minh cải thiện Policy → §5 Monte Carlo & TD → §6 Điều khiển (SARSA, Q-learning) → §7 Đọc code → §8 Ví dụ tính tay → §9 Bài tập.

---

## 1. Bài toán: Markov Decision Process (MDP)

### 1.1 Định nghĩa
MDP là bộ năm $(\mathcal S,\mathcal A,P,R,\gamma)$:

| Thành phần | Ý nghĩa | FrozenLake 4×4 |
|---|---|---|
| $\mathcal S$ | tập trạng thái | 16 ô (0..15) |
| $\mathcal A$ | tập hành động | 4: trái, xuống, phải, lên |
| $P(s'\mid s,a)$ | xác suất chuyển trạng thái | trơn trượt: đi đúng hướng 1/3, lệch trái/phải mỗi bên 1/3 |
| $R(s,a,s')$ | phần thưởng | 1 khi tới đích (ô 15), còn lại 0 |
| $\gamma\in[0,1)$ | hệ số chiết khấu | 0.95 trong notebook |

**Tính chất Markov**: $P(s_{t+1}\mid s_t,a_t,s_{t-1},\dots)=P(s_{t+1}\mid s_t,a_t)$. "Hiện tại đủ để dự đoán tương lai". Nếu quan sát không đủ (ví dụ một khung hình Pong không cho biết vận tốc bóng) ta phải **gộp thêm thông tin** (xếp 4 frame – Week 3) để khôi phục tính Markov.

### 1.2 Policy
Policy $\pi(a\mid s)$ là phân phối xác suất trên hành động cho mỗi trạng thái. Policy tất định: $a=\pi(s)$.

### 1.3 Return và hệ số chiết khấu
$$G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\dots=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$$

* **Giải thích ký hiệu ngay sau phương trình**:
  * $G_t$: Lợi nhuận chiết khấu tích lũy từ bước $t$.
  * $r_{t+k+1}$: Phần thưởng nhận được tại bước $t+k+1$.
  * $\gamma \in [0, 1)$: Hệ số chiết khấu tương lai.

Vì sao chiết khấu?
1. **Hội tụ toán học**: nếu $|r|\le R_{max}$ thì $|G_t|\le R_{max}\sum\gamma^k=\dfrac{R_{max}}{1-\gamma}<\infty$.
2. Tương lai không chắc chắn nên đáng giá ít hơn.
3. Ưu tiên đạt mục tiêu **sớm**: với $\gamma<1$, đường ngắn cho $G$ lớn hơn.

Quan hệ đệ quy (rất quan trọng, mọi thứ sau đây xuất phát từ đây):
$$\boxed{G_t=r_{t+1}+\gamma G_{t+1}}$$

---

## 2. Hàm giá trị và phương trình Bellman

### 2.1 Định nghĩa
$$V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]$$

* **Giải thích ký hiệu**:
  * $V^\pi(s)$: State-value function, giá trị kỳ vọng khi ở trạng thái $s$ và tuân theo chính sách $\pi$.
  * $Q^\pi(s, a)$: Action-value function, giá trị kỳ vọng khi thực hiện hành động $a$ tại $s$ rồi tuân theo $\pi$.

Mối quan hệ qua lại giữa hai hàm:
$$V^\pi(s)=\sum_{a \in \mathcal A}\pi(a\mid s)\,Q^\pi(s,a)$$
$$Q^\pi(s,a)=\sum_{s' \in \mathcal S}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^\pi(s')\big]$$

---

### 2.2 Chứng minh chi tiết Phương trình Bellman kỳ vọng (Derivation of Bellman Expectation Equation)
Ta muốn biểu diễn $V^\pi(s)$ dưới dạng đệ quy theo các trạng thái kế tiếp $s'$.

* **Chứng minh toán học từng bước**:
  * **Bước 1**: Áp dụng định nghĩa của hàm giá trị trạng thái và quan hệ đệ quy $G_t = R_{t+1} + \gamma G_{t+1}$:
    $$V^\pi(s) = \mathbb E_\pi \big[ G_t \mid S_t = s \big] = \mathbb E_\pi \big[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \big]$$
  * **Bước 2**: Áp dụng **Định lý kỳ vọng toàn phần (Law of Total Expectation)** bằng cách điều kiện hoá theo hành động $A_t = a$:
    $$V^\pi(s) = \sum_{a \in \mathcal A} \pi(a \mid s) \, \mathbb E_\pi \big[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a \big]$$
  * **Bước 3**: Tiếp tục điều kiện hoá theo trạng thái kế tiếp $S_{t+1} = s'$ thông qua động học chuyển tiếp $P(s' \mid s, a)$:
    $$V^\pi(s) = \sum_{a \in \mathcal A} \pi(a \mid s) \sum_{s' \in \mathcal S} P(s' \mid s, a) \, \mathbb E_\pi \big[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \big]$$
  * **Bước 4**: Tách kỳ vọng tuyến tính:
    $$\mathbb E_\pi \big[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \big] = R(s, a, s') + \gamma \, \mathbb E_\pi \big[ G_{t+1} \mid S_{t+1} = s' \big]$$
    Theo tính chất Markov, phần thưởng tương lai từ bước $t+1$ trở đi chỉ phụ thuộc vào trạng thái hiện tại lúc đó là $S_{t+1} = s'$, tức $\mathbb E_\pi \big[ G_{t+1} \mid S_{t+1} = s' \big] = V^\pi(s')$.
  * **Bước 5**: Ghép lại ta thu được **Phương trình Bellman kỳ vọng**:
    $$\boxed{V^\pi(s) = \sum_{a \in \mathcal A} \pi(a \mid s) \sum_{s' \in \mathcal S} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma V^\pi(s') \Big]}$$
    và phương trình Bellman kỳ vọng cho $Q^\pi(s, a)$:
    $$\boxed{Q^\pi(s, a) = \sum_{s' \in \mathcal S} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma \sum_{a' \in \mathcal A} \pi(a' \mid s') Q^\pi(s', a') \Big]}$$
  Chứng minh hoàn tất. $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Chuỗi 3 phép thế sinh ra Phương trình Bellman:**
> - **Phép thế 1 (Thế đệ quy của Return $G_t$ vào trong kỳ vọng $V^\pi(s)$):**
>   - *Hàm gốc:* $V^\pi(s) = \mathbb{E}_\pi\big[G_t \mid S_t = s\big]$, trong đó $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$
>   - *Hàm đem thế:* Nhóm nhân tử chung $\gamma$ từ số hạng thứ 2 trở đi:
>     $$G_t = R_{t+1} + \gamma\underbrace{\big(R_{t+2} + \gamma R_{t+3} + \dots\big)}_{= G_{t+1}} = R_{t+1} + \gamma G_{t+1}$$
>   - *Thế vào $V^\pi(s)$:* $V^\pi(s) = \mathbb{E}_\pi\big[R_{t+1} + \gamma G_{t+1} \mid S_t = s\big]$.
> - **Phép thế 2 (Thế kỳ vọng của bước kế tiếp bằng hàm giá trị $V^\pi(s')$):**
>   Sau khi mở rộng kỳ vọng theo xác suất chọn hành động $\pi(a\mid s)$ và xác suất chuyển trạng thái $P(s'\mid s,a)$, bên trong dấu tổng xuất hiện cụm $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$.
>   Theo đúng định nghĩa của hàm $V^\pi(\cdot)$ ở mục 2.1 (chỉ đổi chỉ số thời gian từ $t \to t+1$ và tên biến từ $s \to s'$):
>   $$\underbrace{\mathbb{E}_\pi\big[G_{t+1} \mid S_{t+1} = s'\big]}_{\text{Kỳ vọng của toàn bộ tương lai từ } t+1} = V^\pi(s')$$
>   Thế $V^\pi(s')$ vào chỗ của $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s']$, ta lập tức khép kín được vòng lặp đệ quy của $V^\pi(s)$ theo $V^\pi(s')$!
> - **Phép thế 3 (Thế chéo qua lại giữa 2 hàm $V^\pi(s)$ và $Q^\pi(s,a)$):**
>   Ta có 2 mảnh ghép cơ bản ở mục 2.1:
>   $$(1)\;\; V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a\mid s)\, Q^\pi(s,a) \qquad \text{và} \qquad (2)\;\; Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'\mid s,a)\big[R(s,a,s') + \gamma V^\pi(s')\big]$$
>   - **Nếu đem (2) thế vào chữ $Q^\pi(s,a)$ trong (1):** Ta thu được phương trình tự đệ quy của riêng $V^\pi$:
>     $$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a\mid s) \underbrace{\left(\sum_{s' \in \mathcal{S}} P(s'\mid s,a)\big[R(s,a,s') + \gamma V^\pi(s')\big]\right)}_{\text{Thế từ (2) vào } Q^\pi(s,a)}$$
>   - **Ngược lại, nếu viết (1) tại trạng thái $s'$ là $V^\pi(s') = \sum_{a'} \pi(a'\mid s')Q^\pi(s',a')$ rồi đem thế vào chữ $V^\pi(s')$ trong (2):** Ta thu được phương trình tự đệ quy của riêng $Q^\pi$ (nền tảng của SARSA):
>     $$Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'\mid s,a)\left[R(s,a,s') + \gamma \underbrace{\sum_{a' \in \mathcal{A}} \pi(a'\mid s')Q^\pi(s',a')}_{\text{Thế từ (1) vào } V^\pi(s')}\right]$$

---

## 3. Tính tối ưu & Chứng minh tính co Banach (Banach Contraction Theorem)

### 3.1 Policy tối ưu
$V^*(s)=\max_\pi V^\pi(s)$, $Q^*(s,a)=\max_\pi Q^\pi(s,a)$. Luôn tồn tại một policy tất định tối ưu $\pi^*(s)=\arg\max_a Q^*(s,a)$. Do đó **biết $Q^*$ là biết cách chơi tối ưu** (không cần mô hình môi trường).

### 3.2 Phương trình Bellman tối ưu
Thay vì lấy trung bình có trọng số theo $\pi(a \mid s)$, policy tối ưu luôn chọn hành động cực đại hoá giá trị:
$$V^*(s)=\max_{a \in \mathcal A} \sum_{s' \in \mathcal S} P(s'\mid s,a) \big[ R(s, a, s') + \gamma V^*(s') \big]$$
$$\boxed{Q^*(s,a)=\sum_{s' \in \mathcal S} P(s'\mid s,a)\Big[ R(s, a, s') + \gamma \max_{a' \in \mathcal A} Q^*(s',a') \Big]}$$

Khác với phương trình kỳ vọng, có toán tử $\max$ nên phương trình Bellman tối ưu là **hệ phương trình phi tuyến**, không thể giải bằng nghịch đảo ma trận mà bắt buộc phải giải bằng phương pháp lặp.

---

### 3.3 Chứng minh toán học: Tính co của Toán tử Bellman và Định lý Banach
Định nghĩa toán tử Bellman tối ưu $T^*$ ánh xạ từ không gian hàm giá trị vào chính nó:
$$(T^* V)(s) = \max_{a \in \mathcal A} \left\{ R(s, a) + \gamma \sum_{s' \in \mathcal S} P(s' \mid s, a) V(s') \right\}$$
với chuẩn cực đại (Infinity/Supremum Norm) trên không gian metric Banach $(\mathbb R^{|\mathcal S|}, \|\cdot\|_\infty)$:
$$\|u - v\|_\infty = \max_{s \in \mathcal S} |u(s) - v(s)|$$

* **Bổ đề (Tính bán cộng tính của toán tử Max - Sub-additivity of Supremum)**:
  Với hai hàm thực bị chặn bất kỳ $f, g: \mathcal A \to \mathbb R$, ta luôn có:
  $$|\max_{a \in \mathcal A} f(a) - \max_{a \in \mathcal A} g(a)| \le \max_{a \in \mathcal A} |f(a) - g(a)|$$
  * *Chứng minh bổ đề*: Gọi $a^* = \arg\max_a f(a)$. Khi đó:
    $$\max_a f(a) - \max_a g(a) = f(a^*) - \max_a g(a) \le f(a^*) - g(a^*) \le \max_a |f(a) - g(a)|$$
    Hoán vị vai trò của $f$ và $g$, ta cũng có: $\max_a g(a) - \max_a f(a) \le \max_a |g(a) - f(a)|$.
    Từ hai bất đẳng thức trên suy ra $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$. $\square$

* **Chứng minh toán học: $T^*$ là một $\gamma$-Contraction Mapping**:
  Xét hai hàm giá trị bất kỳ $u, v \in \mathbb R^{|\mathcal S|}$:
  $$\begin{aligned}
  |(T^* u)(s) - (T^* v)(s)| &= \left| \max_a \Big( R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) u(s') \Big) - \max_a \Big( R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) v(s') \Big) \right| \\
  &\le \max_a \left| \Big( R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) u(s') \Big) - \Big( R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) v(s') \Big) \right| \quad (\text{áp dụng Bổ đề}) \\
  &= \max_a \left| \gamma \sum_{s'} P(s' \mid s, a) \big( u(s') - v(s') \big) \right| \\
  &\le \gamma \max_a \sum_{s'} P(s' \mid s, a) \big| u(s') - v(s') \big| \\
  &\le \gamma \max_a \sum_{s'} P(s' \mid s, a) \|u - v\|_\infty \quad (\text{vì } |u(s') - v(s')| \le \|u-v\|_\infty) \\
  &= \gamma \|u - v\|_\infty \cdot \max_a \underbrace{\sum_{s'} P(s' \mid s, a)}_{= 1} \\
  &= \gamma \|u - v\|_\infty
  \end{aligned}$$
  Lấy cực đại trên mọi $s \in \mathcal S$ ở vế trái:
  $$\boxed{\|T^* u - T^* v\|_\infty \le \gamma \|u - v\|_\infty}$$
  Vì $\gamma \in [0, 1)$, $T^*$ là một **phép co chặt hệ số $\gamma$** (Strict Contraction Mapping).

* **Áp dụng Định lý Điểm bất động Banach (Banach Fixed-Point Theorem)**:
  1. Tồn tại **duy nhất** một điểm bất động $V^* \in \mathbb R^{|\mathcal S|}$ thoả mãn $T^* V^* = V^*$.
  2. Với bất kỳ khởi tạo ban đầu $V_0$, dãy lặp $V_{k+1} = T^* V_k$ **chắc chắn hội tụ** về $V^*$:
     $$\lim_{k \to \infty} V_k = V^*$$
  3. Tốc độ hội tụ suy giảm theo cấp số nhân với sai số bị chặn:
     $$\|V_k - V^*\|_\infty \le \frac{\gamma^k}{1-\gamma} \|V_1 - V_0\|_\infty$$
  Đây là hòn đá tảng toán học bảo đảm thuật toán Value Iteration và Q-learning luôn tìm được nghiệm tối ưu toàn cục! $\square$

---

## 4. Quy hoạch động (Dynamic Programming)

### 4.1 Policy Evaluation (dự đoán)
Lặp phương trình Bellman kỳ vọng như một phép gán:
$$V_{k+1}(s)\leftarrow\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)[R(s, a, s')+\gamma V_k(s')]$$
Hội tụ về $V^\pi$ vì toán tử Bellman kỳ vọng $T^\pi$ cũng là một phép co hệ số $\gamma$.

### 4.2 Policy Improvement & Chứng minh Định lý cải thiện chính sách
Với $V^\pi$ đã biết, ta tạo ra policy tham lam mới:
$$\pi'(s) = \arg\max_{a \in \mathcal A} Q^\pi(s, a) = \arg\max_{a \in \mathcal A} \sum_{s'} P(s' \mid s, a) \big[ R(s, a, s') + \gamma V^\pi(s') \big]$$

* **Định lý cải thiện chính sách (Policy Improvement Theorem)**:
  Nếu với mọi trạng thái $s \in \mathcal S$, policy mới thoả mãn $Q^\pi(s, \pi'(s)) \ge V^\pi(s)$, thì giá trị của policy mới không bao giờ kém hơn policy cũ trên toàn bộ không gian:
  $$\boxed{V^{\pi'}(s) \ge V^\pi(s) \quad \forall s \in \mathcal S}$$

* **Chứng minh toán học chi tiết**:
  * Bắt đầu từ giả thiết: $V^\pi(s) \le Q^\pi(s, \pi'(s))$.
  * Khai triển $Q^\pi$ theo định nghĩa Bellman:
    $$V^\pi(s) \le \mathbb E_{\pi'} \big[ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s \big]$$
  * Thay thế bất đẳng thức $V^\pi(S_{t+1}) \le Q^\pi(S_{t+1}, \pi'(S_{t+1}))$ vào vế phải:
    $$V^\pi(s) \le \mathbb E_{\pi'} \Big[ R_{t+1} + \gamma \, \mathbb E_{\pi'} \big[ R_{t+2} + \gamma V^\pi(S_{t+2}) \mid S_{t+1} \big] \mid S_t = s \Big]$$
    $$= \mathbb E_{\pi'} \big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) \mid S_t = s \big]$$
  * Tiếp tục mở rộng vô hạn lần đệ quy (Telescoping unroll):
    $$V^\pi(s) \le \mathbb E_{\pi'} \Big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \mid S_t = s \Big] = V^{\pi'}(s)$$
  Định lý được chứng minh hoàn tất. $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Thế cuốn chiếu (Telescoping Substitution) trong Định lý Cải thiện Chính sách:**
> - **Bất đẳng thức cơ sở (đúng tại mọi trạng thái $x \in \mathcal{S}$):**
>   $$V^\pi(x) \le Q^\pi\big(x, \pi'(x)\big) = \mathbb{E}_{\pi'}\big[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = x\big]$$
> - **Thế bước 1 (Áp dụng tại $x = S_t = s$):**
>   $$V^\pi(s) \le \mathbb{E}_{\pi'}\Big[R_{t+1} + \gamma \underbrace{V^\pi(S_{t+1})}_{\text{Cụm đem thế tiếp}} \,\Big|\, S_t = s\Big]$$
> - **Thế bước 2 (Thay $V^\pi(S_{t+1})$ bằng cận trên của chính nó tại trạng thái $x = S_{t+1}$):**
>   Vì bất đẳng thức cơ sở đúng với mọi trạng thái, nên tại trạng thái $S_{t+1}$ ta cũng có $V^\pi(S_{t+1}) \le \mathbb{E}_{\pi'}[R_{t+2} + \gamma V^\pi(S_{t+2}) \mid S_{t+1}]$. Đem thế vào Bước 1:
>   $$V^\pi(s) \le \mathbb{E}_{\pi'}\Big[R_{t+1} + \gamma \underbrace{\big(R_{t+2} + \gamma V^\pi(S_{t+2})\big)}_{\text{Đã thế cho } V^\pi(S_{t+1})} \,\Big|\, S_t = s\Big] = \mathbb{E}_{\pi'}\Big[R_{t+1} + \gamma R_{t+2} + \gamma^2 \underbrace{V^\pi(S_{t+2})}_{\text{Cụm đem thế tiếp}} \,\Big|\, S_t = s\Big]$$
> - **Thế đến vô hạn ($k \to \infty$):**
>   Sau $k$ lần thế liên tiếp, số hạng chứa hàm cũ $V^\pi$ bị đẩy lùi về $\gamma^k V^\pi(S_{t+k})$. Vì $\gamma \in [0, 1)$ nên khi $k \to \infty$, $\gamma^k V^\pi(S_{t+k}) \to 0$ biến mất hoàn toàn, còn toàn bộ các phần thưởng phía trước hợp lại thành $\mathbb{E}_{\pi'}[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s] = V^{\pi'}(s)$!

### 4.3 Policy Iteration và Value Iteration
- **Policy Iteration**: lặp [đánh giá đến hội tụ → cải thiện]. Hội tụ sau hữu hạn bước vì số policy tất định hữu hạn và mỗi bước không tệ hơn.
- **Value Iteration**: gộp hai bước: $V_{k+1}(s)=\max_a\sum_{s'}P[R+\gamma V_k(s')]$, tức áp trực tiếp toán tử $\mathcal T$.

Hạn chế: cần **biết $P$** và quét toàn bộ trạng thái ("bootstrap + sweep"). RL thực tế thường không có $P$ → học từ mẫu (§5).

---

## 5. Học từ mẫu không cần mô hình (model-free prediction)

Mục tiêu: ước lượng $V^\pi$ từ các episode sinh ra bởi $\pi$.

### 5.1 Trung bình tăng dần (nền tảng của mọi cập nhật)
Trung bình mẫu $\mu_n=\frac1n\sum_{i\le n}x_i$ viết lại được:
$$\mu_n=\mu_{n-1}+\frac1n(x_n-\mu_{n-1})$$
Tổng quát: $\boxed{\text{ước lượng mới}=\text{cũ}+\alpha\,(\text{mục tiêu}-\text{cũ})}$. Số hạng trong ngoặc là **sai số**; $\alpha$ là bước học. Với $\alpha$ hằng, ước lượng là trung bình mũ, ưu tiên mẫu gần (phù hợp môi trường thay đổi/policy thay đổi).

### 5.2 Monte Carlo (MC)
Chờ hết episode, dùng return thực làm mục tiêu:
$$V(s_t)\leftarrow V(s_t)+\alpha\,[\,G_t-V(s_t)\,]$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc quét toàn bộ không gian và tính kỳ vọng giải tích của Quy hoạch động (Dynamic Programming - Bellman).
> * **Khuyết điểm của phương pháp cũ**: Dynamic Programming đòi hỏi biết tường minh ma trận chuyển trạng thái $P(s' \mid s, a)$ và hàm thưởng $R(s, a)$. Trong thực tế, môi trường là "hộp đen" phức tạp không thể biết trước xác suất.
> * **Logic thiết kế toán học**: Dựa trên Luật số lớn (Law of Large Numbers). Thay vì tính kỳ vọng giải tích $\sum_{s'} P(s' \mid s, a) [\dots]$, agent trực tiếp lấy mẫu các quỹ đạo thực tế và tính trung bình mẫu của $G_t$.
> * **Ưu/Nhược điểm**: **Không chệch (Unbiased)**, nhưng **phương sai cực lớn (High Variance)** do tích lũy nhiều bước ngẫu nhiên, và bắt buộc phải chờ đến khi kết thúc episode mới cập nhật được.

### 5.3 Temporal-Difference TD(0)
Dùng ước lượng hiện có của bước kế để thay cho phần đuôi (**bootstrapping**):
$$V(s_t)\leftarrow V(s_t)+\alpha\,[\,\underbrace{r_{t+1}+\gamma V(s_{t+1})}_{\text{TD target}}-V(s_t)\,],\qquad\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Từ Monte Carlo sang TD(0) qua 2 bước thế:**
> - **Bước 1 (Xuất phát từ công thức cập nhật Monte Carlo ở mục 5.2):**
>   $$V(s_t) \leftarrow V(s_t) + \alpha \big[\underbrace{G_t}_{\text{Mục tiêu MC}} - V(s_t)\big]$$
> - **Bước 2 (Thế khai triển đệ quy $G_t = r_{t+1} + \gamma G_{t+1}$ vào vị trí của $G_t$):**
>   $$V(s_t) \leftarrow V(s_t) + \alpha \big[r_{t+1} + \gamma \underbrace{G_{t+1}}_{\text{Chưa biết nếu chưa hết ván}} - V(s_t)\big]$$
> - **Bước 3 (Phép thế Bootstrapping — Thay biến ngẫu nhiên $G_{t+1}$ bằng hàm ước lượng $V(s_{t+1})$):**
>   Vì $V(s_{t+1}) \approx \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s_{t+1}]$ chính là ước lượng hiện tại của agent cho tổng phần thưởng từ bước $t+1$ trở đi, ta **thế trực tiếp $V(s_{t+1})$ vào chỗ của $G_{t+1}$**:
>   $$V(s_t) \leftarrow V(s_t) + \alpha \big[r_{t+1} + \gamma \underbrace{V(s_{t+1})}_{\text{Thế cho } G_{t+1}} - V(s_t)\big]$$
>   *Tác động lên mẫu:* Phép thế $G_{t+1} \leftarrow V(s_{t+1})$ cắt bỏ hoàn toàn đuôi ngẫu nhiên $(r_{t+2}, r_{t+3}, \dots, r_T)$, giúp giảm phương sai mẫu xuống mức tối thiểu và cập nhật được ngay lập tức tại bước $t+1$!

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế mục tiêu toàn phần $G_t = \sum_{k=0}^T \gamma^k R_{t+k+1}$ của Monte Carlo bằng mục tiêu bootstrap 1 bước $R_{t+1} + \gamma V(S_{t+1})$.
> * **Khuyết điểm của phương pháp cũ**: Monte Carlo có phương sai quá lớn khiến việc hội tụ rất chậm, và hoàn toàn bất lực trước các bài toán tương tác liên tục không có hồi kết (Continuous / Non-episodic tasks).
> * **Logic thiết kế toán học**: Áp dụng nguyên lý **Bootstrapping** (tự nâng đỡ bằng ước lượng sẵn có). Dựa trên quan hệ đệ quy Bellman $G_t = R_{t+1} + \gamma G_{t+1}$, TD(0) lấy phần thưởng thực nhận $R_{t+1}$ làm mỏ neo vào thực tế, còn phần đuôi tương lai $G_{t+1}$ được xấp xỉ bằng giá trị kỳ vọng dự đoán hiện tại $V(S_{t+1})$.
> * **Cơ chế giải quyết**: Chỉ chứa đúng 1 bước ngẫu nhiên $\implies$ **Phương sai cực thấp**, cho phép học trực tuyến từng bước thời gian (Online Step-by-Step Learning) ngay cả trong môi trường tương tác vô hạn. Đổi lại, TD(0) có **độ chệch (bias)** ban đầu do $V(S_{t+1})$ chưa chính xác, nhưng bias này sẽ triệt tiêu dần khi hàm giá trị co về điểm bất động Bellman.

| | Monte Carlo | TD(0) |
|---|---|---|
| Mục tiêu | $G_t$ | $r+\gamma V(s')$ |
| Bias | không | có (giảm dần) |
| Variance | cao | thấp |
| Cập nhật | cuối episode | mỗi bước |
| Dùng tính Markov | không | có |

### 5.4 n-step và TD($\lambda$)
- **n-step return**: $G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kr_{t+k+1}+\gamma^nV(s_{t+n})$. $n=1$: TD(0); $n=\infty$: MC. Cân bằng bias/variance.
- **$\lambda$-return**: trung bình mũ của mọi n-step:
$$G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$
- **Eligibility trace** (cách cài đặt "nhìn lùi", cập nhật được online): $e_t(s)=\gamma\lambda e_{t-1}(s)+\mathbb 1[s_t=s]$, rồi $V(s)\leftarrow V(s)+\alpha\,\delta_t\,e_t(s)$ cho mọi $s$. Trạng thái nào vừa được thăm gần đây nhận phần "công/lỗi" lớn hơn.

Ý tưởng $\lambda$/GAE sẽ quay lại ở Week 5.

---

## 6. Điều khiển model-free (tìm policy tối ưu)

Vì không có mô hình, ta ước lượng **Q** (không phải V) để chọn hành động mà không cần $P$.

### 6.1 Khám phá: $\epsilon$-greedy
$$\pi(a|s)=\begin{cases}1-\epsilon+\epsilon/|\mathcal A|&a=\arg\max_{a'}Q(s,a')\\ \epsilon/|\mathcal A|&\text{còn lại}\end{cases}$$

**Mệnh đề**: cải thiện tham lam theo $Q^\pi$ của policy $\epsilon$-soft $\pi$ cho policy $\epsilon$-greedy $\pi'$ không tệ hơn $\pi$.
*Chứng minh*:
$$Q^\pi(s,\pi'(s))=\tfrac{\epsilon}{|\mathcal A|}\sum_aQ^\pi(s,a)+(1-\epsilon)\max_aQ^\pi(s,a)$$
Vì $\max_a Q\ge\sum_a\frac{\pi(a|s)-\epsilon/|\mathcal A|}{1-\epsilon}Q$ (trung bình có trọng số không vượt max; các trọng số cộng lại bằng 1):
$$Q^\pi(s,\pi'(s))\ge\tfrac{\epsilon}{|\mathcal A|}\sum_aQ^\pi(s,a)+\sum_a\big(\pi(a|s)-\tfrac{\epsilon}{|\mathcal A|}\big)Q^\pi(s,a)=\sum_a\pi(a|s)Q^\pi(s,a)=V^\pi(s)\ \blacksquare$$

**GLIE** (Greedy in the Limit with Infinite Exploration): mọi cặp (s,a) được thăm vô hạn lần và $\epsilon_k\to0$ (ví dụ $\epsilon_k=1/k$). Khi đó MC/SARSA hội tụ về $Q^*$. Notebook dùng $\epsilon\leftarrow0.9993\,\epsilon$ mỗi episode: sau $k$ episode $\epsilon=0.9993^k$ (xuống ~0.37 sau 1400 episode, ~$3\times10^{-5}$ sau 15000).

### 6.2 SARSA (on-policy TD control)
Tên đến từ chuỗi $(s,a,r,s',a')$:
$$Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,r+\gamma\,Q(s',a')-Q(s,a)\,]$$
Mục tiêu dùng $a'$ do **chính policy ε-greedy đang chạy** chọn → học giá trị của policy *có khám phá*; kết quả thận trọng hơn (tránh vách đá nguy hiểm gần đường tối ưu).

### 6.3 Q-learning (off-policy TD control)
$$\boxed{Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\,]}$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế mục tiêu cập nhật On-policy $r + \gamma Q(s', a')$ của SARSA.
> * **Khuyết điểm của phương pháp cũ**: SARSA bắt buộc ước lượng giá trị của chính sách đang chạy (có yếu tố thăm dò $\epsilon$-greedy). Khi agent đi gần các vùng hiểm trở (như mép vực trong bài toán Cliff Walking), các cú nhảy ngẫu nhiên $\epsilon$ thường xuyên đẩy nó xuống vực, khiến SARSA gán nhãn đường đi tối ưu là nguy hiểm và học ra một chính sách đi đường vòng quá xa xôi, thận trọng thái quá.
> * **Logic thiết kế toán học**: Tách rời hoàn toàn **Chính sách hành vi (Behavior Policy)** dùng để sinh dữ liệu (vẫn giữ $\epsilon$-greedy để khám phá) và **Chính sách mục tiêu (Target Policy)** dùng để ước lượng giá trị (dùng toán tử $\max_{a'}$, tức chính sách tất định thuần tham lam).
> * **Cơ chế giải quyết**: Trực tiếp giải bài toán xấp xỉ phương trình Bellman tối ưu $T^* Q^* = Q^*$, tìm ra quỹ đạo tối ưu toàn cục thực sự bất chấp dữ liệu được thu thập bởi hành vi thăm dò ngẫu nhiên nào.

**Off-policy vs on-policy**
| | SARSA | Q-learning |
|---|---|---|
| Policy hành vi (sinh dữ liệu) | ε-greedy | ε-greedy |
| Policy đích (được ước lượng) | ε-greedy (chính nó) | greedy |
| Cần importance sampling? | không | không (vì chỉ dùng $\max$ trên $Q$, không cần tỉ lệ xác suất) |

**Expected SARSA**: thay $Q(s',a')$ bằng $\sum_{a'}\pi(a'|s')Q(s',a')$: giảm phương sai, có thể on hoặc off-policy.

### 6.4 Điều kiện hội tụ (Watkins & Dayan)
Q-learning tabular hội tụ về $Q^*$ (xác suất 1) nếu: (i) mọi (s,a) được cập nhật vô hạn lần; (ii) bước học thoả điều kiện Robbins–Monro:
$$\sum_t\alpha_t=\infty,\qquad\sum_t\alpha_t^2<\infty$$
(ví dụ $\alpha_t=1/t$: tổng phân kỳ, tổng bình phương hội tụ.) Điều kiện đầu đảm bảo "đủ sức đến được đích", điều kiện sau triệt tiêu nhiễu. Notebook dùng $\alpha=0.8$ hằng (vi phạm (ii)): vẫn cho kết quả tốt trong thực nghiệm nhưng Q dao động quanh nghiệm thay vì hội tụ chính xác.

### 6.5 Importance sampling (khi cần off-policy thật sự)
Để ước lượng kỳ vọng dưới $\pi$ từ mẫu theo $\mu$: $\mathbb E_{\pi}[f]=\mathbb E_\mu\big[\frac{\pi(a|s)}{\mu(a|s)}f\big]$. Với quỹ đạo dài, tích tỉ lệ $\prod_t\frac{\pi}{\mu}$ gây phương sai nổ → vì thế Q-learning (không cần tỉ lệ) rất tiện. Kỹ thuật này tái xuất hiện ở PPO (Week 5).

---

## 7. Đọc code: `frozenlake_Qlearning.ipynb`

### 7.1 Luồng chạy
```
env.reset() → obs
lặp:
  a ← ε-greedy(Q, obs)                 # select_eps_greedy_action
  obs', r, done ← env.step(a)
  Q[obs,a] += α (r + γ max_a' Q[obs',a'] - Q[obs,a])   # Q_learning
  nếu done: ε ← ε·0.9993; (mỗi 1000 ván) test_game; reset
```

### 7.2 Ánh xạ công thức ↔ code
| Công thức | Code |
|---|---|
| $\max_{a'}Q(s',a')$ | `best_value, _ = best_action_value(table, obs1)` |
| Mục tiêu $y=r+\gamma\max Q$ | `Q_target = reward + GAMMA * best_value` |
| Sai số TD | `Q_error = Q_target - table[(obs0, action)]` |
| Cập nhật | `table[(obs0, action)] += LEARNING_RATE * Q_error` |
| ε-greedy | `random.random() < epsilon` → hành động ngẫu nhiên |
| Bảng Q | `defaultdict(float)` (khởi tạo 0) |

### 7.3 Phân tích chuyên sâu các tham số và cơ chế tác động lên phương trình

Trong thuật toán Q-learning tabular giải FrozenLake, 4 tham số sau chi phối toàn bộ động học hội tụ:

#### (1) Hệ số chiết khấu $\gamma$ (Discount Factor)
* **Xuất hiện tại**:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big]$$
  và phương trình toán tử Bellman $\mathcal T Q = R + \gamma P \max Q$.
* **Cơ chế tác động toán học**:
  * **Hệ số co Banach**: Toán tử Bellman là phép co hệ số $\gamma$ trên chuẩn $\|\cdot\|_\infty$:
    $$\|\mathcal T u - \mathcal T v\|_\infty \le \gamma \|u - v\|_\infty \implies \|Q_k - Q^*\|_\infty \le \gamma^k \|Q_0 - Q^*\|_\infty$$
    Tốc độ hội tụ tỷ lệ nghịch với $\gamma$: $\gamma$ càng nhỏ, tốc độ co hình học $\gamma^k$ càng nhanh.
  * **Áp lực tìm đường ngắn nhất (Shortest Path Pressure)**:
    Trong FrozenLake, phần thưởng chỉ xuất hiện ở đích ($r_{goal} = 1$), mọi bước đi trên băng đều có $r = 0$.
    * Nếu $\gamma = 1$: Một đường đi đến đích sau 6 bước hay đi vòng vèo 100 bước đều có return $G = 1$. Agent **không có áp lực tìm đường ngắn nhất**!
    * Nếu $\gamma = 0.95$: Đường 6 bước có return $0.95^6 \approx 0.735$; đường 20 bước có return $0.95^{20} \approx 0.358$. Hàm $\max$ sẽ ưu tiên đường đi ngắn nhất để tối đa hoá return.
  * **Khi $\gamma \to 0$**: Mục tiêu $y = r$. Agent chỉ học được giá trị tại các ô liền kề đích, các ô khác mãi mãi bằng 0.

---

#### (2) Tốc độ học $\alpha$ (Learning Rate)
* **Xuất hiện tại**: $Q(s, a) \leftarrow (1-\alpha) Q(s, a) + \alpha \big[r + \gamma \max_{a'} Q(s', a')\big]$.
* **Cơ chế tác động toán học**:
  * Trong môi trường trơn trượt (`is_slippery=True`), khi chọn hành động "đi sang phải", xác suất đi đúng chỉ là $1/3$, xác suất trượt lên trên là $1/3$, trượt xuống dưới là $1/3$. Do đó mục tiêu TD $y = r + \gamma \max Q(s', \cdot)$ là một **biến ngẫu nhiên có phương sai cực lớn**.
  * **Khi $\alpha = 0.8$ (như trong notebook)**: Mẫu vừa quan sát chiếm tới $80\%$ trọng số ước lượng. Nếu agent tình cờ trượt chân vào hố, giá trị $Q$ ngay lập tức sụt giảm nghiêm trọng (bị trừ mất $80\%$ giá trị tích lũy). Điều này khiến bảng $Q$ dao động dữ dội quanh điểm tối ưu, không thể hội tụ phẳng.
  * **Khi $\alpha \in [0.05, 0.2]$**: Agent thực hiện trung bình có trọng số qua nhiều lần chuyển trạng thái, triệt tiêu nhiễu của môi trường ngẫu nhiên, ước lượng kỳ vọng $\mathbb E_{s'\sim P}[r + \gamma \max Q]$ mượt mà và chính xác hơn.
  * **Điều kiện hội tụ Robbins-Monro**: $\sum \alpha_t = \infty$ và $\sum \alpha_t^2 < \infty$. Nếu dùng $\alpha$ cố định, $Q$ chỉ hội tụ về một lân cận của $Q^*$. Muốn hội tụ chính xác tuyệt đối, cần giảm $\alpha_t = \frac{\alpha_0}{1 + c \cdot t}$.

---

#### (3) Tỉ lệ khám phá $\epsilon$ và Tốc độ suy giảm `EPS_DECAY_RATE`
* **Xuất hiện tại**: $\epsilon_{k+1} = \epsilon_k \cdot d$ với $d = 0.9993$.
* **Cơ chế tác động toán học**:
  * Giá trị $\epsilon$ sau $k$ episode: $\epsilon_k = \epsilon_0 \cdot d^k = 1.0 \times (0.9993)^k$.
    * Tại $k = 1000$: $\epsilon \approx (0.9993)^{1000} \approx 0.496$ (khám phá $50\%$).
    * Tại $k = 3000$: $\epsilon \approx 0.122$ (khám phá $12\%$).
    * Tại $k = 7000$: $\epsilon \approx 0.007$ (chuyển sang khai thác thuần túy).
  * **Nguy cơ khi decay quá nhanh (ví dụ $d = 0.99$)**:
    Sau 500 episode $\epsilon \approx 0.006$. Nhưng trong 500 episode đầu, do reward cực thưa, có thể agent chưa bao giờ tình cờ đặt chân được vào ô đích. Khi $\epsilon \approx 0$, policy trở thành tham lam trên một bảng $Q$ toàn số 0 $\implies$ agent đứng yên hoặc lặp vô tận, vĩnh viễn không bao giờ giải được game!
  * **Nguy cơ khi decay quá chậm ($d = 0.9999$)**: Agent dành quá nhiều thời gian đi ngẫu nhiên ngay cả khi đã học được đường đi tối ưu, làm giảm tỉ lệ thắng khi chạy thực tế.

---

#### (4) Tham số $\lambda$ trong TD($\lambda$) và Vết tích lũy (Eligibility Traces)
* **Xuất hiện tại**: $e_t(s, a) = \gamma \lambda e_{t-1}(s, a) + \mathbb 1[s_t=s, a_t=a]$ và $\Delta Q(s, a) = \alpha \delta_t e_t(s, a)$.
* **Cơ chế tác động toán học**:
  * $\lambda$ nội suy giữa TD(0) và Monte Carlo:
    * Khi $\lambda = 0$: $e_t(s, a) = \mathbb 1$, chỉ cập nhật đúng cặp $(s_t, a_t)$ vừa đi qua. Bias cao, variance thấp, thông tin lan ngược mỗi bước chỉ 1 ô.
    * Khi $\lambda = 1$: $e_t(s, a) = \gamma^{t - \tau}$, tương đương Monte Carlo. Mọi trạng thái dẫn đến bàn thắng đều được chia thưởng ngay trong episode đó. Không bias nhưng variance cao.
    * Khi $\lambda \in [0.8, 0.95]$: Tốc độ lan truyền giá trị từ đích về điểm xuất phát nhanh gấp nhiều lần so với TD(0), rút ngắn thời gian huấn luyện.

---

### 7.4 Điểm tinh tế trong cài đặt mã nguồn
- **Phần thưởng thưa (Sparse Reward)**: Chỉ ô 15 có $r=1$. Mọi cập nhật trước khi agent vô tình rơi vào ô 15 đều có target $y = 0$, nghĩa là bảng $Q$ không hề thay đổi. Do đó, pha khám phá ban đầu (khi $\epsilon$ lớn) quyết định sự sống còn của quá trình học.
- **Lỗi ngầm trong `best_action_value`**: Khởi tạo `max_value = 0` và điều kiện `if table[(s, a)] > max_value`. Nếu tất cả giá trị trong bảng đều âm (ví dụ môi trường phạt mỗi bước đi $r = -1$ để thúc giục đi nhanh), hàm này sẽ luôn trả về hành động mặc định 0 và bỏ qua hoàn toàn các giá trị $Q$ âm tốt hơn! Cách chuẩn mực là:
  ```python
  best_action = max(range(n_actions), key=lambda a: table[(state, a)])
  max_value = table[(state, best_action)]
  ```
- **Xử lý trạng thái kết thúc (Terminal State)**: Trạng thái hố và đích phải có target $y = r$ (không được cộng $\gamma \max Q(s', \cdot)$). Trong notebook, do các trạng thái kết thúc không bao giờ được dùng làm `obs0` để cập nhật tiếp, giá trị $Q(terminal, \cdot)$ luôn bằng 0, vô tình giữ cho công thức vẫn đúng.

---

## 8. Ví dụ tính tay (kiểm tra bạn đã hiểu cập nhật)

Đặt $\alpha=0.8,\ \gamma=0.95$, Q khởi tạo 0.

**Bước 1**: ở $s_{14}$, chọn "phải", tới đích $s_{15}$ với $r=1$; $Q(s_{15},\cdot)=0$.
$$y=1+0.95\cdot0=1,\quad\text{sai số}=1-0=1,\quad Q(s_{14},\text{phải})=0+0.8\cdot1=\mathbf{0.8}$$

**Bước 2**: ở $s_{13}$, chọn "phải" sang $s_{14}$, $r=0$; $\max_{a'}Q(s_{14},a')=0.8$.
$$y=0+0.95\cdot0.8=0.76,\quad\text{sai số}=0.76-0=0.76,\quad Q(s_{13},\text{phải})=0.8\cdot0.76=\mathbf{0.608}$$

**Bước 3**: lại thăm $(s_{14},\text{phải})$ nhưng lần này bị trượt sang hố, $r=0$, trạng thái kết thúc, $y=0$:
$$\text{sai số}=0-0.8=-0.8,\quad Q=0.8+0.8\cdot(-0.8)=\mathbf{0.16}$$
→ Với $\alpha=0.8$ một lần trượt xoá gần hết giá trị (đây là lý do $\alpha$ lớn gây nhiễu trong môi trường ngẫu nhiên; $\alpha$ nhỏ sẽ trung bình tốt hơn nhưng học chậm).

Phép tính này cho thấy **giá trị lan ngược từ đích**: giá trị ô liền kề đích ≈ $\gamma^0$, cách 2 bước ≈ $\gamma^1$, … (trước khi nhiễu và xác suất trượt làm giảm).

---

## 9. Bài tập và câu hỏi kiểm tra

**Lý thuyết**
1. Chứng minh $V^\pi(s)=\sum_a\pi(a|s)Q^\pi(s,a)$ và suy ra phương trình Bellman cho $Q^\pi$.
2. Chứng minh toán tử Bellman tối ưu là phép co $\gamma$ trong chuẩn $\|\cdot\|_\infty$ (làm lại §3.3 không nhìn).
3. Vì sao TD có bias mà vẫn thường học nhanh hơn MC?
4. Giải thích bằng lời: SARSA và Q-learning sẽ khác nhau thế nào trong bài toán "đi dọc vách đá" (cliff walking)?

**Thực hành**
1. Cài Value Iteration cho FrozenLake bằng `env.unwrapped.P` (mô hình thật) và so sánh $V^*$ với bảng Q học được.
2. Cài SARSA và Expected SARSA; so sánh đường học.
3. Thử $\alpha\in\{0.05,0.2,0.8\}$ và $\alpha_t=1/\sqrt t$; quan sát độ dao động.
4. Vẽ heatmap $\max_aQ(s,a)$ và mũi tên policy tham lam trên lưới 4×4.
5. Sửa `best_action_value` theo §7.4 và kiểm tra kết quả không đổi.

**Công thức cần thuộc**: $G_t=r+\gamma G_{t+1}$ · Bellman kỳ vọng · Bellman tối ưu · cập nhật TD · cập nhật Q-learning/SARSA · điều kiện Robbins–Monro.
