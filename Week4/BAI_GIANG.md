# Week 4 – Policy Gradient và Actor-Critic (REINFORCE, A2C)

> Mục tiêu: tự suy ra và **chứng minh chặt chẽ Định lý Policy Gradient**, hiểu vì sao tính nhân quả (reward-to-go) và baseline triệt tiêu phương sai mà không gây chệch (unbiased), vì sao Actor-Critic dùng advantage, và hiểu các lỗi trong notebook (đã đối chiếu code) bằng chính phép toán.  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

Các file: [`PolicyGradient.ipynb`](PolicyGradient.ipynb) · [`A2C.ipynb`](A2C.ipynb)

**Lộ trình**: §1 Policy-based vs Value-based → §2 Tham số hoá policy → §3 Định lý Policy Gradient & Chứng minh toán học → §4 Giảm phương sai (Chứng minh Nhân quả & Baseline) → §5 REINFORCE → §6 Actor-Critic/A2C → §7 Entropy → §8 Hành động liên tục → §9 Đọc code Vanilla PG → §10 Đọc code A2C và phân tích lỗi → §11 Ví dụ số → §12 Bài tập.

---

## 1. Vì sao cần Policy Gradient?

Value-based (Q-learning, DQN) học $Q$ rồi chọn $\arg\max$. Hạn chế:
- Khó với **hành động liên tục** (cần giải $\max_a Q(s,a)$ ở mỗi bước).
- Policy sinh ra từ $\arg\max$ là **tất định** và có thể thay đổi nhảy vọt khi Q thay đổi chút ít.
- Không biểu diễn được policy **ngẫu nhiên tối ưu** (ví dụ trò kéo-búa-bao, môi trường quan sát một phần).

Policy-based: tham số hoá trực tiếp $\pi_\theta(a|s)$ và **tối đa hoá** hiệu năng bằng leo gradient:
$$J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_{t=0}^{T}\gamma^tr_t\Big],\qquad\theta\leftarrow\theta+\alpha\,\nabla_\theta J(\theta)$$

* **Giải thích ký hiệu**:
  * $\theta \in \mathbb R^d$: Vector trọng số của mạng policy nơ-ron.
  * $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$: Quỹ đạo tương tác hoàn chỉnh.
  * $J(\theta)$: Hàm mục tiêu vô hướng biểu thị kỳ vọng lợi nhuận toàn cục.
  * $\alpha$: Tốc độ học (Learning rate).

Thay đổi nhỏ của $\theta$ ⇒ thay đổi nhỏ (mượt) của policy.

---

## 2. Tham số hoá policy

### 2.1 Softmax cho hành động rời rạc
Mạng ra logits $z_a(s;\theta)$:
$$\pi_\theta(a|s)=\frac{e^{z_a}}{\sum_ke^{z_k}}$$

**Chứng minh Gradient của log-prob theo logits**:
$$\ln\pi_\theta(a \mid s) = z_a - \ln\sum_{k} e^{z_k}$$
Lấy đạo hàm riêng theo logit thứ $j$:
$$\begin{aligned}
\frac{\partial \ln\pi_\theta(a \mid s)}{\partial z_j} &= \frac{\partial z_a}{\partial z_j} - \frac{1}{\sum_k e^{z_k}} \frac{\partial}{\partial z_j}\Big(\sum_k e^{z_k}\Big) \\
&= \mathbb 1[j=a] - \frac{e^{z_j}}{\sum_k e^{z_k}} \\
&= \mathbb 1[j=a] - \pi_\theta(j \mid s)
\end{aligned}$$
Tức là vector (one-hot của hành động đã chọn) trừ đi (xác suất dự đoán của mạng). Cập nhật đẩy logit của hành động đã chọn lên và các hành động khác xuống, tỉ lệ với mức "bất ngờ" của hành động.

---

## 3. Định lý Policy Gradient (Chứng minh giải tích từ đầu)

Xét quỹ đạo ngẫu nhiên $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ sinh ra bởi chính sách $\pi_\theta$ tương tác với môi trường $P$.
Hàm mật độ xác suất đồng thời của quỹ đạo $\tau$ là:
$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t)$$
Đặt tổng phần thưởng của quỹ đạo là $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}$. Mục tiêu tối ưu hoá là:
$$J(\theta) = \mathbb E_{\tau \sim \pi_\theta}[R(\tau)] = \int p_\theta(\tau) R(\tau) \, d\tau$$

* **Chứng minh toán học chi tiết**:
  * **Bước 1 (Log-Derivative Trick / Likelihood Ratio)**:
    Lấy đạo hàm gradient theo $\theta$ dưới dấu tích phân:
    $$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau) \, d\tau = \int \nabla_\theta p_\theta(\tau) \, R(\tau) \, d\tau$$
    Nhận thấy $\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau) \nabla_\theta \ln p_\theta(\tau)$. Thay vào:
    $$\nabla_\theta J(\theta) = \int p_\theta(\tau) \, \nabla_\theta \ln p_\theta(\tau) \, R(\tau) \, d\tau = \mathbb E_{\tau \sim \pi_\theta}\Big[ \nabla_\theta \ln p_\theta(\tau) \, R(\tau) \Big]$$
  * **Bước 2 (Khai triển logarit của xác suất quỹ đạo)**:
    Lấy logarit tự nhiên của $p_\theta(\tau)$:
    $$\ln p_\theta(\tau) = \ln p(s_0) + \sum_{t=0}^{T-1} \ln \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \ln P(s_{t+1} \mid s_t, a_t)$$
  * **Bước 3 (Triệt tiêu các thành phần không phụ thuộc tham số $\theta$)**:
    Lấy gradient theo $\theta$ hai vế:
    * $\nabla_\theta \ln p(s_0) = 0$ (phân phối trạng thái khởi đầu của môi trường không phụ thuộc $\theta$).
    * $\nabla_\theta \ln P(s_{t+1} \mid s_t, a_t) = 0$ (quy luật vật lý của môi trường hoàn toàn độc lập với $\theta$).
    Do đó:
    $$\nabla_\theta \ln p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t)$$
  * **Bước 4 (Kết luận)**:
    Thế kết quả vào biểu thức kỳ vọng:
    $$\boxed{\nabla_\theta J(\theta) = \mathbb E_{\tau \sim \pi_\theta}\left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \right) R(\tau) \right]}$$
  Chứng minh hoàn tất. $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Sơ đồ thế 3 tầng của Định lý Policy Gradient:**
> - **Tầng 1 (Thế $\nabla_\theta p_\theta(\tau)$ bằng hằng đẳng thức đạo hàm logarit):**
>   Từ quy tắc đạo hàm hàm logarit $\nabla_\theta \ln p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$, ta nhân chéo mẫu số để rút ra hàm thế:
>   $$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \ln p_\theta(\tau)$$
>   Đem thế vào tích phân gốc $\nabla_\theta J(\theta) = \int \underbrace{\nabla_\theta p_\theta(\tau)}_{\text{Thế vào đây}} R(\tau)\,d\tau$, ta khôi phục lại được mật độ xác suất $p_\theta(\tau)$ đứng đầu tích phân:
>   $$\nabla_\theta J(\theta) = \int p_\theta(\tau) \Big[\nabla_\theta \ln p_\theta(\tau) R(\tau)\Big] d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\Big[\nabla_\theta \ln p_\theta(\tau) R(\tau)\Big]$$
> - **Tầng 2 (Thế công thức tích chuỗi Markov $p_\theta(\tau)$ vào trong hàm $\ln(\cdot)$):**
>   Thay $p_\theta(\tau) = p(s_0)\prod_{t=0}^{T-1} \pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t, a_t)$ vào $\ln p_\theta(\tau)$ và dùng tính chất $\ln(A \cdot B) = \ln A + \ln B$:
>   $$\ln p_\theta(\tau) = \ln p(s_0) + \sum_{t=0}^{T-1} \ln \pi_\theta(a_t\mid s_t) + \sum_{t=0}^{T-1} \ln P(s_{t+1}\mid s_t, a_t)$$
> - **Tầng 3 (Lấy $\nabla_\theta$ và thế ngược lại vào Kỳ vọng ở Tầng 1):**
>   Khi tác động $\nabla_\theta$ lên Tầng 2, hai cụm $\nabla_\theta \ln p(s_0) = 0$ và $\nabla_\theta \ln P(s_{t+1}\mid s_t, a_t) = 0$ tự động rơi rụng vì không chứa biến $\theta$. Đem phần còn lại $\nabla_\theta \ln p_\theta(\tau) = \sum_{t=0}^{T-1}\nabla_\theta \ln \pi_\theta(a_t\mid s_t)$ **thế ngược vào vị trí $\nabla_\theta \ln p_\theta(\tau)$ ở Tầng 1**, ta thu được công thức cuối cùng ở Bước 4!

* **Ý nghĩa đột phá**: Môi trường thực tế $P(s' \mid s, a)$ có thể cực kỳ phức tạp và không thể biết trước, nhưng đạo hàm của nó **hoàn toàn biến mất** khỏi công thức gradient! Ta có thể tối ưu hoá trực tiếp bằng lấy mẫu Monte Carlo trên $N$ quỹ đạo thực tế:
  $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t^{(i)} \mid s_t^{(i)}) \right) R(\tau^{(i)})$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế toàn bộ nhánh thuật toán Value-based (Q-learning, DQN) vốn phải ước lượng hàm giá trị rồi chọn hành động qua $\arg\max_a Q(s, a)$.
> * **Khuyết điểm của phương pháp cũ**: Không áp dụng được trực tiếp cho không gian hành động liên tục $\mathbb R^m$ (bài toán robot, xe tự hành không thể tính $\max$ trên vô hạn hành động ở mỗi bước), và không thể học được các chính sách ngẫu nhiên tối ưu (Stochastic optimal policies).
> * **Logic thiết kế toán học**: Tham số hóa trực tiếp chính sách $\pi_\theta(a \mid s)$. Nhờ Log-Derivative Trick $\nabla p = p \nabla \ln p$, gradient của hàm mục tiêu toàn cục $J(\theta)$ được chuyển thành kỳ vọng của gradient log-xác suất. Toàn bộ đạo hàm của quy luật môi trường $P(s' \mid s, a)$ triệt tiêu hoàn toàn.
> * **Cơ chế giải quyết**: Cho phép tối ưu trực tiếp chính sách liên tục hoặc rời rạc thông qua lấy mẫu Monte Carlo trên các quỹ đạo thực tế mà không cần mô hình thế giới.

---

## 4. Giảm phương sai (Variance Reduction)

Công thức ở §3 không chệch nhưng có **phương sai cực lớn** vì mọi hành động tại mọi bước $t$ đều bị nhân với cùng một tổng phần thưởng $R(\tau)$ của toàn bộ episode.

### 4.1 Tính nhân quả (Causality) & Chứng minh Reward-to-Go
Hành động tại thời điểm $t$ không thể đảo ngược thời gian để gây ra phần thưởng ở quá khứ $t' < t$. Ta chứng minh rằng phần thưởng quá khứ có kỳ vọng trực giao bằng 0.

* **Chứng minh toán học (Cơ chế loại bỏ phần thưởng quá khứ)**:
  Tách tổng phần thưởng $R(\tau) = \sum_{t'=0}^{t-1} \gamma^{t'} r_{t'+1} + \sum_{t'=t}^{T-1} \gamma^{t'} r_{t'+1}$.
  Xét số hạng chứa phần thưởng quá khứ $t' < t$:
  $$\mathbb E_{\tau}\Big[ \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, r_{t'+1} \Big]$$
  Áp dụng luật kỳ vọng có điều kiện theo toàn bộ lịch sử tương tác đến trạng thái $s_t$, tức $\tau_{0:t} = (s_0, a_0, \dots, s_t)$:
  $$\mathbb E_{\tau}\Big[ \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, r_{t'+1} \Big] = \mathbb E_{\tau_{0:t}}\Big[ \mathbb E_{a_t \sim \pi_\theta(\cdot \mid s_t)}\big[ \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, r_{t'+1} \mid \tau_{0:t} \big] \Big]$$
  Vì $t' < t$, phần thưởng $r_{t'+1}$ đã xảy ra trong quá khứ nên là một hằng số đã biết đối với điều kiện $\tau_{0:t}$. Do đó đưa $r_{t'+1}$ ra ngoài kỳ vọng theo $a_t$:
  $$= \mathbb E_{\tau_{0:t}}\Big[ r_{t'+1} \, \mathbb E_{a_t \sim \pi_\theta(\cdot \mid s_t)}\big[ \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \mid s_t \big] \Big]$$
  Tính kỳ vọng bên trong:
  $$\mathbb E_{a_t \sim \pi_\theta}\big[ \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \mid s_t \big] = \sum_{a_t \in \mathcal A} \pi_\theta(a_t \mid s_t) \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} = \nabla_\theta \sum_{a_t \in \mathcal A} \pi_\theta(a_t \mid s_t) = \nabla_\theta (1) = 0$$
  Vì vậy $r_{t'+1} \cdot 0 = 0$.
  Từ đó, toàn bộ phần thưởng quá khứ triệt tiêu về 0, ta thu được dạng **Reward-to-Go**:
  $$\boxed{\nabla_\theta J(\theta) = \mathbb E_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, G_t \right]},\qquad G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'+1}$$
  Chứng minh hoàn tất. $\square$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế tổng phần thưởng cả quỹ đạo $R(\tau) = \sum_{t'=0}^T r_{t'+1}$.
> * **Khuyết điểm của phương pháp cũ**: Vi phạm tính nhân quả. Nhân hành động ở bước $t$ với phần thưởng ở quá khứ $t' < t$ tạo ra phương sai khổng lồ, gán "công/tội" cho hành động một cách vô lý.
> * **Logic thiết kế toán học**: Dựa trên nguyên lý Nhân quả (Causality). Bằng chứng minh giải tích $\mathbb E[\nabla \ln \pi(a_t \mid s_t) r_{t'}] = 0$ với mọi $t' < t$, ta loại bỏ toàn bộ phần thưởng quá khứ ra khỏi công thức mà không làm lệch kỳ vọng.
> * **Cơ chế giải quyết**: Giảm phương sai của gradient ước lượng xuống đáng kể, bảo đảm tính không chệch hoàn hảo.

---

### 4.2 Baseline $b(s)$ & Chứng minh tính không chệch (Unbiased Baseline)
$$\nabla_\theta J(\theta) = \mathbb E_{\tau}\left[ \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, \big( G_t - b(s_t) \big) \right]$$

* **Chứng minh toán học (Baseline không làm thay đổi kỳ vọng gradient)**:
  Xét số hạng do $b(s_t)$ đưa vào:
  $$\mathbb E_\tau \big[ \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, b(s_t) \big] = \mathbb E_{s_t}\Big[ \mathbb E_{a_t \sim \pi_\theta(\cdot \mid s_t)}\big[ \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, b(s_t) \mid s_t \big] \Big]$$
  Vì $b(s_t)$ chỉ phụ thuộc $s_t$, độc lập với hành động $a_t$:
  $$= \mathbb E_{s_t}\Big[ b(s_t) \, \sum_{a_t} \pi_\theta(a_t \mid s_t) \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} \Big] = \mathbb E_{s_t}\Big[ b(s_t) \, \nabla_\theta \underbrace{\sum_{a_t} \pi_\theta(a_t \mid s_t)}_{=1} \Big] = \mathbb E_{s_t}[b(s_t) \cdot 0] = 0$$
  Chứng minh hoàn tất. $\square$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế tín hiệu phần thưởng tuyệt đối $G_t$ trong hàm gradient.
> * **Khuyết điểm của phương pháp cũ**: Nếu mọi phần thưởng đều dương ($r > 0 \implies G_t > 0$), mọi hành động được thử nghiệm đều nhận tín hiệu dương và bị đẩy tăng xác suất như nhau, khiến việc phân biệt hành động tốt/xấu cực kỳ chậm chạp.
> * **Logic thiết kế toán học**: Trừ đi một đại lượng $b(s_t) = V(s_t)$ độc lập với hành động $a_t$. Vì $\mathbb E[\nabla \ln \pi(a_t \mid s_t) b(s_t)] = 0$, kỳ vọng gradient được bảo toàn nguyên vẹn.
> * **Cơ chế giải quyết**: Tín hiệu $G_t - b(s_t)$ trở thành Hàm lợi thế (Advantage): hành động tốt hơn mức trung bình nhận giá trị dương, kém hơn nhận giá trị âm, triệt tiêu phương sai mạnh mẽ.

### 4.3 Advantage Function
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$
Dạng chuẩn mực của Policy Gradient:
$$\boxed{\nabla_\theta J(\theta) = \mathbb E_{s \sim d^\pi, a \sim \pi_\theta}\Big[ \nabla_\theta \ln \pi_\theta(a \mid s) \, A^\pi(s, a) \Big]}$$

---

## 5. REINFORCE (Vanilla Policy Gradient)

Thuật toán (Monte Carlo, chờ hết episode):
1. Chạy policy hiện tại, thu một (hoặc vài) episode.
2. Tính $G_t$ cho mọi $t$.
3. Tính hàm mục tiêu thay thế (surrogate) và cập nhật:
$$L(\theta)=-\frac1T\sum_t\log\pi_\theta(a_t|s_t)\,\big(G_t-b\big)$$
`loss.backward()` cho đúng $-\nabla_\theta J$ ước lượng (vì $\nabla L=-\frac1T\sum\nabla\log\pi\cdot(G-b)$).

**Tính chất**: on-policy (dữ liệu phải đến từ policy hiện tại → vứt dữ liệu sau mỗi lần cập nhật → kém hiệu quả dữ liệu), phương sai cao, chỉ cập nhật cuối episode.

---

## 6. Actor-Critic và A2C

### 6.1 Ý tưởng
Dùng **critic** $V_\phi(s)$ (mạng học) làm baseline và để **bootstrap**, giảm phương sai nhiều hơn nữa (đánh đổi một chút bias):
- **Actor**: $\pi_\theta(a|s)$.
- **Critic**: $V_\phi(s)\approx V^\pi(s)$ học bằng hồi quy.

### 6.2 TD error là ước lượng advantage
$$\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$$
Nếu $V_\phi=V^\pi$ chính xác thì
$$\mathbb E[\delta_t\mid s_t,a_t]=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)$$
nên $\delta_t$ là ước lượng **không chệch** của advantage (khi critic đúng) và có phương sai thấp (chỉ một bước ngẫu nhiên).

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Thế từ REINFORCE sang Actor-Critic (A2C):**
> - **Phương trình gốc (REINFORCE có Baseline ở mục 4.2):**
>   $$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t\mid s_t) \Big(\underbrace{G_t}_{\text{Thế 1}} - \underbrace{b(s_t)}_{\text{Thế 2}}\Big)\right]$$
> - **Phép thế 1 (Thay Baseline $b(s_t)$ bằng mạng Critic $V_\phi(s_t)$):**
>   Chọn hàm cơ sở là mạng giá trị trạng thái đang học: $b(s_t) = V_\phi(s_t)$.
> - **Phép thế 2 (Thay Return toàn tập $G_t$ bằng mục tiêu Bootstrap 1 bước hoặc $n$ bước của Critic):**
>   Từ quan hệ đệ quy $G_t = r_t + \gamma G_{t+1}$, ta thay phần đuôi ngẫu nhiên $G_{t+1}$ bằng ước lượng của Critic $V_\phi(s_{t+1})$:
>   $$G_t \approx r_t + \gamma V_\phi(s_{t+1}) \qquad \text{(hoặc } n\text{-bước: } G_t \approx \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n})\text{)}$$
> - **Kết quả sau khi thế đồng thời cả 2 hàm vào cụm $(G_t - b(s_t))$:**
>   $$G_t - b(s_t) \;\longrightarrow\; \big(r_t + \gamma V_\phi(s_{t+1})\big) - V_\phi(s_t) = \delta_t$$
>   *Ý nghĩa:* Phép thế kép này biến cụm $(G_t - b(s_t))$ của REINFORCE (phải đợi hết ván và phương sai cao) thành sai số TD $\delta_t$ của Actor-Critic (tính được ngay lập tức và phương sai cực thấp)!

### 6.3 N-step advantage (dùng trong A2C của repo)
$$\hat A_t=\underbrace{\sum_{k=0}^{n-1}\gamma^kr_{t+k}+\gamma^nV_\phi(s_{t+n})}_{R_t^{(n)}}-V_\phi(s_t)$$
Code thu $n$ bước (`n_steps`) rồi tính $R_t$ lùi: nếu episode chưa kết thúc, khởi tạo phần đuôi bằng $V_\phi(s_{cuối})$ ("run_add = critic(obs)"), nếu đã kết thúc thì bằng 0.

### 6.4 Hàm mất mát của A2C
Cực tiểu hoá tổng ba thành phần:
$$L=\underbrace{-\,\mathbb E\big[\log\pi_\theta(a|s)\,\hat A\big]}_{\text{actor}}\;+\;c_v\underbrace{\mathbb E\big[(R-V_\phi(s))^2\big]}_{\text{critic}}\;-\;\beta\underbrace{\mathbb E\big[H(\pi_\theta(\cdot|s))\big]}_{\text{entropy bonus}}$$
- $\hat A$ phải **tách khỏi đồ thị** (`.detach()`): actor không được lan gradient vào critic qua advantage.
- $c_v$ thường 0.5 (repo dùng 1), $\beta\approx0.01$.

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc lấy mẫu Monte Carlo toàn tập $G_t$ trong REINFORCE bằng sai số TD 1 bước hoặc $n$ bước.
> * **Khuyết điểm của phương pháp cũ**: REINFORCE bắt buộc phải chờ đến cuối episode mới cập nhật được và có phương sai lớn.
> * **Logic thiết kế toán học**: Kiến trúc Actor-Critic: Actor học chính sách $\pi_\theta$, Critic học hàm giá trị $V_\phi$. Sử dụng sai số TD $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ làm ước lượng cho Advantage.
> * **Cơ chế giải quyết**: Cho phép cập nhật chính sách từng bước online hoặc theo mini-batch ngắn đa worker song song, vừa giảm phương sai tối đa vừa đẩy nhanh tốc độ hội tụ.

### 6.5 "Synchronous" trong A2C
Chạy $N$ môi trường song song (repo: 40), gom mọi chuyển tiếp thành một batch. Lợi ích: batch **đa dạng** (nhiều trạng thái khác nhau) → giảm tương quan, thay cho replay buffer vốn không dùng được cho on-policy. A3C (asynchronous) cập nhật bất đồng bộ từ nhiều worker; A2C là phiên bản đồng bộ, thường tận dụng GPU tốt hơn.

---

## 7. Entropy regularization

$$H(\pi(\cdot|s))=-\sum_a\pi(a|s)\log\pi(a|s)\ \in[0,\log|\mathcal A|]$$
Entropy lớn = policy gần ngẫu nhiên đều; entropy 0 = tất định. Cộng $+\beta H$ **vào mục tiêu** (tức trừ khỏi loss) ngăn policy co về tất định quá sớm (thoát khám phá → kẹt cực tiểu cục bộ).

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc tối ưu hàm mất mát thuần túy không có ràng buộc đa dạng.
> * **Khuyết điểm của phương pháp cũ**: Khi một hành động tình cờ nhận được advantage dương ban đầu, xác suất của nó vọt lên gần 1.0, các hành động khác tụt về 0.0, khiến gradient đạo hàm bằng 0 $\implies$ **Sụp đổ chính sách sớm (Premature Policy Collapse)** vào cực tiểu địa phương.
> * **Logic thiết kế toán học**: Đưa Shannon Entropy $\mathcal H(\pi) = -\sum \pi \ln \pi$ vào hàm mục tiêu với trọng số $\beta$. Đạo hàm của entropy sẽ tạo ra một lực đẩy ngược kéo các xác suất đang bị ép về 0 trở lại phân phối đều nếu chúng chưa được khám phá đủ.

**Chú ý dấu** (nguồn lỗi hay gặp): với `ent = Σ p·log p` (= $-H$):
- Đúng: `loss = pg_loss + vl_loss + β·ent`   (≡ `… − β·H`).
- Sai: `loss = pg_loss + vl_loss − β·ent`   (≡ `… + β·H`, **phạt** entropy).

---

## 8. Hành động liên tục (xem đầy đủ ở Week 5)
Policy Gaussian: $\log\pi(a|s)=-\dfrac{(a-\mu)^2}{2\sigma^2}-\log\sigma-\dfrac12\log2\pi$. Gradient theo $\mu$: $\dfrac{a-\mu}{\sigma^2}$ — đẩy trung bình về phía hành động có advantage dương. Policy gradient xử lý được trực tiếp mà không cần $\max_aQ$.

---

## 9. Đọc code: `PolicyGradient.ipynb` (CartPole)

**Môi trường**: CartPole, quan sát 4 chiều (vị trí xe, vận tốc xe, góc cột, vận tốc góc), 2 hành động (trái/phải), reward +1 mỗi bước cột còn đứng; `v0` kết thúc ở 200 bước.

**Mạng**: `PG_nn` = Linear(4→64) → ReLU → Linear(64→2) cho logits.

**Vòng lặp** (mỗi episode = một cập nhật):
```
chọn a ~ softmax(logits)  →  env.step  →  lưu Memory
khi done:
  disc_rewards = G_t (lùi)                 # discounted_rewards
  baseline.extend(disc_rewards)            # deque 30000
  disc_rewards -= mean(baseline)           # trừ baseline
  loss = -mean(disc_rewards · logπ(a))  + β·mean(Σ p log p)
  backward; Adam step
```

**Ánh xạ**:
| Công thức | Code |
|---|---|
| $G_t=r_t+\gamma G_{t+1}$ | `run_add = run_add*gamma + reward` (duyệt ngược) |
| baseline $b$ | `np.mean(baseline)` — hằng số toàn cục, **không phụ thuộc state** |
| $-\log\pi(a|s)\cdot(G-b)$ | `-torch.mean(disc_rewards_t * game_act_log_softmax_t)` |
| $+\beta(-H)$ | `l_entropy = β·mean(Σ p log p)`, `loss = loss + l_entropy` (**đúng dấu**) |

### 9.1 Lỗi indexing (quan trọng)
```python
game_act_log_softmax_t = F.log_softmax(acts, dim=1)[:, [e.action for e in experience]]
```
`acts` có shape `[T,2]`; `[:, idx]` với `idx` dài $T$ cho shape **`[T,T]`**, phần tử $(i,j)=\log\pi(a_j|s_i)$ — mọi trạng thái × mọi hành động đã chọn. Khi nhân với `disc_rewards_t` (shape `[T]`, broadcast theo cột) và lấy `mean`:
$$\frac1{T^2}\sum_i\sum_j G_j\,\log\pi(a_j|s_i)$$
thay vì $\frac1T\sum_tG_t\log\pi(a_t|s_t)$. Đây là tổng chéo **sai** (ghép return của bước $j$ với xác suất ở trạng thái $i\ne j$). Cách đúng:
```python
logp = F.log_softmax(acts, dim=1)
actions = torch.tensor([e.action for e in experience])
logp_a = logp.gather(1, actions[:, None]).squeeze(1)      # shape [T]
loss = -(disc_rewards_t * logp_a).mean() + l_entropy
```
Vì CartPole đơn giản và 2 hành động, thuật toán vẫn "học được chút" nhờ các số hạng chéo vẫn có tương quan, nhưng không phải policy gradient đúng.

### 9.2 Điểm khác
- Baseline là **trung bình toàn bộ return trong deque** (không theo state) nên chỉ giảm phương sai một phần; baseline phụ thuộc state $V(s)$ (A2C) tốt hơn.
- Chuẩn hoá return (chia độ lệch chuẩn) thường giúp ổn định thêm.
- `F.softmax(act)` không có `dim` (cảnh báo ở PyTorch mới).

---

## 10. Đọc code: `A2C.ipynb`

### 10.1 Kiến trúc
`A2C_nn`: thân chung Linear(4→64)+ReLU, hai đầu: `policy` (64→2, logits) và `value` (64→1). Chia sẻ thân giúp tái dùng đặc trưng nhưng gradient từ hai loss tranh chấp.

### 10.2 Thu thập dữ liệu
`Env.step(agent)` chạy `n_steps=1` bước/môi trường; 40 môi trường → batch 40 chuyển tiếp mỗi vòng. `discounted_rewards` tính $R_t$ với bootstrap $V(s_{cuối})$ nếu chưa kết thúc.

### 10.3 Ánh xạ loss
| Số hạng | Code |
|---|---|
| $\hat A=R-V$ | `adv = rewards - log_val[1].detach()` |
| actor | `pg_loss = -mean(act_log_softmax * adv)` |
| critic | `vl_loss = F.mse_loss(log_val[1].squeeze(-1), rewards)` |
| $-\beta H$ | `entropy_loss = β·mean(Σ softmax·log_softmax)` |
| tổng | `loss = pg_loss + vl_loss - entropy_loss` |

### 10.4 Hai lỗi giải thích hiện tượng "model không học" (README ghi "Bug Alert")
**(a) Lỗi shape giống §9.1**: `F.log_softmax(...)[:, actions]` → `[N,N]`. Hơn nữa `log_val[1]` có shape `[N,1]` còn `rewards` shape `[N]`: `rewards - log_val[1].detach()` broadcast thành **`[N,N]`** (mỗi return trừ mọi giá trị V). Hậu quả: $\hat A$ không còn là advantage của từng mẫu.
Sửa:
```python
logits, values = net(states); values = values.squeeze(-1)           # [N]
logp = F.log_softmax(logits, 1).gather(1, actions[:,None]).squeeze(1)  # [N]
adv = (rewards - values).detach()                                      # [N]
pg_loss = -(logp * adv).mean()
```
**(b) Sai dấu entropy**. Đặt $e=\sum p\log p=-H$, `entropy_loss=βe=-βH`. Code: `loss = pg + vl - entropy_loss = pg + vl + βH`. Cực tiểu loss ⇒ **giảm** $H$ ⇒ policy tiến tới tất định ⇒ khớp với biểu đồ README: model ưu tiên **luôn cùng một hành động**. Đúng phải là `loss = pg + vl + entropy_loss` (với `entropy_loss=βe`) hay `− β·H`.

**(c) Giải thích "loss giảm rất nhanh"**: nhánh entropy bị đảo dấu cùng với tín hiệu advantage sai ⇒ loss dễ dàng hạ bằng cách làm policy tất định (không cải thiện reward).

### 10.5 Phân tích chuyên sâu các tham số và cơ chế tác động lên hàm mất mát A2C

Hàm mất mát tổng của A2C là sự phối hợp của ba thành phần:
$$L(\theta) = \underbrace{-\frac{1}{N}\sum_{i=1}^N \log \pi_\theta(a_i \mid s_i) \hat A_i}_{L_{pg}\text{ (Actor)}} + c_v \underbrace{\frac{1}{N}\sum_{i=1}^N \big(R_i - V_\theta(s_i)\big)^2}_{L_v\text{ (Critic)}} - \beta \underbrace{\frac{1}{N}\sum_{i=1}^N H\big(\pi_\theta(\cdot \mid s_i)\big)}_{L_{ent}\text{ (Khám phá)}}$$

Các tham số chi phối trực tiếp cân bằng giữa ba thành phần này:

#### (1) Hệ số trọng số Entropy `ENTROPY_BETA` ($\beta$)
* **Xuất hiện tại**: Số hạng $-\beta H(\pi)$ trong hàm mất mát.
* **Cơ chế tác động toán học**:
  * Entropy Shannon: $H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$. Với CartPole có 2 hành động, $H \in [0, \ln 2 \approx 0.693]$.
  * Đạo hàm: $-\beta \nabla_\theta H = \beta \nabla_\theta \sum_a \pi(a) \log \pi(a)$. Khi cực tiểu hoá loss tổng, số hạng $-\beta H$ ép gradient đẩy phân phối $\pi$ về trạng thái đều (maximum uncertainty).
  * **Khi $\beta$ quá lớn ($\beta > 0.1$)**: Lực đẩy entropy lấn át hoàn toàn tín hiệu Policy Gradient $L_{pg}$. Agent trở thành kẻ ngẫu nhiên tuyệt đối (random agent), bất kể phần thưởng môi trường ra sao.
  * **Khi $\beta \to 0$ (hoặc tắt entropy)**: Policy sụp đổ quá sớm (Premature Convergence). Nếu trong vài bước đầu hành động "đi sang phải" tình cờ nhận reward $+1$, xác suất $\pi(\text{phải})$ tăng vọt lên $0.99$, xác suất $\pi(\text{trái})$ co về $0.01$. Do không còn khám phá, agent vĩnh viễn không bao giờ thử lại hành động trái nữa.
  * **Giá trị chuẩn**: $\beta \in [0.001, 0.02]$.
  * **Hậu quả của bug đảo dấu trong code tác giả**: Code viết `loss = pg + vl - entropy_loss` với `entropy_loss = β * sum(p * log p) = -β * H`. Biểu thức thực tế trở thành `loss = pg + vl + β * H`! Cực tiểu hoá $+ \beta H$ chính là bài toán **ép triệt tiêu entropy**, cưỡng bức policy biến thành tất định chỉ sau vài vòng lặp đầu tiên.

---

#### (2) Bước nhìn trước $n$ (`n_steps`) trong Advantage $N$-step
* **Xuất hiện tại**:
  $$\hat A_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)$$
* **Cơ chế tác động**:
  * **Trường hợp `n_steps = 1` (như trong notebook A2C)**:
    $\hat A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t$ (TD error 1 bước).
    Ước lượng advantage phụ thuộc $100\%$ vào giá trị của mạng Critic $V_\phi$. Trong những epoch đầu, mạng Critic chưa học được gì, $V_\phi$ chỉ là các số ngẫu nhiên $\implies \hat A_t$ là **tín hiệu hoàn toàn sai lệch**, dẫn dắt Actor học theo hướng rác.
  * **Trường hợp `n_steps = 5` đến `20` (chuẩn A2C của DeepMind)**:
    Advantage tích lũy từ 5–20 bước phần thưởng thực tế của môi trường trước khi dựa vào ước lượng của Critic. Tín hiệu phần thưởng thật làm loãng sai số của Critic, giúp Actor học ổn định hơn rất nhiều.

---

#### (3) Ngưỡng cắt Gradient `CLIP_GRAD` (Gradient Norm Clipping)
* **Xuất hiện tại**: `clip_grad_norm_(agent_nn.parameters(), CLIP_GRAD)`.
* **Công thức toán học**:
  $$g \leftarrow g \cdot \min\Big(1,\ \frac{\text{CLIP\_GRAD}}{\|g\|_2}\Big)$$
* **Cơ chế tác động**:
  * Policy Gradient có nhược điểm cố hữu là thỉnh thoảng gặp phải một chuyển tiếp có advantage cực lớn, sinh ra vector gradient bùng nổ, phá hủy hoàn toàn trọng số mạng. Cắt gradient là chốt an toàn bắt buộc.
  * **Tại sao `CLIP_GRAD = 0.1` trong notebook là quá nhỏ?**:
    Chuẩn $\|\cdot\|_2$ của gradient trong mạng nơ-ron thường dao động từ $0.5$ đến $5.0$. Khi ép ngưỡng về $0.1$, mọi bước cập nhật đều bị thu nhỏ tới mức tối đa, mạng bị "bóp nghẹt" tốc độ học, khiến đường reward tăng cực kỳ chậm hoặc đứng yên.
  * **Giá trị chuẩn**: $\text{CLIP\_GRAD} \in [0.5, 40.0]$ (OpenAI baselines thường dùng $0.5$).

---

#### (4) Số môi trường song song `N_ENVS` (40)
* **Cơ chế tác động**:
  * Vì Policy Gradient là on-policy, ta không thể dùng Replay Buffer để phá vỡ tương quan thời gian.
  * A2C giải quyết bằng cách chạy $N=40$ bản sao môi trường đồng thời. Tại mỗi bước, mạng nhận 40 trạng thái độc lập từ 40 con lắc khác nhau.
  * Mini-batch này có phân phối đa dạng, phá vỡ tính tương quan chuỗi thời gian, giúp ước lượng gradient $\frac{1}{N}\sum \nabla \log \pi \hat A$ có phương sai giảm theo tỉ lệ $\frac{1}{\sqrt N}$.

---

#### (5) Tốc độ học Adam `LEARNING_RATE` ($0.003$) và cấu trúc chia sẻ mạng
* **Cơ chế tác động**:
  * Trong notebook, Actor và Critic chia sẻ chung một thân mạng Linear(4→64).
  * Critic giải bài toán hồi quy (MSE), có thể chịu được learning rate lớn ($10^{-3}$).
  * Actor giải bài toán tối ưu phân phối, rất nhạy cảm với bước nhảy lớn: nếu Actor nhảy quá xa, phân phối dữ liệu do nó sinh ra sẽ thay đổi đột ngột $\implies$ bất ổn định.
  * Dùng chung một lr $0.003$ cho cả hai nhánh trên cùng một mạng chia sẻ là hơi cao cho Actor. Trong thực tế, người ta thường dùng lr $\sim 7\times 10^{-4}$ cho mạng chia sẻ, hoặc tách thành 2 mạng riêng với 2 optimizer khác nhau.

---

## 11. Ví dụ số

**(1) Gradient softmax**: $z=[1,0]$ ⇒ $\pi=[0.731,0.269]$. Chọn $a=1$ (xác suất 0.731), advantage $A=+2$:
$\nabla_z\log\pi(1)=[1,0]-[0.731,0.269]=[0.269,-0.269]$; cập nhật $z\leftarrow z+\alpha\cdot2\cdot[0.269,-0.269]$ ⇒ logit hành động 1 tăng. Nếu $A=-2$ thì giảm. Nếu hành động có xác suất thấp (0.269) được chọn và có $A>0$: $\nabla=[-0.731,0.731]$ — tăng **mạnh hơn** (bất ngờ lớn).

**(2) Lỗi broadcast** với $N=3$: `rewards=[5,3,1]`, `values=[[2],[2],[2]]` (shape 3×1). `rewards - values` = ma trận 3×3 với hàng $i$ = `[5,3,1]-2`; trung bình là một số **không** là trung bình của $(R_i-V_i)$ theo từng mẫu — mất liên hệ mẫu-với-mẫu.

**(3) Baseline giảm phương sai**: hai hành động, $G\in\{100,102\}$ với xác suất bằng nhau. Không baseline: tín hiệu $\approx100$ rất lớn, mọi hành động bị đẩy lên, sự khác biệt 2 chìm trong nhiễu. Baseline $b=101$: tín hiệu $\{-1,+1\}$ — chính xác thông tin "hành động nào tốt hơn".

---

## 12. Bài tập và câu hỏi kiểm tra

**Lý thuyết**
1. Tự suy lại định lý PG (§3) và chỉ rõ chỗ xác suất chuyển $P$ biến mất.
2. Chứng minh baseline không bias (§4.2). Tại sao baseline **không được** phụ thuộc hành động $a_t$?
3. Chứng minh $\mathbb E[\delta_t|s_t,a_t]=A^\pi(s_t,a_t)$ khi $V_\phi=V^\pi$.
4. Tính gradient của $\log\pi$ theo $\mu$ và $\sigma$ cho policy Gaussian.
5. Nêu hai lý do replay buffer không dùng trực tiếp cho REINFORCE/A2C.

**Thực hành**
1. Sửa lỗi indexing (§9.1, §10.4a) bằng `gather`, sửa dấu entropy; CartPole-v0 phải đạt ~195+ khi ổn định.
2. Vẽ entropy theo thời gian trước/sau khi sửa dấu.
3. Thay baseline hằng bằng critic $V(s)$ trong notebook Vanilla PG (thành Actor-Critic Monte Carlo).
4. Thử $n_{steps}\in\{1,5,20\}$ và `CLIP_GRAD`$\in\{0.1,0.5,5\}$.
5. Cài GAE ($\lambda=0.95$) cho A2C (xem Week 5).
6. Cài A3C bằng `torch.multiprocessing`.

**Công thức cần thuộc**: $\nabla J=\mathbb E[\nabla\log\pi\,\Psi]$ · $\Psi\in\{G,G-b,A,\delta\}$ · $\nabla_z\log\pi(a)=\mathbb 1_a-\pi$ · $\delta=r+\gamma V'-V$ · $L_{A2C}=-\log\pi\,\hat A+c_v(R-V)^2-\beta H$.
