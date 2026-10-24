# Week 5 – Policy Gradient nâng cao: TRPO và PPO

> Mục tiêu: hiểu **vì sao** policy gradient thô không ổn định, chứng minh toán học **Định lý chênh lệch hiệu năng (Performance Difference Lemma)**, cận dưới cải thiện đơn điệu của TRPO, cơ chế kẹp (clip) của PPO, chứng minh công thức chuỗi **GAE**, và đọc hiểu từng dòng [`PPO.py`](PPO.py) (BipedalWalker, hành động liên tục).  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

Tài liệu đi kèm: [`README.md`](README.md) (cách cài đặt, kết quả).

**Lộ trình**: §1 Vì sao PG thô mất ổn định → §2 Bổ đề chênh lệch hiệu năng & Chứng minh → §3 Hàm thay thế (Surrogate) và Cận dưới TRPO → §4 TRPO & Natural Gradient → §5 PPO-Clip & Phân tích giải tích → §6 GAE & Chứng minh toán học → §7 Policy Gaussian → §8 Thuật toán PPO đầy đủ → §9 Đọc code & Phân tích tham số → §10 Lỗi và so sánh với chuẩn → §11 Ví dụ số → §12 Bài tập.

---

## 1. Vì sao policy gradient thô khó ổn định?

Cập nhật $\theta\leftarrow\theta+\alpha\nabla J$ đo bước trong **không gian tham số**, nhưng điều quan trọng là thay đổi trong **không gian policy**. Cùng một $\|\Delta\theta\|$ có thể gây thay đổi hành vi rất nhỏ (tham số ít nhạy) hoặc rất lớn (rất nhạy) tuỳ vị trí.

Hệ quả:
- Bước quá lớn → policy tệ đi → dữ liệu mới (do policy tệ sinh ra) càng xấu → **sụp đổ không phục hồi** (dữ liệu on-policy phụ thuộc vào chính policy).
- Bước quá nhỏ → học rất chậm.
- On-policy: mỗi batch chỉ dùng được cho **một** bước gradient (kém hiệu quả dữ liệu).

Mục tiêu của TRPO/PPO: **cải thiện đơn điệu (hoặc gần đơn điệu)** và tái dùng mỗi batch cho nhiều bước.

---

## 2. Bổ đề chênh lệch hiệu năng (Performance Difference Lemma)

Cho hai chính sách tuỳ ý: $\pi$ (chính sách hiện tại/cũ) và $\pi'$ (chính sách ứng viên mới).

* **Định lý Kakade & Langford (2002)**:
  $$\boxed{J(\pi') - J(\pi) = \mathbb E_{\tau \sim \pi'}\left[ \sum_{t=0}^\infty \gamma^t A^\pi(s_t, a_t) \right] = \frac{1}{1-\gamma} \mathbb E_{s \sim d^{\pi'}, a \sim \pi'}\big[ A^\pi(s, a) \big]}$$

* **Giải thích ký hiệu**:
  * $J(\pi), J(\pi')$: Kỳ vọng tổng phần thưởng chiết khấu của hai chính sách.
  * $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$: Hàm Advantage tính theo chính sách cũ $\pi$.
  * $\tau \sim \pi'$: Quỹ đạo ngẫu nhiên sinh ra bởi chính sách mới $\pi'$.
  * $d^{\pi'}(s) = (1-\gamma)\sum_{t=0}^\infty \gamma^t \mathbb P(s_t = s \mid \pi')$: Phân phối trạng thái chiết khấu chuẩn hoá dưới chính sách $\pi'$.

* **Chứng minh toán học chi tiết**:
  * Theo định nghĩa của Advantage:
    $$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) = \mathbb E_{s_{t+1} \sim P(\cdot \mid s_t, a_t)}\big[ r_t + \gamma V^\pi(s_{t+1}) \big] - V^\pi(s_t)$$
  * Lấy tổng chiết khấu dọc theo quỹ đạo sinh ra bởi chính sách mới $\tau = (s_0, a_0, s_1, \dots) \sim \pi'$:
    $$\mathbb E_{\tau \sim \pi'}\left[ \sum_{t=0}^\infty \gamma^t A^\pi(s_t, a_t) \right] = \mathbb E_{\tau \sim \pi'}\left[ \sum_{t=0}^\infty \gamma^t \Big( r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) \Big) \right]$$
  * Tách tổng thành hai thành phần tuyến tính:
    * **Thành phần phần thưởng**:
      $$\mathbb E_{\tau \sim \pi'}\left[ \sum_{t=0}^\infty \gamma^t r_t \right] = J(\pi')$$
    * **Thành phần sai số lồng nhau (Telescoping Sum)**:
      $$\begin{aligned}
      \sum_{t=0}^\infty \gamma^t \Big( \gamma V^\pi(s_{t+1}) - V^\pi(s_t) \Big) &= \sum_{t=0}^\infty \gamma^{t+1} V^\pi(s_{t+1}) - \sum_{t=0}^\infty \gamma^t V^\pi(s_t) \\
      &= \Big( \gamma V^\pi(s_1) + \gamma^2 V^\pi(s_2) + \dots \Big) - \Big( V^\pi(s_0) + \gamma V^\pi(s_1) + \gamma^2 V^\pi(s_2) + \dots \Big) \\
      &= - V^\pi(s_0)
      \end{aligned}$$
  * Lấy kỳ vọng theo trạng thái khởi đầu $s_0 \sim p(s_0)$:
    $$\mathbb E_{s_0 \sim p}[-V^\pi(s_0)] = - J(\pi)$$
  * Ghép hai thành phần lại:
    $$\mathbb E_{\tau \sim \pi'}\left[ \sum_{t=0}^\infty \gamma^t A^\pi(s_t, a_t) \right] = J(\pi') - J(\pi)$$
  Chuyển vế ta thu được đẳng thức cần chứng minh. $\square$

* **Ý nghĩa sâu sắc**: Hiệu năng của $\pi'$ tăng lên khi và chỉ khi kỳ vọng của $A^\pi$ (đo bằng chính sách cũ) là dương trên các trạng thái mà chính sách mới ghé thăm.

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế bước cập nhật không ràng buộc $\theta \leftarrow \theta + \alpha \nabla J(\theta)$ trong không gian Euclid của Vanilla Policy Gradient.
> * **Khuyết điểm của phương pháp cũ**: Bước nhảy $\Delta\theta$ trong không gian trọng số không tương ứng đồng nhất với mức độ thay đổi hành vi trong không gian policy. Tại những vùng nhạy cảm, một bước nhảy nhỏ của $\theta$ có thể khiến policy bị đảo lộn hoàn toàn, dẫn đến hiện tượng **Sụp đổ chính sách (Policy Collapse)** không thể phục hồi.
> * **Logic thiết kế toán học**: Dựa trên Bổ đề chênh lệch hiệu năng Kakade-Langford. Thay vì nhảy tự do, thuật toán ràng buộc bước cập nhật trong **Vùng tin cậy (Trust Region)** đo bằng phân kỳ Kullback-Leibler: $D_{KL}(\pi_{old} \parallel \pi) \le \delta$.
> * **Cơ chế giải quyết**: Đảm bảo toán học về tính **cải thiện đơn điệu (Monotonic Improvement)** $J(\pi') \ge J(\pi)$.

---

## 3. Hàm thay thế (surrogate) và cận dưới

Vì ta không có mẫu từ $\pi'$ (chưa thể biết $d^{\pi'}$), ta xấp xỉ bằng phân phối trạng thái cũ $d^\pi$:
$$L_\pi(\pi') = J(\pi) + \frac{1}{1-\gamma} \mathbb E_{s \sim d^\pi, a \sim \pi'}\big[ A^\pi(s, a) \big]$$

Dùng **Importance Sampling** để lấy mẫu $a$ từ dữ liệu cũ:
$$\mathbb E_{a \sim \pi'}[A^\pi(s, a)] = \mathbb E_{a \sim \pi}\left[ \frac{\pi'(a \mid s)}{\pi(a \mid s)} A^\pi(s, a) \right]$$
Đặt tỷ số xác suất $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$, ta thu được hàm mục tiêu surrogate:
$$L^{CPI}(\theta) = \mathbb E_t \big[ r_t(\theta) \hat A_t \big]$$

**Định lý Cận dưới Monotonic (Schulman et al. 2015)**:
$$J(\pi') \ge L_\pi(\pi') - \frac{4\varepsilon\gamma}{(1-\gamma)^2} D_{KL}^{\max}(\pi, \pi'),\qquad \varepsilon = \max_{s, a} |A^\pi(s, a)|$$
Tối ưu hóa vế phải bảo đảm thuật toán cải thiện đơn điệu $J(\pi_{k+1}) \ge J(\pi_k)$ (nguyên lý Minorize-Maximize).

---

## 4. TRPO (Trust Region Policy Optimization)

Bài toán tối ưu có ràng buộc:
$$\max_\theta\ \mathbb E_t\Big[\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\hat A_t\Big]\quad\text{s.t.}\quad\mathbb E_t\big[D_{KL}(\pi_{old}(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t))\big]\le\delta$$

Xấp xỉ Taylor bậc 1 cho mục tiêu và bậc 2 cho khoảng cách KL quanh $\theta_{old}$:
$$D_{KL}(\theta_{old} \parallel \theta) \approx \frac{1}{2} \Delta\theta^\top F \Delta\theta$$
với **ma trận thông tin Fisher** $F = \mathbb E \big[ \nabla_\theta \ln \pi_\theta \nabla_\theta \ln \pi_\theta^\top \big]$. Nghiệm đóng thông qua phương pháp nhân tử Lagrange:
$$\Delta\theta = \sqrt{\frac{2\delta}{g^\top F^{-1} g}} \, F^{-1} g,\qquad g = \nabla_\theta L$$
Vector $F^{-1} g$ là **Natural Policy Gradient** chỉ hướng dốc nhất trên đa tạp phân phối xác suất Riemannian. TRPO dùng thuật toán **Conjugate Gradient** để giải $F x = g$ mà không cần nghịch đảo ma trận $F$, kết hợp Line Search để kiểm tra ràng buộc KL.

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Tìm nghiệm đóng TRPO bằng Phép thế Nhân tử Lagrange:**
> - **Bước 1 (Thế khai triển Taylor bậc 1 và bậc 2 vào bài toán tối ưu có ràng buộc):**
>   Đặt bước dịch tham số $\Delta\theta = \theta - \theta_{\text{old}}$.
>   - Thế hàm mục tiêu bằng xấp xỉ tuyến tính bậc 1: $L(\theta) \approx L(\theta_{\text{old}}) + g^\top \Delta\theta$ (với $g = \nabla_\theta L|_{\theta_{\text{old}}}$).
>   - Thế ràng buộc KL bằng xấp xỉ toàn phương bậc 2: $\bar{D}_{\text{KL}}(\theta_{\text{old}} \parallel \theta) \approx \frac{1}{2}\Delta\theta^\top F \Delta\theta \le \delta$.
>   Bài toán quy về cực đại hóa hàm Lagrange (với nhân tử $\nu > 0$):
>   $$\mathcal{L}(\Delta\theta, \nu) = g^\top \Delta\theta - \nu \left(\frac{1}{2}\Delta\theta^\top F \Delta\theta - \delta\right)$$
> - **Bước 2 (Lấy đạo hàm theo $\Delta\theta$ bằng $0$ để rút hàm $\Delta\theta$ theo $\nu$):**
>   $$\nabla_{\Delta\theta}\mathcal{L} = g - \nu F \Delta\theta = \mathbf{0} \iff F\Delta\theta = \frac{1}{\nu}g \implies \Delta\theta = \frac{1}{\nu}F^{-1}g$$
> - **Bước 3 (Thế $\Delta\theta = \frac{1}{\nu}F^{-1}g$ ngược trở lại phương trình biên vùng tin cậy $\frac{1}{2}\Delta\theta^\top F \Delta\theta = \delta$):**
>   $$\frac{1}{2}\underbrace{\left(\frac{1}{\nu}F^{-1}g\right)^\top}_{\Delta\theta^\top} F \underbrace{\left(\frac{1}{\nu}F^{-1}g\right)}_{\Delta\theta} = \delta \iff \frac{1}{2\nu^2} g^\top (F^{-1})^\top \underbrace{F F^{-1}}_{=I} g = \delta$$
>   Vì $F$ đối xứng nên $(F^{-1})^\top = F^{-1}$, suy ra:
>   $$\frac{1}{2\nu^2} g^\top F^{-1} g = \delta \implies \frac{1}{\nu} = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}$$
> - **Bước 4 (Thế hệ số bước nhảy $\frac{1}{\nu}$ vừa tìm được ở Bước 3 vào lại công thức $\Delta\theta$ ở Bước 2):**
>   $$\Delta\theta = \underbrace{\sqrt{\frac{2\delta}{g^\top F^{-1} g}}}_{\text{Thế cho }\frac{1}{\nu}} F^{-1}g$$

---

## 5. PPO-Clip (Proximal Policy Optimization)

### 5.1 Hàm mục tiêu kẹp (Clipped Surrogate Objective)
$$\boxed{L^{CLIP}(\theta) = \mathbb E_t \left[ \min\left( r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t \right) \right]}$$

* **Giải thích ký hiệu**:
  * $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$: Tỷ số xác suất hành động.
  * $\hat A_t$: Ước lượng Advantage tại bước $t$.
  * $\epsilon \in [0.1, 0.3]$: Biên độ kẹp (thường chọn $\epsilon = 0.2$).

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế bài toán tối ưu bậc hai cồng kềnh với ma trận Fisher $F^{-1}$ và thuật toán Conjugate Gradient của TRPO.
> * **Khuyết điểm của phương pháp cũ**: TRPO quá phức tạp, tốn kém bộ nhớ tính toán ma trận đạo hàm bậc hai, không tương thích với các kiến trúc mạng sâu có chia sẻ trọng số (Actor và Critic chung thân mạng).
> * **Logic thiết kế toán học**: Xây dựng một **cận dưới bi quan (Pessimistic Lower Bound)** chỉ dùng đạo hàm bậc nhất. Hàm kẹp `clip` chặt tỷ số $r_t(\theta)$ vào $[1-\epsilon, 1+\epsilon]$. Khi $r_t$ văng ra ngoài vùng an toàn, gradient đạo hàm bị triệt tiêu về đúng $0$, ngăn không cho policy đi quá xa khỏi $\pi_{old}$.
> * **Cơ chế giải quyết**: Đạt được độ ổn định tương đương TRPO nhưng chạy nhanh gấp nhiều lần, cho phép tối ưu hóa nhiều epoch SGD trên cùng một tập rollout, trở thành thuật toán Deep RL phổ biến nhất thế giới.

### 5.2 Phân tích giải tích 4 trường hợp kẹp

| Dấu của $\hat A_t$ | Giá trị tỷ số $r_t(\theta)$ | Biểu thức bên trong $\min$ | Gradient $\nabla_\theta L$ | Hành vi tối ưu hoá |
|---|---|---|---|---|
| $\hat A_t > 0$ (Hành động tốt) | $r_t \le 1+\epsilon$ | $r_t \hat A_t$ | Dương | Tiếp tục tăng xác suất $\pi_\theta(a_t \mid s_t)$ |
| $\hat A_t > 0$ (Hành động tốt) | $r_t > 1+\epsilon$ | $(1+\epsilon)\hat A_t$ | **0** | Đã tăng đủ trong vùng tin cậy; **triệt tiêu gradient** |
| $\hat A_t < 0$ (Hành động tệ) | $r_t \ge 1-\epsilon$ | $r_t \hat A_t$ | Âm | Tiếp tục giảm xác suất $\pi_\theta(a_t \mid s_t)$ |
| $\hat A_t < 0$ (Hành động tệ) | $r_t < 1-\epsilon$ | $(1-\epsilon)\hat A_t$ | **0** | Đã giảm đủ; **triệt tiêu gradient** tránh sụp đổ policy |

---

## 6. GAE – Generalized Advantage Estimation (Chứng minh toán học đầy đủ)

### 6.1 Cơ chế suy diễn công thức chuỗi GAE
Định nghĩa sai số TD 1 bước (Temporal Difference Error):
$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
Các ước lượng Advantage đa bước:
$$\begin{aligned}
\hat A_t^{(1)} &= \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t) \\
\hat A_t^{(2)} &= \delta_t^V + \gamma \delta_{t+1}^V = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t) \\
\hat A_t^{(k)} &= \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V
\end{aligned}$$

GAE được định nghĩa là **trung bình có trọng số suy giảm hàm mũ** theo hệ số $\lambda \in [0, 1]$ của tất cả các ước lượng $k$-step:
$$\hat A_t^{GAE(\gamma, \lambda)} = (1-\lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat A_t^{(k)}$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế ước lượng Advantage 1 bước $\delta_t^V$ (TD) hoặc Advantage Monte Carlo vô hạn bước $G_t - V(s_t)$.
> * **Khuyết điểm của phương pháp cũ**: 1-step TD có bias quá cao (lệch theo mạng Critic chưa học xong), trong khi Monte Carlo có variance quá cao (tích luỹ sai số ngẫu nhiên).
> * **Logic thiết kế toán học**: Lấy trung bình trọng số suy giảm hàm mũ theo hệ số $\lambda \in [0, 1]$ của mọi ước lượng $k$-step Advantage.
> * **Cơ chế giải quyết**: Bằng cách chọn $\lambda \approx 0.95$, GAE giảm thiểu sai số bình phương trung bình MSE của Advantage, cân bằng tối ưu giữa Bias và Variance.

* **Chứng minh toán học đưa về dạng chuỗi vô hạn gọn**:
  Thay công thức của $\hat A_t^{(k)}$ vào:
  $$\hat A_t^{GAE} = (1-\lambda) \sum_{k=1}^\infty \lambda^{k-1} \left( \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V \right)$$
  Đổi thứ tự lấy tổng hai biến $k$ và $l$: chỉ số $l$ chạy từ $0 \to \infty$, và với mỗi $l$ cố định, chỉ số $k$ chạy từ $l+1 \to \infty$:
  $$\hat A_t^{GAE} = (1-\lambda) \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V \left( \sum_{k=l+1}^\infty \lambda^{k-1} \right)$$
  Tính tổng chuỗi hình học bên trong: Đặt biến $j = k - 1 \implies j$ chạy từ $l \to \infty$:
  $$\sum_{k=l+1}^\infty \lambda^{k-1} = \sum_{j=l}^\infty \lambda^j = \lambda^l \sum_{m=0}^\infty \lambda^m = \frac{\lambda^l}{1-\lambda}$$
  Thế vào biểu thức:
  $$\hat A_t^{GAE} = (1-\lambda) \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V \cdot \frac{\lambda^l}{1-\lambda} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$
  Do đó:
  $$\boxed{\hat A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V}$$
  Chứng minh hoàn tất. $\square$

* **Chứng minh dạng truy hồi tính lùi (Backward Recursive Form)**:
  Tách số hạng đầu tiên $l=0$:
  $$\hat A_t^{GAE} = \delta_t^V + \sum_{l=1}^\infty (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + \gamma \lambda \sum_{m=0}^\infty (\gamma \lambda)^m \delta_{t+1+m}^V = \boxed{\delta_t^V + \gamma \lambda (1-d_t) \hat A_{t+1}^{GAE}}$$
  với $d_t$ là cờ kết thúc ván (done flag). Công thức truy hồi này cho phép tính toàn bộ vector Advantage cho quỹ đạo $T$ bước với độ phức tạp tuyến tính $O(T)$! $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Tại sao các hàm giá trị trung gian $V(s_{t+1}), \dots, V(s_{t+k-1})$ lại tự triệt tiêu và làm sao gom thành đệ quy lùi?**
> - **Phép thế 1 (Triệt tiêu cuốn chiếu — Telescoping Cancellation trong $\hat{A}_t^{(k)}$):**
>   Nếu ta thế định nghĩa $\delta_{t+l}^V = r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$ vào tổng $\hat{A}_t^{(2)} = \delta_t^V + \gamma \delta_{t+1}^V$:
>   $$\hat{A}_t^{(2)} = \big(r_t + \cancel{\gamma V(s_{t+1})} - V(s_t)\big) + \gamma\big(r_{t+1} + \gamma V(s_{t+2}) - \cancel{V(s_{t+1})}\big) = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)$$
>   Ta thấy $+\gamma V(s_{t+1})$ của bước $t$ và $-\gamma V(s_{t+1})$ của bước $t+1$ **tự triệt tiêu nhau hoàn toàn**, chứng minh rằng tổng $\sum_{l=0}^{k-1}\gamma^l \delta_{t+l}^V$ chính là Advantage $k$-bước!
> - **Phép thế 2 (Nhận diện và thế hàm bước kế tiếp $\hat{A}_{t+1}^{\text{GAE}}$ để tạo vòng lặp lùi):**
>   Xuất phát từ chuỗi tổng $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V$, ta tách riêng số hạng đầu ($l=0$) và đặt nhân tử chung $(\gamma\lambda)$ ra ngoài phần đuôi ($l \ge 1$):
>   $$\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma\lambda)\underbrace{\Big[\delta_{t+1}^V + (\gamma\lambda)^1 \delta_{t+2}^V + (\gamma\lambda)^2 \delta_{t+3}^V + \dots\Big]}_{\text{Chính là định nghĩa của }\hat{A}_{t+1}^{\text{GAE}}}$$
>   Vì biểu thức trong ngoặc vuông có cấu trúc y hệt $\hat{A}_t^{\text{GAE}}$ nhưng mọi chỉ số thời gian đều dịch lên $+1$ (từ $t \to t+1$), ta **thế nguyên cả chuỗi vô hạn trong ngoặc vuông bằng biến đơn $\hat{A}_{t+1}^{\text{GAE}}$**:
>   $$\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma\lambda)\hat{A}_{t+1}^{\text{GAE}}$$

---
- $\lambda=1$: $\hat A_t=\sum_l\gamma^l\delta_{t+l}=G_t-V(s_t)$ (Monte Carlo).
- $\lambda=0.95$ (repo): cân bằng.

### 6.3 Mục tiêu cho critic
$$V^{targ}_t=\hat A_t+V_{old}(s_t)\quad(\text{λ-return})$$
Code: `reward=run_add + memories[t].value` ([PPO.py L147](PPO.py#L147)), rồi `F.mse_loss(value, rewards)`.

### 6.4 Chuẩn hoá advantage
Trước khi tối ưu: $\hat A\leftarrow(\hat A-\mu)/(\sigma+10^{-7})$ trên cả batch ([L292-L295](PPO.py#L292-L295)). Giữ thang đo ổn định (lr, $\epsilon$ không phụ thuộc thang reward) — không đổi hướng gradient (trừ một hằng/chia hằng).

---

## 7. Policy Gaussian cho hành động liên tục

BipedalWalker: quan sát 24 chiều, hành động 4 chiều trong $[-1,1]$ (mô-men xoắn 4 khớp).

$$a\sim\mathcal N\big(\mu_\theta(s),\ \mathrm{diag}(\sigma^2)\big),\qquad\sigma=e^{\ell},\ \ell=\text{logstd (tham số học, không phụ thuộc }s)$$

**Log-likelihood** (cộng qua các chiều):
$$\log\pi(a|s)=\sum_{i}\Big[-\frac{(a_i-\mu_i)^2}{2\sigma_i^2}-\log\sigma_i-\tfrac12\log2\pi\Big]=\sum_i\Big[-\tfrac12\big(\tfrac{a_i-\mu_i}{e^{\ell_i}}\big)^2-\ell_i-\tfrac12\log2\pi\Big]$$

**Entropy**: $H=\sum_i(\ell_i+\tfrac12\log(2\pi e))$ — tăng $\ell$ ⇒ khám phá nhiều hơn. Thường $\ell$ khởi tạo 0 (σ=1) rồi giảm dần khi policy tự tin.

**Tỉ lệ xác suất** theo từng chiều rồi nhân (cộng log): $r_t=\exp(\log\pi_{new}-\log\pi_{old})$, trong đó log-prob là **tổng** qua các chiều hành động. (Repo giữ shape `[B, n_actions]` và dùng $r$ theo từng chiều — xem §10.)

**Trung bình bị chặn**: mạng ra `tanh(mean)` trong $(-1,1)$; hành động lấy mẫu rồi `np.clip(·,-1,1)` trước khi gửi môi trường (nhưng log-prob tính trên hành động đã clip).

---

## 8. Thuật toán PPO đầy đủ (như trong repo)

```
lặp:
  1) Thu thập T=2049 bước bằng policy hiện tại → lưu (obs, action, reward, done, V(obs))
  2) Tính GAE(γ=0.99, λ=0.95) lùi → advantage Â_t và target V
  3) old_logp ← log π_θ(a_t|s_t)   (cố định)
  4) Chuẩn hoá Â
  5) Lặp K=7 epoch trên các minibatch 64:
        r = exp(logp_new - old_logp)
        L_pg = -mean( min(r·Â, clip(r,0.8,1.2)·Â) )
        L_v  = mean( (V_φ(s) - V_target)² )
        cập nhật θ (Adam lr=4e-4) và φ (Adam lr=1e-3)
  6) Mỗi 100 vòng: đánh giá 5 episode, lưu checkpoint nếu tốt nhất
```

---

## 9. Đọc code: [`PPO.py`](PPO.py)

| Phần | Vị trí | Giải thích toán |
|---|---|---|
| `A2C_policy` | [L16-L39](PPO.py#L16-L39) | MLP 24→32→32, đầu `mean_l` (tanh) ra $\mu\in(-1,1)^4$; `logstd` là tham số độc lập state. `var_l` được khai báo nhưng **không dùng** |
| `A2C_value` | [L41-L57](PPO.py#L41-L57) | MLP 24→32→32→1 cho $V_\phi(s)$ |
| `Env.steps` | [L84-L129](PPO.py#L84-L129) | lấy mẫu $a=\mu+e^{\ell}\,\xi$, $\xi\sim\mathcal N(0,I)$, clip $[-1,1]$; lưu $V(s_t)$ |
| `generalized_advantage_estimation` | [L131-L149](PPO.py#L131-L149) | đệ quy $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$ |
| `log_policy_prob` | [L152-L155](PPO.py#L152-L155) | log-likelihood Gaussian (xem §10) |
| `clipped_PPO_loss` | [L169-L187](PPO.py#L169-L187) | $L^{CLIP}$ và value loss |
| vòng chính | [L283-L347](PPO.py#L283-L347) | 7 epoch × minibatch 64, hai optimizer |
| `test_game` | [L189-L215](PPO.py#L189-L215) | dùng **hành động xác định** (mean), không lấy mẫu |

**Ánh xạ công thức ↔ code**:
| Công thức | Code |
|---|---|
| $\delta_t$ | `sigma = reward + gamma*value[t+1] - value[t]` |
| $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$ | `run_add = sigma + run_add*gamma*gae_lambda` |
| $r_t$ | `rt_theta = torch.exp(new_log_policy - old_log_policy.detach())` |
| $L^{CLIP}$ | `-torch.mean(torch.min(rt*adv, torch.clamp(rt,1-eps,1+eps)*adv))` |
| $V$ loss | `F.mse_loss(value.squeeze(-1), rewards)` |

### 9.1 Phân tích chuyên sâu các tham số và cơ chế tác động lên phương trình PPO & GAE

PPO là thuật toán chuẩn mực của OpenAI, độ ổn định của nó được quyết định bởi sự phối hợp toán học giữa các tham số sau:

#### (1) Ngưỡng cắt `CLIP_EPS` ($\epsilon = 0.2$)
* **Xuất hiện tại**:
  $$L^{CLIP}(\theta) = \mathbb E_t\Big[\min\big(r_t(\theta) \hat A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t\big)\Big]$$
* **Cơ chế tác động toán học**:
  * Tỉ số tầm quan trọng $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$. Khi $\theta = \theta_{old} \implies r_t = 1$.
  * Đoạn cắt $[1-\epsilon, 1+\epsilon] = [0.8, 1.2]$ thiết lập một **vùng tin cậy đại số (Pessimistic Trust Region)**:
    * Khi hành động có $\hat A_t > 0$ (tốt hơn trung bình), gradient cố gắng tăng xác suất hành động này. Nhưng khi $r_t \ge 1 + \epsilon = 1.2$, hàm $\min$ kích hoạt đoạn bằng phẳng $\implies$ **Gradient bị triệt tiêu về 0**, ngăn không cho policy tăng xác suất quá $20\%$ so với policy cũ trong 1 vòng thu thập.
    * Khi hành động có $\hat A_t < 0$ (xấu), khi $r_t \le 1 - \epsilon = 0.8$, gradient cũng bị triệt tiêu về 0, ngăn không cho policy giảm xác suất quá $20\%$.
  * **Khi $\epsilon$ quá lớn ($\epsilon \ge 0.5$)**: Vùng tin cậy quá rộng, PPO thoái hoá về Policy Gradient truyền thống. Một bước nhảy tham số quá đà sẽ làm thay đổi căn bản phân phối dữ liệu $\implies$ sụp đổ hiệu năng không thể cứu vãn.
  * **Khi $\epsilon$ quá nhỏ ($\epsilon \le 0.05$)**: Mọi mẫu vừa chớm thay đổi đều bị cắt gradient, mạng hầu như không cập nhật được gì, tốc độ học bị tê liệt.
  * **Giá trị chuẩn mực**: $\epsilon \in [0.1, 0.2]$.

---

#### (2) Cặp tham số GAE: `GAE_LAMBDA` ($\lambda = 0.95$) và `GAMMA` ($\gamma = 0.99$)
* **Xuất hiện tại**:
  $$\hat A_t^{GAE} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l},\qquad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
* **Cơ chế tác động toán học**:
  * Chuỗi trọng số hình học có công bội là tích số $c = \gamma \lambda = 0.99 \times 0.95 = \mathbf{0.9405}$.
  * **Tầm nhìn hiệu dụng của Advantage**: $H_{adv} \approx \frac{1}{1 - \gamma \lambda} \approx \frac{1}{1 - 0.9405} \approx \mathbf{16.8\text{ bước}}$.
  * **Bản chất của $\lambda$**:
    * **Khi $\lambda = 0$**: $\hat A_t = \delta_t$. Advantage chỉ nhìn 1 bước tương lai và dựa hoàn toàn vào Critic $V_\phi$. **Variance tối thiểu, Bias tối đa**. Nếu mạng Critic có sai số, toàn bộ Advantage bị méo mó.
    * **Khi $\lambda = 1$**: $\hat A_t = \sum_{l=0}^\infty \gamma^l \delta_{t+l} = G_t - V_\phi(s_t)$ (Monte Carlo). Mọi bước đi đều dùng phần thưởng thật của môi trường. **Bias bằng 0, Variance cực đại** do tích luỹ biến động ngẫu nhiên của hàng trăm bước tiếp theo.
    * **Tại $\lambda = 0.95$**: Cân bằng toán học tối ưu. Tín hiệu phần thưởng từ 17 bước kế tiếp đóng góp $65\%$ trọng số, sau đó chuyển giao êm ái cho ước lượng của Critic.

---

#### (3) Số Epoch tối ưu `PPO_EPOCHS` ($K = 7$) và Kích thước Quỹ đạo `TRAJECTORY_SIZE` ($T = 2049$)
* **Cơ chế tác động**:
  * **Hiệu quả mẫu (Sample Efficiency)**: Với A2C, $T = 2049$ bước chỉ được dùng đúng 1 lần rồi vứt bỏ. Với PPO, tập dữ liệu này được chia thành các mini-batch kích thước $64$ (khoảng 32 mini-batches), và được duyệt qua $K = 7$ lần $\implies$ thực hiện tới $7 \times 32 = \mathbf{224\text{ bước cập nhật SGD}}$ trên cùng một lượng tương tác môi trường!
  * **Nguy cơ khi $K$ quá lớn ($K > 15$)**: Dù có cơ chế clip, việc tối ưu quá nhiều lượt trên cùng một tập dữ liệu sẽ tích luỹ dịch chuyển phân phối. Đến epoch thứ 15, policy mới $\pi_\theta$ đã cách rất xa policy lúc thu thập dữ liệu $\pi_{\theta_{old}}$, khoảng cách $D_{KL}(\pi_{\theta_{old}} \parallel \pi_\theta)$ bùng nổ, vi phạm giả định của Bổ đề Kakade-Langford.
  * **`TRAJECTORY_SIZE` = 2049**: BipedalWalker là bài toán điều khiển chân robot, một chu kỳ bước đi hoàn chỉnh kéo dài hàng trăm bước. Nếu $T$ quá nhỏ (ví dụ 128), GAE bị cắt vụn, robot không học được tính chu kỳ của dáng đi.

---

#### (4) Tốc độ học riêng biệt: `POLICY_LR` ($4\times 10^{-4}$) vs `VALUE_LR` ($10^{-3}$)
* **Cơ chế tác động**:
  * Tách rời hoàn toàn optimizer của Actor và Critic là một thiết kế rất chuẩn trong `PPO.py`.
  * Critic giải bài toán hồi quy có giám sát (MSE) với mục tiêu $\hat A_t + V_{old}(s_t)$. Vì bài toán hồi quy có độ ổn định cao, Critic có thể dùng lr lớn ($10^{-3}$) để bám sát giá trị phần thưởng thay đổi.
  * Actor giải bài toán dịch chuyển phân phối xác suất, bước nhảy cần thận trọng hơn nhiều (lr nhỏ hơn 2.5 lần, $4\times 10^{-4}$) để không làm thủng vùng tin cậy.

---

## 10. Lỗi và khác biệt so với chuẩn (đã đối chiếu code)

1. **Log-prob không khớp phân phối lấy mẫu.** Lấy mẫu dùng $\sigma=e^{\ell}$ ([L97](PPO.py#L97)), nhưng
   ```python
   -((mean-actions)**2)/(2*torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2*math.pi*torch.exp(std)))
   ```
   coi $e^{\ell}$ là **phương sai**: $\log\pi=-\dfrac{(a-\mu)^2}{2e^{\ell}}-\tfrac12\log(2\pi e^{\ell})$, tương ứng $\sigma^2=e^{\ell}$, không phải $\sigma=e^{\ell}$. Tỉ lệ $r_t$ vẫn nhất quán (cùng hàm cho old/new) nên thuật toán chạy, nhưng đó không phải log-likelihood thật nên ý nghĩa của $\epsilon$-clip và gradient của logstd sai lệch. Sửa: `Normal(mean, exp(logstd)).log_prob(a).sum(-1)`.
2. **Tỉ lệ theo từng chiều thay vì theo cả hành động.** Log-prob giữ shape `[B,4]` ⇒ clip trên từng chiều độc lập, trong khi chuẩn PPO dùng $r_t$ của cả vector hành động (tổng log-prob). Với 4 chiều khác biệt này không thảm hoạ nhưng không chuẩn.
3. **Mất tín hiệu terminal.** Khi `done`, reward gán 0 ([L107-L109](PPO.py#L107-L109)). BipedalWalker cho −100 khi ngã ⇒ hình phạt bị bỏ; GAE tại bước done lấy $\hat A_t=r_t=0$. Đúng: giữ reward thật, đặt $V(s_{t+1})=0$ khi **terminated**, nhưng **bootstrap** $V(s_{t+1})$ khi **truncated** (hết thời gian).
4. **Bước cuối bị bỏ**, không bootstrap: vòng `range(len(memories)-1)` và `run_add=0` ban đầu. Cần `last_value=V(s_T)` để khởi tạo.
5. **Minibatch không xáo trộn**: `batch[mb:mb+64]` tuần tự ⇒ mẫu liên tiếp tương quan. Nên `np.random.permutation` mỗi epoch.
6. `CLIP_GRADIENT=0.2` **không dùng** (đã import `clip_grad_norm_` nhưng không gọi); không có entropy bonus.
7. `compute_log_policy_prob` dùng biến toàn cục `agent_policy` thay vì tham số `nn_policy` (chạy được vì cùng object).
8. API cũ: `BipedalWalker-v2`, `gym.wrappers.Monitor`, `F.tanh`.

---

## 11. Ví dụ số

**(a) Cắt PPO**: $\epsilon=0.2$.
| $r$ | $\hat A$ | $r\hat A$ | $\mathrm{clip}(r)\hat A$ | $\min$ | Có gradient? |
|---|---|---|---|---|---|
| 1.1 | +2 | 2.2 | 2.2 | 2.2 | có |
| 1.5 | +2 | 3.0 | 2.4 | 2.4 | **không** |
| 0.5 | −2 | −1.0 | −1.6 | −1.6 | **không** (cận $0.8\times(-2)$ nhỏ hơn) |
| 1.5 | −2 | −3.0 | −2.4 | −3.0 | **có** (sửa sai) |

**(b) GAE hai bước**: $\gamma=0.99,\lambda=0.95$; $\delta_t=1,\ \delta_{t+1}=2,\ \delta_{t+2}=0$ (hết episode). $\hat A_{t+2}=0$; $\hat A_{t+1}=2+0.9405\cdot0=2$; $\hat A_t=1+0.9405\cdot2=2.881$. (Với $\gamma\lambda=0.99\cdot0.95=0.9405$.)

**(c) Tỉ lệ**: $\pi_{old}(a|s)=0.20$, $\pi_\theta(a|s)=0.26$ ⇒ $r=1.3>1.2$; nếu $\hat A>0$ ⇒ gradient bằng 0 (tăng đủ). Hàm log: $\log0.26-\log0.20=0.262$; $e^{0.262}=1.3$ ✓.

---

## 12. Bài tập và câu hỏi kiểm tra

**Lý thuyết**
1. Chứng minh bổ đề chênh lệch hiệu năng (telescoping).
2. Vì sao $\nabla L^{CPI}|_{\theta_{old}}$ trùng gradient policy gradient? (dùng $\nabla r=r\nabla\log\pi$ và $r=1$).
3. Giải thích bằng bảng §5.2 vì sao dùng $\min$ chứ không phải chỉ `clip`.
4. Chứng minh hệ số của $\delta_{t+l}$ trong GAE là $(\gamma\lambda)^l$.
5. Vì sao chuẩn hoá advantage không làm đổi hướng gradient policy (trừ trung bình/chia độ lệch chuẩn)? Nó thay đổi tương đối giữa các mẫu thế nào?

**Thực hành**
1. Thay `log_policy_prob` bằng `torch.distributions.Normal` (tổng theo chiều hành động) và so sánh đường học.
2. Thêm xáo trộn minibatch, entropy bonus, bootstrap bước cuối; giữ reward thật khi `done`.
3. Ghi lại **clip fraction** ($\Pr[|r-1|>\epsilon]$) và **KL xấp xỉ** $\mathbb E[\log\pi_{old}-\log\pi_\theta]$ mỗi vòng.
4. Quét $\epsilon\in\{0.1,0.2,0.3\}$, `PPO_EPOCHS`$\in\{3,7,15\}$, $\lambda\in\{0,0.9,0.95,1\}$.
5. Cài early-stopping theo KL (dừng epoch khi KL vượt ngưỡng).
6. Thử `BipedalWalkerHardcore`.

**Công thức cần thuộc**: $r_t=\pi_\theta/\pi_{old}$ · $L^{CLIP}$ · $\delta_t$ · $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$ · $V^{targ}=\hat A+V_{old}$ · log-likelihood Gaussian.
