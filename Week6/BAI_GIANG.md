# Week 6 – Evolution Strategies (ES) và Genetic Algorithms (GA)

> Mục tiêu: hiểu ES như một **bộ ước lượng gradient không cần đạo hàm**, chứng minh toán học công thức gradient qua 2 phương pháp (Log-Derivative Trick & Bổ đề Stein), chứng minh giải tích tại sao Mirrored Sampling triệt tiêu phương sai bậc chẵn, và đọc hiểu [`ES.py`](ES.py) (LunarLanderContinuous).  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

Tài liệu đi kèm: [`README.md`](README.md).

**Lộ trình**: §1 Tối ưu hộp đen → §2 Làm trơn Gaussian & 2 Cách chứng minh gradient ES → §3 Giảm phương sai & Chứng minh Mirrored Sampling → §4 Rank shaping → §5 Mở rộng song song → §6 Genetic Algorithm → §7 So sánh với RL → §8 Đọc code & Phân tích tham số → §9 Lỗi và lưu ý → §10 Ví dụ số → §11 Bài tập.

---

## 1. Tối ưu hộp đen

Ta có hàm "chất lượng" $F(\theta)$ = tổng reward của một episode khi policy có tham số $\theta$. Ta **không** biết (hoặc không muốn dùng) $\nabla_\theta F$: môi trường có thể không khả vi, reward thưa/trễ, có tính ngẫu nhiên. Chỉ có thể **đánh giá** $F$ tại các điểm.

Policy gradient (Week 4–5) vẫn cần backprop qua $\log\pi$ và ước lượng giá trị; ES **bỏ qua** toàn bộ cấu trúc thời gian (MDP): coi cả episode là một lần gọi hàm $F$.

---

## 2. Làm trơn Gaussian và ước lượng gradient

### 2.1 Mục tiêu đã làm trơn
Hàm $F(\theta)$ có thể không liên tục, gồ ghề hoặc có nhiễu. Ta thay bằng kỳ vọng khi nhiễu hoá tham số:
$$J_\sigma(\theta) = \mathbb E_{\varepsilon \sim \mathcal N(0, I)}\big[ F(\theta + \sigma \varepsilon) \big]$$

* **Giải thích ký hiệu**:
  * $\theta \in \mathbb R^d$: Vector tham số trung tâm của mạng nơ-ron ($d$ chiều).
  * $\varepsilon \sim \mathcal N(0, I)$: Vector nhiễu Gauss chuẩn hoá $d$ chiều.
  * $\sigma > 0$: Độ lệch chuẩn nhiễu (Perturbation scale).
  * $J_\sigma(\theta)$: Hàm mục tiêu đã được làm trơn cục bộ (Gaussian Convolution).

### 2.2 Chứng minh toán học công thức Gradient ES

#### Cách 1: Sử dụng Log-Derivative Trick trên không gian tham số
Đặt biến ngẫu nhiên $w = \theta + \sigma \varepsilon \in \mathbb R^d$. Khi đó $w \sim \mathcal N(\theta, \sigma^2 I)$ có hàm mật độ xác suất:
$$p(w; \theta) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp\left( -\frac{\|w - \theta\|_2^2}{2\sigma^2} \right)$$
Hàm mục tiêu làm trơn viết lại dưới dạng tích phân:
$$J_\sigma(\theta) = \int_{\mathbb R^d} F(w) p(w; \theta) \, dw$$

* **Chứng minh chi tiết**:
  * **Bước 1**: Lấy đạo hàm gradient theo $\theta$ dưới dấu tích phân:
    $$\nabla_\theta J_\sigma(\theta) = \int F(w) \nabla_\theta p(w; \theta) \, dw$$
  * **Bước 2**: Sử dụng Log-Derivative Trick $\nabla_\theta p(w; \theta) = p(w; \theta) \nabla_\theta \ln p(w; \theta)$:
    $$\nabla_\theta J_\sigma(\theta) = \int F(w) p(w; \theta) \nabla_\theta \ln p(w; \theta) \, dw = \mathbb E_{w \sim \mathcal N(\theta, \sigma^2 I)}\Big[ F(w) \nabla_\theta \ln p(w; \theta) \Big]$$
  * **Bước 3**: Tính gradient của log-density hàm Gauss:
    $$\ln p(w; \theta) = -\frac{d}{2}\ln(2\pi \sigma^2) - \frac{\|w - \theta\|_2^2}{2\sigma^2}$$
    $$\nabla_\theta \ln p(w; \theta) = -\nabla_\theta \frac{\|w - \theta\|_2^2}{2\sigma^2} = \frac{w - \theta}{\sigma^2}$$
  * **Bước 4**: Đổi biến trở lại $w = \theta + \sigma \varepsilon \implies w - \theta = \sigma \varepsilon$:
    $$\nabla_\theta J_\sigma(\theta) = \mathbb E_{\varepsilon \sim \mathcal N(0, I)}\left[ F(\theta + \sigma \varepsilon) \frac{\sigma \varepsilon}{\sigma^2} \right] = \boxed{\frac{1}{\sigma} \mathbb E_{\varepsilon \sim \mathcal N(0, I)}\big[ F(\theta + \sigma \varepsilon) \, \varepsilon \big]}$$
  Chứng minh hoàn tất. $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Chuyển Đạo hàm từ Hàm hộp đen $F$ sang Phân phối Nhiễu Gaussian:**
> - **Bước 1 (Thế phép đổi biến $w = \theta + \sigma\varepsilon$ để đưa tham số $\theta$ từ bên trong hàm $F$ sang hàm mật độ xác suất $p(w;\theta)$):**
>   Trong biểu thức gốc $J_\sigma(\theta) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[F(\theta + \sigma\varepsilon)]$, biến $\theta$ nằm kẹt bên trong hàm hộp đen $F(\cdot)$ không khả vi. Bằng phép thế biến ngẫu nhiên $w = \theta + \sigma\varepsilon \sim \mathcal{N}(\theta, \sigma^2 I)$, ta đẩy toàn bộ sự phụ thuộc vào $\theta$ sang hàm mật độ chuẩn $p(w;\theta)$:
>   $$J_\sigma(\theta) = \int F(w) \, p(w; \theta)\,dw$$
> - **Bước 2 (Thế $\nabla_\theta p(w;\theta) = p(w;\theta)\nabla_\theta \ln p(w;\theta)$ và thế đạo hàm log-Gaussian):**
>   Với mật độ chuẩn $\ln p(w;\theta) = -\frac{d}{2}\ln(2\pi\sigma^2) - \frac{\|w-\theta\|_2^2}{2\sigma^2}$, đạo hàm theo $\theta$ là $\nabla_\theta \ln p(w;\theta) = \frac{w-\theta}{\sigma^2}$. Đem thế vào kỳ vọng:
>   $$\nabla_\theta J_\sigma(\theta) = \mathbb{E}_{w \sim \mathcal{N}(\theta, \sigma^2 I)}\left[F(w)\underbrace{\frac{w - \theta}{\sigma^2}}_{\text{Thế cho }\nabla_\theta \ln p}\right]$$
> - **Bước 3 (Thế ngược lại $w = \theta + \sigma\varepsilon$ và $w - \theta = \sigma\varepsilon$):**
>   Thay $w$ bằng $\theta + \sigma\varepsilon$ và thay hiệu $w - \theta$ trên tử số bằng $\sigma\varepsilon$, sau đó rút gọn $\frac{\sigma\varepsilon}{\sigma^2} = \frac{\varepsilon}{\sigma}$:
>   $$\nabla_\theta J_\sigma(\theta) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}\left[F(\theta + \sigma\varepsilon)\frac{\cancel{\sigma}\varepsilon}{\sigma^{\cancel{2}}}\right] = \frac{1}{\sigma}\mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}\big[F(\theta + \sigma\varepsilon)\varepsilon\big]$$

#### Cách 2: Sử dụng Tích phân từng phần (Integration by Parts / Bổ đề Stein)
Gọi $\phi(\varepsilon) = \frac{1}{(2\pi)^{d/2}} e^{-\frac{1}{2}\|\varepsilon\|^2}$ là mật độ Gauss chuẩn.
Ta có: $J_\sigma(\theta) = \int F(\theta + \sigma \varepsilon) \phi(\varepsilon) d\varepsilon$.
Lấy đạo hàm theo $\theta$:
$$\nabla_\theta J_\sigma(\theta) = \int \nabla_\theta F(\theta + \sigma \varepsilon) \phi(\varepsilon) d\varepsilon = \frac{1}{\sigma} \int \nabla_\varepsilon F(\theta + \sigma \varepsilon) \phi(\varepsilon) d\varepsilon$$
Áp dụng tích phân từng phần trong không gian $\mathbb R^d$ (với giả thiết $F$ tăng chậm hơn hàm mũ Gauss ở vô cùng):
$$\int \nabla_\varepsilon F(\theta + \sigma \varepsilon) \phi(\varepsilon) d\varepsilon = - \int F(\theta + \sigma \varepsilon) \nabla_\varepsilon \phi(\varepsilon) d\varepsilon$$
Vì $\nabla_\varepsilon \phi(\varepsilon) = -\varepsilon \phi(\varepsilon)$, hai dấu trừ triệt tiêu:
$$= \int F(\theta + \sigma \varepsilon) \, \varepsilon \, \phi(\varepsilon) d\varepsilon = \mathbb E_{\varepsilon}\big[ F(\theta + \sigma \varepsilon) \varepsilon \big]$$
Chia cho $\sigma$, ta thu được chính xác kết quả: $\nabla_\theta J_\sigma(\theta) = \frac{1}{\sigma} \mathbb E_\varepsilon [F(\theta + \sigma \varepsilon) \varepsilon]$. $\square$

* **Ước lượng Monte Carlo thực nghiệm**:
  $$\hat g = \frac{1}{n \sigma} \sum_{i=1}^n F(\theta + \sigma \varepsilon_i) \varepsilon_i, \qquad \theta \leftarrow \theta + \alpha \hat g$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc lan truyền ngược (Backpropagation) qua toàn bộ chuỗi thời gian MDP của Policy Gradient.
> * **Khuyết điểm của phương pháp cũ**: Policy Gradient gặp trở ngại lớn khi môi trường có phần thưởng cực kỳ trễ (hàng ngàn bước mới có thưởng), hoặc hàm chuyển tiếp không khả vi, và việc đồng bộ gradient trên các cụm máy tính GPU/CPU rất tốn băng thông truyền thông.
> * **Logic thiết kế toán học**: Coi toàn bộ episode như một hàm hộp đen $F(\theta)$. Bằng phép làm trơn Gauss $J_\sigma(\theta) = \mathbb E[F(\theta + \sigma \varepsilon)]$, ta chứng minh được gradient của hàm làm trơn chính là kỳ vọng $\frac{1}{\sigma}\mathbb E[F(\theta + \sigma \varepsilon)\varepsilon]$.
> * **Cơ chế giải quyết**: Gradient được ước lượng hoàn toàn không cần đạo hàm, không cần Critic, không cần giải tích vi phân qua thời gian, cho phép mở rộng song song tuyến tính trên hàng ngàn CPU.

---

## 3. Giảm phương sai & Chứng minh Mirrored (Antithetic) Sampling

### 3.1 Baseline tự nhiên
Vì $\mathbb E[\varepsilon] = 0$, việc trừ bất kỳ hằng số $b$ nào khỏi $F$ không làm thay đổi kỳ vọng gradient:
$$\mathbb E\big[(F - b)\varepsilon\big] = \mathbb E[F\varepsilon] - b \mathbb E[\varepsilon] = \mathbb E[F\varepsilon] - 0 = \mathbb E[F\varepsilon]$$
Chọn $b = \bar F = \frac{1}{n}\sum_{i=1}^n F_i$ (trung bình của batch) giúp giảm phương sai tối đa.

### 3.2 Chứng minh giải tích Mirrored (Antithetic) Sampling
Thuật toán lấy các cặp nhiễu đối xứng $(+\varepsilon, -\varepsilon)$:
$$\boxed{\hat g = \frac{1}{2n\sigma} \sum_{i=1}^n \Big[ F(\theta + \sigma \varepsilon_i) - F(\theta - \sigma \varepsilon_i) \Big] \varepsilon_i}$$

* **Chứng minh toán học (Cơ chế triệt tiêu thành phần bậc chẵn qua Taylor)**:
  Khai triển Taylor bậc 2 của hàm $F$ quanh tâm $\theta$ theo hai hướng $+\sigma \varepsilon$ và $-\sigma \varepsilon$:
  $$F(\theta + \sigma \varepsilon) = F(\theta) + \sigma \nabla F(\theta)^\top \varepsilon + \frac{\sigma^2}{2} \varepsilon^\top \nabla^2 F(\theta) \varepsilon + \mathcal O(\sigma^3)$$
  $$F(\theta - \sigma \varepsilon) = F(\theta) - \sigma \nabla F(\theta)^\top \varepsilon + \frac{\sigma^2}{2} \varepsilon^\top \nabla^2 F(\theta) \varepsilon - \mathcal O(\sigma^3)$$
  Lấy hiệu số hai phương trình:
  $$F(\theta + \sigma \varepsilon) - F(\theta - \sigma \varepsilon) = 2\sigma \nabla F(\theta)^\top \varepsilon + \mathcal O(\sigma^3)$$
  Chia cho $2\sigma$ và nhân với vector $\varepsilon$:
  $$\frac{F(\theta + \sigma \varepsilon) - F(\theta - \sigma \varepsilon)}{2\sigma} \varepsilon = \big( \nabla F(\theta)^\top \varepsilon \big) \varepsilon + \mathcal O(\sigma^2) \varepsilon = \big( \varepsilon \varepsilon^\top \big) \nabla F(\theta) + \mathcal O(\sigma^2) \varepsilon$$
  Lấy kỳ vọng theo $\varepsilon \sim \mathcal N(0, I)$:
  Vì $\mathbb E[\varepsilon \varepsilon^\top] = I$ (ma trận đơn vị) và $\mathbb E[\varepsilon] = 0$:
  $$\mathbb E\left[ \frac{F(\theta + \sigma \varepsilon) - F(\theta - \sigma \varepsilon)}{2\sigma} \varepsilon \right] = I \cdot \nabla F(\theta) + 0 = \nabla F(\theta)$$
  * **Kết luận**: Hiệu số đối xứng **triệt tiêu hoàn toàn** hằng số $F(\theta)$ VÀ thành phần độ cong bậc hai $\varepsilon^\top \nabla^2 F \varepsilon$ (vốn là hai nguồn gây phương sai lớn nhất trong lấy mẫu 1 phía). Phương sai của ước lượng gradient giảm đi đáng kể! $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Cách hai khai triển Taylor tự triệt tiêu nhiễu bậc chẵn:**
> - **Hàm gốc cần ước lượng trên mỗi cặp mẫu:** $\hat{g}_i = \frac{F(\theta + \sigma\varepsilon) - F(\theta - \sigma\varepsilon)}{2\sigma}\varepsilon$.
> - **Hai hàm khai triển Taylor đem thế vào tử số:**
>   $$\begin{aligned}
>   (1)\;\; F(\theta + \sigma\varepsilon) &= \mathbf{F(\theta)} + \sigma \nabla F(\theta)^\top \varepsilon + \mathbf{\frac{\sigma^2}{2}\varepsilon^\top H \varepsilon} + \mathcal{O}(\sigma^3) \\
>   (2)\;\; F(\theta - \sigma\varepsilon) &= \mathbf{F(\theta)} - \sigma \nabla F(\theta)^\top \varepsilon + \mathbf{\frac{\sigma^2}{2}\varepsilon^\top H \varepsilon} - \mathcal{O}(\sigma^3)
>   \end{aligned}$$
>   *(Chú ý: Khi thay $+\varepsilon$ bằng $-\varepsilon$, số hạng bậc 1 đổi dấu thành $-\sigma\nabla F^\top \varepsilon$, nhưng số hạng bậc 2 chứa $(-\varepsilon)^\top H (-\varepsilon) = +\varepsilon^\top H \varepsilon$ **giữ nguyên dấu dương**!)*
> - **Thực hiện thế (1) và (2) vào hiệu số $F(\theta + \sigma\varepsilon) - F(\theta - \sigma\varepsilon)$:**
>   $$F(\theta + \sigma\varepsilon) - F(\theta - \sigma\varepsilon) = \big(\cancel{F(\theta)} - \cancel{F(\theta)}\big) + \big(\sigma\nabla F^\top \varepsilon - (-\sigma\nabla F^\top \varepsilon)\big) + \left(\cancel{\frac{\sigma^2}{2}\varepsilon^\top H \varepsilon} - \cancel{\frac{\sigma^2}{2}\varepsilon^\top H \varepsilon}\right) + \mathcal{O}(\sigma^3) = 2\sigma \nabla F(\theta)^\top \varepsilon + \mathcal{O}(\sigma^3)$$
> - **Thế hiệu số vừa rút gọn vào công thức $\hat{g}_i$ và dùng tính chất tích vô hướng $(\nabla F^\top \varepsilon)\varepsilon = (\varepsilon \varepsilon^\top)\nabla F$:**
>   $$\hat{g}_i = \frac{\cancel{2\sigma} \big(\nabla F(\theta)^\top \varepsilon\big)}{\cancel{2\sigma}}\varepsilon + \mathcal{O}(\sigma^2)\varepsilon = \big(\varepsilon \varepsilon^\top\big)\nabla F(\theta) + \mathcal{O}(\sigma^2)\varepsilon$$
>   Khi lấy trung bình trên đám mây mẫu chuẩn $\varepsilon \sim \mathcal{N}(0, I)$, ma trận hiệp phương sai mẫu $\frac{1}{n}\sum \varepsilon_i \varepsilon_i^\top \to I$, trả về đúng vector đạo hàm $\nabla F(\theta)$!

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế phép lấy mẫu nhiễu một phía $F(\theta + \sigma \varepsilon_i)\varepsilon_i$.
> * **Khuyết điểm của phương pháp cũ**: Lấy mẫu một phía có phương sai rất lớn do sự chênh lệch mức nền của $F(\theta)$ và độ cong phi tuyến bậc 2 (Hessian).
> * **Logic thiết kế toán học**: Lấy mẫu theo cặp đối xứng $\pm \varepsilon$. Bằng khai triển Taylor, hiệu số $F(\theta + \sigma \varepsilon) - F(\theta - \sigma \varepsilon)$ triệt tiêu hoàn toàn số hạng bậc chẵn (bậc 0: giá trị nền $F(\theta)$, bậc 2: ma trận Hessian).
> * **Cơ chế giải quyết**: Giảm phương sai tới 50% và chỉ cần gửi hạt giống ngẫu nhiên (seed) giữa các máy tính, giảm chi phí truyền thông từ $O(nd) \to O(n)$.

---

### 3.3 Chất lượng ước lượng và số chiều
Số tham số $d$ lớn ⇒ cần nhiều mẫu $n$: tỉ số tín hiệu/nhiễu của $\hat g$ tăng theo $n$ và giảm khi $d$ tăng (cỡ $\sqrt{n/d}$ về cosine với gradient thật, mang tính định tính). Vì vậy ES hợp với mạng nhỏ hoặc có nhiều CPU. Với $\sigma$:
- **$\sigma$ quá nhỏ**: $F(\theta+\sigma\varepsilon)-F(\theta)$ chìm trong nhiễu đánh giá (môi trường ngẫu nhiên).
- **$\sigma$ quá lớn**: xấp xỉ tuyến tính sai, tìm kiếm quá thô.

---

## 4. Rank shaping (fitness shaping)

Reward thô có thể có outlier (một episode may mắn) và thang đo thay đổi theo bài toán. Thay $F_i$ bằng **hạng đã chuẩn hoá**:
$$u_i=\frac{\mathrm{rank}_i-1}{m-1}-\frac12\in\Big[-\tfrac12,\tfrac12\Big],\qquad m=\text{số mẫu}$$
(rank 1 = tệ nhất). Khi đó cập nhật: $\theta\leftarrow\theta+\dfrac{\alpha}{n\sigma}\sum_iu_i\varepsilon_i$.

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc nhân trực tiếp giá trị phần thưởng thô $F_i$ vào vector nhiễu $\varepsilon_i$.
> * **Khuyết điểm của phương pháp cũ**: Thang đo phần thưởng thay đổi thất thường theo môi trường; các giá trị ngoại lai (outliers - một episode tình cờ đạt điểm cực cao) sẽ bẻ gãy hướng đi của gradient.
> * **Logic thiết kế toán học**: Chuyển đổi toàn bộ điểm số của quần thể thành thứ hạng chuẩn hóa $u_i \in [-0.5, 0.5]$.
> * **Cơ chế giải quyết**: Đạt được tính bất biến với mọi phép co giãn đơn điệu của hàm thưởng, triệt tiêu ảnh hưởng của ngoại lai và tự động đóng vai trò là một baseline trung bình bằng 0.

**Lợi ích**:
- Bất biến với **mọi phép biến đổi đơn điệu** của reward (thang đo, offset).
- Chống outlier.
- $u_i$ trung bình 0 ⇒ tự động đóng vai trò baseline.
- Tương đương tối ưu hoá một mục tiêu "thứ hạng" (giống bản chất của các thuật toán tiến hoá như CMA-ES/NES).

Code: [`normalized_rank`](ES.py#L123-L130) dùng `scipy.stats.rankdata` (xử lý đồng hạng bằng hạng trung bình).

---

## 5. Mở rộng song song bằng "Common Random Numbers"

Vấn đề: nếu mỗi worker gửi lại vector nhiễu $\varepsilon_i\in\mathbb R^d$ về bộ điều phối, chi phí truyền $O(nd)$ rất lớn.

Giải pháp: nhiễu được sinh từ **seed** của bộ sinh số giả ngẫu nhiên. Worker chỉ gửi $(\text{seed}_i,F^+_i,F^-_i)$ — 3 số. Bộ điều phối **tái tạo** $\varepsilon_i$ từ seed đó. Chi phí truyền $O(n)$ thay vì $O(nd)$, nên mở rộng được lên hàng nghìn CPU (bài báo OpenAI: 1440 CPU giải Humanoid trong ~10 phút).

Code:
- Worker ([L92-L120](ES.py#L92-L120)): nhận tham số, chọn `seed=np.random.randint(1e6)`, `np.random.seed(seed)`, sinh `noise`, đánh giá `±noise`, trả `[[pos_rew, neg_rew], seed]`.
- Điều phối ([L190-L200](ES.py#L190-L200)): `np.random.seed(p_seed); noise = sample_noise(actor)` tạo lại **đúng** nhiễu, rồi `batch_noise += [noise, -noise]`.
- Điều kiện đúng: cùng thứ tự tham số và kiểu dữ liệu sinh nhiễu ở worker và điều phối.

---

## 6. Genetic Algorithm (GA)

GA duy trì **quần thể** các tham số và tiến hoá qua thế hệ:
1. **Đánh giá** fitness $F(\theta^{(k)})$ cho mọi cá thể.
2. **Chọn lọc**: giữ top-$T$ (truncation) hoặc chọn theo xác suất ∝ fitness.
3. **Đột biến**: $\theta'=\theta+\sigma\varepsilon$.
4. (Tuỳ chọn) **Lai ghép**: trộn tham số hai bố mẹ.
5. Thay quần thể; lặp.

**So với ES**: GA là **quần thể rời rạc + chọn lọc**, ES cập nhật **một** tâm $\theta$ bằng trung bình có trọng số của nhiễu (gradient ước lượng). Deep Neuroevolution (Such et al. 2017) cho thấy GA đơn giản (truncation selection + đột biến Gauss, không lai ghép) huấn luyện được mạng hàng triệu tham số cho Atari, cạnh tranh DQN/A3C.

**Khi nào dùng**: không cần khả vi; chịu được reward thưa; dễ song song; nhưng kém hiệu quả dữ liệu hơn RL gradient.

---

## 7. So sánh ES với Policy Gradient (PG)

Cả hai đều dạng "$\nabla=\mathbb E[F\cdot\nabla\log p]$", khác nhau ở **không gian nhiễu**:

| | Policy Gradient | Evolution Strategies |
|---|---|---|
| Nhiễu đặt ở | **hành động** (mỗi bước) | **tham số** (mỗi episode) |
| $\nabla\log p$ | $\nabla_\theta\log\pi_\theta(a_t|s_t)$ (backprop) | $\varepsilon/\sigma$ (giải tích, không backprop) |
| Policy | ngẫu nhiên | **tất định** (mạng ra mean) |
| Credit assignment | theo thời gian (return, critic) | không cần: chấm điểm cả episode |
| Phương sai | tăng theo độ dài $T$ | phụ thuộc số chiều $d$, độc lập $T$ |
| Song song | cần đồng bộ gradient | chỉ gửi vài số vô hướng |
| Hiệu quả mẫu | tốt hơn | kém hơn (thường 10–100×) |
| Reward thưa/trễ | khó | ít nhạy |

Hệ quả: với episode **dài** và reward **trễ**, ES tránh phương sai $O(T)$ của PG; với bài toán ngắn/dày reward, PG hiệu quả hơn nhiều.

---

## 8. Đọc code: [`ES.py`](ES.py)

### 8.1 Mạng chính sách
`NeuralNetwork`: MLP 8→32→32 (Tanh), đầu `mean_l` (32→2) rồi `tanh` ⇒ hành động $\in(-1,1)^2$, **tất định**.
Số tham số: $(8\cdot32+32)+(32\cdot32+32)+(32\cdot2+2)=288+1056+66=1410$ tham số thực sự ảnh hưởng đầu ra. Cộng thêm `var_l` (66) và `logstd` (2) là **tham số thừa** (không dùng trong `forward`) ⇒ tổng 1478 chiều bị nhiễu, 68 chiều "chết".

### 8.2 Luồng một vòng lặp ES
```
1. Điều phối đưa BATCH_SIZE=100 bản tham số vào params_queue
2. Mỗi worker lấy 1 bản: sinh seed → noise
      F⁺ = evaluate(θ + σ·noise);   F⁻ = evaluate(θ − σ·noise)
      trả [[F⁺, F⁻], seed]
3. Điều phối: tái tạo noise từ seed; gom 200 (noise, reward)
4. u = normalized_rank(200 reward)
5. update_j = (1/(100·σ)) Σ_i u_i · noise_i[j]
6. p.grad = −update;  Adam.step()       # tối thiểu hoá −J = tối đa hoá J
```

### 8.3 Ánh xạ
| Công thức | Code |
|---|---|
| $\theta+\sigma\varepsilon$ | `p.data += torch.FloatTensor(n * STD_NOISE)` ([L83](ES.py#L83)) |
| $F(\theta\pm\sigma\varepsilon)$ | `evaluate_neuralnet` (chạy 1 episode, cộng reward) |
| $u_i$ | `normalized_rank` |
| $\frac{1}{n\sigma}\sum u_i\varepsilon_i$ | `upd_weights / (BATCH_SIZE*STD_NOISE)` ([L219](ES.py#L219)) |
| leo gradient | `p.grad = −upd_weights; optimizer.step()` (Adam) |

### 8.4 Phân tích chuyên sâu các tham số và cơ chế tác động lên phương trình ES

Evolution Strategies là phương pháp tối ưu hoá ngẫu nhiên không vi phân, động học của nó hoàn toàn phụ thuộc vào việc điều chỉnh 4 tham số sau:

#### (1) Độ lệch chuẩn nhiễu `STD_NOISE` ($\sigma = 0.05$)
* **Xuất hiện tại**:
  $$\theta' = \theta \pm \sigma \varepsilon,\qquad \hat g = \frac{1}{2n\sigma} \sum_{i=1}^n \big(F_i^+ - F_i^-\big) \varepsilon_i$$
* **Cơ chế tác động toán học**:
  * $\sigma$ xác định **bán kính làm trơn Gaussian** của bề mặt mục tiêu: $J_\sigma(\theta) = \mathbb E_{\varepsilon \sim \mathcal N(0, I)}[F(\theta + \sigma \varepsilon)]$.
  * Phép khai triển Taylor cho thấy:
    $$\frac{F(\theta + \sigma \varepsilon) - F(\theta - \sigma \varepsilon)}{2\sigma} = \nabla F(\theta)^\top \varepsilon + \mathcal O(\sigma^2 \|\varepsilon\|^3)$$
  * **Khi $\sigma$ quá lớn ($\sigma \ge 0.3$)**: Sai số bậc cao $\mathcal O(\sigma^2)$ lấn át gradient tuyến tính. Nhiễu làm biến dạng chính sách quá mạnh khiến tàu vũ trụ LunarLander mất kiểm soát ngay từ giây đầu tiên của mọi episode. Mọi cá thể đều nhận điểm sàn (crash -100 điểm) $\implies F^+ - F^- \approx 0$, thuật toán hoàn toàn mù tịt hướng cải thiện.
  * **Khi $\sigma$ quá nhỏ ($\sigma \le 0.005$)**: Hiệu số $F^+ - F^-$ trở nên cực kỳ bé. Trong khi đó, môi trường LunarLander có tính ngẫu nhiên (vị trí spawn ban đầu, vận tốc gió). Biến động ngẫu nhiên của môi trường sẽ áp đảo hoàn toàn tín hiệu do $\sigma \varepsilon$ tạo ra $\implies$ gradient ước lượng bị nhiễu hạt phá hủy.
  * **Giá trị chuẩn mực**: $\sigma \in [0.02, 0.1]$.

---

#### (2) Kích thước Batch đánh giá `BATCH_SIZE` ($n = 100$ cặp $\implies 200$ episodes/vòng)
* **Cơ chế tác động toán học**:
  * Mạng nơ-ron có $d \approx 1410$ tham số. Theo lý thuyết xác suất trong không gian nhiều chiều, hai vector ngẫu nhiên độc lập trong $\mathbb R^d$ hầu như luôn trực giao với nhau.
  * Phương sai của ước lượng gradient tỷ lệ thuận với số chiều $d$ và tỷ lệ nghịch với kích thước mẫu $n$:
    $$\operatorname{Var}(\hat g) \propto \frac{d}{n}$$
  * **Khi $n$ quá nhỏ ($n < 20$)**: Với 1410 chiều, chỉ lấy 20 hướng $\varepsilon$ ngẫu nhiên là hoàn toàn không đủ để quét không gian. Hướng cập nhật $\sum u_i \varepsilon_i$ hầu như trực giao với hướng dốc thực sự của hàm mục tiêu $\implies$ agent đi lạc không phương hướng.
  * **Khi $n$ lớn ($n \ge 1000$)**: Ước lượng gradient cực kỳ chuẩn xác, nhưng đòi hỏi thời gian tính toán rất lớn. Đó là lý do OpenAI dùng 1440 CPUs song song để chạy $n = 10000$. Với máy tính cá nhân 4 cores, $n = 100$ là điểm cân bằng thực tiễn.

---

#### (3) Chuẩn hoá thứ hạng `normalized_rank` ($u_i \in [-0.5, 0.5]$)
* **Xuất hiện tại**: $u_i = \frac{\operatorname{rank}(F_i) - 1}{m - 1} - 0.5$ với $m = 2n = 200$.
* **Cơ chế tác động**:
  * **Khử hiện tượng Outlier phá vỡ gradient**: Giả sử trong 200 episode có 199 episode tàu rơi nhận $-100$ điểm, nhưng có 1 episode may mắn gió thổi hạ cánh trúng đích nhận $+250$ điểm. Nếu dùng phần thưởng thô, một mẫu duy nhất này sẽ áp đảo toàn bộ $99\%$ gradient, kéo trọng số nhảy vọt về một hướng có thể chỉ là do may mắn nhất thời.
  * Qua phép biến đổi hạng, cá thể tốt nhất nhận giá trị cố định $+0.5$, cá thể tệ nhất nhận $-0.5$. Đóng góp của mọi thứ hạng là đều đặn, không có cá thể nào có thể độc chiếm gradient.
  * **Bất biến với thang đo**: Thuật toán cho kết quả y hệt dù môi trường trả về reward $[0, 1]$ hay $[-1000, +1000]$.

---

#### (4) Tốc độ học Adam `LEARNING_RATE` ($\alpha = 0.01$)
* **Cơ chế tác động**:
  * Vector cập nhật $g = \frac{1}{n\sigma}\sum u_i \varepsilon_i$ được đưa trực tiếp vào `p.grad` để Adam xử lý.
  * Vì $u_i$ đã bị giới hạn trong $[-0.5, 0.5]$, độ lớn của gradient ước lượng không bị biến động thất thường. Điều này cho phép ES dùng learning rate $0.01$ (cao gấp 10–20 lần so với PPO hay DQN) mà không sợ nổ mạng, giúp agent thích nghi nhanh chóng qua từng vòng lặp.

---

## 9. Lưu ý (đã đối chiếu code)

1. **Số episode**: mỗi vòng có $2\times100=200$ episode (`+noise` và `-noise` mỗi worker call). README ghi "100 game/vòng ⇒ 65.000 game sau 650 vòng"; theo code thì $\approx130.000$.
2. **Tham số thừa** `var_l`, `logstd` bị nhiễu hoá dù không ảnh hưởng đầu ra ⇒ lãng phí chiều tìm kiếm và thêm nhiễu cho ước lượng.
3. `sample_noise` trả `np.array(list_of_arrays)` với shape khác nhau ⇒ mảng đối tượng ("ragged"); NumPy ≥1.24 báo lỗi. Dùng `list`.
4. Truyền `actor.state_dict()` qua Queue **mỗi mẫu** (100 lần/vòng) tốn kém; ES chuẩn gửi tham số một lần mỗi vòng cho mỗi worker.
5. `evaluate_noisy_net` đánh giá **một** episode cho mỗi nhiễu — phương sai của $F$ (do ngẫu nhiên môi trường) lớn. Rank shaping giúp, nhưng có thể cải thiện bằng trung bình nhiều episode.
6. API cũ: `LunarLanderContinuous-v2`, `gym.wrappers.Monitor`.

---

## 10. Ví dụ số

**Ước lượng gradient 1 chiều**: $F(\theta)=-\theta^2$ (cực đại tại 0), $\theta=1$, $\sigma=0.1$. Mirrored với $\varepsilon=+1$: $F(1.1)=-1.21$, $F(0.9)=-0.81$. Ước lượng
$$\hat g=\frac{F(\theta+\sigma\varepsilon)-F(\theta-\sigma\varepsilon)}{2\sigma}\varepsilon=\frac{-1.21-(-0.81)}{0.2}\cdot1=-2$$
Gradient thật: $-2\theta=-2$ ✓ (chính xác vì $F$ bậc hai: sai phân trung tâm không sai số).

**Rank shaping**: 4 reward $[−300,\,−100,\,50,\,2000]$ ⇒ hạng $[1,2,3,4]$ ⇒ $u=\frac{\text{rank}-1}{3}-0.5=[-0.5,-0.167,0.167,0.5]$. Outlier 2000 không còn chi phối: chỉ ảnh hưởng thông qua thứ hạng 4.

**Cập nhật**: nhiễu $\varepsilon=[+1,-1]$ cho $F^+$ cao hơn $F^-$ ⇒ $u^+=+0.5,u^-=-0.5$ ⇒ đóng góp $0.5\cdot\varepsilon+(-0.5)\cdot(-\varepsilon)=\varepsilon$: cùng hướng $\varepsilon$ (nhận gấp đôi do cặp đối xứng).

---

## 11. Bài tập và câu hỏi kiểm tra

**Lý thuyết**
1. Tự suy ra $\nabla J_\sigma=\frac1\sigma\mathbb E[F(\theta+\sigma\varepsilon)\varepsilon]$ (§2.2).
2. Vì sao $\mathbb E[F(\theta)\varepsilon]=0$? Từ đó chứng minh trừ baseline không làm bias.
3. Chứng minh mirrored sampling khử các số hạng bậc chẵn của Taylor.
4. Vì sao rank shaping bất biến với phép biến đổi đơn điệu của reward?
5. Chi phí truyền thông của ES với seed là bao nhiêu so với truyền vector nhiễu? Giải thích điều kiện để tái tạo đúng.
6. So sánh $\nabla\log p$ giữa ES và PG (bảng §7).

**Thực hành**
1. Chạy ES **không** rank shaping (dùng reward thô chuẩn hoá z-score) và so sánh.
2. Bỏ mirrored sampling (dùng chỉ `+noise`, trừ baseline trung bình) và so sánh phương sai đường học.
3. Quét $\sigma\in\{0.02,0.05,0.1,0.2\}$ và `BATCH_SIZE`$\in\{25,50,100\}$.
4. Xoá `var_l`, `logstd` khỏi mạng và quan sát tốc độ học.
5. Cài GA đơn giản (top-20% giữ lại + đột biến σ) trên cùng mạng; so sánh số episode cần để đạt ≥200.
6. Đo thời gian/vòng với 1, 2, 4, 8 worker (tính speedup).

**Công thức cần thuộc**: $\nabla J_\sigma=\frac1\sigma\mathbb E[F(\theta+\sigma\varepsilon)\varepsilon]$ · mirrored · $u_i=\frac{\text{rank}-1}{m-1}-\frac12$ · $\theta\leftarrow\theta+\frac{\alpha}{n\sigma}\sum u_i\varepsilon_i$.
