# Week 7 – Model-Based Reinforcement Learning

> Mục tiêu: hiểu cách **học mô hình môi trường** rồi **lập kế hoạch** trên mô hình; tại sao model-based tiết kiệm mẫu, chứng minh toán học **Bổ đề mô phỏng (Simulation Lemma)** giải thích sai số tích luỹ $O(H^2 \varepsilon)$, chứng minh cơ chế làm trơn Tikhonov của nhiễu đầu vào, MPC và random shooting hoạt động ra sao, và đọc hiểu [`model_based.py`](model_based.py) (RoboschoolAnt, theo Nagabandi et al. 2018).  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

Mã nguồn thực hành đi kèm: [`model_based.py`](model_based.py), sơ đồ thuật toán [`imgs/pseudocode.png`](imgs/pseudocode.png).

**Lộ trình**: §1 Model-free vs model-based → §2 Học mô hình động học (kèm chứng minh Tikhonov) → §3 Học mô hình reward → §4 Sai số tích luỹ & Chứng minh Simulation Lemma → §5 Lệch phân phối và aggregation → §6 Lập kế hoạch (MPC, shooting) → §7 Thuật toán trong repo → §8 Đọc code & Phân tích tham số → §9 Lưu ý → §10 Ví dụ số → §11 Mở rộng → §12 Bài tập.

---

## 1. Model-free và model-based

**Model-free** (Week 2–6): học thẳng $Q$ hoặc $\pi$ từ tương tác; không giữ mô hình thế giới. Hiệu năng cuối tốt nhưng cần **hàng triệu** mẫu.

**Model-based**: học
$$\hat f_\phi(s,a)\approx s'\quad(\text{động học}),\qquad\hat r_\psi(s,a)\approx r\quad(\text{reward})$$

* **Giải thích ký hiệu**:
  * $\hat f_\phi(s, a)$: Mạng nơ-ron tham số hoá bởi $\phi$ xấp xỉ chuyển tiếp động học.
  * $\hat r_\psi(s, a)$: Mạng nơ-ron tham số hoá bởi $\psi$ xấp xỉ hàm phần thưởng.
  * $s \in \mathcal S, a \in \mathcal A$: Trạng thái và hành động hiện tại.

rồi dùng mô hình để **mô phỏng "trong đầu"** và chọn hành động.

| | Model-free | Model-based |
|---|---|---|
| Hiệu quả mẫu | thấp | **cao** (học có giám sát từ mọi chuyển tiếp) |
| Hiệu năng tiệm cận | cao | thấp hơn nếu mô hình sai (*model bias*) |
| Tính toán lúc chạy | rẻ (forward 1 mạng) | **đắt** (lập kế hoạch mỗi bước) |
| Tái dùng cho nhiệm vụ khác | khó | dễ: đổi hàm reward, giữ mô hình |

**Vì sao hiệu quả mẫu cao**: mô hình học bằng **hồi quy có giám sát** với nhãn chắc chắn ($s'$ thật), tín hiệu dày hơn nhiều so với reward vô hướng (có thể thưa). Mỗi chuyển tiếp cho $\dim(s)$ giá trị nhãn thay vì 1.

---

## 2. Học mô hình động học

### 2.1 Bài toán hồi quy
Dữ liệu $\mathcal D=\{(s_i,a_i,s'_i)\}$. Học $\hat f_\phi$ bằng cực tiểu sai số bình phương:
$$\min_\phi\ \frac1{|\mathcal D|}\sum_i\big\|\Delta_i-\hat f_\phi(s_i,a_i)\big\|^2,\qquad\Delta_i=s'_i-s_i$$

### 2.2 Dự đoán **hiệu** $\Delta s$ thay vì $s'$
$s'=s+\Delta s$. Lý do:
- Với bước thời gian nhỏ, $s'\approx s$; dự đoán $s'$ trực tiếp buộc mạng phải "sao chép" $s$ (lãng phí năng lực).
- $\Delta s$ có độ lớn nhỏ, phân phối gần $0$, thang đo ổn định.
- Tương tự khối **residual**: $\hat s'=s+\hat f(s,a)$ — sai số một bước giữ nhỏ.
Code: [L144](model_based.py#L144) `y_env_train = y_env_train - obs`, và khi mô phỏng [L286](model_based.py#L286) `m_obs = pred_obs + m_obs`.

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Thế Cuộn Residual (Residual Rollout) qua $H$ bước tương lai:**
> - **Phương trình chuyển tiếp 1 bước:** $\hat{s}_{t+1} = \hat{s}_t + \hat{f}_\phi(\hat{s}_t, a_t)$ (với trạng thái xuất phát thực tế $\hat{s}_t = s_t$).
> - **Thế bước 1 vào bước 2 ($\hat{s}_{t+2} = \hat{s}_{t+1} + \hat{f}_\phi(\hat{s}_{t+1}, a_{t+1})$):**
>   Thay $\hat{s}_{t+1}$ bằng $s_t + \hat{f}_\phi(s_t, a_t)$:
>   $$\hat{s}_{t+2} = \underbrace{\big[s_t + \hat{f}_\phi(s_t, a_t)\big]}_{\text{Thế cho }\hat{s}_{t+1}} + \hat{f}_\phi\Big(\underbrace{s_t + \hat{f}_\phi(s_t, a_t)}_{\text{Thế cho }\hat{s}_{t+1}},\, a_{t+1}\Big)$$
> - **Thế liên tiếp đến bước $t+H$:**
>   $$\hat{s}_{t+H} = s_t + \sum_{k=0}^{H-1} \hat{f}_\phi(\hat{s}_{t+k}, a_{t+k})$$
>   *Ý nghĩa toán học:* Nhờ phép thế cộng dồn kiểu ResNet ($\hat{s}_{t+1} = \hat{s}_t + \Delta\hat{s}_t$), đạo hàm riêng $\frac{\partial \hat{s}_{t+1}}{\partial \hat{s}_t} = I + \frac{\partial \hat{f}_\phi}{\partial \hat{s}_t}$ luôn có ma trận đơn vị $I$ làm "đường cao tốc", giữ cho trạng thái không bị biến dạng về $0$ khi cuộn $H$ bước!

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc huấn luyện mạng nơ-ron dự đoán trực tiếp toàn bộ vector trạng thái kế tiếp $\hat s_{t+1} = \hat f_\phi(s_t, a_t)$.
> * **Khuyết điểm của phương pháp cũ**: Trong mô phỏng vật lý với $\Delta t$ rất nhỏ, trạng thái $s_{t+1} \approx s_t$. Mạng nơ-ron dự đoán trực tiếp $s'$ sẽ dành $99\%$ tham số để học "hàm đồng nhất" (copy $s_t$ sang $s_{t+1}$), khiến các thay đổi vi mô mang tính quyết định của vận tốc và gia tốc bị chìm lấp trong sai số số học.
> * **Logic thiết kế toán học**: Áp dụng nguyên lý mạng phần dư **ResNet (Residual Learning)**. Mạng chỉ chuyên tâm mô hình hóa biến thiên vi phân $\Delta s = s_{t+1} - s_t$, vốn có giá trị nhỏ tập trung quanh 0.
> * **Cơ chế giải quyết**: Giảm thiểu sai số dự đoán 1 bước và tăng độ nhạy đối với các biến động động học nhanh.

### 2.3 Chuẩn hoá (standardization)
Mỗi biến được đưa về trung bình 0, độ lệch chuẩn 1: $\tilde x=(x-\mu)/\sigma$. Áp dụng cho
- đầu vào $(s,a)$ (`input_scaler`),
- nhãn $\Delta s$ (`env_output_scaler`),
- nhãn reward (`rew_output_scaler`).

Vì các chiều trạng thái có thang đo rất khác nhau (vị trí, vận tốc, lực chạm…), không chuẩn hoá thì MSE bị chi phối bởi chiều lớn nhất. Khi dự đoán phải **đảo ngược**: $\Delta\hat s=\sigma_\Delta\,\hat y+\mu_\Delta$ (`inverse_transform`).

### 2.4 Nhiễu huấn luyện & Chứng minh Tikhonov Regularization
Cộng nhiễu Gauss nhỏ $\xi \sim \mathcal N(0, \sigma_\xi^2 I)$ vào vector đầu vào $X = [s; a]$ khi huấn luyện ([L187](model_based.py#L187)).

* **Chứng minh toán học (Cơ chế phạt độ dốc ma trận Jacobian)**:
  Xét hàm mất mát bình phương khi có nhiễu đầu vào:
  $$\mathcal L_{noisy} = \mathbb E_{\xi \sim \mathcal N(0, \sigma_\xi^2 I)}\left[ \big\| \hat f_\phi(X + \xi) - Y \big\|_2^2 \right]$$
  Khai triển Taylor bậc 1 của $\hat f_\phi(X + \xi)$ quanh $X$:
  $$\hat f_\phi(X + \xi) = \hat f_\phi(X) + J_f(X) \xi + \mathcal O(\|\xi\|^2)$$
  với $J_f(X) = \nabla_X \hat f_\phi(X)$ là ma trận Jacobian đạo hàm riêng của mô hình. Thay vào hàm mất mát:
  $$\hat f_\phi(X + \xi) - Y = \big( \hat f_\phi(X) - Y \big) + J_f(X) \xi + \mathcal O(\|\xi\|^2)$$
  Bình phương chuẩn Euclid:
  $$\big\| \hat f_\phi(X + \xi) - Y \big\|_2^2 \approx \big\| \hat f_\phi(X) - Y \big\|_2^2 + 2 \big( \hat f_\phi(X) - Y \big)^\top J_f(X) \xi + \xi^\top J_f(X)^\top J_f(X) \xi$$
  Lấy kỳ vọng theo $\xi$: Vì $\mathbb E[\xi] = 0$ và $\mathbb E[\xi \xi^\top] = \sigma_\xi^2 I$:
  $$\mathbb E_\xi\big[ 2 (\hat f_\phi(X) - Y)^\top J_f(X) \xi \big] = 0$$
  $$\mathbb E_\xi\big[ \xi^\top J_f(X)^\top J_f(X) \xi \big] = \operatorname{Tr}\Big( J_f(X)^\top J_f(X) \mathbb E[\xi \xi^\top] \Big) = \sigma_\xi^2 \operatorname{Tr}\Big( J_f(X)^\top J_f(X) \Big) = \sigma_\xi^2 \big\| J_f(X) \big\|_F^2$$
  Do đó:
  $$\boxed{\mathcal L_{noisy} \approx \big\| \hat f_\phi(X) - Y \big\|_2^2 + \sigma_\xi^2 \big\| \nabla_X \hat f_\phi(X) \big\|_F^2}$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Tại sao cộng nhiễu $\xi$ vào đầu vào $X$ lại biến thành cộng chuẩn Frobenius $\|\nabla_X \hat{f}_\phi\|_F^2$ vào hàm Loss?**
> - **Bước 1 (Thế khai triển Taylor bậc 1 của $\hat{f}_\phi(X+\xi)$ vào vector sai số):**
>   Đặt vector sai số gốc khi không có nhiễu là $\mathbf{e} = \hat{f}_\phi(X) - Y$.
>   Thế xấp xỉ Taylor $\hat{f}_\phi(X + \xi) \approx \hat{f}_\phi(X) + J_f(X)\xi$ vào hiệu số bên trong chuẩn $L_2^2$:
>   $$\hat{f}_\phi(X + \xi) - Y \approx \underbrace{\big(\hat{f}_\phi(X) - Y\big)}_{=\mathbf{e}} + J_f(X)\xi = \mathbf{e} + J_f(X)\xi$$
> - **Bước 2 (Thế hằng đẳng thức tích vô hướng $\|\mathbf{u} + \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + 2\mathbf{u}^\top \mathbf{v} + \mathbf{v}^\top \mathbf{v}$ với $\mathbf{u} = \mathbf{e}, \mathbf{v} = J_f(X)\xi$):**
>   $$\big\|\mathbf{e} + J_f(X)\xi\big\|_2^2 = \|\mathbf{e}\|_2^2 + 2\mathbf{e}^\top J_f(X)\xi + \big(J_f(X)\xi\big)^\top \big(J_f(X)\xi\big) = \|\mathbf{e}\|_2^2 + 2\mathbf{e}^\top J_f(X)\xi + \xi^\top J_f(X)^\top J_f(X)\xi$$
> - **Bước 3 (Thế tính chất Vết ma trận $\mathbf{v}^\top \mathbf{M} \mathbf{v} = \operatorname{Tr}(\mathbf{v}^\top \mathbf{M} \mathbf{v}) = \operatorname{Tr}(\mathbf{M}\mathbf{v}\mathbf{v}^\top)$ vào số hạng bậc 2 và lấy kỳ vọng $\mathbb{E}_\xi$):**
>   - Số hạng chéo bậc 1: $\mathbb{E}_\xi[2\mathbf{e}^\top J_f(X)\xi] = 2\mathbf{e}^\top J_f(X)\underbrace{\mathbb{E}[\xi]}_{=\mathbf{0}} = 0$.
>   - Số hạng bậc 2: Thế $\xi^\top J_f^\top J_f \xi = \operatorname{Tr}\big(J_f^\top J_f (\xi\xi^\top)\big)$ rồi thế tiếp hiệp phương sai của nhiễu $\mathbb{E}[\xi\xi^\top] = \sigma_\xi^2 I$:
>     $$\mathbb{E}_\xi\big[\xi^\top J_f^\top J_f \xi\big] = \operatorname{Tr}\Big(J_f^\top J_f \underbrace{\mathbb{E}[\xi\xi^\top]}_{=\sigma_\xi^2 I}\Big) = \sigma_\xi^2 \underbrace{\operatorname{Tr}\big(J_f^\top J_f\big)}_{=\|J_f\|_F^2} = \sigma_\xi^2 \|J_f(X)\|_F^2$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc huấn luyện MSE thuần túy trên dữ liệu đầu vào không có nhiễu.
> * **Khuyết điểm của phương pháp cũ**: Mạng nơ-ron dễ bị hiện tượng quá khớp (Overfitting) vào các điểm mẫu cá biệt, khiến hàm động học có độ dốc cực lớn $\|\nabla f\| \gg 1$. Khi cuộn mô hình nhiều bước trong lập kế hoạch, các sai lệch nhỏ ở đầu vào bị khuếch đại dữ dội.
> * **Logic thiết kế toán học**: Bơm nhiễu $\xi$ vào đầu vào tương đương giải tích với **Chính quy hóa Tikhonov** phạt trực tiếp chuẩn Frobenius của ma trận Jacobi đạo hàm động học.
> * **Cơ chế giải quyết**: Ép bề mặt hàm chuyển tiếp phải trơn tru cục bộ, ngăn ngừa bùng nổ sai số khi cuộn mô hình nhiều bước.

---

## 3. Học mô hình reward

Bài báo gốc giả định **hàm reward đã biết** (là hàm của $s,a$ thiết kế sẵn). Repo (khác bài báo) **học** reward bằng mạng thứ hai, vì Roboschool không cho công thức reward trực tiếp:
$$\min_\psi\ \frac1{|\mathcal D|}\sum_i\big(r_i-\hat r_\psi(s_i,a_i)\big)^2$$
Hệ quả: sai số reward cộng vào sai số động học khi lập kế hoạch ⇒ cần nhiều dữ liệu hơn và hyperparameter được tinh chỉnh lại (README).

---

## 4. Sai số tích luỹ & Chứng minh Bổ đề mô phỏng (Simulation Lemma)

Gọi $\varepsilon$ là sai số dự đoán **một bước** của mô hình động học $\hat f$. Khi mô phỏng chuỗi $H$ bước liên tiếp:
$$\hat s_{t+k+1} = \hat s_{t+k} + \hat f(\hat s_{t+k}, a_{t+k})$$
Nếu động học có hằng số Lipschitz $L_f = \|\nabla_s f\| > 1$, sai số sau $H$ bước bùng nổ theo cấp số nhân:
$$\|\hat s_{t+H} - s_{t+H}\| \le \varepsilon \sum_{j=0}^{H-1} L_f^j = \varepsilon \frac{L_f^H - 1}{L_f - 1} \approx \mathcal O(L_f^H \varepsilon)$$

* **Chứng minh toán học Bổ đề mô phỏng (Simulation Lemma)**:
  Xét MDP thật $M = (P, R, \gamma)$ và MDP mô hình $\hat M = (\hat P, \hat R, \gamma)$. Giả sử sai số mô hình bị chặn bởi:
  $$\max_{s, a} \|P(\cdot \mid s, a) - \hat P(\cdot \mid s, a)\|_{TV} \le \varepsilon_P, \qquad \max_{s, a} |R(s, a) - \hat R(s, a)| \le \varepsilon_R$$
  Gọi $V_M^\pi$ và $V_{\hat M}^\pi$ là hàm giá trị của cùng một chính sách $\pi$ trên hai môi trường.
  * **Bước 1**: Áp dụng phương trình ma trận Bellman:
    $$V_M^\pi = R^\pi + \gamma P^\pi V_M^\pi \implies V_M^\pi = (I - \gamma P^\pi)^{-1} R^\pi$$
    $$V_{\hat M}^\pi = \hat R^\pi + \gamma \hat P^\pi V_{\hat M}^\pi \implies V_{\hat M}^\pi = (I - \gamma \hat P^\pi)^{-1} \hat R^\pi$$
  * **Bước 2**: Khai triển hiệu số $V_M^\pi - V_{\hat M}^\pi$:
    $$\begin{aligned}
    V_M^\pi - V_{\hat M}^\pi &= (I - \gamma P^\pi)^{-1} R^\pi - V_{\hat M}^\pi \\
    &= (I - \gamma P^\pi)^{-1} \Big( R^\pi - (I - \gamma P^\pi) V_{\hat M}^\pi \Big) \\
    &= (I - \gamma P^\pi)^{-1} \Big( R^\pi - V_{\hat M}^\pi + \gamma P^\pi V_{\hat M}^\pi \Big) \\
    &= (I - \gamma P^\pi)^{-1} \Big( R^\pi - (\hat R^\pi + \gamma \hat P^\pi V_{\hat M}^\pi) + \gamma P^\pi V_{\hat M}^\pi \Big) \\
    &= (I - \gamma P^\pi)^{-1} \Big( (R^\pi - \hat R^\pi) + \gamma (P^\pi - \hat P^\pi) V_{\hat M}^\pi \Big)
    \end{aligned}$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Thủ thuật "Nhân rồi Thế" trong chứng minh Simulation Lemma:**
> - **Mục tiêu:** Ta muốn gom hiệu hai hàm giá trị $V_M^\pi - V_{\hat{M}}^\pi$ về thành hiệu hai hàm thưởng $(R^\pi - \hat{R}^\pi)$ và hiệu hai ma trận chuyển tiếp $(P^\pi - \hat{P}^\pi)$.
> - **Bước thế 1 (Đặt nhân tử chung ma trận nghịch đảo $(I - \gamma P^\pi)^{-1}$):**
>   Ta viết $V_{\hat{M}}^\pi = \underbrace{(I - \gamma P^\pi)^{-1}(I - \gamma P^\pi)}_{=I} V_{\hat{M}}^\pi$ và thế vào $V_M^\pi - V_{\hat{M}}^\pi$:
>   $$V_M^\pi - V_{\hat{M}}^\pi = (I - \gamma P^\pi)^{-1}\Big[R^\pi - (I - \gamma P^\pi)V_{\hat{M}}^\pi\Big] = (I - \gamma P^\pi)^{-1}\Big[R^\pi - \underbrace{V_{\hat{M}}^\pi}_{\text{Thế ở Bước 2}} + \gamma P^\pi V_{\hat{M}}^\pi\Big]$$
> - **Bước thế 2 (Thế phương trình Bellman của mô hình ảo $V_{\hat{M}}^\pi = \hat{R}^\pi + \gamma \hat{P}^\pi V_{\hat{M}}^\pi$ vào đúng chữ $V_{\hat{M}}^\pi$ đứng ở giữa):**
>   *(Chú ý: Chỉ thế vào chữ $V_{\hat{M}}^\pi$ mang dấu trừ, giữ nguyên cụm $+\gamma P^\pi V_{\hat{M}}^\pi$ ở cuối để ghép cặp!)*
>   $$= (I - \gamma P^\pi)^{-1}\Big[R^\pi - \underbrace{\big(\hat{R}^\pi + \gamma \hat{P}^\pi V_{\hat{M}}^\pi\big)}_{\text{Đã thế cho }V_{\hat{M}}^\pi} + \gamma P^\pi V_{\hat{M}}^\pi\Big] = (I - \gamma P^\pi)^{-1}\Big[(R^\pi - \hat{R}^\pi) + \gamma(P^\pi - \hat{P}^\pi)V_{\hat{M}}^\pi\Big]$$
  * **Bước 3**: Lấy chuẩn cực đại $\|\cdot\|_\infty$ hai vế:
    $$\|V_M^\pi - V_{\hat M}^\pi\|_\infty \le \big\|(I - \gamma P^\pi)^{-1}\big\|_\infty \left( \|R^\pi - \hat R^\pi\|_\infty + \gamma \big\|(P^\pi - \hat P^\pi) V_{\hat M}^\pi\big\|_\infty \right)$$
    Nhận thấy $\big\|(I - \gamma P^\pi)^{-1}\big\|_\infty = \big\|\sum_{t=0}^\infty (\gamma P^\pi)^t\big\|_\infty \le \sum_{t=0}^\infty \gamma^t = \frac{1}{1-\gamma}$.
    Lại có $\|V_{\hat M}^\pi\|_\infty \le \frac{R_{\max}}{1-\gamma}$ và $\|(P^\pi - \hat P^\pi) V_{\hat M}^\pi\|_\infty \le 2 \varepsilon_P \|V_{\hat M}^\pi\|_\infty \le \frac{2 R_{\max} \varepsilon_P}{1-\gamma}$.
  * **Bước 4**: Thay vào bất đẳng thức:
    $$\|V_M^\pi - V_{\hat M}^\pi\|_\infty \le \frac{1}{1-\gamma} \left( \varepsilon_R + \frac{2\gamma R_{\max} \varepsilon_P}{1-\gamma} \right) = \frac{\varepsilon_R}{1-\gamma} + \frac{2\gamma R_{\max} \varepsilon_P}{(1-\gamma)^2}$$
  * **Kết luận với bài toán chân trời hữu hạn $H$**:
    Khi $\gamma \to 1$, đại lượng $\frac{1}{1-\gamma}$ tương ứng với độ dài $H$. Do đó:
    $$\boxed{|J_{\text{thật}}(\pi) - J_{\text{mô hình}}(\pi)| \le \mathcal O\Big( H^2 R_{\max} \varepsilon_P + H \varepsilon_R \Big)}$$
  Đẳng thức bậc $H^2$ được chứng minh tường minh! $\square$

* **Hệ quả thực dụng**: Vì sai số tích lũy tăng theo bậc bình phương $H^2$ của tầm nhìn, ta **bắt buộc phải chọn $H$ ngắn** ($H=10$ trong repo) và kết hợp cơ chế Model Predictive Control (MPC) để tái lập kế hoạch liên tục ở mọi bước.

---

## 5. Lệch phân phối và Data Aggregation

### 5.1 Vấn đề
Mô hình huấn luyện trên dữ liệu **ngẫu nhiên** chỉ chính xác ở vùng không gian mà hành động ngẫu nhiên đi tới (quanh trạng thái ban đầu, kiến "run rẩy"). Controller tốt sẽ đi tới những trạng thái **khác** (chạy nhanh) nơi mô hình chưa thấy dữ liệu ⇒ dự đoán tệ (**distribution shift**) ⇒ kế hoạch tệ ⇒ lại đi vào vùng lạ.

### 5.2 Giải pháp (kiểu DAgger)
Lặp:
1. Huấn luyện mô hình trên $\mathcal D$.
2. Chạy controller trên môi trường thật, thu dữ liệu $\mathcal D_{new}$ ở vùng nó thực sự đi tới.
3. $\mathcal D\leftarrow\mathcal D\cup\mathcal D_{new}$.

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc chỉ huấn luyện mô hình một lần trên tập dữ liệu tĩnh ban đầu (Offline Static Training / Behavioral Cloning).
> * **Khuyết điểm của phương pháp cũ**: **Trôi dạt phân phối (Covariate Shift / Distribution Drift)**. Bộ điều khiển tối ưu đưa robot vào những tư thế chuyển động mới chưa từng có trong tập dữ liệu ngẫu nhiên ban đầu. Mô hình dự đoán sai $\implies$ bộ điều khiển ra quyết định sai $\implies$ robot bị ngã văng ra ngoài không gian dữ liệu.
> * **Logic thiết kế toán học**: Nguyên lý **DAgger (Dataset Aggregation)**. Bắt buộc mô hình phải học trên chính phân phối trạng thái $d^{\pi_{controller}}$ do bộ điều khiển sinh ra.
> * **Cơ chế giải quyết**: Đảm bảo sai số tích luỹ bị chặn tuyến tính $\mathcal O(H \varepsilon)$ thay vì bùng nổ bậc hai $\mathcal O(H^2 \varepsilon)$, giúp mô hình có khả năng tự sửa sai khi đi vào vùng trạng thái mới.

---

## 6. Lập kế hoạch: MPC và Random Shooting

### 6.1 Bài toán tối ưu chuỗi hành động
Tại trạng thái thật $s_t$, tìm
$$A^*=\arg\max_{a_{t},\dots,a_{t+H-1}}\ \sum_{k=0}^{H-1}\hat r(\hat s_{t+k},a_{t+k}),\qquad\hat s_{t+k+1}=\hat s_{t+k}+\hat f(\hat s_{t+k},a_{t+k}),\ \hat s_t=s_t$$

### 6.2 Model Predictive Control (receding horizon)
Chỉ **thực thi hành động đầu tiên** $a_t^*$ của chuỗi tối ưu, quan sát $s_{t+1}$ thật, rồi **giải lại** từ đầu.

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc thực thi trọn vẹn toàn bộ chuỗi $H$ hành động đã lập kế hoạch (Open-Loop Trajectory Execution).
> * **Khuyết điểm của phương pháp cũ**: Do sai số mô hình bùng nổ theo cấp số nhân $\mathcal O(L_f^H \varepsilon)$, từ bước thứ 3 hoặc thứ 4 trở đi, thế giới thực đã phân kỳ hoàn toàn so với mô phỏng. Nếu cứ mù quáng thực thi hết chuỗi hành động, robot sẽ va chạm hoặc ngã sụp.
> * **Logic thiết kế toán học**: Triết lý **Chân trời lùi dần (Receding Horizon / Closed-Loop Feedback)**. Agent chỉ tin tưởng kế hoạch trong đúng 1 bước đầu tiên ($a_t = A_0^*$). Ngay sau khi thực hiện, agent quan sát trạng thái thực $s_{t+1}$ từ cảm biến thế giới thực, "xóa bỏ" kế hoạch cũ và giải lại bài toán tối ưu trên trạng thái thực mới.
> * **Cơ chế giải quyết**: Liên tục neo kế hoạch vào trạng thái thực tế, triệt tiêu sự bùng nổ sai số đa bước và bù trừ hoàn hảo cho nhiễu môi trường.

### 6.3 Random Shooting
Tối ưu bằng cách **thử ngẫu nhiên**: lấy $K$ chuỗi hành động $A^{(1..K)}$ ngẫu nhiên (đều trong biên hành động), mô phỏng từng chuỗi bằng mô hình, chọn chuỗi có tổng reward dự đoán cao nhất, thực thi hành động đầu của nó.
- Repo: $K=20000$, $H=10$ ⇒ mỗi bước môi trường tốn $20000\times10=200\,000$ lượt mô phỏng mô hình, **được vector hoá** thành $H$ lần forward với batch $K$ ([`multi_model_based_control`](model_based.py#L252-L305)).
- **Hạn chế**: không gian chuỗi có $H\times\dim(a)=10\times8=80$ chiều; xác suất ngẫu nhiên rơi vào vùng gần tối ưu giảm theo hàm mũ theo chiều ("lời nguyền số chiều"). Random shooting vẫn hiệu quả vì chỉ cần chuỗi **đủ tốt** (MPC sửa sai mỗi bước).
- **Nâng cấp**: CEM (lặp: lấy mẫu → giữ top-$k$ → cập nhật Gauss), MPPI (trung bình có trọng số $\exp(\text{return}/\lambda)$), gradient qua mô hình.

---

## 7. Thuật toán trong repo (tóm tắt)

```
D_rand ← 1000 quỹ đạo ngẫu nhiên
D_rl   ← ∅
for it in 1..3:
    huấn luyện f̂ (Δs) và r̂ trên D_rand ∪ D_rl   (55 epoch, batch 512, Adam 1e-3)
    reset env
    while số bước thu < 20000:
        a ← MPC random-shooting(f̂, r̂, s; K=20000, H=10)
        s', r, done ← env.step(a)
        D_rl ← D_rl ∪ {(s,s',r,done,a)}
        nếu done: reset
```
Chỉ phần **model-based** được cài; bài báo còn khởi tạo policy bằng bắt chước controller (imitation) rồi tinh chỉnh bằng **model-free** (TRPO) để thành MB-MF.

---

## 8. Đọc code: [`model_based.py`](model_based.py)

### 8.1 Bản đồ hàm
| Hàm | Vai trò | Toán |
|---|---|---|
| `NNDynamicModel` | MLP dự đoán $\Delta s$ | $\hat f_\phi$ |
| `NNRewardModel` | MLP dự đoán $r$ | $\hat r_\psi$ |
| `gather_random_trajectories` | thu $\mathcal D_{rand}$ | lấy mẫu $a\sim\mathcal U$ |
| `train_dyna_model` | chuẩn hoá, chia train/valid, huấn luyện 2 mạng | MSE |
| `multi_model_based_control` | random shooting batch | $\arg\max\sum\hat r$ |
| vòng chính | aggregation + MPC | §7 |

### 8.2 Chi tiết `train_dyna_model`
1. Nối dữ liệu ngẫu nhiên và RL (từ vòng 2).
2. **Chia**: 80% train, 20% valid (`D_valid = d_concat[-num_examples_added/5:]`, tức cuối tập).
3. Tạo $X=[s;a]$, nhãn động học $\Delta s=s'-s$, nhãn reward.
4. Chuẩn hoá 3 bộ (`StandardScaler`) và **lưu** `norm` để dùng khi lập kế hoạch.
5. Với mỗi epoch, mỗi minibatch: cộng nhiễu, cập nhật mạng động học rồi mạng reward (hai optimizer độc lập).
6. Mỗi 10 epoch in loss validation.

### 8.3 Chi tiết `multi_model_based_control` (vector hoá)
```
m_obs      = lặp s thật K lần                       # [K, ds]
unroll_rew = 0                                      # [K, 1]
for t in range(H):
    a_k   = K hành động ngẫu nhiên                  # lấy từ env.action_space.sample
    inp   = input_scaler.transform([m_obs ; a_k])
    Δŝ    = env_output_scaler.inverse_transform(f̂(inp))
    m_obs = m_obs + Δŝ                              # trạng thái dự đoán mới
    unroll_rew += r̂(inp)                            # cộng reward dự đoán (đã chuẩn hoá)
    nếu t == 0: lưu a_k làm "hành động đầu"
best = argmax(unroll_rew); trả a_first[best]
```
Lưu ý: `unroll_rew` cộng giá trị reward **ở thang đã chuẩn hoá** ($\tilde r=(r-\mu_r)/\sigma_r$); vì phép biến đổi **tăng, affine, giống nhau cho mọi chuỗi** nên không đổi thứ hạng `argmax` (tổng $H$ số hạng, hằng $H\mu_r$ không ảnh hưởng). ✔.

### 8.4 Phân tích chuyên sâu các tham số và cơ chế tác động trong Model-Based RL & MPC

Hệ thống Model-Based RL trong [`model_based.py`](model_based.py#L308-L328) được vận hành bởi tổ hợp các tham số kiểm soát quá trình học mô hình động học và thuật toán điều khiển dự đoán MPC:

| Tên siêu tham số | Giá trị trong Code | Ký hiệu Toán học | Vai trò & Phạm vi kiểm soát |
|---|---|---|---|
| `HORIZION_LENGTH` | `10` | $H$ | Độ dài tầm nhìn tương lai khi "cuộn" (rollout) mô hình động học |
| `NUM_ACTIONS_SEQUENCES` | `20000` | $K$ | Số lượng chuỗi hành động ngẫu nhiên được lấy mẫu song song (Random Shooting) |
| `AGGR_ITER` | `3` | $M$ | Số vòng lặp thu thập dữ liệu và tái huấn luyện mô hình (DAgger Aggregation) |
| `STEPS_PER_AGGR` | `20000` | $N_{aggr}$ | Số bước môi trường thực tế thu thập ở mỗi vòng aggregation |
| `NUM_RAND_TRAJECTORIES` | `1000` | $N_{rand}$ | Số quỹ đạo ngẫu nhiên thu thập ban đầu để khởi tạo tập $\mathcal D_{rand}$ |
| `scale` (Nhiễu đầu vào) | `0.001` | $\sigma_{\xi}$ | Độ lệch chuẩn của nhiễu Gauss cộng vào đầu vào khi huấn luyện mô hình |
| `TRAIN_ITER_MODEL` | `55` | $E_{model}$ | Số epoch huấn luyện mạng động học $\hat{f}_\phi$ và mạng reward $\hat{r}_\psi$ |
| `BATCH_SIZE` | `512` | $B$ | Kích thước minibatch cho bộ tối ưu hóa Adam |
| `ENV/REW_LEARNING_RATE` | `1e-3` | $\alpha_f, \alpha_r$ | Tốc độ học của mạng động học và mạng reward |

---

#### 1. Tầm nhìn lập kế hoạch $H$ (`HORIZION_LENGTH = 10`)
Tham số $H$ quyết định số bước thời gian mà MPC mô phỏng về phía trước trong bài toán tối ưu quỹ đạo:
$$A^* = \arg\max_{a_{t:t+H-1}} \sum_{k=0}^{H-1} \hat{r}\big(\hat{s}_{t+k}, a_{t+k}\big),\qquad \hat{s}_{t+k+1} = \hat{s}_{t+k} + \hat{f}_\phi(\hat{s}_{t+k}, a_{t+k})$$

* **Cơ chế tác động lên Compounding Error (Sai số tích luỹ)**:
  Giả sử sai số dự đoán 1 bước của mô hình bị chặn bởi $\|\hat{f}(s, a) - f(s, a)\| \le \varepsilon$, và động học có hằng số Lipschitz $L_f = \|\nabla_s f\|$.
  - Nếu $L_f \le 1$ (hệ thống tiêu tán ổn định): Sai số sau $H$ bước tích luỹ tuyến tính $O(H \varepsilon)$, và theo **Simulation Lemma**, sai số của hàm giá trị mục tiêu bị chặn bởi bậc:
    $$\big|J_{\text{thật}}(A) - J_{\text{mô hình}}(A)\big| \le \frac{R_{\max} \varepsilon}{(1-\gamma)^2} \approx O(H^2 R_{\max} \varepsilon)$$
  - Nếu $L_f > 1$ (hệ thống phi tuyến nhạy cảm, hỗn loạn như khớp chân robot Ant khi tiếp đất): Sai số bùng nổ theo cấp số nhân:
    $$\|\hat{s}_{t+H} - s_{t+H}\| \le \varepsilon \sum_{j=0}^{H-1} L_f^j = \varepsilon \frac{L_f^H - 1}{L_f - 1} \approx O(L_f^H \varepsilon)$$
* **Hành vi khi $H \to 1$ (Quá ngắn)**: MPC suy biến thành bộ điều khiển tham lam 1 bước (Myopic Controller). Agent không nhìn thấy hậu quả tương lai (ví dụ: chân đạp mạnh về phía trước giúp tăng vận tốc tức thời ở bước $t$, nhưng dẫn đến lật ngửa robot và kết thúc episode ở bước $t+3$).
* **Hành vi khi $H \gg 20$ (Quá dài)**: Sai số tích luỹ $O(L_f^H \varepsilon)$ lấn át hoàn toàn tín hiệu thật. Mô hình bắt đầu sinh ra hiện tượng "ảo giác" (hallucination) — tối ưu hóa trên trạng thái tưởng tượng phi vật lý, dẫn đến chuỗi hành động thực tế gây sụp đổ toàn diện. Đồng thời, không gian tìm kiếm bùng nổ theo cấp số nhân $O(|\mathcal A|^H)$.
* **Điểm tối ưu thực nghiệm $H = 10$**: Tương đương khoảng 0.1 - 0.2 giây vật lý trong Roboschool/MuJoCo. Đủ dài để agent dự đoán được quán tính và phản lực mặt sàn, nhưng đủ ngắn để sai số $L_f^{10}\varepsilon$ vẫn nằm trong biên độ mà cơ chế Receding Horizon của MPC có thể bù trừ ở bước kế tiếp.

---

#### 2. Số lượng chuỗi hành động thử nghiệm $K$ (`NUM_ACTIONS_SEQUENCES = 20000`)
Tham số $K$ quyết định mật độ lấy mẫu trong không gian hành động của thuật toán **Random Shooting**.

* **Cơ chế bao phủ không gian (Action Space Coverage)**:
  Không gian tối ưu chuỗi hành động là một siêu khối liên tục có số chiều:
  $$D = H \times \dim(\mathcal A) = 10 \times 8 = 80 \text{ chiều}$$
  Thể tích không gian tìm kiếm chuẩn hoá $[-1, 1]^{80}$ là $V = 2^{80} \approx 1.2 \times 10^{24}$.
  Giả sử vùng chuỗi hành động "chất lượng cao" (sub-optimal basin) chiếm một tỷ lệ thể tích $\rho \in (0, 1)$ cực nhỏ. Xác suất để trong $K$ mẫu ngẫu nhiên có ít nhất 1 chuỗi lọt vào vùng này là:
  $$P(\text{thành công}) = 1 - (1 - \rho)^K \approx 1 - \exp(-K \rho)$$
* **Hành vi khi $K$ nhỏ ($K < 1000$)**: Xác suất $1 - e^{-K\rho} \approx 0$ trong không gian 80 chiều. MPC không tìm được chuỗi phối hợp nhịp nhàng giữa 8 khớp động cơ; robot di chuyển giật cục, run rẩy và nhanh chóng ngã gục.
* **Hành vi khi $K$ lớn ($K = 20000$)**: Mật độ mẫu đủ dày để bắt được các chuỗi hành động có xu hướng đẩy thân robot về phía trước. Chi phí tính toán là $K \times H = 200\,000$ lượt forward MLP mỗi bước môi trường. Nhờ vector hoá hoàn toàn trên GPU ([L252-L305](model_based.py#L252-L305)), phép tính này chỉ mất vài mili-giây.
* **Hạn chế ($K \to \infty$)**: Do random shooting lấy mẫu đều độc lập qua các bước thời gian, hành vi tiệm cận bị chặn bởi **Model Bias**. Khi cần độ chính xác cao hơn, người ta chuyển sang thuật toán Cross-Entropy Method (CEM) hoặc MPPI để lấy mẫu thích nghi thay vì tăng $K$ vô hạn.

---

#### 3. Số vòng lặp Data Aggregation $M$ (`AGGR_ITER = 3`) và $N_{aggr}$ (`STEPS_PER_AGGR = 20000`)
Hai tham số này thiết lập cơ chế **DAgger (Dataset Aggregation)** nhằm giải quyết vấn đề phân phối trôi dạt (Distribution Shift):

$$\mathcal D^{(0)} = \mathcal D_{rand},\qquad \mathcal D^{(m)} = \mathcal D^{(m-1)} \cup \mathcal D_{\pi_{MPC}}^{(m)}$$

* **Cơ chế toán học khắc phục Covariate Shift**:
  Khi chỉ học trên $\mathcal D_{rand}$, mạng nơ-ron học phân phối $p_{rand}(s, a)$. Sai số kỳ vọng khi chạy MPC policy $\pi_{MPC}$ bị khuếch đại bởi khoảng cách phân phối:
  $$\mathbb E_{(s, a)\sim \pi_{MPC}}\big[\|\Delta s - \hat{f}(s, a)\|^2\big] \le \varepsilon_{train} + C \cdot D_{TV}\big(p_{\pi_{MPC}}(s, a) \parallel p_{rand}(s, a)\big)$$
  Mỗi vòng `AGGR_ITER` bổ sung $20000$ bước trạng thái mà chính bộ điều khiển MPC ghé thăm vào tập huấn luyện, ép khoảng cách biến thiên toàn phần $D_{TV} \to 0$.
* **Hành vi khi `AGGR_ITER = 1` (Không aggregation)**: Mô hình chỉ biết các trạng thái "co giật" ngẫu nhiên gần điểm xuất phát. Khi robot bắt đầu chạy nhanh về phía trước, trạng thái văng ra khỏi miền hỗ trợ của dữ liệu huấn luyện, mô hình sụp đổ hoàn toàn.
* **Hành vi khi `AGGR_ITER` tăng (3 - 5 vòng)**: Chất lượng mô hình hội tụ tiệm cận về stationary distribution của bộ điều khiển tối ưu. Sau vòng 3, robot Ant trong repo đạt điểm số $\approx 800$ (tương đương học có giám sát trên 60,000 bước trạng thái thực tế).

---

#### 4. Nhiễu làm trơn đầu vào $\sigma_{\xi}$ (`scale = 0.001`)
Trong quá trình huấn luyện mạng động học, một lượng nhiễu Gauss $\xi \sim \mathcal N(0, \sigma_{\xi}^2 I)$ được cộng vào vector đầu vào $X = [s; a]$ ([L187](model_based.py#L187)).

* **Bản chất giải tích (Tikhonov Regularization & Lipschitz Smoothing)**:
  Khai triển Taylor bậc 2 của hàm mất mát kỳ vọng khi có nhiễu đầu vào:
  $$\mathbb E_{\xi}\Big[\big\|\hat{f}(X + \xi) - Y\big\|^2\Big] \approx \big\|\hat{f}(X) - Y\big\|^2 + \sigma_{\xi}^2 \big\|\nabla_X \hat{f}(X)\big\|_F^2$$
  Việc cộng nhiễu tương đương về mặt toán học với việc **phạt ma trận Jacobian / Frobenius norm của gradient** $\|\nabla_X \hat{f}(X)\|_F^2$.
* **Tác động lên quá trình mô phỏng MPC**: Khi cuộn $H$ bước, đầu vào ở bước $t+k$ chính là đầu ra $\hat{s}_{t+k}$ của bước trước, vốn luôn mang một sai số nhỏ. Nếu mạng $\hat{f}$ có độ dốc Lipschitz lớn, sai số này sẽ bị khuếch đại dữ dội. Nhiễu $\sigma_{\xi}$ ép mô hình phải phẳng và trơn cục bộ (locally robust), ngăn hiện tượng bùng nổ sai số dọc theo quỹ đạo mô phỏng.
* **Lưu ý thực thi trong repo**: Do lệnh `X_mb += np.random.normal(...)` can thiệp trực tiếp lên mảng nhớ in-place ([§9.2](BAI_GIANG.md#9-luu-y-da-doi-chieu-code)), phương sai nhiễu bị tích luỹ qua các epoch. Cần sửa thành `X_mb = X_mb + noise` để giữ nguyên $\sigma_{\xi} = 0.001$.

---

#### 5. Tốc độ học $\alpha_f, \alpha_r$ (`1e-3`) và Epochs $E_{model}$ (`55`)
* **`ENV_LEARNING_RATE` & `REW_LEARNING_RATE = 1e-3`**: Hai mạng nơ-ron hồi quy phi tuyến độc lập được tối ưu bằng Adam. Mức $10^{-3}$ là tiêu chuẩn giúp hàm mất mát MSE hội tụ nhanh chóng trong vòng 55 epoch trên batch size 512.
* **Nguy cơ Overfitting khi $E_{model}$ quá lớn**: Nếu huấn luyện quá nhiều epoch ($E_{model} > 200$), mạng nơ-ron sẽ ghi nhớ máy móc các chuyển tiếp trong $\mathcal D$, làm mất khả năng nội suy mượt mà (smooth generalization) tại các trạng thái nằm giữa các quỹ đạo đã quan sát.

---

## 9. Lưu ý (đã đối chiếu code)

1. **Biến toàn cục**: [L180-L184](model_based.py#L180-L184) dùng `BATCH_SIZE` thay vì tham số `batch_size`; cùng giá trị 512 nên chưa lỗi.
2. **Nhiễu cộng dồn**: `X_mb += np.random.normal(...)` là phép cộng tại chỗ trên một **view** của `X_train`, nên nhiễu bị cộng dồn **vĩnh viễn** vào dữ liệu qua các epoch (độ lệch chuẩn tích luỹ $\approx0.001\sqrt{55}\approx0.007$ — nhỏ nhưng sai ý đồ). Sửa: `X_mb = X_mb + noise`. Ngoài ra comment ghi "variance 0.0001" nhưng `scale=0.001` là **độ lệch chuẩn** (variance = $10^{-6}$).
3. **Vòng thu thập** kết thúc theo episode nên vượt `STEPS_PER_AGGR`; không dự đoán `done` (reward model không biết khi nào ngã/kết thúc).
4. **Chia train/valid**: `D_valid` là phần **cuối** tập (không xáo trước khi chia) ⇒ từ vòng 2 nó toàn dữ liệu RL (phân phối khác train); vẫn hợp lý để đo độ chính xác trên vùng quan tâm.
5. **`device='cuda'`** cố định; 20000 chuỗi × 10 bước mỗi hành động ⇒ chậm trên CPU.
6. **Roboschool** đã ngừng phát triển ⇒ dùng `Ant-v4/v5` của MuJoCo.
7. Reward model ra thang chuẩn hoá nhưng log in `unnorm_rew` dùng `inverse_transform([float(...)])` — chỉ để in.

---

## 10. Ví dụ số

**Dự đoán $\Delta s$ và chuẩn hoá**: một chiều vận tốc có $\mu_\Delta=0.01,\ \sigma_\Delta=0.2$. Mạng ra $\hat y=1.5$ ⇒ $\Delta\hat s=0.2\cdot1.5+0.01=0.31$. Trạng thái hiện tại $s=2.0$ ⇒ $\hat s'=2.31$.

**Tích luỹ sai số**: giả sử mỗi bước sai $\varepsilon=0.05$ và độ nhạy động học $\|\nabla_sf\|=1.1$. Sai số sau $k$ bước $\approx\varepsilon\sum_{j<k}1.1^j=0.05\cdot\frac{1.1^k-1}{0.1}$: $k=5\Rightarrow0.31$, $k=10\Rightarrow0.80$, $k=20\Rightarrow2.9$. ⇒ Horizon 10 trong repo là vừa phải; 50 bước sẽ thành $\approx58$ (vô nghĩa).

**Chi phí MPC**: $K\times H=20000\times10=2\times10^5$ lượt mô hình mỗi bước môi trường; một episode 1000 bước ⇒ $2\times10^8$ lượt (nhờ batch GPU vẫn khả thi nhưng đắt, đối chiếu PPO chỉ 1 forward/bước).

---

## 11. Mở rộng cần biết

| Phương pháp | Ý tưởng |
|---|---|
| **Dyna-Q** (Sutton) | dùng mô hình để sinh thêm kinh nghiệm giả, cập nhật Q thêm (kết hợp planning + learning) |
| **PETS** | ensemble mô hình xác suất + CEM, tách bất định aleatoric/epistemic |
| **MBPO** | rollout ngắn từ trạng thái thật + SAC, giảm compounding error |
| **World Models / Dreamer** | học mô hình trong không gian tiềm ẩn từ ảnh, "mơ" để học policy |
| **MuZero** | mô hình tiềm ẩn chỉ học thứ cần cho lập kế hoạch (reward, value, policy) |
| **AlphaZero** | mô hình luật chơi **đã biết** + MCTS (tuần 8) |

---

## 12. Bài tập và câu hỏi kiểm tra

**Lý thuyết**
1. Vì sao dự đoán $\Delta s$ tốt hơn $s'$? Viết lại sai số $s'$ theo sai số $\Delta s$.
2. Giải thích compounding error và vì sao MPC giảm thiểu nó.
3. Vì sao chỉ thực thi hành động đầu rồi lập kế hoạch lại (receding horizon)?
4. Nêu hai lý do cần aggregation dữ liệu; liên hệ với DAgger.
5. Chứng minh `argmax` không đổi khi cộng hằng $H\mu_r$ và nhân $1/\sigma_r>0$ vào tổng reward (§8.3).
6. Mô tả vì sao random shooting kém khi chiều hành động hoặc $H$ lớn.

**Thực hành**
1. Sửa nhiễu cộng dồn (§9.2); so sánh loss validation.
2. Đo **lỗi dự đoán theo horizon**: $\|\hat s_{t+k}-s_{t+k}\|$ với $k=1..20$ trên dữ liệu valid; vẽ đồ thị.
3. Thay random shooting bằng **CEM** (5 vòng lặp, top 10%, cập nhật mean/std Gauss) và so sánh reward với cùng ngân sách mô hình.
4. Thử $H\in\{5,10,20\}$ và $K\in\{1000,5000,20000\}$.
5. Huấn luyện một **ensemble 5 mô hình**, dùng phương sai dự đoán làm hình phạt bất định.
6. So sánh **số mẫu môi trường** cần để đạt ≈800 điểm giữa MB (tuần này) và PPO (tuần 5) trên cùng môi trường.

**Công thức cần thuộc**: $\min\|\Delta s-\hat f(s,a)\|^2$ · $\hat s'=s+\hat f$ · $A^*=\arg\max\sum\hat r$ · lỗi mô hình $\sim H^2\varepsilon$ · $K\cdot H$ lượt mô phỏng mỗi bước.
