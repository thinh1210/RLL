# Week 8 – Các khái niệm nâng cao & Đồ án tốt nghiệp (Advanced RL & Capstone)

> Mục tiêu: làm chủ các lý thuyết hiện đại ở ranh giới nghiên cứu (SOTA) của Deep RL: **Suy luận xác suất & Điều khiển (Chứng minh MaxEnt RL, SAC)**, **Học tăng cường nghịch đảo & Bắt chước (DAgger, MaxEnt IRL, GAIL)**, **Khám phá nâng cao (Curiosity, RND)**, **Meta-RL (MAML, RL²)**, kiến trúc vô địch cờ **AlphaGo Zero / AlphaZero**, hệ thống **OpenAI Five**, và **Cầu nối RLHF & DPO cho LLM** (kèm chứng minh nghiệm đóng).  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

**Lộ trình**: §1 Suy luận xác suất & Chứng minh MaxEnt RL (SAC) → §2 Imitation Learning & Inverse RL → §3 Cơ chế khám phá nâng cao → §4 Meta-Learning & Transfer → §5 AlphaGo Zero & MCTS → §6 OpenAI Five & Hệ thống phân tán → §7 Cầu nối với RLHF & LLM (kèm chứng minh nghiệm đóng) → §8 Phân tích chuyên sâu tham số thuật toán nâng cao → §9 Đề xuất đồ án Capstone → §10 Tổng kết toàn khóa.

---

## 1. Suy luận xác suất & Maximum Entropy RL (SAC)

### 1.1 Khung lý thuyết: Điều khiển như bài toán suy luận (Control as Inference)
Thay vì chỉ tối đa hoá phần thưởng, ta coi bài toán RL là bài toán **suy luận xác suất Bayes** trên đồ hình xác suất (Probabilistic Graphical Model).
Định nghĩa một biến ngẫu nhiên nhị phân $O_t \in \{0, 1\}$ thể hiện **"bước $t$ có tối ưu hay không"** (Optimality event):
$$p(O_t=1 \mid s_t, a_t) = \exp\big(\beta\, r(s_t, a_t)\big)$$

Xác suất đồng thời của quỹ đạo $\tau = (s_0, a_0, \dots, s_T, a_T)$ và toàn bộ chuỗi sự kiện tối ưu $O_{1:T}=1$:
$$p(\tau, O_{1:T}=1) = p(s_0) \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \exp\big(\beta\, r(s_t, a_t)\big)$$

Khi ta điều kiện hoá theo $O_{1:T}=1$, phân phối hậu nghiệm chuẩn tắc trên quỹ đạo tối ưu là:
$$p(\tau \mid O_{1:T}=1) = \frac{1}{Z} p(s_0) \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \exp\Big(\beta \sum_{t=0}^T r(s_t, a_t)\Big)$$
với $Z = \int p(\tau, O_{1:T}=1) d\tau$ là hằng số chuẩn hoá phân vùng (partition function).

* **Chứng minh toán học: Suy dẫn mục tiêu Maximum Entropy RL**:
  Ta muốn tìm một chính sách hành vi $\pi_\theta(a_t \mid s_t)$ sao cho phân phối quỹ đạo sinh ra bởi chính sách:
  $$p_\pi(\tau) = p(s_0) \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \pi(a_t \mid s_t)$$
  tiệm cận phân phối hậu nghiệm tối ưu $p(\tau \mid O_{1:T}=1)$ bằng cách cực tiểu hoá phân kỳ Kullback-Leibler:
  $$\min_\pi D_{KL}\Big( p_\pi(\tau) \parallel p(\tau \mid O_{1:T}=1) \Big)$$
  
  Khai triển định nghĩa của $D_{KL}$:
  $$\begin{aligned}
  D_{KL}\Big( p_\pi(\tau) \parallel p(\tau \mid O_{1:T}=1) \Big) &= \int p_\pi(\tau) \ln \frac{p_\pi(\tau)}{p(\tau \mid O_{1:T}=1)} d\tau \\
  &= \mathbb E_{\tau \sim \pi}\left[ \ln \frac{p(s_0) \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \pi(a_t \mid s_t)}{\frac{1}{Z} p(s_0) \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \exp\big(\beta r(s_t, a_t)\big)} \right]
  \end{aligned}$$
  *Nhận xét kỳ diệu*: Toàn bộ phân phối khởi đầu $p(s_0)$ và động học môi trường $P(s_{t+1} \mid s_t, a_t)$ ở tử số và mẫu số **triệt tiêu hoàn toàn lẫn nhau**!
  $$\begin{aligned}
  &= \mathbb E_{\tau \sim \pi}\left[ \ln Z + \sum_{t=0}^T \Big( \ln \pi(a_t \mid s_t) - \beta r(s_t, a_t) \Big) \right] \\
  &= \ln Z - \beta \sum_{t=0}^T \mathbb E_{(s_t, a_t) \sim \pi}\left[ r(s_t, a_t) - \frac{1}{\beta} \ln \pi(a_t \mid s_t) \right] \\
  &= \ln Z - \beta \sum_{t=0}^T \mathbb E_{(s_t, a_t) \sim \pi}\Big[ r(s_t, a_t) + \alpha \mathcal H\big(\pi(\cdot \mid s_t)\big) \Big]
  \end{aligned}$$
  trong đó $\alpha = \frac{1}{\beta} > 0$ và $\mathcal H(\pi(\cdot \mid s_t)) = -\mathbb E_{a \sim \pi}[\ln \pi(a \mid s_t)]$ là Shannon entropy.
  Vì $\ln Z$ là hằng số không phụ thuộc $\pi$ và $\beta > 0$, bài toán $\min_\pi D_{KL}$ tương đương chính xác với:
  $$\boxed{\max_\pi J_{MaxEnt}(\pi) = \sum_{t=0}^T \mathbb E_{(s_t, a_t) \sim \pi}\Big[ r(s_t, a_t) + \alpha \mathcal H\big(\pi(\cdot \mid s_t)\big) \Big]}$$
  Chứng minh toán học hoàn tất trọn vẹn! $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Thế Tỉ số Phân phối Quỹ đạo vào $D_{\text{KL}}$ và Thế định nghĩa Entropy $\mathcal{H}$:**
> - **Bước 1 (Thế hai hàm phân phối quỹ đạo $p_\pi(\tau)$ và $p(\tau \mid O_{1:T}=1)$ vào phân số bên trong $\ln(\cdot)$ của $D_{\text{KL}}$):**
>   $$\frac{p_\pi(\tau)}{p(\tau \mid O_{1:T}=1)} = \frac{\cancel{p(s_0)} \prod_{t=0}^T \cancel{P(s_{t+1}\mid s_t, a_t)} \, \pi(a_t\mid s_t)}{\frac{1}{Z}\cancel{p(s_0)} \prod_{t=0}^T \cancel{P(s_{t+1}\mid s_t, a_t)} \, \exp\big(\beta r(s_t, a_t)\big)} = Z \prod_{t=0}^T \frac{\pi(a_t\mid s_t)}{\exp\big(\beta r(s_t, a_t)\big)}$$
>   *(Nhờ cả hai phân phối đều đi qua cùng một môi trường vật lý $P(s_{t+1}\mid s_t, a_t)$, toàn bộ các thừa số môi trường ở tử và mẫu tự gạch chéo triệt tiêu nhau!)*
> - **Bước 2 (Thế tỉ số đã rút gọn vào hàm $\ln(\cdot)$ và dùng $\ln\big(\frac{A}{e^B}\big) = \ln A - B$):**
>   $$\ln \frac{p_\pi(\tau)}{p(\tau \mid O_{1:T}=1)} = \ln Z + \sum_{t=0}^T \Big[\ln \pi(a_t\mid s_t) - \beta r(s_t, a_t)\Big]$$
> - **Bước 3 (Thế định nghĩa Shannon Entropy $\mathcal{H}\big(\pi(\cdot\mid s_t)\big) = \mathbb{E}_{a_t \sim \pi}[-\ln \pi(a_t\mid s_t)]$ và $\alpha = \frac{1}{\beta}$):**
>   Đặt $-\beta$ làm nhân tử chung: $\mathbb{E}[\ln \pi(a_t\mid s_t) - \beta r(s_t, a_t)] = -\beta \, \mathbb{E}\big[r(s_t, a_t) + \frac{1}{\beta}(-\ln \pi(a_t\mid s_t))\big]$.
>   Thay $\mathbb{E}_{a_t \sim \pi}[-\ln \pi(a_t\mid s_t)]$ bằng $\mathcal{H}(\pi(\cdot\mid s_t))$ và thay $\frac{1}{\beta}$ bằng $\alpha$, bài toán cực tiểu hóa $-\beta J_{\text{MaxEnt}}(\pi)$ chuyển thành cực đại hóa $J_{\text{MaxEnt}}(\pi)$!

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế các thuật toán Actor-Critic tất định hoặc tối ưu hóa phần thưởng thuần tuý như DDPG, TD3.
> * **Khuyết điểm của phương pháp cũ**: DDPG/TD3 có chính sách tất định, rất dễ bị sụp đổ sớm vào các cực tiểu địa phương nghèo nàn và cực kỳ nhạy cảm với việc chọn siêu tham số.
> * **Logic thiết kế toán học**: Dẫn xuất từ khung lý thuyết **Điều khiển như bài toán suy luận xác suất (Control as Inference)**. Đưa thêm số hạng Shannon Entropy vào hàm mục tiêu với trọng số nhiệt độ $\alpha$. Thuật toán đồng thời tối đa hóa phần thưởng VÀ độ bất định của hành động.
> * **Cơ chế giải quyết**: Chính sách khám phá tự nhiên khắp không gian trạng thái, nắm bắt được các nghiệm **đa mốt (Multimodal)**, chống chịu nhiễu tối ưu, kết hợp với kỹ thuật Dual Gradient Descent để tự động điều chỉnh nhiệt độ $\alpha$ hướng về mức entropy mục tiêu $\bar{\mathcal H} = -\dim(\mathcal A)$.

* **Giải thích ký hiệu**:
  * $O_t \in \{0, 1\}$: Biến nhị phân thể hiện trạng thái tối ưu.
  * $\beta > 0$: Hệ số độ nhạy phần thưởng trong mô hình đồ hình.
  * $\alpha = 1/\beta$: Hệ số nhiệt độ entropy (Entropy temperature).
  * $\mathcal H(\pi)$: Độ đo đa dạng entropy của chính sách.

**Ưu điểm của MaxEnt RL**:
1. Policy học được mọi cách giải khả dĩ (đa mốt - multimodal), không bị sụp đổ về một phương án duy nhất.
2. Bền vững trước nhiễu và thay đổi bất ngờ của môi trường.
3. Thăm dò (exploration) tự nhiên mà không cần các thủ thuật chắp vá như $\epsilon$-greedy.

---

### 1.2 Phương trình Soft Bellman & Soft Value Iteration
Trong MaxEnt RL, phương trình Bellman được biến đổi thành dạng "mềm" (Soft):

$$V_{soft}(s) = \alpha \log \int_{\mathcal A} \exp\Big(\frac{1}{\alpha} Q_{soft}(s, a)\Big) da = \alpha \operatorname{soft}\max_{a} \frac{Q_{soft}(s, a)}{\alpha}$$
$$Q_{soft}(s, a) = r(s, a) + \gamma \mathbb E_{s' \sim P}\big[V_{soft}(s')\big]$$

Policy tối ưu tương ứng phân phối Boltzmann:
$$\pi^*(a \mid s) \propto \exp\Big(\frac{1}{\alpha} Q_{soft}(s, a)\Big)$$

---

### 1.3 Thuật toán Soft Actor-Critic (SAC - Haarnoja et al., 2018)
SAC là thuật toán **off-policy actor-critic** chuẩn mực nhất cho không gian hành động liên tục hiện nay, kết hợp MaxEnt RL với ba kỹ thuật:

#### (a) Clipped Double-Q trick (Chống thổi phồng Q)
Duy trì hai mạng critic $Q_{\theta_1}$ và $Q_{\theta_2}$, cùng hai mạng target $Q_{\bar{\theta}_1}, Q_{\bar{\theta}_2}$.
Mục tiêu Bellman mềm dùng giá trị tối thiểu của hai critic:
$$y = r + \gamma \Big(\min_{j=1,2} Q_{\bar{\theta}_j}(s', \tilde{a}') - \alpha \log \pi_\phi(\tilde{a}' \mid s')\Big),\qquad \tilde{a}' \sim \pi_\phi(\cdot \mid s')$$
Loss huấn luyện Critic ($j \in \{1, 2\}$):
$$J_Q(\theta_j) = \mathbb E_{(s, a, r, s')\sim \mathcal D}\Big[\big(Q_{\theta_j}(s, a) - y\big)^2\Big]$$

#### (b) Reparameterization Trick & Tanh Squashing cho Actor
Để gradient truyền được qua quá trình lấy mẫu hành động liên tục bị chặn trong khoảng $[-1, 1]$:
$$a_\phi(s, \xi) = \tanh\big(\mu_\phi(s) + \sigma_\phi(s) \odot \xi\big),\qquad \xi \sim \mathcal N(0, I)$$
Do phép biến đổi phi tuyến $a = \tanh(u)$, hàm mật độ xác suất phải điều chỉnh theo định lý biến đổi biến ngẫu nhiên (Jacobian determinant):
$$\log \pi_\phi(a \mid s) = \log \mu(u \mid s) - \sum_{i=1}^{\dim(\mathcal A)} \log\big(1 - \tanh^2(u_i)\big)$$
Loss huấn luyện Actor:
$$J_\pi(\phi) = \mathbb E_{s \sim \mathcal D, \xi \sim \mathcal N}\Big[\alpha \log \pi_\phi\big(a_\phi(s, \xi) \mid s\big) - \min_{j=1,2} Q_{\theta_j}\big(s, a_\phi(s, \xi)\big)\Big]$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Phép thế Tái tham số hóa (Reparameterization) & Định thức Jacobian trong SAC:**
> - **Bước 1 (Bài toán gốc bị tắc nghẽn đạo hàm):**
>   Hàm mất mát gốc của Actor là kỳ vọng trên phân phối hành động $a \sim \pi_\phi(\cdot\mid s)$:
>   $$J_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}}\Big[\mathbb{E}_{a \sim \pi_\phi(\cdot\mid s)}\big[\alpha \log \pi_\phi(a\mid s) - Q_\theta(s, a)\big]\Big]$$
>   Vì phép lấy mẫu $a \sim \pi_\phi$ phụ thuộc vào $\phi$, ta không thể đưa đạo hàm $\nabla_\phi$ chui trực tiếp vào trong $Q_\theta(s, a)$ nếu giữ nguyên biến $a$.
> - **Bước 2 (Phép thế Tái tham số hóa — Biểu diễn biến ngẫu nhiên $a$ thành hàm tất định của nhiễu chuẩn $\xi \sim \mathcal{N}(0, I)$):**
>   Đặt biến trung gian (trước khi nén) $u = \mu_\phi(s) + \sigma_\phi(s) \odot \xi$ và biến hành động sau khi nén $a = a_\phi(s, \xi) = \tanh(u)$.
>   **Thế mọi chữ $a$ trong hàm mất mát bằng hàm $a_\phi(s, \xi)$**, đồng thời chuyển kỳ vọng từ $\mathbb{E}_{a \sim \pi_\phi}$ sang $\mathbb{E}_{\xi \sim \mathcal{N}(0, I)}$ (không còn phụ thuộc vào $\phi$!):
>   $$J_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D},\, \xi \sim \mathcal{N}(0, I)}\Big[\alpha \log \pi_\phi\big(\underbrace{a_\phi(s, \xi)}_{\text{Thế cho }a}\mid s\big) - Q_\theta\big(s, \underbrace{a_\phi(s, \xi)}_{\text{Thế cho }a}\big)\Big]$$
>   Nhờ phép thế này, $\nabla_\phi Q_\theta(s, a_\phi(s, \xi)) = \nabla_a Q_\theta(s, a)\big|_{a=a_\phi} \cdot \nabla_\phi a_\phi(s, \xi)$ truyền thẳng đạo hàm từ Critic xuyên qua hành động $a$ về mạng Actor $\phi$!
> - **Bước 3 (Phép thế Đổi biến Mật độ Xác suất qua hàm $a = \tanh(u)$):**
>   Theo định lý đổi biến xác suất với ánh xạ khả nghịch $a = \tanh(u)$, mật độ của $a$ liên hệ với mật độ Gauss $\mu(u\mid s)$ qua định thức ma trận Jacobian:
>   $$\pi_\phi(a\mid s) = \mu(u\mid s) \cdot \left|\det \left(\frac{\partial u}{\partial a}\right)\right| = \mu(u\mid s) \cdot \left|\det \left(\frac{\partial a}{\partial u}\right)\right|^{-1}$$
>   Vì hàm $\tanh$ tác động theo từng chiều độc lập nên ma trận Jacobian $\frac{\partial a}{\partial u} = \operatorname{diag}\big(1 - \tanh^2(u_1), \dots, 1 - \tanh^2(u_D)\big)$ là ma trận đường chéo có định thức bằng tích đường chéo $\prod_{i=1}^D (1 - \tanh^2(u_i))$. Đem thế vào và lấy $\log$ hai vế:
>   $$\log \pi_\phi(a\mid s) = \log \mu(u\mid s) - \sum_{i=1}^D \log\big(1 - \tanh^2(u_i)\big)$$

#### (c) Tự động điều chỉnh hệ số Entropy $\alpha$ (Dual Gradient Descent)
Thay vì cố định $\alpha$, ràng buộc entropy kỳ vọng không nhỏ hơn ngưỡng $\bar{\mathcal H}$ (thường chọn $\bar{\mathcal H} = -\dim(\mathcal A)$):
$$J(\alpha) = \mathbb E_{s \sim \mathcal D, a \sim \pi_\phi}\Big[-\alpha \log \pi_\phi(a \mid s) - \alpha \bar{\mathcal H}\Big]$$

---

## 2. Bắt chước (Imitation Learning) & Học tăng cường nghịch đảo (Inverse RL)

Khi phần thưởng quá khó để thiết kế thủ công, ta học từ tập quỹ đạo mẫu của chuyên gia $\mathcal D^* = \{\tau_1^*, \dots, \tau_N^*\}$.

```
Imitation Learning
├── Behavioral Cloning (Học có giám sát: s → a*)
│    └── Khuyết điểm: Lệch phân phối tích luỹ O(εT²)
├── DAgger (Dataset Aggregation: Hỏi chuyên gia ở trạng thái mới)
│    └── Ưu điểm: Sai số co về O(εT)
└── Inverse Reinforcement Learning (Tìm hàm Reward ngầm R)
     ├── MaxEnt IRL (So khớp tần suất ghé thăm trạng thái)
     └── GAIL (GAN + RL: Discriminator phân biệt Expert vs Agent)
```

### 2.1 Lệch phân phối trong Behavioral Cloning (Covariate Shift)
Behavioral Cloning (BC) huấn luyện policy bằng hồi quy / phân loại có giám sát: $\min_\theta \mathbb E_{(s, a)\sim \mathcal D^*}[\mathcal L(\pi_\theta(s), a)]$.
Nếu xác suất mắc lỗi tại mỗi bước là $\epsilon$:
1. Bước đầu tiên agent lệch nhẹ khỏi quỹ đạo chuyên gia.
2. Trạng thái kế tiếp $s_1$ nằm ngoài phân phối huấn luyện $\mathcal D^*$.
3. Sai số tại các bước tiếp theo bùng nổ dây chuyền.
$$\mathbb E\big[\text{Tổng số lỗi}\big] \le \epsilon T + \epsilon (1-\epsilon) (T-1) + \dots \implies \mathcal O(\epsilon T^2)$$

### 2.2 Thuật toán DAgger (Ross et al., 2011)
Khắc phục covariate shift bằng cách **cho chuyên gia dán nhãn lại các trạng thái mà agent tự đi vào**:
1. Huấn luyện $\pi_1$ ban đầu bằng BC trên $\mathcal D^*$.
2. Cho $\pi_k$ chạy trên môi trường thực tế, thu thập tập trạng thái $\{s_t\}$.
3. Yêu cầu chuyên gia cung cấp hành động chuẩn $a_t^* = \pi^*(s_t)$ cho các trạng thái này.
4. Gộp dữ liệu mới vào tập tích lũy: $\mathcal D \leftarrow \mathcal D \cup \{(s_t, a_t^*)\}$.
5. Huấn luyện lại $\pi_{k+1}$ trên $\mathcal D$.
**Kết quả lý thuyết**: Kỳ vọng sai số giảm từ $\mathcal O(\epsilon T^2)$ xuống tuyến tính $\mathcal O(\epsilon T)$.

---

### 2.3 Maximum Entropy Inverse Reinforcement Learning (Ziebart et al., 2008)
IRL giả định chuyên gia hành động để tối đa hoá một hàm phần thưởng tuyến tính ngầm $R_\theta(s) = \theta^\top \phi(s)$.
Xác suất một quỹ đạo tỉ lệ với phần thưởng tích luỹ (nguyên lý MaxEnt):
$$p(\tau \mid \theta) = \frac{1}{Z(\theta)} \exp\big(R_\theta(\tau)\big) = \frac{1}{Z(\theta)} \exp\Big(\theta^\top \sum_{t=0}^T \phi(s_t)\Big)$$
Cực đại hoá log-likelihood của tập dữ liệu chuyên gia $\mathcal D^*$:
$$\mathcal L(\theta) = \sum_{\tau^* \in \mathcal D^*} \log p(\tau^* \mid \theta) = \sum_{\tau^*} \theta^\top \Phi(\tau^*) - M \log Z(\theta)$$
Đạo hàm theo tham số $\theta$:
$$\nabla_\theta \mathcal L(\theta) = \sum_{\tau^* \in \mathcal D^*} \Phi(\tau^*) - M \sum_\tau p(\tau \mid \theta) \Phi(\tau) = M \Big(\mathbb E_{\tau \sim \pi^*}\big[\Phi(\tau)\big] - \mathbb E_{\tau \sim \pi_\theta}\big[\Phi(\tau)\big]\Big)$$
$$\boxed{\nabla_\theta \mathcal L(\theta) = M \sum_{s} \big(\mu^*(s) - \mu_\theta(s)\big) \phi(s)}$$
trong đó $\mu(s)$ là **Tần suất ghé thăm trạng thái (State Visitation Frequency - SVF)**. Gradient bằng 0 khi tần suất ghé thăm các đặc trưng trạng thái của agent trùng khít với chuyên gia.

---

### 2.4 GAIL: Generative Adversarial Imitation Learning (Ho & Ermon, 2016)
GAIL bỏ qua bước giải bài toán RL nghịch đảo tốn kém để tìm reward, mà thiết lập một **trò chơi Minimax kiểu GAN**:

$$\min_\pi \max_{D \in (0, 1)} \mathbb E_{(s, a)\sim \pi^*}\big[\log D(s, a)\big] + \mathbb E_{(s, a)\sim \pi}\big[\log (1 - D(s, a))\big] - \lambda \mathcal H(\pi)$$

* **Discriminator $D(s, a)$**: Đóng vai trò bộ phân biệt — chấm điểm cao cho cặp $(s, a)$ từ chuyên gia, điểm thấp cho agent.
* **Generator / Policy $\pi(a \mid s)$**: Tối ưu bằng PPO hoặc TRPO với phần thưởng nội tại:
$$r_{GAIL}(s, a) = -\log\big(1 - D(s, a)\big)$$
Khi $D$ không thể phân biệt nổi ($D \to 0.5$), phân phối chiếm dụng trạng thái–hành động $\rho_\pi(s, a)$ tiệm cận hoàn toàn phân phối của chuyên gia $\rho_{\pi^*}(s, a)$.

---

## 3. Các cơ chế khám phá nâng cao (Advanced Exploration)

Trong các môi trường phần thưởng cực thưa (sparse reward) như game giải đố Montezuma's Revenge hay Pitfall, thuật toán truyền thống không bao giờ tình cờ tìm thấy điểm thưởng. Ta bổ sung **phần thưởng nội tại (intrinsic reward)**:
$$r_t^{total} = r_t^{ext} + \beta\, r_t^{int}$$

```
Cơ chế Khám phá (Exploration)
├── Count-based & Density Models (Ghé thăm ít → Thưởng nhiều)
│    └── r_int ∝ 1 / √(N(s))
├── Curiosity / ICM (Tò mò dựa trên sai số dự đoán động học)
│    └── Loại bỏ nhiễu "Noisy TV" qua Inverse Dynamic Features
└── RND (Random Network Distillation)
     └── Sai số giữa mạng cố định và mạng học → Đơn giản, cực kỳ hiệu quả
```

### 3.1 Intrinsic Curiosity Module (ICM - Pathak et al., 2017)
Nếu chỉ dùng mạng dự đoán trạng thái kế tiếp thô $\hat{s}_{t+1} = f(s_t, a_t)$, agent sẽ bị "nghiện" đứng nhìn màn hình tivi nhiễu hạt ngẫu nhiên (**Noisy TV problem**) vì nhiễu ngẫu nhiên không thể dự đoán, tạo ra lỗi dự đoán vĩnh viễn cao.

ICM giải quyết triệt để vấn đề này bằng cách chiếu trạng thái vào không gian đặc trưng $\phi(s)$ chỉ giữ lại thông tin **mà hành động của agent có thể tác động**:
1. **Inverse Dynamic Model**: Dự đoán hành động đã thực hiện từ hai trạng thái liên tiếp:
   $$\hat{a}_t = g\big(\phi(s_t), \phi(s_{t+1})\big),\qquad \min \mathcal L_{inv}(\hat{a}_t, a_t)$$
2. **Forward Dynamic Model**: Dự đoán đặc trưng trạng thái kế tiếp trong không gian $\phi$:
   $$\hat{\phi}(s_{t+1}) = f\big(\phi(s_t), a_t\big),\qquad \mathcal L_{forw} = \frac{1}{2}\big\|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\big\|_2^2$$
3. Phần thưởng tò mò:
   $$\boxed{r_t^{int} = \frac{\eta}{2}\big\|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\big\|_2^2}$$

---

### 3.2 Random Network Distillation (RND - Burda et al., 2018)
RND là bước đột phá đưa PPO lần đầu tiên vượt qua màn 1 của *Montezuma's Revenge* mà không cần dữ liệu chuyên gia.
Cơ chế gồm 2 mạng nơ-ron:
1. **Target Network $f(s)$**: Khởi tạo ngẫu nhiên, **đóng băng trọng số vĩnh viễn**.
2. **Predictor Network $\hat{f}_\theta(s)$**: Huấn luyện liên tục để dự đoán đầu ra của mạng Target:
   $$\min_\theta \big\|\hat{f}_\theta(s) - f(s)\big\|_2^2$$

Phần thưởng nội tại:
$$\boxed{r_t^{int} = \big\|\hat{f}_\theta(s_t) - f(s_t)\big\|_2^2}$$

* **Logic toán học**: 
  * Ở các trạng thái quen thuộc (được thăm nhiều lần), $\hat{f}_\theta$ đã được tối ưu để khớp với $f$, sai số MSE $\to 0$.
  * Ở các trạng thái mới lạ, $\hat{f}_\theta$ chưa từng thấy nên dự đoán sai lệch lớn so với $f$, sai số MSE cao $\to$ thưởng lớn.
  * Vì target $f(s)$ là hàm tất định cố định (deterministic), không có tính ngẫu nhiên, nên hoàn toàn miễn nhiễm với bài toán Noisy TV.

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế cơ chế tạo tò mò dựa trên sai số dự đoán động học của Intrinsic Curiosity Module (ICM).
> * **Khuyết điểm của phương pháp cũ**: Vấn nạn **"Tivi nhiễu hạt" (Noisy TV Problem)**. Nếu môi trường có một chiếc màn hình tivi phát nhiễu ngẫu nhiên không thể đoán trước, mạng dự đoán động học liên tục ghi nhận sai số khổng lồ, khiến agent bị "thôi miên", đứng nhìn tivi mãi mãi vì tưởng rằng đó là vùng đất mới lạ cần khám phá.
> * **Logic thiết kế toán học**: Sử dụng một mạng mục tiêu cố định $f(s)$ được khởi tạo ngẫu nhiên và **đóng băng vĩnh viễn trọng số**, cùng một mạng dự đoán $\hat f_\theta(s)$ học cách bắt chước đầu ra của mạng mục tiêu.
> * **Cơ chế giải quyết**: Trạng thái nào đã ghé thăm nhiều lần thì $\hat f_\theta$ sẽ chưng cất thành công $\implies r^{int} \to 0$. Chiếc tivi phát nhiễu dù ngẫu nhiên nhưng các trạng thái ảnh của nó khi qua $f(s)$ sẽ dần được $\hat f_\theta$ khớp dữ liệu, triệt tiêu ảo giác tò mò và giải quyết hoàn toàn bài toán Sparse Reward (như game kinh điển Montezuma's Revenge).

---

## 4. Meta-Learning & Transfer Learning trong RL

Mục tiêu của Meta-RL là học một thuật toán học tăng cường nhanh: huấn luyện trên phân phối các bài toán $\mathcal T \sim p(\mathcal T)$ sao cho khi đối mặt với một môi trường mới, agent thích nghi chỉ sau 1 đến vài episode.

### 4.1 RL²: Fast Reinforcement Learning via Slow Reinforcement Learning (Duan et al., Wang et al., 2016)
* Biến thuật toán RL thành một mạng hồi quy (RNN / LSTM / GRU).
* Đầu vào của RNN tại mỗi bước chứa cả lịch sử đầy đủ:
  $$x_t = (s_t, a_{t-1}, r_{t-1}, d_{t-1})$$
* Trọng số của RNN được cố định sau khi huấn luyện (gọi là "Slow learning").
* Ẩn số trạng thái (hidden state $h_t$) của RNN đóng vai trò như một **bộ nhớ đệm**, tự động chạy thuật toán học tăng cường trong nội bộ ("Fast learning") mà không cần tính đạo hàm gradient khi inference.

---

### 4.2 MAML for RL (Model-Agnostic Meta-Learning - Finn et al., 2017)
Tìm bộ trọng số khởi tạo ban đầu $\theta$ sao cho chỉ cần 1 bước cập nhật Policy Gradient trên bài toán mới $\mathcal T_i$:
$$\theta_i' = \theta + \alpha \nabla_\theta J_{\mathcal T_i}(\pi_\theta)$$
thì policy $\pi_{\theta_i'}$ đạt hiệu năng tối đa trên bài toán đó:

$$\max_\theta \sum_{\mathcal T_i \sim p(\mathcal T)} J_{\mathcal T_i}\big(\pi_{\theta_i'}\big) = \sum_{\mathcal T_i} J_{\mathcal T_i}\Big(\pi_{\theta + \alpha \nabla_\theta J_{\mathcal T_i}(\pi_\theta)}\Big)$$

Cập nhật meta-gradient (Gradient of Gradient - đạo hàm bậc 2 Hessian-vector product):
$$\theta \leftarrow \theta + \beta \sum_{\mathcal T_i} \Big(I + \alpha \nabla_\theta^2 J_{\mathcal T_i}(\theta)\Big) \nabla_{\theta_i'} J_{\mathcal T_i}(\theta_i')$$

---

## 5. MCTS, AlphaGo Zero & AlphaZero (Silver et al. - DeepMind)

AlphaGo Zero loại bỏ toàn bộ dữ liệu ván cờ của con người, tự chơi với chính mình từ con số không (Tabula Rasa) và đánh bại nhà vô địch thế giới.

```
       Chu trình Self-Play của AlphaGo Zero
   ┌──────────────────────────────────────────┐
   │ Trạng thái bàn cờ s                      │
   │    │                                     │
   │    ▼                                     │
   │ MCTS Search (Nhiều mô phỏng ngầm)       │
   │    │                                     │
   │    ▼                                     │
   │ Phân phối thăm dò π (Hành động cải thiện) │
   │    │                                     │
   │    ▼ Chọn nước đi a ~ π                  │
   │ Chơi đến hết ván → Kết quả z ∈ {+1, -1}  │
   │    │                                     │
   │    ▼                                     │
   │ Huấn luyện mạng nơ-ron: (p, v) = f_θ(s)  │
   │    - Policy head p khớp với π             │
   │    - Value head v khớp với z              │
   └──────────────────────────────────────────┘
```

### 5.1 Kiến trúc mạng tích hợp (Dual-Headed ResNet)
Một mạng tích chập sâu (ResNet 20 hoặc 40 blocks) nhận đầu vào là trạng thái bàn cờ $s$, dự đoán đồng thời hai đầu ra:
$$(\mathbf p, v) = f_\theta(s)$$
* $\mathbf p = \Pr(a \mid s)$: Vector xác suất ưu tiên cho các nước đi hợp lệ (Policy head).
* $v \in [-1, 1]$: Dự đoán xác suất người chơi hiện tại sẽ thắng (Value head).

---

### 5.2 Thuật toán MCTS có mạng nơ-ron dẫn hướng (PUCT)
Tại mỗi trạng thái $s$, thuật toán thực hiện hàng nghìn lượt mô phỏng cây (simulation). Mỗi cạnh $(s, a)$ lưu trữ:
$\{N(s, a): \text{số lần ghé thăm},\ W(s, a): \text{tổng giá trị},\ Q(s, a) = W/N,\ P(s, a): \text{tiền nghiệm từ mạng}\}$.

Bốn pha của MCTS:
1. **Selection (Chọn nhánh)**: Đi từ gốc xuống lá, tại mỗi nút chọn hành động theo công thức biến thể UCB (PUCT):
   $$\boxed{a_t = \arg\max_a \Big[ Q(s, a) + U(s, a) \Big] = \arg\max_a \Big[ Q(s, a) + c_{puct}\, P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \Big]}$$
   * Giai đoạn đầu ($N$ nhỏ): $U(s, a)$ chiếm ưu thế $\to$ ưu tiên nước đi mạng nơ-ron gợi ý $P(s, a)$.
   * Giai đoạn sau ($N$ lớn): $Q(s, a)$ chiếm ưu thế $\to$ chọn nước đi đã chứng minh bằng mô phỏng là tốt.

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế thuật toán Minimax Alpha-Beta Pruning và UCT truyền thống (dùng mô phỏng ngẫu nhiên rollout ngẫu nhiên).
> * **Khuyết điểm của phương pháp cũ**: Không gian trạng thái cờ vây quá khổng lồ ($10^{170}$ trạng thái, phân nhánh $b \approx 250$). Minimax chỉ nhìn được vài bước, còn rollout ngẫu nhiên trong UCT không có chiều sâu chiến thuật vì các nước đi ngẫu nhiên hoàn toàn phi lý.
> * **Logic thiết kế toán học**: Tích hợp mạng nơ-ron hai đầu (Policy $\mathbf p$ và Value $v$) vào cây MCTS. Thay thế việc mô phỏng ngẫu nhiên bằng giá trị dự đoán tức thì $v(s)$, và dùng xác suất tiên nghiệm $P(s, a)$ từ mạng policy để định hướng tìm kiếm PUCT.
> * **Cơ chế giải quyết**: MCTS đóng vai trò như một **bộ khuếch đại chính sách (Policy Improvement Operator)**: phân phối lượt thăm nom $N(s, a)$ sau khi tìm kiếm là một chính sách mạnh hơn nhiều so với policy gốc $\mathbf p(s)$. Mạng nơ-ron sau đó học lại chính phân phối MCTS này trong vòng lặp Self-play để đạt tới trình độ siêu phàm không cần tri thức con người.

2. **Expansion & Evaluation (Mở rộng & Đánh giá)**: Khi gặp nút lá chưa có trên cây, gọi mạng $f_\theta(s_{leaf}) \to (\mathbf p, v)$. Gán $P(s_{leaf}, \cdot) = \mathbf p$.
3. **Backup (Cập nhật ngược)**: Lan truyền giá trị $v$ ngược lên toàn bộ các nút cha dọc theo đường đã đi:
   $$N(s, a) \leftarrow N(s, a) + 1,\qquad W(s, a) \leftarrow W(s, a) + v,\qquad Q(s, a) \leftarrow \frac{W(s, a)}{N(s, a)}$$

---

### 5.3 Lấy mẫu hành động và Hàm mất mát Self-Play
Sau khi hoàn thành mô phỏng MCTS, phân phối hành động cải thiện $\mathbf \pi$ được tính từ tần suất ghé thăm của các nhánh:
$$\pi(a \mid s) = \frac{N(s, a)^{1/\tau}}{\sum_b N(s, b)^{1/\tau}}$$
(với $\tau = 1$ ở 30 nước đầu để tăng khám phá, sau đó $\tau \to 0$ để chọn nước đi có $N$ cao nhất).

Hàm mất mát toàn cục huấn luyện mạng $f_\theta$:
$$\boxed{\mathcal L(\theta) = (z - v)^2 - \mathbf \pi^\top \log \mathbf p + c \|\theta\|_2^2}$$
* $(z - v)^2$: Sai số MSE giữa giá trị dự đoán $v$ và kết quả thắng/thua thực tế của ván đấu $z \in \{+1, -1\}$.
* $-\mathbf \pi^\top \log \mathbf p$: Cross-entropy loss ép vector ưu tiên của mạng $\mathbf p$ tiến dần về chính sách đã được MCTS cải thiện $\mathbf \pi$.
* $c\|\theta\|^2$: L2 regularization chống overfitting.

---

## 6. OpenAI Five & Hệ thống RL phân tán quy mô lớn

OpenAI Five là hệ thống AI đầu tiên đánh bại các nhà vô địch thế giới bộ môn thể thao điện tử Dota 2 (vào tháng 4 năm 2019). Đây là môi trường cực kỳ khắc nghiệt: không gian quan sát liên tục ~20,000 chiều, không gian hành động phân cấp khổng lồ, thời gian chơi ~45 phút (≈90,000 frames) và thông tin không hoàn hảo.

### 6.1 Kiến trúc huấn luyện (Rapid Framework)
* **Quy mô tính toán**: Chạy PPO phân tán trên 256 GPU P100 và 128,000 nhân CPU.
* Mỗi ngày hệ thống tích luỹ được lượng kinh nghiệm tương đương **180 năm chơi game của con người**.
* Mô hình dùng mạng nơ-ron khổng lồ với lõi là một tầng LSTM 1024 đơn vị ẩn để ghi nhớ lịch sử dài hạn.

### 6.2 Kỹ thuật Team Spirit (Chia sẻ phần thưởng đồng đội)
Dota 2 là game phối hợp 5 đấu 5. Nếu mỗi agent chỉ tối ưu phần thưởng cá nhân $r_{ind}$, các người chơi sẽ tranh giành tài nguyên (farm lính).
Định nghĩa phần thưởng tổng hợp thông qua tham số $\tau \in [0, 1]$:
$$r_i(t) = (1 - \tau) r_{ind, i}(t) + \tau\, \bar{r}_{team}(t),\qquad \bar{r}_{team}(t) = \frac{1}{5}\sum_{j=1}^5 r_{ind, j}(t)$$
Trong suốt quá trình huấn luyện hàng tháng trời, $\tau$ được tăng dần từ $0$ (tập trung học kỹ năng cá nhân như di chuyển, ăn lính) lên $1$ (hoàn toàn hy sinh vì chiến thắng chung của đội).

### 6.3 Phẫu thuật mạng nơ-ron (Surgical Surgery)
Khi cập nhật phiên bản game hoặc thêm tính năng (thêm tướng, mở rộng item), kích thước vector quan sát và hành động bị thay đổi. Nếu huấn luyện lại từ đầu sẽ mất hàng tháng trời.
OpenAI phát triển kỹ thuật **phẫu thuật trọng số**:
* Mở rộng ma trận trọng số của lớp Linear: các hàng/cột mới được khởi tạo sao cho giá trị đầu ra ban đầu bằng 0, không ảnh hưởng đến các đặc trưng đã học.
* Cập nhật và tiếp tục huấn luyện mà không làm suy giảm hiệu năng hiện có.

---

## 7. Cầu nối hiện đại: Reinforcement Learning from Human Feedback (RLHF)

Thuật toán PPO (được học tại [Week 5](../Week5/BAI_GIANG.md)) chính là xương sống của công nghệ căn chỉnh mô hình ngôn ngữ lớn (LLM Alignment như InstructGPT, ChatGPT, Claude).

```
               Quy trình RLHF kinh điển
┌─────────────────────────────────────────────────────────┐
│ 1. Pretraining LLM (Mô hình ngôn ngữ ban đầu π_init)    │
│    │                                                    │
│ 2. Reward Modeling (Huấn luyện r_ψ từ phản hồi người)   │
│    │  min -E[log σ(r_ψ(x, y_win) - r_ψ(x, y_lose))]     │
│    │                                                    │
│ 3. PPO Fine-Tuning:                                     │
│    max E [ r_ψ(x, y) - β D_KL(π_θ(y|x) || π_init(y|x)) ]│
└─────────────────────────────────────────────────────────┘
```

Mục tiêu huấn luyện của giai đoạn 3 chính là biến thể **Maximum Entropy PPO**:
$$\max_\theta \mathbb E_{x \sim \mathcal D, y \sim \pi_\theta(\cdot \mid x)}\Big[r_\psi(x, y) - \beta\, D_{KL}\big(\pi_\theta(y \mid x) \parallel \pi_{init}(y \mid x)\big)\Big]$$

* **Giải thích ký hiệu**:
  * $x$: Câu lệnh gợi ý đầu vào (Prompt context).
  * $y$: Chuỗi token câu trả lời do mô hình sinh ra.
  * $r_\psi(x, y)$: Điểm số phần thưởng do mạng Reward Model chấm.
  * $\pi_{init}$: Mô hình ngôn ngữ gốc sau giai đoạn Supervised Fine-Tuning (SFT).
  * $\beta > 0$: Hệ số phạt phân kỳ KL chống "hack reward".

* **Chứng minh toán học: Nghiệm tối ưu giải tích đóng (Closed-Form Solution & Tiền đề DPO)**:
  Với mỗi prompt $x$ cố định, ta giải bài toán cực đại hoá hàm mục tiêu theo phân phối $\pi(y \mid x)$:
  $$\max_\pi \sum_y \pi(y \mid x) r_\psi(x, y) - \beta \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi_{init}(y \mid x)}$$
  Nhóm các số hạng theo thừa số chung $-\beta$:
  $$\begin{aligned}
  &= -\beta \sum_y \pi(y \mid x) \left[ \ln \frac{\pi(y \mid x)}{\pi_{init}(y \mid x)} - \frac{1}{\beta} r_\psi(x, y) \right] \\
  &= -\beta \sum_y \pi(y \mid x) \ln \left( \frac{\pi(y \mid x)}{\pi_{init}(y \mid x) \exp\big(\frac{1}{\beta} r_\psi(x, y)\big)} \right)
  \end{aligned}$$
  Định nghĩa hàm phân phối tối ưu chuẩn hoá $\pi^*(y \mid x)$ với hằng số chuẩn hoá $Z(x) = \sum_y \pi_{init}(y \mid x) \exp\big(\frac{1}{\beta} r_\psi(x, y)\big)$:
  $$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{init}(y \mid x) \exp\left( \frac{1}{\beta} r_\psi(x, y) \right)$$
  Thay vào biểu thức trên:
  $$\begin{aligned}
  &= -\beta \sum_y \pi(y \mid x) \ln \left( \frac{\pi(y \mid x)}{Z(x) \pi^*(y \mid x)} \right) \\
  &= -\beta \sum_y \pi(y \mid x) \ln \frac{\pi(y \mid x)}{\pi^*(y \mid x)} + \beta \ln Z(x) \sum_y \pi(y \mid x) \\
  &= \beta \ln Z(x) - \beta D_{KL}\Big( \pi(y \mid x) \parallel \pi^*(y \mid x) \Big)
  \end{aligned}$$
  Vì $D_{KL}(p \parallel q) \ge 0$ và đạt cực tiểu bằng $0$ khi và chỉ khi $p = q$, giá trị lớn nhất đạt được khi $\pi(y \mid x) \equiv \pi^*(y \mid x)$. Do đó nghiệm tối ưu đóng là:
  $$\boxed{\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{init}(y \mid x) \exp\left( \frac{1}{\beta} r_\psi(x, y) \right)}$$
  * **Ý nghĩa đột phá**: Công thức này chứng minh rằng policy tối ưu trong RLHF có dạng phân phối Boltzmann gia quyền trên mô hình nền $\pi_{init}$. Đảo ngược công thức này để biểu diễn $r(x, y)$ theo $\ln \frac{\pi^*(y \mid x)}{\pi_{init}(y \mid x)}$ chính là nền tảng ra đời của thuật toán **Direct Preference Optimization (DPO - Rafailov et al., 2023)**, cho phép căn chỉnh LLM trực tiếp mà không cần huấn luyện mạng Reward Model riêng hay chạy PPO! $\square$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế toàn bộ quy trình PPO-RLHF truyền thống gồm 4 mô hình (Actor $\pi_\theta$, Critic $V_\phi$, Reference Model $\pi_{ref}$, Reward Model $r_\psi$).
> * **Khuyết điểm của phương pháp cũ**: PPO-RLHF cực kỳ phức tạp và thiếu ổn định: phải huấn luyện riêng Reward Model, sau đó chạy vòng lặp tương tác Actor-Critic với hàng loạt siêu tham số nhạy cảm, dễ gặp hiện tượng Reward Hacking và sụp đổ phân phối từ ngữ.
> * **Logic thiết kế toán học**: Xuất phát từ nghiệm đóng giải tích của bài toán cực đại phần thưởng có ràng buộc KL:
>   $$r(x, y) = \beta \ln \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \ln Z(x)$$
>   Thay nghiệm đóng này trực tiếp vào mô hình xác suất Bradley-Terry lựa chọn của con người $p(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$. Hằng số phân vùng $Z(x)$ **triệt tiêu hoàn toàn**!
> * **Cơ chế giải quyết**: Biến toàn bộ bài toán Reinforcement Learning phức tạp thành bài toán **Hồi quy Logistic / Binary Cross-Entropy có giám sát** trực tiếp trên mô hình ngôn ngữ, loại bỏ hoàn toàn mạng Reward Model và mạng Critic, đơn giản hóa và tối ưu hóa vượt bậc quá trình căn chỉnh LLM.

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Chuỗi 3 phép thế biến RLHF thành DPO (Direct Preference Optimization):**
> - **Bước 1 (Cô lập hàm phần thưởng $r(x,y)$ từ nghiệm đóng $\pi^*(y\mid x)$):**
>   Từ phương trình nghiệm đóng vừa chứng minh ở trên:
>   $$\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{init}}(y\mid x)\exp\left(\frac{1}{\beta}r(x, y)\right) \iff \frac{\pi^*(y\mid x)}{\pi_{\text{init}}(y\mid x)} = \frac{1}{Z(x)}\exp\left(\frac{1}{\beta}r(x, y)\right)$$
>   Lấy logarit tự nhiên $\ln(\cdot)$ hai vế rồi nhân với $\beta$, ta rút được hàm phần thưởng $r(x,y)$ theo $\pi^*$:
>   $$\ln \frac{\pi^*(y\mid x)}{\pi_{\text{init}}(y\mid x)} = \frac{1}{\beta}r(x, y) - \ln Z(x) \implies \boxed{r(x, y) = \beta \ln \frac{\pi^*(y\mid x)}{\pi_{\text{init}}(y\mid x)} + \beta \ln Z(x)}$$
>   *(Khó khăn lớn nhất ở đây là tổng phân hoạch $Z(x) = \sum_{y \in \mathcal{V}^L} \pi_{\text{init}}(y\mid x)\exp(\frac{1}{\beta}r(x,y))$ chạy trên mọi câu trả lời khả dĩ của từ điển nên **không thể tính được**!)*
> - **Bước 2 (Viết hàm $r(x,y)$ cho câu trả lời thắng $y_w$ và câu trả lời thua $y_l$ trên cùng một prompt $x$):**
>   $$\begin{aligned}
>   (1)\;\; r(x, y_w) &= \beta \ln \frac{\pi^*(y_w\mid x)}{\pi_{\text{init}}(y_w\mid x)} + \beta \ln Z(x) \\
>   (2)\;\; r(x, y_l) &= \beta \ln \frac{\pi^*(y_l\mid x)}{\pi_{\text{init}}(y_l\mid x)} + \beta \ln Z(x)
>   \end{aligned}$$
> - **Bước 3 (Thế (1) và (2) vào hiệu số $r(x, y_w) - r(x, y_l)$ trong mô hình Bradley-Terry $P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$):**
>   Vì cả $y_w$ và $y_l$ đều trả lời cho **cùng một câu hỏi $x$**, nên số hạng $\beta \ln Z(x)$ ở (1) và (2) **giống hệt nhau và tự trừ nhau bằng $0$**:
>   $$r(x, y_w) - r(x, y_l) = \left(\beta \ln \frac{\pi^*(y_w\mid x)}{\pi_{\text{init}}(y_w\mid x)} + \cancel{\beta \ln Z(x)}\right) - \left(\beta \ln \frac{\pi^*(y_l\mid x)}{\pi_{\text{init}}(y_l\mid x)} + \cancel{\beta \ln Z(x)}\right) = \beta \ln \frac{\pi^*(y_w\mid x)}{\pi_{\text{init}}(y_w\mid x)} - \beta \ln \frac{\pi^*(y_l\mid x)}{\pi_{\text{init}}(y_l\mid x)}$$
>   Thế hiệu số sạch bóng $Z(x)$ này vào hàm Sigmoid $\sigma(\cdot)$, ta thu được ngay hàm mất mát **DPO Loss** chỉ phụ thuộc trực tiếp vào mạng ngôn ngữ $\pi_\theta$:
>   $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{init}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\ln \sigma\left(\beta \ln \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{init}}(y_w\mid x)} - \beta \ln \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{init}}(y_l\mid x)}\right)\right]$$

---

## 8. Phân tích chuyên sâu các tham số cốt lõi trong các thuật toán nâng cao

Các thuật toán tiên tiến nhất trong Deep RL được định hình bởi các tham số cân bằng giữa tính ngẫu nhiên, sự phối hợp đa tác tử, khả năng mở rộng quy mô và tính ổn định toán học:

| Thuật toán | Siêu tham số | Ký hiệu Toán | Vai trò & Biểu thức chứa tham số |
|---|---|---|---|
| **SAC** | Hệ số nhiệt độ entropy | $\alpha$ | Cân bằng Reward vs Entropy: $J(\pi) = \mathbb E[r + \alpha \mathcal H(\pi)]$ |
| **SAC** | Entropy mục tiêu | $\bar{\mathcal H}$ | Ngưỡng điều khiển Dual Gradient: $\nabla_\alpha J(\alpha) = -\mathbb E[\log \pi + \bar{\mathcal H}]$ |
| **SAC** | Hệ số cập nhật mềm | $\tau$ | Polyak target update: $\bar{\theta} \leftarrow (1-\tau)\bar{\theta} + \tau \theta$ |
| **ICM / RND** | Trọng số phần thưởng tò mò | $\beta$ | Tích hợp ngoại tại & nội tại: $r_t = r_t^{ext} + \beta r_t^{int}$ |
| **AlphaGo Zero** | Hệ số khám phá PUCT | $c_{puct}$ | Cân bằng Value vs Prior trong MCTS: $Q(s, a) + c_{puct} P(s, a) \frac{\sqrt{\sum N}}{1 + N(s, a)}$ |
| **AlphaGo Zero** | Nhiệt độ chọn nước đi | $\tau_{mcts}$ | Làm phẳng/nhọn phân phối ghé thăm: $\pi(a \mid s) \propto N(s, a)^{1/\tau_{mcts}}$ |
| **OpenAI Five** | Hệ số Team Spirit | $\tau_{team}$ | Cân bằng cá nhân & đồng đội: $r_i = (1-\tau_{team})r_{ind, i} + \tau_{team}\bar{r}_{team}$ |
| **RLHF** | Hệ số phạt KL | $\beta_{KL}$ | Chống Reward Hacking: $\max \mathbb E[r_\psi(x, y) - \beta_{KL} D_{KL}(\pi_\theta \parallel \pi_{init})]$ |

---

### 8.1 Nhóm Soft Actor-Critic (SAC)

#### 1. Hệ số nhiệt độ entropy $\alpha$ (Entropy Temperature)
$\alpha$ xuất hiện đồng thời trong 3 phương trình cốt lõi của SAC:
* **Hàm mục tiêu toàn cục**: $J(\pi) = \sum_t \mathbb E\big[r(s_t, a_t) + \alpha \mathcal H(\pi(\cdot \mid s_t))\big]$.
* **Mục tiêu Soft Bellman cho Critic**:
  $$y = r(s, a) + \gamma \Big(\min_{j=1,2} Q_{\bar{\theta}_j}(s', \tilde{a}') - \alpha \log \pi_\phi(\tilde{a}' \mid s')\Big),\qquad \tilde{a}' \sim \pi_\phi(\cdot \mid s')$$
* **Hàm mất mát của Actor**:
  $$J_\pi(\phi) = \mathbb E_{s \sim \mathcal D, \xi \sim \mathcal N}\Big[\alpha \log \pi_\phi\big(a_\phi(s, \xi) \mid s\big) - \min_{j=1,2} Q_{\theta_j}\big(s, a_\phi(s, \xi)\big)\Big]$$

* **Cơ chế tác động & Phân tích giới hạn**:
  * **Khi $\alpha \to 0$**: Thành phần $-\alpha \log \pi$ biến mất hoàn toàn. SAC suy biến thành thuật toán **TD3 (Twin Delayed DDPG)** với chính sách tất định. Policy sụp đổ về phân phối Dirac delta $\pi(a \mid s) \to \delta(a - a^*)$. Agent mất khả năng khám phá tự nhiên, dễ bị mắc kẹt vĩnh viễn ở các cực tiểu cục bộ (local optima) trong các môi trường có địa hình reward gồ ghề.
  * **Khi $\alpha \to \infty$**: Phần thưởng môi trường $r(s, a)$ bị áp đảo hoàn toàn. Policy trở thành phân phối đều ngẫu nhiên (Uniform Distribution) trên không gian hành động, biến agent thành kẻ lang thang vô định phớt lờ nhiệm vụ.
  * **Cơ chế tự động điều chỉnh $\alpha$ (Dual Gradient Descent)**:
    Do mỗi giai đoạn huấn luyện đòi hỏi mức độ khám phá khác nhau, SAC tối ưu hoá $\alpha$ qua bài toán đối ngẫu Lagrange với ràng buộc $\mathbb E[-\log \pi(a \mid s)] \ge \bar{\mathcal H}$:
    $$J(\alpha) = \mathbb E_{s \sim \mathcal D, a \sim \pi_\phi}\big[-\alpha \log \pi_\phi(a \mid s) - \alpha \bar{\mathcal H}\big] \implies \nabla_\alpha J(\alpha) = -\mathbb E_{a \sim \pi_\phi}\big[\log \pi_\phi(a \mid s) + \bar{\mathcal H}\big]$$
    - Khi policy quá tập trung (entropy thấp, $-\log \pi < \bar{\mathcal H}$): Đạo hàm $\nabla_\alpha J(\alpha) < 0 \implies$ bước gradient descent $-\eta_\alpha \nabla_\alpha$ sẽ **làm tăng $\alpha$**, kích thích policy bung rộng trở lại.
    - Khi policy quá phân tán (entropy cao, $-\log \pi > \bar{\mathcal H}$): Đạo hàm dương làm **giảm $\alpha$**, hướng agent tập trung vào khai thác phần thưởng.

#### 2. Ngưỡng Entropy mục tiêu $\bar{\mathcal H}$
Giá trị chuẩn mực được Haarnoja et al. đề xuất là:
$$\boxed{\bar{\mathcal H} = -\dim(\mathcal A)}$$

* **Nguồn gốc giải tích**:
  Xét phân phối chuẩn nhiều chiều $u \sim \mathcal N(\mu, \sigma^2 I)$ trên $\mathbb R^D$ (với $D = \dim(\mathcal A)$). Entropy vi phân là:
  $$\mathcal H(u) = \sum_{i=1}^D \frac{1}{2} \ln(2\pi e \sigma_i^2) = \frac{D}{2} \ln(2\pi e) + \sum_{i=1}^D \ln(\sigma_i)$$
  Để sau khi qua hàm bọc $a = \tanh(u) \in [-1, 1]^D$, hành động vẫn giữ được độ linh hoạt vừa phải (độ lệch chuẩn $\sigma_i \approx 1/e \approx 0.37$), ta có:
  $$\ln(\sigma_i) \approx -1 \implies \frac{1}{2}\ln(2\pi e) - 1 = \frac{1}{2}\ln(2\pi) - \frac{1}{2} \approx 0.919 - 0.5 = 0.419 \approx 0$$
  Khi đó $\mathcal H \approx -D = -\dim(\mathcal A)$.
  - Nếu đặt $\bar{\mathcal H} > 0$: Policy bị ép duy trì phương sai cực lớn, khiến hầu hết các hành động sau khi qua $\tanh$ đều bị đẩy dính chặt vào hai vách biên $\{-1, +1\}$, làm bão hòa gradient $\nabla_u \tanh(u) \to 0$.
  - Nếu đặt $\bar{\mathcal H} \ll -\dim(\mathcal A)$: Ràng buộc quá lỏng lẻo, $\alpha$ nhanh chóng tụt về $0$, triệt tiêu ưu thế của MaxEnt RL.

#### 3. Hệ số Polyak Target Update $\tau$ (`\tau = 0.005`)
Mạng target $Q_{\bar{\theta}}$ trong SAC không cập nhật gián đoạn theo chu kỳ $C$ bước như DQN, mà được làm trơn liên tục ở mọi bước huấn luyện:
$$\bar{\theta} \leftarrow (1 - \tau)\bar{\theta} + \tau \theta$$
* **Khi $\tau \to 1$**: Target network lập tức đồng nhất với critic hiện tại. Vòng lặp hồi tiếp dương giữa giá trị dự đoán và mục tiêu Bellman gây nổ phương sai và phân kỳ hàm giá trị.
* **Khi $\tau \to 0$ ($\tau = 10^{-5}$)**: Mạng target biến thiên quá chậm, thông tin phần thưởng TD lan truyền cực kỳ chậm chạp, kéo dài thời gian hội tụ gấp hàng chục lần.
* Mức $\tau = 0.005$ tương đương với việc target network là một bộ lọc thông thấp (exponential moving average) với "chu kỳ bán rã" khoảng $\approx \frac{1}{\tau} = 200$ bước cập nhật, bảo đảm đường mục tiêu trơn tru hoàn hảo.

---

### 8.2 Nhóm Khám phá nâng cao (Curiosity & RND)

#### Trọng số phần thưởng tò mò $\beta$ trong $r_t^{total} = r_t^{ext} + \beta\, r_t^{int}$
* **Khi $\beta \to 0$**: Agent quay về model-free RL truyền thống. Trong môi trường thưa thưởng (sparse reward như Montezuma's Revenge), tín hiệu gradient $\nabla_\theta J = 0$ ở hầu hết mọi quỹ đạo, quá trình học tê liệt hoàn toàn.
* **Khi $\beta \gg 1$**: Xảy ra hiện tượng **"Nghiện tò mò" (Curiosity trap / delusion)**. Agent bị cuốn theo việc khám phá các ngóc ngách lạ lẫm của bản đồ và phớt lờ hoàn toàn mục tiêu chính (nhặt chìa khóa, mở cửa, bảo toàn mạng sống).
* **Chuẩn hóa động (Running Normalization)**:
  Do sai số MSE dự đoán của mạng Predictor $\hat f_\theta(s)$ trong RND có biên độ thay đổi liên tục theo thời gian, nếu giữ nguyên $r_t^{int}$ thô, độ lớn của gradient nội tại sẽ lấn át gradient ngoại tại. Giải pháp bắt buộc là chia $r_t^{int}$ cho độ lệch chuẩn trôi $\hat \sigma_{int}$:
  $$\tilde r_t^{int} = \frac{r_t^{int}}{\hat \sigma_{int} + \epsilon}$$
  nhờ đó giữ cho tỷ số tín hiệu ngoại/nội luôn ổn định qua mọi giai đoạn huấn luyện.

---

### 8.3 Nhóm MCTS & AlphaGo Zero

#### 1. Hệ số thăm dò PUCT $c_{puct}$
Trong công thức chọn nhánh cây MCTS:
$$a_t = \arg\max_a \Big[ Q(s, a) + c_{puct}\, P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \Big]$$
* **Khi $c_{puct} \to 0$**: Nút cây được chọn thuần túy theo giá trị trung bình $Q(s, a)$ của các lần mô phỏng trước (Greedy Exploitation). Các nước cờ đột phá mà mạng nơ-ron gợi ý với xác suất tiền nghiệm $P(s, a)$ cao nhưng chưa được mô phỏng đủ số lần sẽ bị bỏ qua hoàn toàn.
* **Khi $c_{puct} \to \infty$**: Giá trị mô phỏng $Q(s, a)$ bị phớt lờ. Cây MCTS suy biến thành quá trình lấy mẫu thuần túy theo phân phối tiên nghiệm của mạng nơ-ron $P(s, a)$.
* **Khoảng tối ưu thực nghiệm**: $c_{puct} \in [1.0, 2.5]$. AlphaGo Zero dùng $c_{puct} = \approx 1.25$ (hoặc lên tới $2.5$ ở các phiên bản sau), cho phép mở rộng nhanh chóng các biến thế cờ mới ở giai đoạn đầu mô phỏng trước khi hội tụ về nhánh có $Q$ cao nhất.

#### 2. Nhiệt độ phân phối hành động $\tau_{mcts}$
$$\pi(a \mid s) = \frac{N(s, a)^{1/\tau_{mcts}}}{\sum_b N(s, b)^{1/\tau_{mcts}}}$$
* **Giai đoạn khai cuộc (30 nước đầu - Self-play Exploration)**: Đặt $\tau_{mcts} = 1.0$. Phân phối xác suất chọn nước đi $\pi(a \mid s)$ tỉ lệ thuận tuyến tính với số lần ghé thăm $N(s, a)$. Nhờ đó, kho ván cờ tự chơi có độ đa dạng cực lớn, không ván nào lặp lại ván nào.
* **Giai đoạn trung cuộc và tàn cuộc (Nước thứ 31 trở đi & Thi đấu cạnh tranh)**: Đặt $\tau_{mcts} \to 0$. Phân phối $\pi(a \mid s)$ trở thành hàm one-hot:
  $$\pi(a \mid s) = \begin{cases} 1 & \text{nếu } a = \arg\max_b N(s, b) \\ 0 & \text{ngược lại} \end{cases}$$
  Loại bỏ hoàn toàn tính ngẫu nhiên, bảo đảm agent luôn ra đòn tối ưu và chính xác tuyệt đối.

---

### 8.4 Nhóm Hệ thống Multi-Agent & OpenAI Five

#### Hệ số tinh thần đồng đội $\tau_{team} \in [0, 1]$ (Team Spirit Annealing)
$$r_i(t) = (1 - \tau_{team}) r_{ind, i}(t) + \tau_{team}\, \bar{r}_{team}(t),\qquad \bar{r}_{team}(t) = \frac{1}{5}\sum_{j=1}^5 r_{ind, j}(t)$$
* **Giai đoạn đầu ($\tau_{team} = 0$)**: Mỗi tác tử (tướng) chỉ quan tâm đến phần thưởng cá nhân $r_{ind, i}$ (tiêu diệt lính, nhận kinh nghiệm, không để chết). Nếu đặt $\tau_{team} = 1$ ngay từ đầu, xảy ra vấn đề **Credit Assignment đa tác tử**: tín hiệu thắng/thua của cả đội bị san phẳng cho 5 người chơi, gradient bị nhiễu loạn cực lớn khiến từng tướng không học nổi các thao tác vi mô cơ bản.
* **Lộ trình tăng dần (Curriculum Schedule)**: $\tau_{team}$ được tăng tuyến tính từ $0.0$ lên $1.0$ trong suốt hàng ngàn giờ huấn luyện. Khi các kỹ năng cá nhân đã thuần thục, $\tau_{team} \to 1.0$ ép toàn bộ 5 người chơi phải biết hy sinh tài nguyên (support nhường tiền cho carry, che chắn đòn đánh để đồng đội rút lui), đưa lối chơi của tập thể lên tầm vô địch thế giới.

---

### 8.5 Nhóm Căn chỉnh Ngôn ngữ lớn (RLHF)

#### Hệ số phạt phân kỳ $\beta_{KL}$ trong tối ưu hóa PPO cho LLM
$$\max_\theta \mathbb E_{x \sim \mathcal D, y \sim \pi_\theta}\Big[r_\psi(x, y) - \beta_{KL}\, D_{KL}\big(\pi_\theta(y \mid x) \parallel \pi_{init}(y \mid x)\big)\Big]$$
* **Khi $\beta_{KL} \to 0$**: Hiện tượng **"Reward Hacking" / "Goodhart's Law"**. LLM phát hiện ra các lỗ hổng của mạng Reward Model $r_\psi$ và sinh ra các chuỗi văn bản kỳ dị (ví dụ: lặp đi lặp lại những cụm từ nịnh bợ, chèn hàng loạt từ khóa ưa thích của RM) để giành điểm số cực cao, làm phá hủy hoàn toàn khả năng ngôn ngữ tự nhiên.
* **Khi $\beta_{KL} \gg 1$**: Khoảng cách KL bị siết quá chặt, $\pi_\theta$ bị giam lỏng tại mô hình nền ban đầu $\pi_{init}$, khiến LLM không tiếp thu được bất kỳ định hướng an toàn hay phong cách trả lời nào từ dữ liệu con người.
* **Bộ điều khiển thích nghi $\beta_{KL}$ (Adaptive Controller)**:
  Các hệ thống thực tế (như InstructGPT) điều chỉnh $\beta_{KL}$ động sau mỗi batch:
  $$\beta_{new} = \beta_{old} \cdot \exp\Big(K_p \cdot \big(D_{KL}^{batch} - D_{KL}^{target}\big)\Big)$$
  giữ cho độ chệch phân phối của mô hình luôn dao động ổn định quanh mức mong muốn $D_{KL}^{target} \approx 6.0\text{ nats}$.

---

## 9. Đề xuất đồ án Capstone (Final Project Ideas)

Dưới đây là 4 hướng đồ án kinh điển để bạn hiện thực hoá và đưa vào portfolio chuyên môn:

| Hướng đồ án | Mô tả bài toán | Thuật toán đề xuất | Thách thức chính |
|---|---|---|---|
| **1. Multi-Agent Pommerman** | Game đặt bom đối kháng 4 người trên lưới | PPO + Centralized Critic / Self-Play | Quan sát một phần, phối hợp đồng đội, tránh tự sát |
| **2. Continuous Locomotion** | Điều khiển robot Mujoco Ant / Humanoid vượt chướng ngại vật | SAC (Soft Actor-Critic) hoặc PPO | Không gian liên tục số chiều cao, nhạy cảm với khởi tạo |
| **3. World Models & Latent RL** | Game đua xe CarRacing-v0 học từ ảnh thô | VAE (Vision) + MDN-RNN (Memory) + Linear Controller | Học biểu diễn tiềm ẩn, "mơ" trong tiềm thức để lập kế hoạch |
| **4. Mini-RLHF cho LLM** | Căn chỉnh mô hình ngôn ngữ nhỏ (GPT-2) | HuggingFace TRL (PPO + LoRA) | Reward hacking, cân bằng tham số phạt KL |

---

## 10. Tổng kết toàn khóa: Bản đồ tư duy 8 Tuần

```
                                  [RL ROADMAP]
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
     [Model-Free RL]                                       [Model-Based RL]
            │                                                     │
    ┌───────┴───────────────┐                                     ▼
    ▼                       ▼                            [Week 7: MB-MF]
[Value-Based]        [Policy-Based]                     (Học động học s' & r,
    │                       │                            MPC Random Shooting)
    ├─ Week 2: Tabular      ├─ Week 4: PG & A2C                   │
    │  (Q-Learning, SARSA)  │  (REINFORCE, Baseline,              ▼
    │                       │   Advantage)               [Week 8: Advanced]
    └─ Week 3: Deep Q       │                            - AlphaGo Zero (MCTS)
       (DQN, Double,        ├─ Week 5: PPO & TRPO        - SAC (MaxEnt RL)
        Dueling, N-Step)    │  (Clipped objective, GAE)  - GAIL & IRL
                            │                            - RND (Curiosity)
                            └─ Week 6: Black-Box         - OpenAI Five & RLHF
                               (Evolution Strategies)
```

**Chúc mừng bạn đã hoàn thành trọn vẹn lý thuyết và phân tích mã nguồn của toàn bộ lộ trình Học tăng cường sâu!**
Mọi nền tảng toán học, trực giác logic và kinh nghiệm thực chiến phân tích bug từ Week 1 đến Week 8 đã sẵn sàng để bạn tự tin xây dựng và tinh chỉnh các hệ thống AI tăng cường tự chủ.
