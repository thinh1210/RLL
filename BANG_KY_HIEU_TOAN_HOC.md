# BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC TRONG HỌC TĂNG CƯỜNG
## (Standard Mathematical Notation Reference for Reinforcement Learning)

> **Mục đích tài liệu**: Chuẩn hoá toàn bộ ký hiệu, biến số, toán tử và không gian toán học xuất hiện trong toàn bộ 8 tuần học tại repository [`Reinforcement-Learning`](./README.md).  
> Khi đọc bất kỳ công thức nào trong các bài giảng tuần ([Week 1](./Week1/BAI_GIANG.md) đến [Week 8](./Week8/BAI_GIANG.md)) hoặc [Cẩm nang tham số](./CAM_NANG_THAM_SO_RL.md), hãy đối chiếu với bảng ký hiệu này để nắm chính xác bản chất không gian toán học và ý nghĩa vật lý/giải tích của từng thành phần.

---

## MỤC LỤC
1. [Không gian, Tập hợp & Phần tử cơ bản](#1-không-gian-tập-hợp--phần-tử-cơ-bản)
2. [Xác suất, Phân phối & Kỳ vọng](#2-xác-suất-phân-phối--kỳ-vọng)
3. [Phần thưởng, Chiết khấu & Lợi nhuận tích luỹ](#3-phần-thưởng-chiết-khấu--lợi-nhuận-tích-luỹ)
4. [Hàm giá trị & Hàm Advantage](#4-hàm-giá-trị--hàm-advantage)
5. [Toán tử Bellman & Phép biến đổi](#5-toán-tử-bellman--phép-biến-đổi)
6. [Tối ưu hoá, Gradient & Ma trận giải tích](#6-tối-ưu-hoá-gradient--ma-trận-giải-tích)
7. [Lý thuyết thông tin, Entropy & Độ phân kỳ](#7-lý-thuyết-thông-tin-entropy--độ-phân-kỳ)
8. [Ký hiệu chuyên biệt theo từng họ thuật toán (Week 1 – 8)](#8-ký-hiệu-chuyên-biệt-theo-từng-họ-thuật-toán-week-1--8)

---

## 1. Không gian, Tập hợp & Phần tử cơ bản

| Ký hiệu | Tên gọi chuẩn | Bản chất toán học | Ý nghĩa trong RL |
|---|---|---|---|
| $\mathcal S$ | State Space (Không gian trạng thái) | Tập rời rạc hữu hạn hoặc đa tạp liên tục $\mathbb R^{d_s}$ | Tập hợp tất cả các trạng thái có thể có của môi trường |
| $s, s_t \in \mathcal S$ | State (Trạng thái) | Điểm trong không gian $\mathcal S$ hoặc vector $s_t \in \mathbb R^{d_s}$ | Tình trạng cụ thể của môi trường tại bước thời gian $t$ |
| $\mathcal A$ | Action Space (Không gian hành động) | Tập hữu hạn $\{a_1, \dots, a_K\}$ hoặc siêu khối liên tục $[-1, 1]^{d_a}$ | Tập hợp các hành động khả dĩ mà tác tử (agent) có thể thực thi |
| $a, a_t \in \mathcal A$ | Action (Hành động) | Phần tử rời rạc hoặc vector điều khiển $a_t \in \mathbb R^{d_a}$ | Quyết định cụ thể do agent đưa ra tại thời điểm $t$ |
| $\tau$ | Trajectory / Episode (Quỹ đạo) | Chuỗi ngẫu nhiên $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ | Lịch sử tương tác hoàn chỉnh từ đầu đến khi kết thúc ván đấu |
| $T$ | Episode Horizon (Độ dài quỹ đạo) | Số nguyên dương $T \in \mathbb N^*$ hoặc $\infty$ | Số bước thời gian tối đa của một tập hợp thể nghiệm (episode) |
| $t$ | Time step (Bước thời gian) | Chỉ số thời gian rời rạc $t \in \{0, 1, 2, \dots\}$ | Thời điểm tương tác hiện tại |
| $\mathcal D$ | Dataset / Replay Buffer | Tập hợp các bộ chuyển tiếp $\{(s_i, a_i, r_i, s'_i, d_i)\}$ | Kho lưu trữ kinh nghiệm quá khứ dùng cho huấn luyện off-policy |
| $d_s, d_a$ | State / Action Dimension | Số nguyên dương $d_s, d_a \in \mathbb N^*$ | Số chiều của vector quan sát trạng thái và vector hành động |

---

## 2. Xác suất, Phân phối & Kỳ vọng

| Ký hiệu | Tên gọi chuẩn | Bản chất toán học | Ý nghĩa trong RL |
|---|---|---|---|
| $P(s' \mid s, a)$ | Transition Dynamics (Động học môi trường) | Hàm mật độ xác suất điều kiện $\mathcal S \times \mathcal A \to \Delta(\mathcal S)$ | Xác suất môi trường chuyển sang $s'$ khi ở $s$ và nhận hành động $a$ |
| $\pi(a \mid s)$ | Policy (Chính sách) | Phân phối xác suất điều kiện $\pi: \mathcal S \to \Delta(\mathcal A)$ | Chiến lược ra quyết định của agent: xác suất chọn hành động $a$ tại trạng thái $s$ |
| $\pi_\theta(a \mid s)$ | Parameterized Policy | Hàm khả vi tham số hoá bởi vector trọng số $\theta \in \mathbb R^d$ | Chính sách biểu diễn bằng mạng nơ-ron với trọng số $\theta$ |
| $\mathbb P(\tau \mid \pi)$ | Trajectory Distribution | Mật độ xác suất đồng thời trên không gian quỹ đạo | $\mathbb P(\tau \mid \pi) = p(s_0) \prod_{t=0}^{T-1} \pi(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$ |
| $\mathbb E_{x \sim p}[\cdot]$ | Mathematical Expectation (Kỳ vọng toán học) | Tích phân $\int p(x) f(x) dx$ hoặc tổng $\sum p(x) f(x)$ | Giá trị trung bình kỳ vọng của đại lượng ngẫu nhiên |
| $\operatorname{Var}_{x \sim p}[\cdot]$ | Variance (Phương sai) | $\mathbb E[(X - \mathbb E[X])^2]$ | Mức độ dao động/bất định của biến ngẫu nhiên quanh giá trị trung bình |
| $\rho_\pi(s)$ hoặc $d^\pi(s)$ | State Visitation Distribution | Phân phối dừng hoặc phân phối chiết khấu ghé thăm trạng thái | Tần suất xuất hiện của trạng thái $s$ khi agent tuân theo chính sách $\pi$ |
| $\sim$ | Sampled from (Được lấy mẫu từ) | Quan hệ phân phối ngẫu nhiên $x \sim p(x)$ | Đại lượng $x$ được sinh ngẫu nhiên theo luật phân phối xác suất $p$ |

---

## 3. Phần thưởng, Chiết khấu & Lợi nhuận tích luỹ

| Ký hiệu | Tên gọi chuẩn | Bản chất toán học | Ý nghĩa trong RL |
|---|---|---|---|
| $R(s, a)$ hoặc $r(s, a)$ | Reward Function (Hàm phần thưởng) | Hàm vô hướng $\mathcal S \times \mathcal A \to \mathbb R$ | Điểm thưởng tức thời môi trường phản hồi sau khi nhận cặp $(s, a)$ |
| $R_{t+1}$ hoặc $r_t$ | Instantaneous Reward | Biến ngẫu nhiên vô hướng $r_t \in \mathbb R$ | Giá trị phần thưởng cụ thể nhận được tại bước thời gian $t$ |
| $\gamma$ | Discount Factor (Hệ số chiết khấu) | Tham số thực $\gamma \in [0, 1)$ | Trọng số giảm dần theo thời gian, đo lường mức độ ưu tiên tương lai |
| $G_t$ hoặc $R(\tau)$ | Cumulative Discounted Return | Biến ngẫu nhiên $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$ | Tổng điểm thưởng chiết khấu tích luỹ từ bước $t$ tới cuối ván đấu |
| $H_{\text{eff}} = \frac{1}{1-\gamma}$ | Effective Horizon | Số thực dương $H_{\text{eff}} \in [1, \infty)$ | Tầm nhìn hiệu dụng tương lai của agent trước khi phần thưởng bị triệt tiêu |
| $J(\pi)$ hoặc $J(\theta)$ | Policy Objective (Mục tiêu tối ưu chính sách) | Hàm mục tiêu vô hướng $J(\theta) = \mathbb E_{\tau \sim \pi_\theta}[G_0]$ | Điểm số kỳ vọng của toàn bộ quỹ đạo mà thuật toán RL cần cực đại hoá |

---

## 4. Hàm giá trị & Hàm Advantage

| Ký hiệu | Tên gọi chuẩn | Bản chất toán học | Ý nghĩa trong RL |
|---|---|---|---|
| $V^\pi(s)$ | State-Value Function | $V^\pi(s) = \mathbb E_\pi \big[ G_t \mid S_t = s \big]$ | Giá trị kỳ vọng khi bắt đầu từ trạng thái $s$ và tuân thủ chính sách $\pi$ |
| $Q^\pi(s, a)$ | Action-Value Function | $Q^\pi(s, a) = \mathbb E_\pi \big[ G_t \mid S_t = s, A_t = a \big]$ | Giá trị kỳ vọng khi từ $s$, thực hiện hành động $a$, sau đó tuân theo $\pi$ |
| $V^*(s), Q^*(s, a)$ | Optimal Value Functions | $V^*(s) = \max_\pi V^\pi(s),\ Q^*(s, a) = \max_\pi Q^\pi(s, a)$ | Giá trị tối đa có thể đạt được bởi bất kỳ chính sách khả dĩ nào |
| $A^\pi(s, a)$ | Advantage Function | $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ | Mức độ vượt trội (lợi thế) của hành động $a$ so với hành động trung bình tại $s$ |
| $\delta_t$ hoặc $\delta_t^V$ | TD Error (Sai số Temporal Difference) | $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ | Độ lệch giữa ước lượng mới (1-step bootstrap) và ước lượng hiện tại của giá trị |
| $\hat{A}_t$ | Advantage Estimator | Ước lượng thực nghiệm của $A(s_t, a_t)$ | Vector lợi thế tính bằng GAE hoặc n-step bootstrap dùng để nhân với Policy Gradient |

---

## 5. Toán tử Bellman & Phép biến đổi

| Ký hiệu | Tên gọi chuẩn | Bản chất toán học | Ý nghĩa trong RL |
|---|---|---|---|
| $T^\pi$ | Bellman Expectation Operator | Toán tử tuyến tính $(T^\pi V)(s) = R^\pi(s) + \gamma \sum_{s'} P^\pi(s' \mid s) V(s')$ | Ánh xạ cập nhật hàm giá trị theo chính sách $\pi$ đã cho |
| $T^*$ | Bellman Optimality Operator | Toán tử phi tuyến $(T^* Q)(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')$ | Ánh xạ co Bellman tối ưu, hội tụ về $Q^*$ theo định lý Banach |
| $\|\cdot\|_\infty$ | Chebyshev / Infinity Norm | $\|f\|_\infty = \max_{x} |f(x)|$ | Chuẩn cực đại trên không gian hàm, dùng để chứng minh tính chất co chặt |
| $\|\cdot\|_2$ | Euclidean / $L_2$ Norm | $\|x\|_2 = \sqrt{\sum_i x_i^2}$ | Chuẩn Euclid đo độ dài vector |
| $\|\cdot\|_F$ | Frobenius Norm | $\|A\|_F = \sqrt{\operatorname{Tr}(A^\top A)}$ | Chuẩn ma trận đo độ lớn của ma trận Jacobian hoặc trọng số mạng nơ-ron |

---

## 6. Tối ưu hoá, Gradient & Ma trận giải tích

| Ký hiệu | Tên gọi chuẩn | Bản chất toán học | Ý nghĩa trong RL |
|---|---|---|---|
| $\nabla_\theta$ | Gradient Operator | Vector đạo hàm riêng $\nabla_\theta f = \left[ \frac{\partial f}{\partial \theta_1}, \dots, \frac{\partial f}{\partial \theta_d} \right]^\top$ | Hướng tăng nhanh nhất của hàm số theo không gian tham số $\theta$ |
| $\nabla_\theta^2$ | Hessian Matrix | Ma trận đối xứng $d \times d$ đạo hàm riêng bậc hai | Đo độ cong cục bộ của bề mặt hàm mất mát |
| $\mathcal F(\theta)$ | Fisher Information Matrix | $\mathcal F = \mathbb E_{s, a} \big[ \nabla_\theta \log \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)^\top \big]$ | Ma trận Riemann đo độ cong của không gian phân phối xác suất trong Natural PG / TRPO |
| $\alpha$ hoặc $\eta$ | Learning Rate (Tốc độ học) | Số thực dương nhỏ $\alpha \in (0, 1)$ | Kích thước bước nhảy cập nhật tham số $\theta \leftarrow \theta + \alpha \nabla_\theta J$ |
| $\operatorname{clip}(x, a, b)$ | Clipping Function | $\min(\max(x, a), b)$ | Hàm cắt xén giá trị của $x$ vào đoạn $[a, b]$ |
| $\operatorname{Tr}(A)$ | Trace (Vết của ma trận) | Tổng các phần tử trên đường chéo chính $\sum_i A_{ii}$ | Xuất hiện khi tính kỳ vọng dạng toàn phương và chuẩn ma trận |

---

## 7. Lý thuyết thông tin, Entropy & Độ phân kỳ

| Ký hiệu | Tên gọi chuẩn | Bản chất toán học | Ý nghĩa trong RL |
|---|---|---|---|
| $\mathcal H(p)$ | Shannon Entropy | $\mathcal H(p) = -\mathbb E_{x \sim p}[\ln p(x)] = -\sum_x p(x) \ln p(x)$ | Độ hỗn loạn, tính ngẫu nhiên hay độ đa dạng của phân phối $p$ |
| $D_{KL}(p \parallel q)$ | Kullback-Leibler Divergence | $D_{KL}(p \parallel q) = \mathbb E_{x \sim p}\left[ \ln \frac{p(x)}{q(x)} \right] \ge 0$ | Độ phân kỳ thông tin đo khoảng cách định hướng giữa hai phân phối $p$ và $q$ |
| $D_{TV}(p, q)$ | Total Variation Distance | $D_{TV}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)| = \sup_A |p(A) - q(A)|$ | Khoảng cách biến thiên toàn phần giữa hai phân phối xác suất ($D_{TV} \le \sqrt{\frac{1}{2} D_{KL}}$) |
| $\beta$ hoặc $\alpha_{ent}$ | Entropy Coefficient | Số thực dương nhỏ | Trọng số khuyến khích tính ngẫu nhiên trong Policy Gradient / SAC |

---

## 8. Ký hiệu chuyên biệt theo từng họ thuật toán (Week 1 – 8)

### 8.1 Multi-Armed Bandits & Thăm dò (Week 1)
* $N_t(a)$: Số lần cần gạt $a$ đã được kéo tính đến thời điểm $t$.
* $\hat{Q}_t(a)$: Trung bình mẫu phần thưởng của cần gạt $a$: $\hat{Q}_t(a) = \frac{1}{N_t(a)} \sum_{i: a_i=a} R_i$.
* $c$: Hệ số tin cậy trong UCB1 ($c = \sqrt{2} \approx 1.414$ theo bất đẳng thức Hoeffding).
* $\epsilon$: Xác suất chọn hành động ngẫu nhiên đều trong chiến lược $\epsilon$-greedy.

### 8.2 Q-Learning & DQN (Week 2, 3)
* $\theta$: Vector trọng số của mạng nơ-ron Q-Network chính (Online Network).
* $\theta^-$: Vector trọng số của mạng mục tiêu (Target Network).
* $C$: Chu kỳ cập nhật cố định (Hard Update Interval) sao chép $\theta \to \theta^-$ sau mỗi $C$ bước.
* $\tau$: Hệ số cập nhật mềm Polyak (Soft Target Update): $\theta^- \leftarrow (1-\tau)\theta^- + \tau \theta$.
* $y_t^{DQN}$: Mục tiêu Bellman scalar: $y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$.
* $y_t^{DDQN}$: Mục tiêu Double DQN chống thổi phồng: $y_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta); \theta^-)$.

### 8.3 Policy Gradients & A2C (Week 4)
* $b(s)$: Hàm đường cơ sở phụ thuộc trạng thái (State-dependent Baseline), thường là $V(s)$.
* $\phi$: Vector trọng số của mạng Critic xấp xỉ hàm giá trị $V_\phi(s)$.
* $g_{\max}$: Ngưỡng cắt chuẩn độ dốc (Gradient Clipping Threshold) chống nổ gradient.
* $N_{envs}$: Số môi trường độc lập chạy song song đồng thời thu thập kinh nghiệm.

### 8.4 PPO & GAE (Week 5)
* $r_t(\theta)$: Tỷ số xác suất hành động (Probability Ratio): $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$.
* $\epsilon_{clip}$: Biên độ cắt xén của hàm mục tiêu kẹp surrogate (thường $\epsilon_{clip} = 0.2$).
* $\lambda$: Hệ số suy giảm cấp số nhân trong Generalized Advantage Estimation (GAE).
* $\hat{A}_t^{GAE}$: Lợi thế tổng quát tính theo chuỗi lặp: $\hat{A}_t^{GAE} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$.
* $K$: Số epoch huấn luyện lặp lại trên cùng một tập đệm rollout buffer PPO.

### 8.5 Evolution Strategies (Week 6)
* $\epsilon_i \sim \mathcal N(0, I)$: Vector nhiễu Gauss độc lập chuẩn hoá gán vào tham số chính sách.
* $\sigma$: Độ lệch chuẩn nhiễu tham số (Perturbation scale).
* $n$: Kích thước quần thể (Population size / Number of perturbed candidates).
* $d$: Tổng số tham số (chiều không gian trọng số) của mạng nơ-ron.
* $F(\theta)$: Hàm độ thích nghi (Fitness / Total trajectory reward của cá thể).

### 8.6 Model-Based RL & MPC (Week 7)
* $\hat{f}_\phi(s, a)$: Mạng nơ-ron dự đoán động học chuyển tiếp: $\hat{s}' = s + \hat{f}_\phi(s, a)$.
* $\hat{r}_\psi(s, a)$: Mạng nơ-ron dự đoán hàm phần thưởng vô hướng.
* $H$: Tầm nhìn lập kế hoạch phía trước (Planning Horizon) trong thuật toán MPC.
* $K$: Số chuỗi hành động ngẫu nhiên được lấy mẫu đồng thời trong Random Shooting.
* $M$: Số vòng lặp thu thập dữ liệu và tái huấn luyện mô hình (DAgger Data Aggregation rounds).
* $L_f$: Hằng số Lipschitz của hàm động học: $\|\nabla_s f\| \le L_f$.

### 8.7 Advanced SOTA Algorithms (Week 8)
* $\alpha_{SAC}$: Nhiệt độ Entropy trong Soft Actor-Critic cân bằng phần thưởng và độ ngẫu nhiên.
* $\bar{\mathcal H}$: Entropy mục tiêu trong bài toán quy hoạch đối ngẫu SAC (thường chọn $-\dim(\mathcal A)$).
* $c_{puct}$: Hệ số cân bằng giữa giá trị mô phỏng và tiên nghiệm mạng trong thuật toán PUCT (AlphaGo Zero).
* $\tau_{mcts}$: Nhiệt độ làm mềm phân phối chọn nước đi từ tần suất ghé thăm nhánh cây $N(s, a)^{1/\tau_{mcts}}$.
* $\tau_{team}$: Hệ số tinh thần đồng đội chuyển hoá phần thưởng cá nhân sang cả đội trong OpenAI Five.
* $\beta_{KL}$: Trọng số phạt phân kỳ KL giữa policy mới và policy nền ban đầu trong PPO-RLHF cho LLM.

---
*Bảng ký hiệu này được tham chiếu chéo tự động bởi tất cả các tệp bài giảng trong khoá học.*
