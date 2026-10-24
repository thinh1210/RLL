# CẨM NANG TOÀN DIỆN VỀ THAM SỐ TRONG HỌC TĂNG CƯỜNG
## (Reinforcement Learning Hyperparameter & Parameter Handbook)

> **Tài liệu tham chiếu chuẩn mực**: Tổng hợp và giải mã toàn bộ tham số, siêu tham số, hằng số toán học xuất hiện trong toàn bộ 8 tuần học tại repository [`Reinforcement-Learning`](./README.md).  
> 📐 **Quy chuẩn ký hiệu**: Đối chiếu định nghĩa chi tiết của mọi biến số, toán tử và không gian tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](./BANG_KY_HIEU_TOAN_HOC.md).  
> Tài liệu phân tích rõ: **Bản chất toán học**, **Phương trình chứa tham số**, **Cơ chế tác động giải tích**, **Hành vi tại các giới hạn biên ($\to 0, \to 1, \to \infty$)**, **Ma trận độ nhạy**, và **Cẩm nang chẩn đoán lỗi thực chiến (Debugging Guide)**.

---

## MỤC LỤC
1. [Bảng tra cứu tham số tổng thể (Master Parameter Index)](#1-bảng-tra-cứu-tham-số-tổng-thể-master-parameter-index)
2. [Bản đồ tiến hoá thuật toán & Logic thiết kế các hàm toán học (Algorithmic Evolution & Mathematical Design Logic)](#2-bản-đồ-tiến-hoá-thuật-toán--logic-thiết-kế-các-hàm-toán-học-algorithmic-evolution--mathematical-design-logic)
   - [§2.1 Week 1: Từ Trung bình mẫu tới UCB1 & Policy Gradient](#21-week-1-từ-trung-bình-mẫu-tới-ucb1--policy-gradient)
   - [§2.2 Week 2: Từ Quy hoạch động tới Monte Carlo, TD(0) & Q-Learning](#22-week-2-từ-quy-hoạch-động-tới-monte-carlo-td0--q-learning)
   - [§2.3 Week 3: Sự tiến hóa của Deep Q-Networks (DQN, Double, Dueling, N-step, Noisy)](#23-week-3-sự-tiến-hóa-của-deep-q-networks-dqn-double-dueling-n-step-noisy)
   - [§2.4 Week 4: Từ Q-learning tới Policy Gradient, Reward-to-Go, Baseline & Actor-Critic](#24-week-4-từ-q-learning-tới-policy-gradient-reward-to-go-baseline--actor-critic)
   - [§2.5 Week 5: Từ Vanilla PG tới TRPO, PPO-Clip & GAE](#25-week-5-từ-vanilla-pg-tới-trpo-ppo-clip--gae)
   - [§2.6 Week 6: Từ Lan truyền ngược MDP tới Evolution Strategies & Mirrored Sampling](#26-week-6-từ-lan-truyền-ngược-mdp-tới-evolution-strategies--mirrored-sampling)
   - [§2.7 Week 7: Từ Model-Free tới Model-Based RL, Residual Dynamics, MPC & DAgger](#27-week-7-từ-model-free-tới-model-based-rl-residual-dynamics-mpc--dagger)
   - [§2.8 Week 8: Các bước nhảy vọt hiện đại (MaxEnt SAC, RND Curiosity, AlphaGo PUCT, DPO)](#28-week-8-các-bước-nhảy-vọt-hiện-đại-maxent-sac-rnd-curiosity-alphago-puct-dpo)
3. [Giải mã tham số theo từng tuần & Phương trình chi tiết](#3-giải-mã-tham-số-theo-từng-tuần--phương-trình-chi-tiết)
   - [§3.1 Week 1: Multi-Armed Bandits & Thăm dò cổ điển](#31-week-1-multi-armed-bandits--thăm-dò-cổ-điển)
   - [§3.2 Week 2: Tabular MDP, Bellman Operators & TD Learning](#32-week-2-tabular-mdp-bellman-operators--td-learning)
   - [§3.3 Week 3: Deep Q-Networks (DQN) & Value-Based Deep RL](#33-week-3-deep-q-networks-dqn--value-based-deep-rl)
   - [§3.4 Week 4: Policy Gradient, REINFORCE & Advantage Actor-Critic (A2C)](#34-week-4-policy-gradient-reinforce--advantage-actor-critic-a2c)
   - [§3.5 Week 5: Proximal Policy Optimization (PPO) & GAE](#35-week-5-proximal-policy-optimization-ppo--gae)
   - [§3.6 Week 6: Black-Box Optimization & Evolution Strategies (ES)](#36-week-6-black-box-optimization--evolution-strategies-es)
   - [§3.7 Week 7: Model-Based RL, Model Predictive Control (MPC) & DAgger](#37-week-7-model-based-rl-model-predictive-control-mpc--dagger)
   - [§3.8 Week 8: Advanced SOTA (SAC, Curiosity/RND, AlphaGo Zero, OpenAI Five, RLHF)](#38-week-8-advanced-sota-sac-curiosityrnd-alphago-zero-openai-five-rlhf)
4. [Ma trận độ nhạy & Các cặp đánh đổi toán học (Trade-off Matrix)](#4-ma-trận-độ-nhạy--các-cặp-đánh-đổi-toán-học-trade-off-matrix)
5. [Cẩm nang chẩn đoán lỗi lâm sàng trong Deep RL (Debugging Guide by Symptoms)](#5-cẩm-nang-chẩn-đoán-lỗi-lâm-sàng-trong-deep-rl-debugging-guide-by-symptoms)
6. [Quy tắc vàng thực chiến khi tinh chỉnh (Rules of Thumb)](#6-quy-tắc-vàng-thực-chiến-khi-tinh-chỉnh-rules-of-thumb)

---

## 1. Bảng tra cứu tham số tổng thể (Master Parameter Index)

| Ký hiệu | Tên tham số trong Code | Xuất hiện tại | Khoảng giá trị chuẩn | Vai trò chính |
|---|---|---|---|---|
| $\gamma$ | `GAMMA`, `gamma` | Week 1–5, 7, 8 | $[0.95, 0.999]$ | Hệ số chiết khấu tương lai (Discount factor) |
| $\alpha$ | `LR`, `learning_rate` | Week 1–8 | $[10^{-5}, 10^{-2}]$ | Tốc độ học của thuật toán tối ưu (Optimizer step size) |
| $\epsilon$ | `EPSILON`, `eps` | Week 1–3 | $1.0 \to 0.01$ | Xác suất hành động ngẫu nhiên ($\epsilon$-greedy exploration) |
| $c$ | `c` | Week 1 | $\sqrt{2} \approx 1.414$ | Hệ số tin cậy cận trên trong UCB1 (Confidence bound scale) |
| $\lambda$ | `LAMBDA`, `gae_lambda` | Week 2, 5 | $[0.90, 0.98]$ | Hệ số làm trơn TD($\lambda$) và trọng số GAE |
| $C$ | `TARGET_UPDATE_INTERVAL` | Week 3 | $[100, 10000]$ bước | Chu kỳ cập nhật cứng mạng Target trong DQN (Hard update) |
| $n$ | `N_STEP` | Week 3, 4 | $[1, 5]$ bước | Số bước thời gian tích luỹ phần thưởng (Multi-step return) |
| $N$ | `BUFFER_SIZE`, `capacity` | Week 3, 7 | $[5\cdot 10^4, 10^6]$ | Dung lượng bộ đệm trải nghiệm (Replay buffer size) |
| $\beta$ | `ENTROPY_WEIGHT`, `ent_coef` | Week 4, 5 | $[10^{-4}, 0.05]$ | Trọng số khuyến khích entropy cho policy (Entropy bonus) |
| $N_{envs}$ | `N_ENVS`, `num_workers` | Week 4, 5 | $[4, 64]$ | Số môi trường song song (Parallel vectorized workers) |
| $g_{clip}$ | `CLIP_GRAD`, `max_grad_norm` | Week 4, 5 | $[0.5, 1.0]$ | Ngưỡng cắt xén chuẩn độ dốc chống nổ gradient (Gradient clipping) |
| $\epsilon_{clip}$ | `CLIP_EPS`, `clip_range` | Week 5 | $[0.1, 0.3]$ | Biên độ cắt xén tỷ số xác suất trong PPO surrogate loss |
| $K$ | `EPOCHS`, `ppo_epochs` | Week 5 | $[3, 10]$ | Số epoch tối ưu hóa trên cùng một tập rollout buffer PPO |
| $\sigma$ | `SIGMA`, `noise_std` | Week 6, 7 | $[0.001, 0.1]$ | Độ lệch chuẩn nhiễu tham số (ES) hoặc nhiễu đầu vào (MBRL) |
| $n_{pop}$ | `POPULATION_SIZE` | Week 6 | $[40, 10000]$ | Số lượng cá thể trong một quần thể tìm kiếm Evolution Strategies |
| $H$ | `HORIZION_LENGTH` | Week 7 | $[5, 30]$ | Tầm nhìn dự đoán tương lai của bộ điều khiển MPC |
| $K_{seq}$ | `NUM_ACTIONS_SEQUENCES` | Week 7 | $[10^3, 5\cdot 10^4]$ | Số chuỗi hành động ngẫu nhiên được lấy mẫu trong Random Shooting |
| $\alpha_{SAC}$ | `alpha` | Week 8 | $[0.01, 1.0]$ hoặc tự học | Nhiệt độ entropy trong Maximum Entropy RL (SAC) |
| $\bar{\mathcal H}$ | `target_entropy` | Week 8 | $-\dim(\mathcal A)$ | Ngưỡng entropy mục tiêu trong Dual Gradient Descent của SAC |
| $\tau$ | `TAU`, `tau` | Week 8 | $[0.001, 0.01]$ | Hệ số Polyak cập nhật mềm mạng Target ($\approx 0.005$) |
| $c_{puct}$ | `c_puct` | Week 8 | $[1.0, 2.5]$ | Hằng số cân bằng khám phá/khai thác trong tìm kiếm MCTS PUCT |
| $\tau_{mcts}$ | `temperature` | Week 8 | $1.0 \to 0$ | Nhiệt độ làm phẳng phân phối nước đi từ cây MCTS |
| $\tau_{team}$ | `team_spirit` | Week 8 | $0.0 \to 1.0$ | Hệ số chuyển hóa phần thưởng cá nhân sang đồng đội (OpenAI Five) |
| $\beta_{KL}$ | `kl_penalty` | Week 8 | $[0.01, 0.2]$ | Trọng số phạt phân kỳ KL trong tinh chỉnh PPO-RLHF cho LLM |

---

## 2. Bản đồ tiến hoá thuật toán & Logic thiết kế các hàm toán học (Algorithmic Evolution & Mathematical Design Logic)

Mỗi phương trình trong Học tăng cường không tự nhiên sinh ra; chúng là **kết quả giải tích trực tiếp** nhằm khắc phục một lỗ hổng cụ thể hoặc thay thế một giả định phi thực tế của thuật toán đi trước. Dưới đây là bản đồ tiến hoá logic toàn khoá học:

### 2.1 Week 1: Từ Trung bình mẫu tới UCB1 & Policy Gradient

#### 1. Công thức cập nhật tăng dần với bước học hằng số
* **Phương trình toán học**:
  $$Q_{k+1} = Q_k + \alpha (R_k - Q_k) = (1-\alpha)Q_k + \alpha R_k$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế công thức tính trung bình số học thông thường $Q_n = \frac{1}{n-1}\sum_{i=1}^{n-1} R_i$.
* **Lỗ hổng của phương pháp cũ**:
  1. *Chi phí bộ nhớ*: Cần lưu trữ toàn bộ lịch sử phần thưởng quá khứ ($O(n)$ không gian bộ nhớ) hoặc phải đếm mẫu $n$ tăng dần.
  2. *Môi trường không dừng (Non-stationary)*: Khi phân phối xác suất của phần thưởng biến đổi theo thời gian, công thức trung bình mẫu chia đều trọng số $\frac{1}{n} \to 0$, khiến agent bị "hóa đá", không thể thích nghi với các biến động mới.
* **Bản chất Logic & Thiết kế toán học**: Sử dụng bước học hằng số $\alpha \in (0, 1]$ biến ước lượng thành **trung bình có trọng số suy giảm theo hàm mũ (Exponential Recency-Weighted Average)**: trọng số của phần thưởng cách hiện tại $i$ bước là $\alpha(1-\alpha)^i$. Phần thưởng mới nhận luôn giữ nguyên tỷ trọng $\alpha$, thông tin cũ suy biến dần về $0$.
* **Cơ chế giải quyết**: Cập nhật trực tuyến $O(1)$ bộ nhớ và duy trì độ nhạy liên tục trước các biến động của môi trường.

#### 2. Thuật toán UCB1 (Upper Confidence Bound)
* **Phương trình toán học**:
  $$a_t = \arg\max_{a} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế chiến lược $\epsilon$-greedy ($a_t$ chọn ngẫu nhiên đều với xác suất $\epsilon$).
* **Lỗ hổng của phương pháp cũ**: $\epsilon$-greedy khám phá hoàn toàn "mù quáng" (blind exploration). Nó đối xử với một hành động đã biết là cực kỳ tệ hại ngang hàng với một hành động chưa hề được thử nghiệm. Điều này dẫn đến **Hối tiếc tích lũy tăng tuyến tính $\Omega(T)$**, agent liên tục lãng phí tài nguyên vào các nhánh vô vọng.
* **Bản chất Logic & Thiết kế toán học**: Dựa trên nguyên lý **"Lạc quan khi đối mặt với sự bất định" (Optimism in the Face of Uncertainty)**. Dẫn xuất từ Bất đẳng thức nồng độ Hoeffding với xác suất sai lệch đuôi giảm theo $t^{-4}$. Đại lượng $U_t(a) = c \sqrt{\frac{\ln t}{N_t(a)}}$ đo lường bán kính khoảng tin cậy trên của giá trị hành động. Nếu một hành động ít được chọn ($N_t(a)$ nhỏ), bán kính bất định tăng vọt, buộc agent phải ưu tiên thử nghiệm nó. Khi đã thử nhiều ($N_t(a)$ lớn), bán kính co lại, nhường chỗ cho giá trị khai thác $Q_t(a)$.
* **Cơ chế giải quyết**: Đạt được mức **hối tiếc tích lũy tối ưu lý thuyết $O(\ln T)$** (theo định lý Lai & Robbins), triệt tiêu sự lãng phí tài nguyên của $\epsilon$-greedy.

---

### 2.2 Week 2: Từ Quy hoạch động tới Monte Carlo, TD(0) & Q-Learning

#### 1. Monte Carlo (MC) và Temporal Difference TD(0)
* **Phương trình toán học**:
  $$\text{Monte Carlo: } V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$
  $$\text{TD(0): } V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**:
  1. *Monte Carlo thay thế Quy hoạch động (Dynamic Programming - Bellman)*: Thay thế việc tính kỳ vọng giải tích đòi hỏi ma trận xác suất chuyển trạng thái $P(s' \mid s, a)$ và hàm thưởng $R(s, a)$.
  2. *TD(0) thay thế Monte Carlo*: Thay thế mục tiêu toàn phần $G_t = \sum_{k=0}^T \gamma^k R_{t+k+1}$ bằng mục tiêu bootstrap 1 bước $R_{t+1} + \gamma V(S_{t+1})$.
* **Lỗ hổng của phương pháp cũ**:
  - *Dynamic Programming*: Bất khả thi trong thực tế vì môi trường là "hộp đen" (không ai cung cấp ma trận $P$).
  - *Monte Carlo*: Bắt buộc phải chờ đến khi episode kết thúc hoàn toàn mới cập nhật được (không dùng được cho các bài toán tương tác liên tục không hồi kết). Quan trọng nhất: $G_t$ là tích lũy của hàng trăm biến ngẫu nhiên dọc theo toàn bộ chuỗi thời gian, dẫn đến **phương sai cực đại (Huge Variance)**, làm tốc độ học chậm chạp.
* **Bản chất Logic & Thiết kế toán học**: TD(0) áp dụng nguyên lý **Bootstrapping** (tự nâng đỡ bằng ước lượng sẵn có). Khai triển đệ quy Bellman $G_t = R_{t+1} + \gamma G_{t+1}$, TD(0) giữ lại phần thưởng thực nhận $R_{t+1}$ (để neo vào thực tế) và xấp xỉ phần đuôi chưa xảy ra bằng giá trị dự đoán hiện tại $V(S_{t+1})$.
* **Cơ chế giải quyết**: Chỉ chứa 1 bước ngẫu nhiên duy nhất $\implies$ **Phương sai cực thấp**, cho phép học trực tuyến từng bước thời gian (Online step-by-step learning) ngay cả trong môi trường vô hạn.

#### 2. Q-Learning (Off-Policy TD Control)
* **Phương trình toán học**:
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \big]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế SARSA ($R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$).
* **Lỗ hổng của phương pháp cũ**: SARSA là thuật toán On-policy, mục tiêu cập nhật phụ thuộc vào hành động $A_{t+1}$ do chính sách thăm dò $\epsilon$-greedy lựa chọn. Nếu agent đứng gần vực thẳm nguy hiểm, các bước nhảy ngẫu nhiên $\epsilon$ thường xuyên đẩy nó xuống vực, khiến hàm giá trị của SARSA bị bóp méo thành một chính sách "quá sợ hãi", không tìm ra con đường ngắn nhất tối ưu (ví dụ kinh điển trong bài toán Cliff Walking).
* **Bản chất Logic & Thiết kế toán học**: Tách rời **Chính sách hành vi (Behavior Policy)** dùng để thu thập mẫu (vẫn giữ $\epsilon$-greedy để khám phá) và **Chính sách mục tiêu (Target Policy)** dùng để ước lượng giá trị (dùng toán tử $\max_{a'}$, tức chính sách tất định thuần tham lam).
* **Cơ chế giải quyết**: Trở thành thuật toán **Off-policy** hoàn chỉnh, trực tiếp xấp xỉ phương trình tối ưu Bellman $T^* Q^* = Q^*$, cho phép học chính sách tối ưu toàn cục bất chấp dữ liệu được thu thập bởi hành vi thăm dò ngẫu nhiên nào.

---

### 2.3 Week 3: Sự tiến hóa của Deep Q-Networks (DQN, Double, Dueling, N-step, Noisy)

#### 1. Replay Buffer & Target Network trong DQN
* **Phương trình toán học**:
  $$\mathcal L(\theta) = \mathbb E_{(s, a, r, s') \sim \mathcal D}\left[ \Big( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \Big)^2 \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế việc cập nhật Q-learning trực tiếp trên chuỗi chuyển tiếp nối tiếp thời gian $(s_t, a_t, r_{t+1}, s_{t+1})$ bằng 1 mạng nơ-ron duy nhất $\theta$.
* **Lỗ hổng của phương pháp cũ**: Gây ra hiện tượng phân kỳ **"Deadly Triad"**:
  1. Dữ liệu nối tiếp thời gian có tính **tương quan cao (strongly correlated)**, vi phạm giả định độc lập cùng phân phối (i.i.d) của Gradient Descent.
  2. **Mục tiêu di động (Moving Target)**: Mạng vừa cập nhật $\theta$ thì mục tiêu $y = r + \gamma \max Q(s', a'; \theta)$ cũng nhảy vọt theo, tạo vòng lặp phản hồi dương khiến giá trị $Q$ bùng nổ vô hạn.
* **Bản chất Logic & Thiết kế toán học**:
  - *Replay Buffer $\mathcal D$*: Lưu trữ hàng trăm ngàn chuyển tiếp và lấy mẫu ngẫu nhiên đồng đều (Uniform Random Sampling), chuyển đổi bài toán tương tác nối tiếp về bài toán học có giám sát chuẩn i.i.d.
  - *Target Network $\theta^-$*: Đóng băng tham số của hàm mục tiêu trong chu kỳ $C$ bước, biến bài toán thành bài toán hồi quy với nhãn mục tiêu cố định tạm thời (Semi-Gradient).
* **Cơ chế giải quyết**: Ổn định hóa hoàn toàn quá trình huấn luyện mạng nơ-ron sâu cho Q-learning, mở ra kỷ nguyên Deep RL hiện đại.

#### 2. Double DQN (Decoupled Selection and Evaluation)
* **Phương trình toán học**:
  $$y_t^{DoubleQ} = R_{t+1} + \gamma Q\Big( S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta);\, \theta^- \Big)$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế mục tiêu chuẩn của DQN $y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$.
* **Lỗ hổng của phương pháp cũ**: Hiện tượng **Thiên lệch cực đại hoá (Maximization Bias)**. Toán tử $\mathbb E[\max(X_1, X_2)] \ge \max(\mathbb E[X_1], \mathbb E[X_2])$ (Bất đẳng thức Jensen). Khi mạng nơ-ron có sai số ước lượng ngẫu nhiên mang phân phối chuẩn $\mathcal N(0, \sigma^2)$, toán tử $\max$ luôn chọn lấy sai số dương lớn nhất, làm giá trị $Q$ bị thổi phồng một lượng $\approx \frac{\sigma}{\sqrt{\pi}}$. Sai số này tích luỹ qua các bước bootstrap khiến agent đánh giá sai nghiêm trọng chất lượng các hành động.
* **Bản chất Logic & Thiết kế toán học**: Phân rã hai nhiệm vụ: **Chọn hành động tốt nhất** (Action Selection) và **Định giá hành động đó** (Action Evaluation) cho hai bộ trọng số khác nhau. Mạng chính $\theta$ chọn hành động $a^* = \arg\max_{a'} Q(S_{t+1}, a'; \theta)$, nhưng mạng Target $\theta^-$ mới là bên đưa ra điểm số $Q(S_{t+1}, a^*; \theta^-)$.
* **Cơ chế giải quyết**: Vì sai số ngẫu nhiên của $\theta$ và $\theta^-$ độc lập với nhau, kỳ vọng của sai số định giá triệt tiêu về 0, loại bỏ hoàn toàn hiện tượng thổi phồng giá trị $Q$.

#### 3. Dueling DQN (State Value & Advantage Decomposition)
* **Phương trình toán học**:
  $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal A|} \sum_{a' \in \mathcal A} A(s, a'; \theta, \alpha) \right)$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế kiến trúc mạng nguyên khối dự đoán trực tiếp $Q(s, a)$.
* **Lỗ hổng của phương pháp cũ**: Trong nhiều trạng thái (ví dụ xe đang chạy trên đường thẳng an toàn), việc rẽ trái hay rẽ phải chưa tạo ra sự khác biệt, nhưng trạng thái đó tự bản thân nó đã rất có giá trị. Mạng DQN cũ buộc phải trải nghiệm qua từng hành động độc lập mới cập nhật được giá trị $Q(s, a)$ cho hành động đó, cực kỳ lãng phí dữ liệu.
* **Bản chất Logic & Thiết kế toán học**: Tách giá trị của trạng thái $V(s)$ (giá trị nội tại) và lợi thế $A(s, a)$ (mức độ hơn kém giữa các hành động). Để giải quyết bài toán **Bất định danh (Unidentifiability)** (vô số cặp $V$ và $A$ có cùng tổng), ta trừ đi trung bình của $A$. Khi đó, trung bình của thành phần Advantage bằng đúng 0, neo chắc chắn $V(s)$ vào giá trị kỳ vọng của trạng thái.
* **Cơ chế giải quyết**: Mỗi lần cập nhật gradient cho bất kỳ hành động nào, mạng đều cập nhật luôn $V(s)$, lan truyền thông tin giá trị trạng thái cho **toàn bộ các hành động khác cùng lúc** chỉ trong 1 bước học.

---

### 2.4 Week 4: Từ Q-learning tới Policy Gradient, Reward-to-Go, Baseline & Actor-Critic

#### 1. Định lý Policy Gradient (Log-Derivative Trick)
* **Phương trình toán học**:
  $$\nabla_\theta J(\theta) = \mathbb E_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^T \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \, R(\tau) \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế hoàn toàn nhánh Value-based (DQN chọn $\arg\max_a Q(s, a)$).
* **Lỗ hổng của phương pháp cũ**:
  1. Không thể áp dụng trực tiếp cho **Không gian hành động liên tục** (Continuous Action Space) như góc quay vô-lăng, lực đẩy động cơ tên lửa (vì không thể tính $\max_{a}$ trên vô hạn hành động ở mỗi mili-giây).
  2. Chính sách $\arg\max$ là tất định và thay đổi đột ngột khi $Q$ dao động nhỏ, không thể mô hình hóa các chính sách ngẫu nhiên tối ưu (như bài toán bluffing trong Poker, oẳn tù tì).
* **Bản chất Logic & Thiết kế toán học**: Sử dụng thủ thuật đạo hàm logarit $\nabla p = p \nabla \ln p$. Gradient của hàm mục tiêu toàn cục được chuyển thành kỳ vọng của gradient log-xác suất nhân với phần thưởng.
* **Cơ chế giải quyết**: Đạo hàm của quy luật vật lý môi trường $P(s' \mid s, a)$ **triệt tiêu hoàn toàn** khỏi biểu thức! Cho phép tối ưu trực tiếp chính sách liên tục $\pi_\theta$ mà không cần biết mô hình động học của môi trường.

#### 2. Reward-to-Go & Baseline (Variance Reduction)
* **Phương trình toán học**:
  $$\nabla_\theta J(\theta) = \mathbb E_{\tau}\left[ \sum_{t=0}^T \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \Big( \underbrace{\sum_{t'=t}^T \gamma^{t'-t} r_{t'+1}}_{G_t \text{ (Reward-to-go)}} - \underbrace{b(s_t)}_{\text{Baseline}} \Big) \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế tổng phần thưởng cả quỹ đạo $R(\tau) = \sum_{t'=0}^T r_{t'+1}$.
* **Lỗ hổng của phương pháp cũ**: Phương sai khổng lồ. Hành động ở bước $t=100$ bị nhân với cả phần thưởng nhận được ở bước $t=1$, gán "công/tội" cho một hành động không hề gây ra kết quả đó (vi phạm tính nhân quả). Đồng thời, nếu mọi phần thưởng đều dương ($r > 0$), mọi hành động đều được tăng xác suất như nhau.
* **Bản chất Logic & Thiết kế toán học**:
  - *Reward-to-go*: Dựa trên nguyên lý **Nhân quả (Causality)** — hành động hiện tại không thể thay đổi quá khứ. Đã chứng minh giải tích $\mathbb E[\nabla \ln \pi(a_t \mid s_t) r_{t'}] = 0$ với mọi $t' < t$.
  - *Baseline $b(s_t) = V(s_t)$*: Đã chứng minh $\mathbb E[\nabla \ln \pi(a_t \mid s_t) b(s_t)] = 0$. Trừ đi giá trị trung bình kỳ vọng $V(s_t)$ biến mục tiêu thành **Hàm lợi thế (Advantage)**: hành động tốt hơn kỳ vọng nhận giá trị dương, kém hơn nhận giá trị âm.
* **Cơ chế giải quyết**: Giảm phương sai của gradient ước lượng xuống hàng bậc độ lớn mà **hoàn toàn không làm chệch kỳ vọng (strictly unbiased)**.

#### 3. Advantage Actor-Critic (A2C)
* **Phương trình toán học**:
  $$\hat A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế việc lấy mẫu Monte Carlo $G_t$ trong REINFORCE.
* **Lỗ hổng của phương pháp cũ**: REINFORCE vẫn phải chờ đến cuối episode mới cập nhật được, và $G_t$ dù đã dùng reward-to-go vẫn tích luỹ phương sai của nhiều bước tương lai.
* **Bản chất Logic & Thiết kế toán học**: Dùng một mạng nơ-ron thứ hai (Critic $V_\phi$) để thực hiện Bootstrapping 1 bước giống như TD(0). Sai số TD $\delta_t$ chính là một ước lượng không chệch của Advantage $A(s_t, a_t)$ khi Critic hội tụ.
* **Cơ chế giải quyết**: Cho phép cập nhật chính sách **từng bước thời gian (Online)** hoặc theo từng chuỗi ngắn $n$-step, loại bỏ việc phải chờ hết episode và triệt tiêu phương sai mạnh mẽ.

---

### 2.5 Week 5: Từ Vanilla PG tới TRPO, PPO-Clip & GAE

#### 1. Performance Difference Lemma & Trust Region (TRPO)
* **Phương trình toán học**:
  $$J(\pi') - J(\pi) = \frac{1}{1-\gamma} \mathbb E_{s \sim d^{\pi'}, a \sim \pi'}\big[ A^\pi(s, a) \big]$$
  $$\max_\theta \mathbb E_{s \sim d^{\theta_{old}}, a \sim \pi_{\theta_{old}}}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A^{\theta_{old}}(s, a) \right] \quad \text{s.t.} \quad \mathbb E\big[ D_{KL}(\pi_{\theta_{old}} \parallel \pi_\theta) \big] \le \delta$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế bước nhảy tham số không ràng buộc $\theta \leftarrow \theta + \alpha \nabla J(\theta)$ trong không gian Euclid của Vanilla Policy Gradient.
* **Lỗ hổng của phương pháp cũ**: **Sụp đổ chính sách sớm (Policy Collapse)**. Bước nhảy $\alpha$ cố định trong không gian trọng số $\theta$ không tương thích với độ nhạy trong không gian phân phối xác suất. Tại vùng nhạy cảm, một bước nhảy nhỏ của $\theta$ có thể làm policy biến đổi hoàn toàn, rơi vào vùng hành vi thảm họa; dữ liệu mới thu thập được lại càng tệ hại, khiến mô hình vĩnh viễn không phục hồi được.
* **Bản chất Logic & Thiết kế toán học**: Dựa trên Bổ đề chênh lệch hiệu năng Kakade-Langford. Thay vì nhảy tự do, thuật toán giới hạn bước cập nhật trong một **Vùng tin cậy (Trust Region)** đo bằng phân kỳ Kullback-Leibler $D_{KL} \le \delta$.
* **Cơ chế giải quyết**: Đảm bảo toán học về tính **cải thiện đơn điệu (Monotonic Improvement)**: chính sách mới luôn có hiệu năng không thấp hơn chính sách cũ.

#### 2. PPO-Clip (Proximal Policy Optimization)
* **Phương trình toán học**:
  $$\mathcal L^{CLIP}(\theta) = \hat{\mathbb E}_t \left[ \min\left( r_t(\theta)\hat A_t,\ \operatorname{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big)\hat A_t \right) \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế bài toán tối ưu bậc hai phức tạp kèm ma trận Fisher $F^{-1}$ và thuật toán Conjugate Gradient của TRPO.
* **Lỗ hổng của phương pháp cũ**: TRPO quá cồng kềnh, tính toán ma trận đạo hàm bậc hai $F$ cực kỳ tốn kém bộ nhớ và không tương thích tốt với các mạng nơ-ron có chia sẻ trọng số hoặc kiến trúc sâu phức tạp.
* **Bản chất Logic & Thiết kế toán học**: Xây dựng một **cận dưới bi quan (Pessimistic Bound)** chỉ dùng đạo hàm bậc nhất. Hàm kẹp `clip` chặt tỷ số xác suất $r_t(\theta)$ vào khoảng $[1-\epsilon, 1+\epsilon]$.
  - Nếu hành động tốt ($\hat A_t > 0$): Không cho phép $r_t$ vượt quá $1+\epsilon$. Nếu vượt quá, đạo hàm bị triệt tiêu về $0$.
  - Nếu hành động tệ ($\hat A_t < 0$): Không cho phép $r_t$ giảm sâu hơn $1-\epsilon$. Nếu thấp hơn, đạo hàm triệt tiêu về $0$.
* **Cơ chế giải quyết**: Giữ được sự ổn định tương đương TRPO nhưng chỉ sử dụng bộ tối ưu SGD/Adam bậc nhất thông thường, cho phép tái sử dụng tập rollout để tối ưu hóa $K$ epoch liên tục, trở thành thuật toán chuẩn mực số 1 trong công nghiệp RL.

#### 3. Generalized Advantage Estimator (GAE)
* **Phương trình toán học**:
  $$\hat A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + \gamma \lambda \hat A_{t+1}^{GAE}$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế ước lượng Advantage 1 bước $\delta_t^V$ (TD) hoặc Advantage Monte Carlo vô hạn bước $G_t - V(s_t)$.
* **Lỗ hổng của phương pháp cũ**: 1-step TD có bias quá cao (lệch theo mạng Critic chưa hoàn thiện), trong khi Monte Carlo có variance quá cao.
* **Bản chất Logic & Thiết kế toán học**: Lấy trung bình có trọng số suy giảm hàm mũ theo hệ số $\lambda \in [0, 1]$ của mọi chân trời $k$-step Advantage. Tạo ra sự pha trộn nhịp nhàng giữa Bias và Variance.
* **Cơ chế giải quyết**: Bằng việc điều chỉnh $\lambda \approx 0.95$, GAE giảm thiểu đáng kể sai số bình phương trung bình của Advantage, tăng tốc độ hội tụ của PPO lên vượt bậc.

---

### 2.6 Week 6: Từ Lan truyền ngược MDP tới Evolution Strategies & Mirrored Sampling

#### 1. Evolution Strategies (Gaussian Parameter Smoothing)
* **Phương trình toán học**:
  $$\nabla_\theta \mathbb E_{\varepsilon \sim \mathcal N(0, I)}[F(\theta + \sigma \varepsilon)] = \frac{1}{\sigma} \mathbb E_{\varepsilon}\big[ F(\theta + \sigma \varepsilon) \varepsilon \big]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế toàn bộ cơ chế lan truyền ngược (Backpropagation) qua chuỗi thời gian MDP của Policy Gradient.
* **Lỗ hổng của phương pháp cũ**: Policy Gradient gặp trở ngại lớn khi môi trường có phần thưởng cực kỳ trễ (hàng ngàn bước mới có thưởng), hoặc hàm chuyển tiếp không khả vi, có độ trễ truyền thông lớn khi huấn luyện phân tán trên cụm máy tính GPU/CPU.
* **Bản chất Logic & Thiết kế toán học**: Biến bài toán tối ưu thành **Tối ưu hộp đen (Black-Box Optimization)**. Thay vì tối ưu trực tiếp $F(\theta)$, ta tối ưu phiên bản đã làm trơn bằng tích chập Gauss $J_\sigma(\theta)$.
* **Cơ chế giải quyết**: Đạo hàm được ước lượng thuần túy thông qua các giá trị vô hướng $F(\theta + \sigma \varepsilon)$ nhân với vector nhiễu $\varepsilon$, hoàn toàn không cần lan truyền ngược, không cần hàm giá trị Critic. Khả năng mở rộng tuyến tính hoàn hảo trên hàng ngàn CPU song song.

#### 2. Mirrored Sampling (Antithetic Perturbations)
* **Phương trình toán học**:
  $$\hat g_{mirror} = \frac{1}{2 n \sigma} \sum_{i=1}^n \big( F(\theta + \sigma \varepsilon_i) - F(\theta - \sigma \varepsilon_i) \big) \varepsilon_i$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế phép lấy mẫu nhiễu một phía $F(\theta + \sigma \varepsilon_i) \varepsilon_i$.
* **Lỗ hổng của phương pháp cũ**: Nhiễu một phía có phương sai rất lớn do sự chênh lệch nền của hàm mục tiêu $F(\theta)$, và bị ảnh hưởng nặng nề bởi độ cong phi tuyến bậc hai (Hessian).
* **Bản chất Logic & Thiết kế toán học**: Lấy mẫu theo cặp đối xứng hoàn hảo $\pm \varepsilon$. Khai triển chuỗi Taylor giải tích:
  $$F(\theta + \sigma \varepsilon) - F(\theta - \sigma \varepsilon) = 2\sigma \nabla F(\theta)^\top \varepsilon + \mathcal O(\sigma^3)$$
  Mọi số hạng bậc chẵn (bậc 0: giá trị nền $F(\theta)$, bậc 2: ma trận Hessian $\varepsilon^\top \nabla^2 F \varepsilon$) **triệt tiêu hoàn toàn lẫn nhau**!
* **Cơ chế giải quyết**: Giảm phương sai ước lượng gradient tới một nửa, đồng thời chỉ cần truyền hạt giống ngẫu nhiên (random seed) giữa các máy tính phân tán, cắt giảm băng thông mạng từ $O(n \cdot d)$ xuống $O(n)$.

---

### 2.7 Week 7: Từ Model-Free tới Model-Based RL, Residual Dynamics, MPC & DAgger

#### 1. Dự đoán vi sai động học (Residual Dynamics $\Delta s$)
* **Phương trình toán học**:
  $$\hat s_{t+1} = s_t + \hat f_\phi(s_t, a_t) \quad \iff \quad \hat f_\phi(s_t, a_t) \approx s_{t+1} - s_t$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế việc huấn luyện mạng nơ-ron dự đoán trực tiếp trạng thái tiếp theo $\hat s_{t+1} = \hat f_\phi(s_t, a_t)$.
* **Lỗ hổng của phương pháp cũ**: Trong các hệ thống điều khiển với bước thời gian ngắn ($\Delta t$ nhỏ, ví dụ $0.01$s), trạng thái $s_{t+1} \approx s_t$. Mạng nơ-ron học dự đoán $s'$ sẽ lãng phí $99\%$ dung lượng tham số chỉ để học "hàm đồng nhất" (sao chép $s_t$ sang $s_{t+1}$), khiến các thay đổi vi mô mang tính quyết định của vận tốc và gia tốc bị chìm lấp trong sai số số học.
* **Bản chất Logic & Thiết kế toán học**: Tương tự nguyên lý mạng **ResNet (Residual Learning)**. Mạng chỉ tập trung mô hình hóa biến thiên vi phân $\Delta s = s_{t+1} - s_t$, vốn có giá trị tập trung quanh 0.
* **Cơ chế giải quyết**: Giảm mạnh sai số dự đoán 1 bước, bảo toàn các thành phần động học nhạy cảm.

#### 2. Model Predictive Control (MPC) Receding Horizon
* **Phương trình toán học**:
  $$A^* = \arg\max_{a_{t:t+H-1}} \sum_{k=0}^{H-1} \hat r(\hat s_{t+k}, a_{t+k}), \quad \text{chỉ thực thi } a_t = A^*_0, \quad \text{tái lập kế hoạch tại } t+1$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế việc thực thi toàn bộ chuỗi hành động đã lập kế hoạch (Open-Loop Trajectory Execution).
* **Lỗ hổng của phương pháp cũ**: Hiện tượng **Bùng nổ sai số tích lũy (Compounding Error)** theo Bổ đề mô phỏng (Simulation Lemma): sai số tích lũy sau $H$ bước tăng theo quy luật hàm mũ $\mathcal O(L_f^H \varepsilon)$. Nếu chạy hết cả chuỗi $H$ hành động, ở các bước cuối agent sẽ rơi vào trạng thái hoàn toàn khác với tưởng tượng, dẫn đến lật xe hoặc va chạm vật lý.
* **Bản chất Logic & Thiết kế toán học**: Nguyên lý **Chân trời lùi dần (Receding Horizon)**. Agent chỉ tin tưởng mô hình trong đúng 1 bước đầu tiên ($a_t = A^*_0$). Sau khi thực hiện và nhận được trạng thái thực $s_{t+1}$ từ thế giới thật, agent lập tức "vứt bỏ" kế hoạch cũ và giải lại bài toán tối ưu từ đầu trên trạng thái thực mới.
* **Cơ chế giải quyết**: Triệt tiêu sự tích lũy sai số nhiều bước, liên tục neo mô hình về trạng thái thực của môi trường.

#### 3. DAgger (Dataset Aggregation)
* **Phương trình toán học**:
  $$\mathcal D_{k+1} = \mathcal D_k \cup \big\{ (s, \pi^*(s)) \mid s \sim d^{\pi_{\theta_k}} \big\}$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế phương pháp học bắt chước hành vi tĩnh (Behavioral Cloning).
* **Lỗ hổng của phương pháp cũ**: Hiện tượng **Trôi dạt phân phối (Covariate Shift / Distribution Drift)**. Mô hình chỉ được học trên quỹ đạo của chuyên gia. Khi tự điều khiển ngoài đời thực, một sai số nhỏ đưa agent vào một trạng thái lạ mà chuyên gia chưa từng gặp, mô hình không biết xử lý, gây sai số lớn hơn, đẩy agent văng vĩnh viễn ra khỏi miền dữ liệu an toàn.
* **Bản chất Logic & Thiết kế toán học**: Ép chuyên gia (hoặc bộ điều khiển MPC) gán nhãn cho chính các trạng thái mà **chính sách hiện tại của agent** đang ghé thăm, rồi gộp vào tập huấn luyện qua các vòng lặp.
* **Cơ chế giải quyết**: Đảm bảo sai số tích luỹ bị chặn tuyến tính $\mathcal O(\varepsilon H)$ thay vì bậc hai $\mathcal O(\varepsilon H^2)$, giúp mô hình có khả năng tự phục hồi khi gặp sự cố.

---

### 2.8 Week 8: Các bước nhảy vọt hiện đại (MaxEnt SAC, RND Curiosity, AlphaGo PUCT, DPO)

#### 1. Soft Actor-Critic (Maximum Entropy RL)
* **Phương trình toán học**:
  $$J_{MaxEnt}(\pi) = \mathbb E_\pi \left[ \sum_{t=0}^\infty \gamma^t \Big( r(s_t, a_t) + \alpha \mathcal H\big(\pi(\cdot \mid s_t)\big) \Big) \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế các thuật toán Actor-Critic tất định hoặc tối ưu hóa phần thưởng thuần tuý như DDPG, TD3.
* **Lỗ hổng của phương pháp cũ**: DDPG/TD3 có chính sách tất định, rất dễ bị sụp đổ sớm vào các cực tiểu địa phương nghèo nàn và cực kỳ nhạy cảm với việc chọn siêu tham số.
* **Bản chất Logic & Thiết kế toán học**: Dẫn xuất từ khung lý thuyết **Điều khiển như bài toán suy luận xác suất (Control as Inference)**. Đưa thêm số hạng Shannon Entropy vào hàm mục tiêu với trọng số nhiệt độ $\alpha$. Thuật toán đồng thời tối đa hóa phần thưởng VÀ độ bất định của hành động.
* **Cơ chế giải quyết**: Chính sách khám phá tự nhiên khắp không gian trạng thái, nắm bắt được các nghiệm **đa mốt (Multimodal)**, chống chịu nhiễu tối ưu, kết hợp với kỹ thuật Dual Gradient Descent để tự động điều chỉnh nhiệt độ $\alpha$ hướng về mức entropy mục tiêu $\bar{\mathcal H} = -\dim(\mathcal A)$.

#### 2. Random Network Distillation (RND Exploration)
* **Phương trình toán học**:
  $$r_t^{int} = \big\| \hat f_\theta(s_{t+1}) - f_{target}(s_{t+1}) \big\|_2^2$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế cơ chế tạo tò mò bằng sai số dự đoán động học (Intrinsic Curiosity Module - ICM).
* **Lỗ hổng của phương pháp cũ**: Vấn nạn **"Tivi nhiễu hạt" (Noisy TV Problem)**. Nếu môi trường có một chiếc tivi phát nhiễu ngẫu nhiên không thể đoán trước, mạng dự đoán động học liên tục ghi nhận sai số khổng lồ, khiến agent bị "thôi miên", đứng nhìn tivi mãi mãi vì tưởng rằng đó là vùng đất mới lạ cần khám phá.
* **Bản chất Logic & Thiết kế toán học**: Sử dụng một mạng mục tiêu cố định $f_{target}$ được khởi tạo ngẫu nhiên và **đóng băng vĩnh viễn trọng số**, cùng một mạng dự đoán $\hat f_\theta$ học cách bắt chước đầu ra của mạng mục tiêu.
* **Cơ chế giải quyết**: Trạng thái nào đã ghé thăm nhiều lần thì $\hat f_\theta$ sẽ chưng cất thành công $\implies r^{int} \to 0$. Chiếc tivi phát nhiễu dù ngẫu nhiên nhưng các trạng thái ảnh của nó khi qua $f_{target}$ sẽ dần được $\hat f_\theta$ khớp dữ liệu, triệt tiêu ảo giác tò mò và giải quyết hoàn toàn bài toán Sparse Reward (như game kinh điển Montezuma's Revenge).

#### 3. AlphaGo Zero / AlphaZero PUCT MCTS
* **Phương trình toán học**:
  $$a_t = \arg\max_a \left[ Q(s, a) + c_{puct} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế thuật toán Minimax Alpha-Beta Pruning và UCT truyền thống (dùng mô phỏng ngẫu nhiên rollout ngẫu nhiên).
* **Lỗ hổng của phương pháp cũ**: Không gian trạng thái cờ vây quá khổng lồ ($10^{170}$ trạng thái, phân nhánh $b \approx 250$). Minimax chỉ nhìn được vài bước, còn rollout ngẫu nhiên trong UCT không có chiều sâu chiến thuật vì các nước đi ngẫu nhiên hoàn toàn phi lý.
* **Bản chất Logic & Thiết kế toán học**: Tích hợp mạng nơ-ron hai đầu (Policy $\mathbf p$ và Value $v$) vào cây MCTS. Thay thế việc mô phỏng ngẫu nhiên bằng giá trị dự đoán tức thì $v(s)$, và dùng xác suất tiên nghiệm $P(s, a)$ từ mạng policy để định hướng tìm kiếm PUCT.
* **Cơ chế giải quyết**: MCTS đóng vai trò như một **bộ khuếch đại chính sách (Policy Improvement Operator)**: phân phối lượt thăm nom $N(s, a)$ sau khi tìm kiếm là một chính sách mạnh hơn nhiều so với policy gốc $\mathbf p(s)$. Mạng nơ-ron sau đó học lại chính phân phối MCTS này trong vòng lặp Self-play để đạt tới trình độ siêu phàm không cần tri thức con người.

#### 4. Direct Preference Optimization (DPO) trong LLM Alignment
* **Phương trình toán học**:
  $$\mathcal L_{DPO}(\theta; \pi_{ref}) = -\mathbb E_{(x, y_w, y_l) \sim \mathcal D}\left[ \ln \sigma\left( \beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)} \right) \right]$$
* **Thuật toán tiền nhiệm & Thành phần bị thay thế**: Thay thế toàn bộ quy trình PPO-RLHF truyền thống gồm 4 mô hình (Actor $\pi_\theta$, Critic $V_\phi$, Reference Model $\pi_{ref}$, Reward Model $r_\psi$).
* **Lỗ hổng của phương pháp cũ**: PPO-RLHF cực kỳ phức tạp và thiếu ổn định: phải huấn luyện riêng Reward Model, sau đó chạy vòng lặp tương tác Actor-Critic với hàng loạt siêu tham số nhạy cảm, dễ gặp hiện tượng Reward Hacking và sụp đổ phân phối từ ngữ.
* **Bản chất Logic & Thiết kế toán học**: Xuất phát từ nghiệm đóng giải tích của bài toán cực đại phần thưởng có ràng buộc KL:
  $$r(x, y) = \beta \ln \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \ln Z(x)$$
  Thay nghiệm đóng này trực tiếp vào mô hình xác suất Bradley-Terry lựa chọn của con người $p(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$. Hằng số phân vùng $Z(x)$ **triệt tiêu hoàn toàn**!
* **Cơ chế giải quyết**: Biến toàn bộ bài toán Reinforcement Learning phức tạp thành bài toán **Hồi quy Logistic / Binary Cross-Entropy có giám sát** trực tiếp trên mô hình ngôn ngữ, loại bỏ hoàn toàn mạng Reward Model và mạng Critic, đơn giản hóa và tối ưu hóa vượt bậc quá trình căn chỉnh LLM.

---

## 3. Giải mã tham số theo từng tuần & Phương trình chi tiết

### 3.1 Week 1: Multi-Armed Bandits & Thăm dò cổ điển
Tham chiếu bài giảng: [`Week1/BAI_GIANG.md`](Week1/BAI_GIANG.md#6-phân-tích-chuyên-sâu-các-tham-số-trong-phương-trình)

#### 1. Hệ số chiết khấu $\gamma$ (Discount Factor)
* **Phương trình**:
  $$G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$
* **Ý nghĩa toán học**: Bán kính hội tụ chuỗi hình học, bảo đảm $G_t \le \frac{R_{\max}}{1-\gamma} < \infty$.
* **Tầm nhìn hiệu dụng (Effective Horizon)**:
  $$H_{\text{eff}} = \frac{1}{1-\gamma}$$
  * $\gamma = 0.9 \implies H_{\text{eff}} = 10$ bước.
  * $\gamma = 0.99 \implies H_{\text{eff}} = 100$ bước.
  * $\gamma = 0.999 \implies H_{\text{eff}} = 1000$ bước.
* **Biên giới hạn**:
  * $\gamma \to 0$: Agent hoàn toàn tham lam ngắn hạn (Myopic), chỉ tối đa hóa phần thưởng ngay bước kế tiếp, bỏ qua mọi hệ quả lâu dài.
  * $\gamma \to 1$: Agent coi trọng phần thưởng ở vô hạn như phần thưởng tức thì; phương sai của tổng $G_t$ tăng cao, làm chậm quá trình hội tụ trong môi trường ngẫu nhiên.

#### 2. Tốc độ cập nhật $\alpha$ trong ước lượng trung bình trôi
* **Phương trình**:
  $$Q_{k+1} = Q_k + \alpha (R_k - Q_k) = (1-\alpha)Q_k + \alpha R_k$$
* **Ý nghĩa toán học**: Trọng số của phân phối hàm mũ giảm dần $w_i = \alpha(1-\alpha)^{k-i}$ gán cho phần thưởng ở quá khứ.
* **Biên giới hạn**:
  * $\alpha = 0$: Mô hình không học gì ($Q_{k+1} = Q_0$).
  * $\alpha = 1$: Mô hình hoàn toàn mất trí nhớ, chỉ lưu giá trị phần thưởng vừa nhận ($Q_{k+1} = R_k$).
  * Điều kiện Robbins-Monro để hội tụ về kỳ vọng thực tế trong bài toán ngẫu nhiên dừng: $\sum_{k=1}^\infty \alpha_k = \infty$ và $\sum_{k=1}^\infty \alpha_k^2 < \infty$.

#### 3. Hệ số tin cậy $c$ trong thuật toán UCB1
* **Phương trình**:
  $$a_t = \arg\max_{a} \left[ \hat{Q}_t(a) + c \sqrt{\frac{2 \ln t}{N_t(a)}} \right]$$
* **Ý nghĩa toán học**: Dẫn xuất từ bất đẳng thức nồng độ Hoeffding với xác suất đuôi $p \le t^{-4}$. Giá trị lý thuyết chuẩn tắc là $c = \sqrt{2} \approx 1.414$.
* **Biên giới hạn**:
  * $c \to 0$: Suy biến thành greedy chọn theo $\hat{Q}(a)$, dễ bị "kẹt tay" vĩnh viễn ở cần gạt dưới tối ưu do phương sai mẫu ban đầu thấp.
  * $c \gg 1$: Thành phần khám phá lấn át hoàn toàn giá trị ước lượng, agent phân bổ số lần giật cần gạt gần như đều nhau, hối tiếc (Regret) tăng tuyến tính thay vì logarit $O(\ln T)$.

---

### 3.2 Week 2: Tabular MDP, Bellman Operators & TD Learning
Tham chiếu bài giảng: [`Week2/BAI_GIANG.md`](Week2/BAI_GIANG.md#73-phân-tích-chuyên-sâu-các-tham-số-và-cơ-chế-tác-động-lên-phương-trình)

#### 1. Hệ số chiết khấu $\gamma$ và Định lý điểm bất động Banach
* **Phương trình toán học**:
  $$\|T^* Q_1 - T^* Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
* **Ý nghĩa toán học**: Toán tử Bellman $T^*$ là một phép co chặt (Contraction Mapping) trên không gian metric Banach khi và chỉ khi $\gamma \in [0, 1)$. Tốc độ co hình học là $\gamma^k$.
* **Biên giới hạn**:
  * Nếu $\gamma \ge 1$ trong đồ hình có chu trình (cyclic MDP): Toán tử Bellman mất tính chất co, chuỗi lặp giá trị phân kỳ ra $\pm \infty$.
  * Trong bài toán tìm đường ngắn nhất (Shortest Path) với phần thưởng âm mỗi bước $r = -1$: Bắt buộc $\gamma = 1$ để agent không bị khuyến khích "tự sát" hoặc đi lòng vòng.

#### 2. Hệ số làm trơn $\lambda$ trong TD($\lambda$)
* **Phương trình**:
  $$G_t^\lambda = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n}$$
* **Cơ chế cân bằng Bias - Variance**:
  * $\lambda = 0$ (TD(0)): Chỉ dùng 1-step bootstrap $R_{t+1} + \gamma V(S_{t+1})$. Phương sai tối thiểu (thấp nhất), nhưng độ chệch (bias) cao nhất (phụ thuộc nặng nề vào giá trị khởi tạo $V$).
  * $\lambda = 1$ (Monte Carlo): Chạy hết episode để lấy $G_t$. Hoàn toàn không có bias (unbiased), nhưng phương sai cực đại (bị ảnh hưởng bởi toàn bộ biến ngẫu nhiên dọc theo quỹ đạo).
  * Khoảng tối ưu thực nghiệm: $\lambda \in [0.90, 0.95]$ giúp tối ưu hóa sai số bình phương trung bình (MSE).

---

### 3.3 Week 3: Deep Q-Networks (DQN) & Value-Based Deep RL
Tham chiếu bài giảng: [`Week3/BAI_GIANG.md`](Week3/BAI_GIANG.md#111-phân-tích-chuyên-sâu-các-tham-số-và-cơ-chế-tác-động-lên-phương-trình-dqn)

#### 1. Chu kỳ cập nhật Target Network $C$ (`TARGET_UPDATE_INTERVAL`)
* **Phương trình hàm mất mát**:
  $$\mathcal L(\theta) = \mathbb E_{(s, a, r, s') \sim \mathcal D}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
* **Bản chất toán học**: Đóng băng trọng số $\theta^-$ trong $C$ bước để triệt tiêu vòng lặp phản hồi dương (Positive Feedback Loop):
  $$\nabla_\theta \mathcal L(\theta) \approx -2 \mathbb E \left[ \delta_t \cdot \nabla_\theta Q(s, a; \theta) \right]$$
* **Cơ chế tác động**:
  * $C \to 1$: Mục tiêu $y_t$ di chuyển liên tục theo $\theta_t$. Gradient vừa đẩy $Q(s, a)$ tiến về $y_t$ thì $y_t$ lại vọt ra xa hơn, gây dao động dữ dội và phân kỳ hàm giá trị.
  * $C \gg 1$ ($C > 20000$ bước): Target network bị "hóa đá", thông tin TD lan truyền qua các trạng thái cực chậm, làm agent học rất lâu.
  * Khoảng chuẩn: $C \in [500, 2000]$ bước (hoặc Polyak $\tau = 0.005$ cập nhật mềm).

#### 2. Kích thước Replay Buffer $N$ (`BUFFER_CAPACITY = 100000`)
* **Ý nghĩa**: Phá vỡ tính phụ thuộc tương quan thời gian (Temporal Correlation) giữa các chuyển tiếp liên tiếp $\mathbb E_{(s_t, s_{t+1})} \ne \mathbb E_{\mathcal D}$, và đảm bảo giả định độc lập cùng phân phối (i.i.d) của tối ưu hóa ngẫu nhiên (SGD).
* **Biên giới hạn**:
  * $N$ quá nhỏ ($N < 5000$): Bộ đệm chỉ chứa các hành động gần nhất, mạng nơ-ron bị quên thảm họa (Catastrophic Forgetting) các vùng không gian trạng thái trước đó.
  * $N$ quá lớn ($N > 2\cdot 10^6$): Bộ đệm chứa quá nhiều dữ liệu từ chính sách cổ xưa (lỗi thời), làm loãng các chuyển tiếp chất lượng cao mà policy hiện tại đang khám phá được (Off-policy Distribution Gap).

#### 3. Chân trời đa bước $n$ trong Multi-Step DQN (`N_STEP = 3`)
* **Phương trình**:
  $$y_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_{a'} Q(S_{t+n}, a'; \theta^-)$$
* **Tác động**: Rút ngắn thời gian lan truyền phần thưởng về quá khứ theo hệ số $n$.
* **Cảnh báo toán học**: Khi $n > 1$, dữ liệu trong buffer là **Off-policy** (được sinh ra bởi $\pi_{old} \ne \pi_{current}$). Do đó, chuỗi hành động $a_{t+1}, \dots, a_{t+n-1}$ không phải do policy hiện tại chọn. Nếu chọn $n > 5$ mà không có trọng số quan trọng (Importance Sampling correction), mục tiêu $y_t^{(n)}$ sẽ bị chệch nghiêm trọng.

---

### 3.4 Week 4: Policy Gradient, REINFORCE & Advantage Actor-Critic (A2C)
Tham chiếu bài giảng: [`Week4/BAI_GIANG.md`](Week4/BAI_GIANG.md#105-phân-tích-chuyên-sâu-các-tham-số-và-cơ-chế-tác-động-lên-hàm-mất-mát-a2c)

#### 1. Trọng số Entropy $\beta$ (`ENTROPY_WEIGHT = 0.001` - `0.01`)
* **Hàm mất mát toàn cục**:
  $$\mathcal L_{total}(\theta, \phi) = \mathcal L_{PG}(\theta) + c_1 \mathcal L_V(\phi) - \beta \mathcal H(\pi_\theta)$$
* **Phương trình Entropy**:
  $$\mathcal H(\pi(\cdot \mid s)) = -\sum_{a \in \mathcal A} \pi(a \mid s) \ln \pi(a \mid s)$$
* **Đạo hàm điều khiển**:
  $$\nabla_\theta \left( -\beta \mathcal H \right) = \beta \sum_{a} \big( 1 + \ln \pi(a \mid s) \big) \nabla_\theta \pi(a \mid s)$$
* **Phân tích tác động**:
  * $\beta \to 0$: Khi một hành động tình cờ nhận được advantage dương ban đầu, $\pi(a \mid s)$ vọt lên $1.0$, xác suất các hành động khác tụt về $0.0$, gradient đạo hàm bằng 0. Xảy ra **sụp đổ chính sách sớm (Premature Policy Collapse)**.
  * $\beta \gg 1$: Entropy chiếm ưu thế tuyệt đối, policy không hội tụ mà duy trì phân phối đều (Random Uniform).
* **Lưu ý lỗi kinh điển trong code**: Trong [`Week4/A2C.ipynb`](Week4/A2C.ipynb), tác giả định nghĩa `entropy_loss = beta * (p * log_p).sum() = -\beta H`, sau đó viết `loss = pg + vl - entropy_loss` $\implies \mathcal L = \mathcal L_{PG} + c_1 \mathcal L_V + \beta \mathcal H$, dấu trừ biến thành dấu cộng làm gradient **triệt tiêu entropy nhanh gấp đôi** thay vì khuyến khích!

#### 2. Ngưỡng cắt xén chuẩn độ dốc $g_{\max}$ (`CLIP_GRAD = 0.5`)
* **Phương trình**:
  $$g \leftarrow g \cdot \min\left( 1, \frac{g_{\max}}{\|g\|_2 + 10^{-6}} \right)$$
* **Ý nghĩa toán học**: Trong Policy Gradient, khi policy vô tình bước vào trạng thái có $\pi(a \mid s) \to 0$ nhưng advantage cực lớn, số hạng $\frac{\hat{A}_t}{\pi(a \mid s)}$ làm $\|\nabla_\theta \mathcal L\|_2 \to \infty$. $g_{\max}$ chặn đứng sự nhảy vọt của trọng số ra khỏi miền tối ưu địa phương.

---

### 3.5 Week 5: Proximal Policy Optimization (PPO) & GAE
Tham chiếu bài giảng: [`Week5/BAI_GIANG.md`](Week5/BAI_GIANG.md#91-phân-tích-chuyên-sâu-các-tham-số-và-cơ-chế-tác-động-lên-phương-trình-ppo--gae)

#### 1. Biên độ cắt xén $\epsilon_{clip}$ (`CLIP_RANGE = 0.2`)
* **Phương trình PPO-Clip Surrogate Loss**:
  $$\mathcal L^{CLIP}(\theta) = \hat{\mathbb E}_t \left[ \min\left( r_t(\theta)\hat{A}_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
  với $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$.
* **Phân tích giải tích điều khiển**:
  * Khi $\hat{A}_t > 0$ (hành động tốt): Nếu $r_t(\theta) \ge 1+\epsilon \implies$ biểu thức bên trong bị kẹp tại $(1+\epsilon)\hat{A}_t$ (hằng số độc lập với $\theta$). Đạo hàm $\nabla_\theta \mathcal L = 0$. Thuật toán triệt tiêu gradient, không cho phép policy tăng xác suất hành động này thêm nữa trong đợt cập nhật hiện tại.
  * Khi $\hat{A}_t < 0$ (hành động tệ): Nếu $r_t(\theta) \le 1-\epsilon \implies$ biểu thức bị kẹp tại $(1-\epsilon)\hat{A}_t$, đạo hàm triệt tiêu về 0, bảo vệ policy không bị giảm quá sâu gây bất ổn.
* **Biên giới hạn**:
  * $\epsilon \to 0$: Mọi bước cập nhật đều bị kẹp chặt, policy không thay đổi được gì ($\theta \approx \theta_{old}$).
  * $\epsilon \ge 1$: Mất hoàn toàn tác dụng cắt xén, PPO thoái hóa thành Vanilla Policy Gradient không có Trust Region, dễ sụp đổ hiệu năng.

#### 2. Cặp tham số $\gamma - \lambda$ trong GAE (Generalized Advantage Estimator)
* **Phương trình**:
  $$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V,\qquad \delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
* **Bản chất suy giảm hàm mũ**: Hệ số suy giảm của sai số TD là $(\gamma \lambda)^l$.
* **Chân trời hiệu dụng của Advantage**:
  $$H_{\text{eff}}^{GAE} = \frac{1}{1 - \gamma \lambda}$$
  * Với $\gamma = 0.99, \lambda = 0.95 \implies \gamma \lambda \approx 0.9405 \implies H_{\text{eff}} \approx 16.8$ bước thời gian.
  * $\lambda = 0 \implies \hat{A}_t = \delta_t^V$ (Variance thấp nhất, Bias phụ thuộc hoàn toàn vào $V$).
  * $\lambda = 1 \implies \hat{A}_t = \sum_{l=0}^\infty \gamma^l R_{t+l+1} - V(S_t)$ (Unbiased Monte Carlo Advantage, Variance cực đại).

#### 3. Số Epoch tái sử dụng dữ liệu $K$ (`EPOCHS = 4` - `10`)
* Tận dụng tính chất Trust Region của hàm kẹp để tối ưu hóa nhiều lượt SGD trên cùng một tập rollout batch mà không sợ bị lệch phân phối quá đà.
* Nếu $K > 15$: Dữ liệu bị overfitting, policy trôi dạt ra xa khỏi $\pi_{old}$ bất chấp hàm clip, gây suy giảm hiệu năng sau các update cycle.

---

### 3.6 Week 6: Black-Box Optimization & Evolution Strategies (ES)
Tham chiếu bài giảng: [`Week6/BAI_GIANG.md`](Week6/BAI_GIANG.md#84-phân-tích-chuyên-sâu-các-tham-số-và-cơ-chế-tác-động-lên-phương-trình-es)

#### 1. Độ lệch chuẩn nhiễu tham số $\sigma$ (`NOISE_STD = 0.05` - `0.1`)
* **Phương trình xấp xỉ đạo hàm**:
  $$\nabla_\theta \mathbb E_{\epsilon \sim \mathcal N(0, I)}[F(\theta + \sigma \epsilon)] = \frac{1}{\sigma} \mathbb E_{\epsilon}\big[ F(\theta + \sigma \epsilon) \epsilon \big]$$
* **Khai triển Taylor**:
  $$\frac{F(\theta + \sigma \epsilon) - F(\theta - \sigma \epsilon)}{2\sigma} = \nabla_\theta F(\theta)^\top \epsilon + \mathcal O(\sigma^2 \|\nabla^3 F\|)$$
* **Đánh đổi giải tích**:
  * $\sigma \to 0$: Phần dư Taylor $\mathcal O(\sigma^2) \to 0$ (ước lượng đạo hàm không chệch), nhưng chia cho $\sigma \to 0$ ở mẫu số khuếch đại sai số số học (Numerical Cancellation / Floating Point Roundoff error) khiến gradient ước lượng dao động dữ dội.
  * $\sigma \gg 1$: Bề mặt hàm mục tiêu bị làm mờ quá mức, agent mất khả năng nắm bắt các đặc trưng hình học sắc nét của hàm mục tiêu, gradient chỉ phản ánh trung bình của vùng lân cận bán kính $\sigma$.

#### 2. Kích thước quần thể $n$ (`POPULATION_SIZE`) so với số chiều tham số $d$
* **Phương sai của ước lượng gradient**:
  $$\operatorname{Var}\big(\hat{g}_{ES}\big) \propto \frac{d}{n}$$
  trong đó $d$ là tổng số trọng số mạng nơ-ron (thường từ $10^4$ đến $10^7$).
* **Quy luật khả thi**: Để tín hiệu gradient vượt qua nhiễu (Signal-to-Noise Ratio $\text{SNR} > 1$), bắt buộc $n$ phải đủ lớn ($n \ge 100$ đến vài ngàn cá thể). Do đó, ES chỉ vượt trội khi có hạ tầng cụm máy tính hàng ngàn CPU để song song hóa việc đánh giá $F(\theta + \sigma \epsilon_i)$.

---

### 3.7 Week 7: Model-Based RL, Model Predictive Control (MPC) & DAgger
Tham chiếu bài giảng: [`Week7/BAI_GIANG.md`](Week7/BAI_GIANG.md#84-phân-tích-chuyên-sâu-các-tham-số-và-cơ-chế-tác-động-trong-model-based-rl--mpc)

#### 1. Tầm nhìn lập kế hoạch $H$ (`HORIZION_LENGTH = 10`)
* **Phương trình bài toán tối ưu quỹ đạo**:
  $$A^* = \arg\max_{a_{t:t+H-1}} \sum_{k=0}^{H-1} \hat{r}\big(\hat{s}_{t+k}, a_{t+k}\big),\qquad \hat{s}_{t+k+1} = \hat{s}_{t+k} + \hat{f}_\phi(\hat{s}_{t+k}, a_{t+k})$$
* **Cơ chế bùng nổ sai số (Compounding Error & Simulation Lemma)**:
  Giả sử sai số 1 bước của mô hình động học là $\varepsilon$, hệ số Lipschitz động học là $L_f = \|\nabla_s f\|$.
  $$\|\hat{s}_{t+H} - s_{t+H}\| \le \varepsilon \frac{L_f^H - 1}{L_f - 1} \approx \mathcal O(L_f^H \varepsilon)$$
  Sai số của hàm giá trị mục tiêu bị chặn bởi $\mathcal O(H^2 R_{\max} \varepsilon)$.
* **Đánh đổi**:
  * $H \to 1$: Myopic greedy, không nhìn thấy nguy cơ ngã hoặc lật xe trong vài bước tới.
  * $H > 20$: Sai số bùng nổ hàm mũ, mô hình bị "ảo giác" (hallucination) — tối ưu hóa trên thế giới tưởng tượng sai lệch, dẫn đến quyết định ngoài đời thực thảm họa.
  * Điểm cân bằng: $H = 10$ cho phép lập kế hoạch vừa đủ cho quán tính vật lý mà không để sai số kịp phân kỳ.

#### 2. Số chuỗi hành động ngẫu nhiên $K$ (`NUM_ACTIONS_SEQUENCES = 20000`)
* **Xác suất bao phủ không gian tối ưu**:
  Không gian chuỗi hành động liên tục có số chiều $D = H \times \dim(\mathcal A) = 10 \times 8 = 80$ chiều.
  Nếu miền chuỗi hành động tốt chiếm tỷ lệ thể tích $\rho$:
  $$P(\text{thành công}) \approx 1 - \exp(-K \rho)$$
* $K = 20000$ đảm bảo mật độ mẫu đủ dày đặc để bắt được các quỹ đạo bước đi hiệu quả của robot Ant mà vẫn chạy mượt mà trong vài mili-giây nhờ tính toán ma trận song song trên GPU.

#### 3. Nhiễu Gauss làm trơn đầu vào $\sigma_{\xi}$ (`scale = 0.001`)
* **Bản chất giải tích**: Tương đương Tikhonov Regularization phạt chuẩn Frobenius của ma trận Jacobi đạo hàm động học:
  $$\mathbb E_\xi \big[\|\hat{f}(X + \xi) - Y\|^2\big] \approx \|\hat{f}(X) - Y\|^2 + \sigma_\xi^2 \|\nabla_X \hat{f}(X)\|_F^2$$
  Giúp bề mặt hàm chuyển tiếp trơn tru, ngăn ngừa hiện tượng khuếch đại sai số khi cuộn mô hình nhiều bước.

---

### 3.8 Week 8: Advanced SOTA (SAC, Curiosity/RND, AlphaGo Zero, OpenAI Five, RLHF)
Tham chiếu bài giảng: [`Week8/BAI_GIANG.md`](Week8/BAI_GIANG.md#8-phân-tích-chuyên-sâu-các-tham-số-cốt-lõi-trong-các-thuật-toán-nâng-cao)

#### 1. Hệ số nhiệt độ $\alpha$ và Entropy mục tiêu $\bar{\mathcal H}$ trong SAC
* **Mục tiêu Soft Bellman**:
  $$y = r + \gamma \Big( \min_{j=1,2} Q_{\bar{\theta}_j}(s', \tilde{a}') - \alpha \log \pi_\phi(\tilde{a}' \mid s') \Big)$$
* **Tự động điều chỉnh $\alpha$ qua Dual Gradient**:
  $$\nabla_\alpha J(\alpha) = -\mathbb E_{a \sim \pi} \big[ \log \pi(a \mid s) + \bar{\mathcal H} \big]$$
* **Quy chuẩn entropy mục tiêu**: $\bar{\mathcal H} = -\dim(\mathcal A)$. Đảm bảo sau phép biến đổi co $a = \tanh(u)$, độ lệch chuẩn trước khi squashing duy trì quanh $\sigma \approx 1/e \approx 0.37$, giữ hành động không bị dính vào hai vách tường biên $\pm 1$.

#### 2. Trọng số phần thưởng tò mò $\beta$ trong Khám phá nâng cao (ICM / RND)
* **Phương trình**:
  $$r_t^{total} = r_t^{ext} + \beta\, \frac{r_t^{int}}{\hat{\sigma}_{int} + \epsilon}$$
* Bắt buộc chuẩn hóa running standard deviation $\hat{\sigma}_{int}$ để ngăn $r_t^{int}$ lấn át tín hiệu môi trường khi agent chuyển từ giai đoạn tìm tòi sang giai đoạn hoàn thành nhiệm vụ.

#### 3. Hệ số PUCT $c_{puct}$ và Nhiệt độ $\tau_{mcts}$ trong AlphaGo Zero
* **PUCT Selection**:
  $$a_t = \arg\max_a \left[ Q(s, a) + c_{puct} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$
  $c_{puct} \approx 1.25 - 2.5$ bảo đảm mạng nơ-ron dẫn đường hiệu quả mà không bỏ sót các nhánh chưa thử nghiệm.
* **Nhiệt độ $\tau_{mcts}$**: $\tau = 1$ ở 30 nước khai cuộc (đa dạng hóa thế cờ self-play), $\tau \to 0$ (argmax) ở các nước sau để thi đấu chính xác tuyệt đối.

#### 4. Hệ số Team Spirit $\tau_{team}$ trong OpenAI Five
* **Phương trình**:
  $$r_i = (1 - \tau_{team}) r_{ind, i} + \tau_{team} \bar{r}_{team}$$
* Curriculum tăng dần từ $0 \to 1$ giải quyết bài toán mâu thuẫn giữa Credit Assignment vi mô của từng hero và mục tiêu chiến thắng toàn cục của đội 5 người.

#### 5. Hệ số phạt phân kỳ $\beta_{KL}$ trong PPO-RLHF cho LLM
* **Phương trình**:
  $$\max_\theta \mathbb E \big[ r_\psi(x, y) - \beta_{KL} D_{KL}(\pi_\theta(y \mid x) \parallel \pi_{init}(y \mid x)) \big]$$
* Ngăn chặn hiện tượng **Reward Hacking** (mô hình lừa điểm Reward Model bằng các chuỗi từ lặp dị thường) trong khi vẫn tiếp thu tốt định hướng an toàn của con người.

---

## 4. Ma trận độ nhạy & Các cặp đánh đổi toán học (Trade-off Matrix)

| Tham số | Khi TĂNG ($\uparrow$) | Khi GIẢM ($\downarrow$) | Cặp đánh đổi cốt lõi (Trade-off) |
|---|---|---|---|
| **$\gamma$ (Discount factor)** | Tầm nhìn dài hạn $\uparrow$, Variance của $G_t$ $\uparrow$, Hội tụ chậm $\downarrow$ | Myopic ngắn hạn $\uparrow$, Variance $\downarrow$, Hội tụ nhanh $\uparrow$ | **Long-term foresight vs. Sample variance** |
| **$\alpha$ (Learning rate)** | Học nhanh $\uparrow$, Nguy cơ nổ loss & phân kỳ $\uparrow$ | Học ổn định $\uparrow$, Nguy cơ kẹt yên ngựa & học chậm $\downarrow$ | **Convergence speed vs. Optimization stability** |
| **$\epsilon$ / $\tau_{boltzmann}$** | Khám phá rộng $\uparrow$, Hiệu năng thu thập tức thời $\downarrow$ | Khai thác sâu $\uparrow$, Nguy cơ kẹt cực tiểu địa phương $\uparrow$ | **Exploration vs. Exploitation** |
| **$\lambda$ (GAE / TD($\lambda$))** | Giảm Bias $\downarrow$, Tăng Variance $\uparrow$ | Giảm Variance $\downarrow$, Tăng Bias $\uparrow$ (phụ thuộc vào $V$) | **Bias vs. Variance** |
| **$C$ (Target interval)** | Hàm mục tiêu ổn định $\uparrow$, Lan truyền TD chậm $\downarrow$ | Lan truyền TD nhanh $\uparrow$, Dao động mục tiêu di động $\uparrow$ | **Target stability vs. Information propagation** |
| **$B$ (Batch size)** | Gradient ước lượng chính xác $\uparrow$, Chi phí tính toán $\uparrow$ | Khám phá nhờ nhiễu SGD $\uparrow$, Gradient dao động mạnh $\uparrow$ | **Gradient accuracy vs. Regularization noise** |
| **$\beta$ (Entropy weight)** | Ngăn sụp đổ policy $\uparrow$, Kéo dài thời gian hội tụ $\downarrow$ | Tăng tốc hội tụ về chính sách thuần $\uparrow$, Dễ kẹt local optima $\uparrow$ | **Policy diversity vs. Convergence sharpness** |
| **$H$ (Planning horizon)** | Tầm nhìn MPC xa $\uparrow$, Compounding error $\mathcal O(H^2\varepsilon)$ nổ $\uparrow$ | Ít sai số mô hình $\uparrow$, Thiển cận ngắn hạn $\uparrow$ | **Model foresight vs. Compounding hallucination** |

---

## 5. Cẩm nang chẩn đoán lỗi lâm sàng trong Deep RL (Debugging Guide by Symptoms)

Khi huấn luyện mô hình RL gặp trục trặc, hãy đối chiếu các triệu chứng lâm sàng dưới đây để xác định chính xác tham số gây lỗi:

```
                          [SƠ ĐỒ CHẨN ĐOÁN LỖI LÂM SÀNG]
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   [Loss nổ NaN/Inf]          [Return không tăng]          [Sụp đổ sau đỉnh cao]
           │                            │                            │
  ┌────────┴────────┐          ┌────────┴────────┐          ┌────────┴────────┐
  ▼                 ▼          ▼                 ▼          ▼                 ▼
Learning Rate     Thiếu      Zero              Reward     Target Update     Buffer quá nhỏ
quá cao        Gradient Clip Exploration    quá thưa     quá dồn dập       hoặc Policy Clip
(α > 1e-2)     (g_clip=None) (ε=0, β=0)     (Cần RND)    (C quá nhỏ, τ→1)  quá lỏng (ε>0.3)
```

### Triệu chứng 1: Loss nổ ra $NaN$ hoặc $\pm\infty$ ngay từ các epoch đầu
* **Nguyên nhân chính**:
  1. `learning_rate` ($\alpha$) quá lớn ($> 10^{-2}$ với mạng sâu).
  2. Không có `clip_grad_norm` dẫn đến một bước cập nhật gradient khổng lồ thổi bay trọng số ra khỏi miền số học khả dĩ.
  3. Giá trị $Q$ hoặc Advantage chưa được chuẩn hóa, gặp episode có return cực lớn làm sai số MSE nổ bậc hai.
* **Phác đồ điều trị**:
  - Hạ `learning_rate` xuống $10^{-4}$ hoặc $3\cdot 10^{-4}$ (mức chuẩn của Adam trong RL).
  - Bổ sung `torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)`.
  - Kiểm tra hàm chia trong PPO: Luôn cộng $\epsilon_{num} = 10^{-8}$ vào mẫu số: $\hat{A} = (A - \mu) / (\sigma + 10^{-8})$.

### Triệu chứng 2: Return kẹt vĩnh viễn ở mức ngẫu nhiên (Agent không học được gì)
* **Nguyên nhân chính**:
  1. Thiếu cơ chế khám phá: $\epsilon$ trong DQN giảm quá nhanh về 0 trước khi agent tình cờ chạm vào reward đầu tiên.
  2. Trong Policy Gradient/PPO: Entropy bonus $\beta = 0$, policy lập tức sụp đổ thành phân phối tất định chọn duy nhất 1 hành động sai.
  3. Môi trường có Sparse Reward (phần thưởng cực thưa như mê cung, Montezuma): Agent ngẫu nhiên không bao giờ chạm tới đích.
* **Phác đồ điều trị**:
  - Kéo dài lịch trình suy giảm $\epsilon$ trong DQN (ví dụ: giảm tuyến tính qua $100,000$ bước thay vì $1,000$ bước).
  - Tăng hệ số entropy $\beta$ lên $0.01$ hoặc $0.05$.
  - Tích hợp Intrinsic Reward (ICM hoặc RND - Week 8) với trọng số $\beta_{int} \approx 1.0$.

### Triệu chứng 3: Đường cong học tập đạt đỉnh cao rồi bất ngờ lao dốc thảm hại (Performance Collapse)
* **Nguyên nhân chính**:
  1. **Moving Target Collapse**: Trong DQN, $C$ quá nhỏ hoặc trong SAC Polyak $\tau$ quá lớn, mạng Target thay đổi quá nhanh tạo ra phản hồi dương làm bùng nổ giá trị $Q$.
  2. **Catastrophic Forgetting**: Kích thước Replay Buffer $N$ quá nhỏ, mạng quên sạch các kỹ năng ở giai đoạn đầu sau khi phân phối trạng thái dịch chuyển.
  3. **PPO Step quá lỏng**: $\epsilon_{clip}$ quá lớn ($> 0.3$) kết hợp số epoch $K > 15$ làm policy mới văng ra quá xa khỏi Trust Region.
* **Phác đồ điều trị**:
  - Tăng chu kỳ Target Update $C$ lên ít nhất $1000$ bước (hoặc giảm Polyak $\tau$ xuống $0.005$).
  - Nâng dung lượng Replay Buffer lên tối thiểu $100,000$ transitions.
  - Khóa chặt PPO: Đặt $\epsilon_{clip} = 0.1$ hoặc $0.2$, giới hạn $K \le 4$ epoch.

### Triệu chứng 4: Agent "run rẩy / co giật tại chỗ" hoặc chỉ lặp lại một chuỗi hành động ngớ ngẩn
* **Nguyên nhân chính**:
  1. Trong Model-Based RL (Week 7): Tầm nhìn $H$ quá dài kết hợp mô hình động học có sai số lớn $\implies$ Agent tối ưu trên ảo giác (Model Hallucination).
  2. Entropy bonus $\beta$ quá lớn khiến agent chỉ tối đa hóa tính hỗn loạn của hành động thay vì tối đa hóa reward.
* **Phác đồ điều trị**:
  - Rút ngắn horizon $H$ từ $20 \to 10$.
  - Thêm các vòng Data Aggregation (DAgger) để mô hình học chính xác vùng phân phối mà agent đang thực sự đi vào.
  - Hạ trọng số entropy $\beta$ xuống bậc $10^{-3}$.

---

## 6. Quy tắc vàng thực chiến khi tinh chỉnh (Rules of Thumb)

1. **Quy tắc Andrej Karpathy cho RL**: Trước khi chạy mạng nơ-ron phức tạp, hãy luôn kiểm tra thuật toán trên môi trường đồ chơi đơn giản (CartPole, CliffWalking) và đảm bảo agent giải được trong vòng 2 phút.
2. **Quy tắc Khởi tạo Learning Rate**: Với hầu hết các thuật toán Deep RL (DQN, A2C, PPO, SAC), `learning_rate = 3e-4` với bộ tối ưu **Adam** là điểm khởi đầu an toàn và mạnh mẽ nhất trên 90% các bài toán benchmark.
3. **Quy tắc Vectorized Environments**: Trong A2C và PPO, luôn ưu tiên tăng số lượng worker môi trường song song (`N_ENVS = 8` đến `32`) thay vì chỉ tăng kích thước batch, vì các worker song song thu thập các mẫu hoàn toàn độc lập theo thời gian thực, triệt tiêu phương sai hiệu quả nhất.
4. **Quy tắc Chuẩn hóa Quan sát và Reward**:
   - Luôn chuẩn hóa trạng thái đầu vào về trung bình 0, phương sai 1 (Running Mean-Std Filter).
   - Không chuẩn hóa reward bằng cách trừ trung bình trong bài toán vô hạn (infinite horizon) vì sẽ làm biến đổi bản chất của bài toán Markov. Thay vào đó, hãy chia reward cho độ lệch chuẩn trôi: $r \leftarrow r / \sigma_r$.
5. **Quy tắc Kiểm tra Tính Chắc chắn (Ablation Study)**: Không bao giờ thay đổi đồng thời 2 siêu tham số trong một lần thử nghiệm. Luôn cố định tất cả các tham số khác và chỉ quét qua một dải giá trị đơn lẻ (Grid/Random sweep) với ít nhất 3 đến 5 hạt giống ngẫu nhiên (random seeds) độc lập để loại trừ may rủi thống kê.

---
*Cẩm nang này là tài liệu đồng hành xuyên suốt toàn bộ giáo trình [Tuần 1 đến Tuần 8](./README.md).*
