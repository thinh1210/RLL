# Kiến thức chi tiết theo từng phần của repo (kèm đối chiếu code)

Tài liệu này bổ sung cho [KIEN_THUC_CAN_HOC.md](KIEN_THUC_CAN_HOC.md). Mỗi tuần gồm: **lý thuyết cốt lõi → ánh xạ vào code → lỗi/điểm cần lưu ý đã kiểm tra → bài tập → cập nhật cho thư viện mới**.

> Các nhận xét về lỗi bên dưới là kết quả đọc code, chưa chạy thực nghiệm (repo viết cho Gym cũ). Chúng được đánh dấu **[Lỗi]** (sai thật sự), **[Lưu ý]** (khác chuẩn/giảm chất lượng) hoặc **[Cũ]** (API lỗi thời).

## Mục lục
0. [Ký hiệu và công thức tổng hợp](#0-ký-hiệu-và-công-thức-tổng-hợp)
1. [Week 1 – Giới thiệu](#1-week-1--giới-thiệu)
2. [Week 2 – Q-learning tabular](#2-week-2--q-learning-tabular)
3. [Week 3 – DQN và biến thể](#3-week-3--dqn-và-biến-thể)
4. [Week 4 – Policy Gradient và A2C](#4-week-4--policy-gradient-và-a2c)
5. [Week 5 – PPO](#5-week-5--ppo)
6. [Week 6 – Evolution Strategies](#6-week-6--evolution-strategies)
7. [Week 7 – Model-Based RL](#7-week-7--model-based-rl)
8. [Week 8 – Chủ đề nâng cao](#8-week-8--chủ-đề-nâng-cao)
9. [Bảng tổng hợp lỗi và cách sửa](#9-bảng-tổng-hợp-lỗi-và-cách-sửa)
10. [Chuyển sang thư viện hiện đại](#10-chuyển-sang-thư-viện-hiện-đại)
11. [Cách tự kiểm tra mức độ hiểu](#11-cách-tự-kiểm-tra-mức-độ-hiểu)

---

## 0. Ký hiệu và công thức tổng hợp

| Ký hiệu | Ý nghĩa |
|---|---|
| $s, a, r$ | trạng thái, hành động, phần thưởng |
| $\pi_\theta(a\mid s)$ | policy tham số hoá bởi $\theta$ |
| $G_t=\sum_{k\ge 0}\gamma^k r_{t+k}$ | return chiết khấu |
| $V^\pi(s)=\mathbb E_\pi[G_t\mid s]$ | hàm giá trị trạng thái |
| $Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s,a]$ | hàm giá trị hành động |
| $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ | advantage: hành động tốt hơn trung bình bao nhiêu |

**Các công thức xuyên suốt**

- Bellman expectation: $V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\,[r+\gamma V^\pi(s')]$
- Bellman optimality: $Q^*(s,a)=\mathbb E[r+\gamma\max_{a'}Q^*(s',a')]$
- TD error: $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$
- Policy gradient: $\nabla_\theta J=\mathbb E[\nabla_\theta\log\pi_\theta(a|s)\,\Psi_t]$, với $\Psi_t\in\{G_t,\ G_t-b(s),\ A_t,\ \delta_t\}$ (càng về sau càng ít phương sai, càng nhiều bias).
- GAE: $\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$

**Phân loại thuật toán trong repo**

| Tuần | Thuật toán | Value-based? | Policy-based? | On/Off-policy | Hành động |
|---|---|---|---|---|---|
| 2 | Q-learning | có | ngầm (greedy) | off | rời rạc |
| 3 | DQN (+Double/Dueling/Noisy/N-step) | có | ngầm | off | rời rạc |
| 4 | REINFORCE, A2C | critic | có | on | rời rạc |
| 5 | PPO | critic | có | on (tái dùng ít epoch) | liên tục |
| 6 | ES | không | trực tiếp trên tham số | – (hộp đen) | liên tục |
| 7 | MB-MF (shooting MPC) | model | planning | – | liên tục |

---

## 1. Week 1 – Giới thiệu

Repo không có code cho tuần này; nội dung là bài đọc/video.

**Kiến thức cần nắm**
- Vòng lặp tương tác: agent quan sát $s_t$ → chọn $a_t$ → môi trường trả $r_t, s_{t+1}$.
- Vì sao RL khác học có giám sát: dữ liệu do chính agent sinh ra (không i.i.d.), nhãn đúng không có sẵn, phần thưởng trễ (*credit assignment*), cần cân bằng khám phá/khai thác.
- *Pong from Pixels* (Karpathy): policy gradient tối giản bằng NumPy, hiểu "tăng log-prob của hành động dẫn đến thắng".
- Phân loại: model-free/model-based, value-based/policy-based, on/off-policy.

**Bài tập đề xuất**
1. Cài Gymnasium, chạy `CartPole-v1` với agent ngẫu nhiên, đo reward trung bình (~20).
2. Viết agent luật cứng (`action = obs[2] > 0`) để thấy baseline ~40–50.
3. Đọc Sutton & Barto chương 1 và trả lời: vì sao tic-tac-toe học được bằng TD mà không cần mô hình đối thủ?

---

## 2. Week 2 – Q-learning tabular

File: [`Week2/frozenlake_Qlearning.ipynb`](Week2/frozenlake_Qlearning.ipynb)

### 2.1 Lý thuyết
Q-learning cập nhật bảng Q bằng mục tiêu TD off-policy:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\,]$$

- **Off-policy**: hành động chọn ε-greedy để khám phá, nhưng mục tiêu dùng $\max$ (policy greedy). SARSA thay $\max$ bằng $Q(s',a')$ với $a'$ thực sự được chọn (on-policy).
- **Hội tụ** về $Q^*$ nếu mọi cặp (s,a) được thăm vô hạn lần và $\alpha$ giảm đúng lịch (trong thực tế dùng $\alpha$ hằng).
- **ε-greedy + decay**: $\epsilon\leftarrow\epsilon\cdot 0.9993$ sau mỗi episode (xem ô "MAIN PROCEDURE"); sau 15000 episode $\epsilon\approx 3\times10^{-5}$.

### 2.2 Ánh xạ vào notebook
| Thành phần | Hàm/biến |
|---|---|
| Bảng Q | `table = defaultdict(float)` khoá `(state, action)` |
| Chọn hành động | `select_eps_greedy_action`, `select_greedy_action` |
| Cập nhật | `Q_learning(table, obs0, obs1, reward, action)` |
| Siêu tham số | `GAMMA=0.95`, `LEARNING_RATE=0.8`, `EPS_DECAY_RATE=0.9993` |
| Đánh giá | `test_game` (greedy, 100 episode) |

`FrozenLake-v0` mặc định **trơn trượt** (`is_slippery=True`): hành động chỉ đi đúng hướng với xác suất 1/3, nên môi trường ngẫu nhiên; vì vậy dùng $\alpha$ lớn 0.8 vẫn học được nhưng nhiễu. Reward chỉ bằng 1 khi tới đích, còn lại 0 (reward thưa).

### 2.3 Điểm cần lưu ý
- **[Lưu ý]** `best_action_value` khởi tạo `max_value = 0` và chỉ chọn khi `Q > max_value`. Q âm bị bỏ qua và khi mọi Q bằng 0 luôn trả action 0. Với FrozenLake reward không âm nên vẫn chạy, nhưng sai nếu môi trường có reward âm (ví dụ Taxi). Cách đúng: `max(range(n), key=lambda a: table[(s,a)])`.
- **[Lưu ý]** Không xử lý terminal tường minh: target đúng ở trạng thái kết thúc là $r$ (không cộng $\gamma\max Q$). Ở đây ổn vì bảng Q của trạng thái kết thúc không bao giờ được cập nhật nên luôn bằng 0.
- **[Lưu ý]** Bài toán chỉ giải được cho không gian trạng thái rời rạc, nhỏ. Với không gian liên tục phải lượng tử hoá hoặc dùng mạng nơ-ron (Week 3).

### 2.4 Bài tập
1. Đổi sang SARSA (chọn $a'$ ε-greedy trước khi cập nhật), so sánh đường học.
2. Thử `is_slippery=False`, chứng minh Q hội tụ trong vài trăm episode.
3. Vẽ heatmap $\max_a Q(s,a)$ trên lưới 4×4 và quan sát giá trị lan ngược từ đích.
4. Thử $\alpha\in\{0.05,0.2,0.8\}$ và $\gamma\in\{0.9,0.99\}$.

---

## 3. Week 3 – DQN và biến thể

Các file: [`main.py`](Week3/main.py) · [`agent.py`](Week3/agent.py) · [`central_control.py`](Week3/central_control.py) · [`neural_net.py`](Week3/neural_net.py) · [`buffers.py`](Week3/buffers.py) · [`atari_wrappers.py`](Week3/atari_wrappers.py)

### 3.1 Lý thuyết
**Vì sao cần DQN**: với ảnh Atari không thể dùng bảng Q, ta xấp xỉ $Q(s,a;\theta)$ bằng CNN. Kết hợp *function approximation + bootstrapping + off-policy* ("deadly triad") dễ phân kỳ, nên DQN thêm hai thủ thuật:

1. **Experience replay**: lưu $(s,a,r,s',done)$, lấy mẫu ngẫu nhiên để phá tương quan thời gian và tái dùng dữ liệu.
2. **Target network** $\theta^-$ (cập nhật chậm): mục tiêu $y=r+\gamma\max_{a'}Q(s',a';\theta^-)$ không "chạy trốn" theo mạng đang học. Loss: $L=\mathbb E[(y-Q(s,a;\theta))^2]$.

**Các biến thể (gói trong Rainbow)**

| Biến thể | Ý tưởng | Công thức/điểm chính |
|---|---|---|
| Double DQN | Tránh đánh giá quá cao do $\max$ trên ước lượng nhiễu | $y=r+\gamma\,Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$ (chọn bằng mạng học, đánh giá bằng mạng đích) |
| Dueling | Tách $V(s)$ và advantage | $Q=V+A-\overline{A}$ (trừ trung bình để định danh được) |
| Multi-step (n-step) | Cân bằng bias/variance | $y=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^nQ(s_{t+n},\cdot)$ |
| Noisy Nets | Khám phá bằng nhiễu có tham số học | $w=\mu+\sigma\odot\varepsilon$ thay ε-greedy |
| (Chưa cài) PER, Distributional | Ưu tiên mẫu có TD error lớn; học phân phối return | – |

### 3.2 Ánh xạ vào code
| Khái niệm | Vị trí |
|---|---|
| Tiền xử lý Atari: lặp hành động 4 frame và lấy max của 2 frame cuối, FireReset, resize 84×84 xám, stack 4 frame, scale [0,1] | [`atari_wrappers.py`](Week3/atari_wrappers.py#L160-L169) (`MaxAndSkipEnv`, `FireResetEnv`, `WarpFrame`, `ImageToPyTorch`, `FrameStack`, `ScaledFloatFrame`) |
| CNN theo Nature-DQN | `DQN` trong [`neural_net.py`](Week3/neural_net.py#L105-L149) |
| Hình dạng chuẩn của Nature-DQN: 4×84×84 → conv8/4 → 20×20 → conv4/2 → 9×9 → conv3/1 → 7×7×64 = **3136** → FC 512 → số hành động. (Xem lưu ý 8 bên dưới: code này thực tế khác chuẩn.) | `_get_conv_out` |
| Dueling | `DuelingDQN` ([`neural_net.py`](Week3/neural_net.py#L58-L102)) |
| ε giảm tuyến tính 1.0→0.02 qua $10^5$ bước | [`agent.py`](Week3/agent.py#L94) |
| N-step lookahead, dừng khi gặp `done` | [`buffers.py`](Week3/buffers.py#L24-L64) |
| Loss, Double DQN, Bellman | [`central_control.py`](Week3/central_control.py#L70-L105) |
| Vòng lặp: act → step → lưu → sample → tối ưu | [`main.py`](Week3/main.py#L70-L94) |

Siêu tham số mặc định: `gamma=0.99`, `lr=5e-5`, `batch=32`, `buffer_capacity=15000` (bài báo gốc dùng 1.000.000), `target update=1000` bước, `n_multi_step=2`.

### 3.3 Lỗi và điểm cần lưu ý (quan trọng)
1. **[Lỗi] Mạng đích bị gộp với mạng học.** [`update_target`](Week3/central_control.py#L50-L55) chạy `load_state_dict(...)` rồi gán `self.target_nn = self.moving_nn`. Sau lần cập nhật đầu tiên hai biến trỏ cùng một object, nên mục tiêu thay đổi theo mạng đang học và mất tác dụng ổn định hoá. Sửa: chỉ giữ dòng `load_state_dict`, bỏ dòng gán.
2. **[Lỗi] Không che trạng thái kết thúc.** [`done_t`](Week3/central_control.py#L83) được tạo nhưng không dùng. Công thức đúng:
   `expected = rewards + gamma**n * next_state_values * (1 - done)`. Nếu thiếu, giá trị của trạng thái kết thúc bị bootstrap sai.
3. **[Lỗi đã được tác giả ghi chú] NoisyLinear không hoạt động** ([`neural_net.py`](Week3/neural_net.py#L13)): epsilon sinh bằng `.data.normal_()` tại chỗ và chưa dùng factorised noise của bài báo. Ngoài ra khi `noisy_net=True` agent luôn greedy, dù nhiễu chưa được đặt/đánh giá đúng cách.
4. **[Lưu ý]** Có `BatchNorm2d` (không có trong bài báo) và mạng đích không chuyển `eval()`, nên thống kê BN lệch giữa lúc chọn hành động (batch=1) và lúc học (batch=32).
5. **[Lưu ý]** `np.random.choice(..., replace=False)` trên deque và lookahead qua `self.buffer[i+n]` (deque truy cập giữa chừng chậm, $O(n)$). Với buffer lớn nên dùng mảng NumPy vòng.
6. **[Cũ]** `namedtuple(..., verbose=False)` bị xoá từ Python 3.7; `gym.wrappers.Monitor` bị xoá khỏi Gym mới.
7. **[Lưu ý]** `MAX_N_GAMES=3000` và `buffer=15000` nhỏ, nhưng README cho biết ~120 game đã học tốt với ε giảm nhanh.
8. **[Lưu ý] Thứ tự wrapper làm 4 frame ghép ngang thay vì xếp theo kênh.** Trong [`make_env`](Week3/atari_wrappers.py#L160-L169), `ImageToPyTorch` (đổi sang CxHxW = 1×84×84) chạy **trước** `FrameStack`, mà `LazyFrames` nối theo `axis=2` (chiều rộng). Kết quả suy ra từ code: quan sát là 1×84×336 (4 frame cạnh nhau, 1 kênh) và `input_shape[0]=1`, chứ không phải 4×84×84. Mạng vẫn chạy (đầu ra conv khoảng 64×7×38) nhưng mất cấu trúc "4 kênh thời gian" của bài báo. Chuẩn của baselines là `WarpFrame → FrameStack → ImageToPyTorch`. Đây là suy luận từ đọc code, chưa chạy kiểm chứng: hãy `print(env.reset().shape)` để xác nhận.

### 3.4 Bài tập
1. Sửa lỗi 1 và 2, so sánh đường reward trước/sau (kỳ vọng ổn định hơn).
2. Bật từng biến thể (Double → Dueling → N-step) và vẽ so sánh trên cùng seed.
3. Tự cài Prioritized Experience Replay (sum-tree) hoặc sửa NoisyLinear theo factorised Gaussian.
4. Đo Q trung bình theo thời gian để phát hiện overestimation (Double DQN nên thấp hơn).

---

## 4. Week 4 – Policy Gradient và A2C

Các file: [`PolicyGradient.ipynb`](Week4/PolicyGradient.ipynb) · [`A2C.ipynb`](Week4/A2C.ipynb)

### 4.1 Lý thuyết
**Mục tiêu**: tối đa hoá $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0]$. Định lý policy gradient:

$$\nabla_\theta J=\mathbb E\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,\Psi_t\Big]$$

- **REINFORCE**: $\Psi_t=G_t$ (Monte Carlo, không bias, **phương sai cao**).
- **Baseline** $b(s)$: trừ đi giá trị không phụ thuộc hành động để giảm phương sai mà không thêm bias. Notebook dùng baseline là trung bình return trong deque (không phụ thuộc state).
- **Actor-Critic**: critic $V_\phi(s)$ ước lượng baseline, $\Psi_t=\hat A_t=R_t-V_\phi(s_t)$ (n-step). **A2C** chạy nhiều môi trường đồng bộ để có batch đa dạng.
- **Entropy bonus** $H(\pi)=-\sum_a\pi\log\pi$: thêm $+\beta H$ vào mục tiêu để chống policy suy biến quá sớm.
- Loss tổng của A2C (cực tiểu hoá): $L=-\log\pi(a|s)\hat A+c_v\,(R-V)^2-\beta H$.

### 4.2 Ánh xạ vào code
| Khái niệm | PolicyGradient | A2C |
|---|---|---|
| Mạng | `PG_nn` (policy 64 ẩn) | `A2C_nn` chia sẻ thân, 2 đầu policy/value |
| Return | `discounted_rewards` | `Env.discounted_rewards` (bootstrap bằng $V(s_{last})$ nếu chưa kết thúc) |
| Lấy mẫu hành động | softmax + `np.random.choice` | như vậy, `N_ENVS=40` môi trường |
| Ổn định | baseline trung bình | `clip_grad_norm_(…, 0.1)`, Adam `eps=1e-3` |

### 4.3 Lỗi tìm thấy (giải thích hiện tượng "Bug Alert" của tác giả)
Tác giả ghi README rằng "A2C cho kết quả lạ" (loss giảm rất nhanh, agent luôn chọn một hành động). Đối chiếu code cho thấy **ít nhất hai lỗi** có thể giải thích điều này:

1. **[Lỗi] Lấy log-prob sai chiều.**
   ```python
   act_log_softmax = F.log_softmax(log_val[0], dim=1)[:, np.array([m.action for m in memories])]
   ```
   Với `[:, idx]` kết quả có shape `[N, N]` (mọi hàng × mọi hành động đã chọn) thay vì `[N]`. Khi nhân với `adv`/`disc_rewards`, phép broadcast ghép sai return của bước $j$ với log-prob của bước $t$. Dùng `gather`:
   ```python
   logp = F.log_softmax(logits, dim=1).gather(1, actions[:, None]).squeeze(1)
   ```
   Lỗi tương tự có ở `PolicyGradient.ipynb` (`game_act_log_softmax_t`). Ở A2C, `adv = rewards - value` còn có shape `[N,1]-[N]`→`[N,N]` vì `value` có chiều thừa. Cần `value.squeeze(-1)`.
2. **[Lỗi] Sai dấu entropy trong A2C.** `entropy_loss = β·mean(Σ p log p) = −β·H`, rồi `loss = pg + vl − entropy_loss = pg + vl + βH`. Cực tiểu hoá loss lúc này **làm giảm entropy**, tức đẩy policy về tất định (phù hợp với hiện tượng luôn chọn một hành động). Đúng phải là `loss = pg + vl - β·H` (tức `+ entropy_loss`). Notebook PolicyGradient dùng `loss + l_entropy` (đúng dấu).
3. **[Lưu ý]** Critic và actor dùng chung thân và chung learning rate; `CLIP_GRAD=0.1` rất nhỏ nên giới hạn cập nhật; `N_ENVS=40` và `n_steps=1` làm advantage chỉ là TD(0) một bước.
4. **[Lưu ý]** PG: baseline lấy trung bình toàn bộ deque nên bị lệch khi policy thay đổi; chuẩn hoá advantage (chia std) thường ổn định hơn.
5. **[Cũ]** `F.softmax(pol_val[0])` không có `dim`; `CartPole-v0` giới hạn 200 bước, còn `v1` là 500.

### 4.4 Bài tập
1. Sửa lỗi 1 và 2 rồi chạy lại A2C; CartPole phải đạt ~195+ (v0).
2. Vẽ entropy theo thời gian trước/sau khi sửa dấu.
3. Cài A3C (đa tiến trình) hoặc GAE cho A2C.
4. Giải thích bằng toán: vì sao trừ baseline không đổi kỳ vọng gradient ($\mathbb E[\nabla\log\pi\cdot b(s)]=0$).

---

## 5. Week 5 – PPO

File: [`Week5/PPO.py`](Week5/PPO.py) (BipedalWalker, hành động liên tục)

### 5.1 Lý thuyết
**Vấn đề**: bước cập nhật policy gradient quá lớn có thể làm tụt hiệu năng không phục hồi được. TRPO ràng buộc $KL(\pi_{old}\|\pi_\theta)\le\delta$ (cần đạo hàm bậc hai/conjugate gradient). **PPO** xấp xỉ bằng cách cắt tỉ lệ xác suất:

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)},\qquad L^{CLIP}=\mathbb E\big[\min(r_t\hat A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)\big]$$

- Tỉ lệ này là **importance sampling**, cho phép dùng lại cùng một batch dữ liệu cho nhiều epoch.
- **GAE** cân bằng bias/variance bằng $\lambda$ ($\lambda=0$: TD(0); $\lambda=1$: Monte Carlo).
- **Policy Gaussian** cho hành động liên tục: $a\sim\mathcal N(\mu_\theta(s),\sigma^2)$ với $\log\sigma$ là tham số học `logstd`.
- Mục tiêu critic: $V\to\hat A+V_{old}$ (λ-return).

### 5.2 Ánh xạ vào code
| Khái niệm | Vị trí |
|---|---|
| Policy (mean tanh) và `logstd` độc lập state | `A2C_policy` ([PPO.py](Week5/PPO.py#L16-L39)) |
| Critic riêng | `A2C_value` |
| Thu thập quỹ đạo `TRAJECTORY_SIZE=2049` | `Env.steps` |
| GAE ngược thời gian | `generalized_advantage_estimation` ([L131-L149](Week5/PPO.py#L131-L149)) |
| Log-prob Gaussian | `log_policy_prob` ([L152-L155](Week5/PPO.py#L152-L155)) |
| Loss cắt + value loss | `clipped_PPO_loss` ([L169-L187](Week5/PPO.py#L169-L187)) |
| 7 epoch, minibatch 64, Adam lr policy 4e-4, value 1e-3, ε=0.2, γ=0.99, λ=0.95 | vòng lặp chính |
| Chuẩn hoá advantage | [L292-L295](Week5/PPO.py#L292-L295) |

### 5.3 Lỗi và điểm cần lưu ý
1. **[Lỗi/Lưu ý] Log-prob không khớp phân phối lấy mẫu.** Lấy mẫu dùng độ lệch chuẩn $\sigma=e^{\text{logstd}}$ ([L97](Week5/PPO.py#L97)), nhưng `log_policy_prob` coi $e^{\text{logstd}}$ là **phương sai** ($\sigma^2$). Tỉ lệ $r_t$ vẫn nhất quán giữa old/new nên thuật toán chạy, nhưng đó không phải log-likelihood thật, làm ý nghĩa của ε-clip và gradient của `logstd` sai lệch. Chuẩn: $\log\pi=-\frac{(a-\mu)^2}{2\sigma^2}-\log\sigma-\frac12\log 2\pi$ (hoặc dùng `torch.distributions.Normal`).
2. **[Lưu ý] Mất tín hiệu terminal.** Khi `done`, reward bị ghi đè bằng 0 ([L107-L109](Week5/PPO.py#L107-L109)). Trong BipedalWalker, ngã cho −100, nên tín hiệu "ngã là xấu" bị bỏ qua, chỉ còn giá trị V ngầm. Hơn nữa không phân biệt `terminated` với `truncated` (hết giờ cần bootstrap).
3. **[Lưu ý] Bước cuối bị bỏ và không bootstrap.** Vòng GAE chạy `range(len-1)` và khởi tạo `run_add=0` ở cuối quỹ đạo, tức bỏ bước cuối và bỏ $V(s_T)$, gây bias nhẹ. Cần thêm `last_value` để bootstrap.
4. **[Lưu ý] Minibatch không xáo trộn**: cắt tuần tự `batch[mb:mb+64]` nên các mẫu liên tiếp tương quan. Nên `np.random.permutation` mỗi epoch.
5. **[Lưu ý]** `CLIP_GRADIENT=0.2` được khai báo và `clip_grad_norm_` được import nhưng **không được dùng**. Không có entropy bonus. `var_l` định nghĩa mà không dùng.
6. **[Lưu ý]** `compute_log_policy_prob` dùng biến toàn cục `agent_policy` thay vì tham số `nn_policy`.
7. **[Cũ]** `BipedalWalker-v2`, `Monitor`, `F.tanh` đã lỗi thời. Giá trị thành công: 300 điểm trung bình 100 episode.

### 5.4 Bài tập
1. Sửa log-prob bằng `torch.distributions.Normal(mean, exp(logstd))` và so sánh.
2. Thêm xáo trộn minibatch, entropy bonus, và bootstrap giá trị cuối.
3. Vẽ phân phối $r_t$ qua các epoch để thấy clip hoạt động; theo dõi "clip fraction" và KL xấp xỉ.
4. Thử giá trị ε∈{0.1,0.2,0.3} và `PPO_EPOCHS`∈{3,7,15}.

---

## 6. Week 6 – Evolution Strategies

File: [`Week6/ES.py`](Week6/ES.py) (LunarLanderContinuous)

### 6.1 Lý thuyết
ES là tối ưu hộp đen không cần backprop qua môi trường. Xấp xỉ gradient của hàm kỳ vọng làm trơn:

$$\nabla_\theta\,\mathbb E_{\varepsilon\sim\mathcal N(0,I)}[F(\theta+\sigma\varepsilon)]\approx\frac{1}{n\sigma}\sum_{i=1}^n F(\theta+\sigma\varepsilon_i)\,\varepsilon_i$$

Các kỹ thuật trong bài báo OpenAI (Salimans et al., 2017) được cài:
- **Mirrored (antithetic) sampling**: đánh giá cả $+\varepsilon$ và $-\varepsilon$ để giảm phương sai.
- **Chuẩn hoá theo hạng (fitness shaping)**: thay reward bằng thứ hạng đã chuẩn hoá về $[-0.5,0.5]$ để chống outlier và bất biến theo tỉ lệ reward.
- **Common random numbers**: worker chỉ gửi về *seed* và reward; máy chủ tái tạo nhiễu từ seed, nên truyền thông chỉ vài byte/worker (cho phép mở rộng lên hàng nghìn CPU).

Ưu điểm: song song hoá cực tốt, không cần credit assignment theo thời gian, bền với reward thưa. Nhược điểm: cần nhiều mẫu hơn RL gradient (10–100×).

### 6.2 Ánh xạ vào code
| Khái niệm | Vị trí |
|---|---|
| Mạng chính sách tất định (tanh) | `NeuralNetwork` ([ES.py](Week6/ES.py#L17-L40)) |
| Nhiễu cho mọi tham số | `sample_noise` |
| Đánh giá tác nhân bị nhiễu rồi khôi phục tham số | `evaluate_noisy_net` |
| Worker + mirrored | `worker` ([L92-L120](Week6/ES.py#L92-L120)) |
| Rank chuẩn hoá | `normalized_rank` ([L123-L130](Week6/ES.py#L123-L130)) |
| Cập nhật: $\sum r\cdot\varepsilon$ đưa vào `p.grad` (dấu âm) rồi Adam | [L210-L225](Week6/ES.py#L210-L225) |
| Siêu tham số: σ=0.05, 100 cặp/lần lặp, lr=0.01, 4 worker | đầu file |

### 6.3 Lưu ý
- **[Lưu ý]** Mỗi lần lặp đánh giá $2\times100=200$ episode (mỗi worker chạy $+\varepsilon$ và $-\varepsilon$), nên 650 lần lặp ≈ 130.000 episode, không phải 65.000 như README ghi.
- **[Lưu ý]** Phép chia `BATCH_SIZE*STD_NOISE` chỉ là hằng số tỉ lệ (có 2·BATCH_SIZE mẫu), được Adam hấp thụ phần lớn. Dùng Adam trên gradient ES là lựa chọn của bài báo.
- **[Lưu ý]** `var_l` và `logstd` bị nhiễu hoá dù không ảnh hưởng đầu ra (lãng phí chiều tìm kiếm).
- **[Lưu ý]** `np.array(nn_noise)` tạo mảng không đều (dtype object); chạy được ở NumPy cũ, NumPy ≥1.24 sẽ báo lỗi. Dùng `list`.
- **[Lưu ý]** Truyền `state_dict()` qua Queue mỗi lần là tốn kém; ES chuẩn truyền tham số một lần mỗi vòng cho mỗi worker.

### 6.4 Bài tập
1. Vẽ so sánh có/không rank-normalization và có/không mirrored sampling.
2. Thử σ∈{0.02,0.05,0.1}: σ quá nhỏ làm gradient nhiễu, quá lớn làm xấp xỉ sai.
3. Đọc *Deep Neuroevolution* (GA) và cài GA đơn giản (chọn lọc + đột biến) trên cùng mạng.

---

## 7. Week 7 – Model-Based RL

File: [`Week7/model_based.py`](Week7/model_based.py) (RoboschoolAnt, theo Nagabandi et al. 2018)

### 7.1 Lý thuyết
Model-free học thẳng $\pi$ hoặc $Q$ từ tương tác. Model-based học mô hình động học $\hat f_\phi(s,a)\approx s'$ (và reward $\hat r_\phi(s,a)$) rồi **lập kế hoạch** trên mô hình đó, nên hiệu quả mẫu cao hơn nhiều.

Quy trình trong code (pseudocode ở [`Week7/imgs/pseudocode.png`](Week7/imgs/pseudocode.png)):
1. Thu thập dữ liệu ngẫu nhiên $\mathcal D_{rand}$ (1000 quỹ đạo).
2. Huấn luyện MLP học **hiệu** $\Delta s=s'-s$ (dễ học hơn $s'$) và MLP reward, với dữ liệu đã chuẩn hoá (StandardScaler) và thêm nhiễu Gauss nhỏ.
3. **MPC bằng random shooting**: tại mỗi bước, lấy 20000 chuỗi hành động ngẫu nhiên dài $H=10$, mô phỏng bằng mô hình, chọn chuỗi có tổng reward dự đoán lớn nhất và **chỉ thực thi hành động đầu tiên**, rồi lập kế hoạch lại.
4. **Data aggregation** (giống DAgger): thêm dữ liệu từ controller vào tập huấn luyện, huấn luyện lại (3 vòng × 20000 bước).

Hạn chế: sai số mô hình tích luỹ theo horizon (*compounding error*); random shooting không tối ưu cho chiều hành động lớn; mô hình MLP tất định khó diễn tả tính ngẫu nhiên. Cải tiến: ensemble mô hình (PETS), CEM/MPPI thay cho random shooting, fine-tune bằng model-free (đoạn "MF" của MB-MF; **repo chỉ cài phần model-based**).

### 7.2 Ánh xạ vào code
| Khái niệm | Vị trí |
|---|---|
| Hai mô hình | `NNDynamicModel`, `NNRewardModel` ([model_based.py](Week7/model_based.py#L26-L65)) |
| Thu dữ liệu ngẫu nhiên | `gather_random_trajectories` |
| Huấn luyện có chuẩn hoá và validation | `train_dyna_model` |
| MPC random shooting (batch hoá 20000 chuỗi) | `multi_model_based_control` ([L252-L305](Week7/model_based.py#L252-L305)) |
| Vòng aggregation | [L358-L411](Week7/model_based.py#L358-L411) |

### 7.3 Lưu ý
- **[Lưu ý]** [L180-L184](Week7/model_based.py#L180-L184) dùng biến toàn cục `BATCH_SIZE` thay vì tham số `batch_size` (cùng giá trị 512 nên chưa lỗi).
- **[Lưu ý]** `X_mb += noise` là phép cộng tại chỗ trên một *view* của `X_train`, nên nhiễu bị cộng dồn vĩnh viễn vào dữ liệu qua các epoch (độ lệch chuẩn tích luỹ ≈ 0.001·√55). Nên dùng `X_mb = X_mb + noise`. Comment ghi "variance 0.0001" nhưng `scale=0.001` là độ lệch chuẩn.
- **[Lưu ý]** Vòng cuối `while num_examples_added < STEPS_PER_AGGR` kết thúc theo episode nên vượt 20000 bước một chút; reward model không dự đoán việc kết thúc episode.
- **[Lưu ý]** `device='cuda'` cố định, 20000 chuỗi × 10 bước mỗi lần hành động → chậm (cần GPU).
- **[Cũ]** Roboschool đã ngừng phát triển; thay bằng `Ant-v4/v5` của MuJoCo (Gymnasium).

### 7.4 Bài tập
1. Vẽ lỗi dự đoán $\|\hat s_{t+k}-s_{t+k}\|$ theo $k$ để thấy compounding error.
2. Thay random shooting bằng CEM (lặp: lấy mẫu → giữ top-k → cập nhật phân phối).
3. So sánh số mẫu môi trường cần để đạt cùng reward giữa MB và PPO (Week 5).

---

## 8. Week 8 – Chủ đề nâng cao

Không có code trong repo; đây là các chủ đề để chọn đồ án.

| Chủ đề | Ý chính | Gợi ý đọc |
|---|---|---|
| Inference & Control (MaxEnt RL) | Coi điều khiển tối ưu như suy luận xác suất; tối đa hoá reward + entropy | Soft Q-learning, SAC |
| Inverse RL / Imitation | Suy ra reward từ chuyên gia; DAgger sửa lệch phân phối của behavior cloning | GAIL, DAgger |
| Exploration | Reward thưa cần khám phá có chủ đích: count-based, curiosity (ICM), RND | Bellemare 2016; Pathak 2017 |
| Transfer, Multi-task, Meta-RL | Tái dùng kỹ năng; học cách học | MAML, RL² |
| Offline RL | Học từ dữ liệu cố định, không tương tác | CQL, IQL |
| Trò chơi cổ điển | AlphaGo Zero = MCTS + self-play + mạng policy/value | David Silver "Classic Games" |
| Multi-agent | Pommerman, Dota 2 (OpenAI Five dùng PPO quy mô lớn) | MADDPG, MAPPO |

**Đề xuất đồ án theo độ khó**: (dễ) sửa các lỗi trong bảng mục 9 và chứng minh bằng thực nghiệm; (vừa) chuyển DQN/PPO sang Gymnasium với CleanRL làm tham chiếu; (khó) SAC hoặc PETS trên MuJoCo.

---

## 9. Bảng tổng hợp lỗi và cách sửa

| # | Vị trí | Loại | Vấn đề | Cách sửa |
|---|---|---|---|---|
| 1 | [central_control.py L54-L55](Week3/central_control.py#L50-L55) | Lỗi | Target net bị gán bằng moving net | Bỏ dòng `self.target_nn = self.moving_nn` |
| 2 | [central_control.py L83, L103](Week3/central_control.py#L83) | Lỗi | `done` không che bootstrap | Nhân `(1-done)` vào số hạng bootstrap |
| 3 | [neural_net.py L9-L55](Week3/neural_net.py#L9-L55) | Lỗi | NoisyLinear sai (in-place, không factorised) | Cài theo bài báo, dùng `register_buffer` và cập nhật ngoài graph |
| 4 | A2C.ipynb `calculate_loss` | Lỗi | `[:, actions]` → `[N,N]`, `adv` broadcast `[N,N]` | `gather(1, a[:,None]).squeeze(1)`; `value.squeeze(-1)` |
| 5 | A2C.ipynb `calculate_loss` | Lỗi | Dấu entropy đảo ngược | `loss = pg + vl + entropy_loss` (với `entropy_loss=β·Σp log p`) |
| 6 | PolicyGradient.ipynb | Lỗi | Cùng lỗi indexing như #4 | Dùng `gather` |
| 7 | [PPO.py L152-L155](Week5/PPO.py#L152-L155) | Lưu ý | σ vs σ² không khớp giữa sampling và log-prob | Dùng `Normal(mean, exp(logstd))` |
| 8 | [PPO.py L107-L111, L138](Week5/PPO.py#L107-L111) | Lưu ý | Reward terminal bị xoá; không bootstrap bước cuối | Giữ reward thật, truyền `last_value`, tách terminated/truncated |
| 9 | [PPO.py L304](Week5/PPO.py#L304) | Lưu ý | Minibatch không xáo trộn | Hoán vị chỉ số mỗi epoch |
| 10 | [ES.py L51](Week6/ES.py#L43-L51) | Lưu ý | `np.array` ragged | Trả về `list` |
| 11 | [model_based.py L187](Week7/model_based.py#L187) | Lưu ý | Nhiễu cộng dồn tại chỗ vào `X_train` | `X_mb = X_mb + noise` |
| 12 | Week2 `best_action_value` | Lưu ý | `max_value=0` bỏ qua Q âm | Dùng `max` với `key` |

---

## 10. Chuyển sang thư viện hiện đại

| Cũ trong repo | Mới |
|---|---|
| `import gym` | `import gymnasium as gym` |
| `obs = env.reset()` | `obs, info = env.reset(seed=…)` |
| `obs, r, done, info = env.step(a)` | `obs, r, terminated, truncated, info = env.step(a)`; `done = terminated or truncated` |
| `gym.wrappers.Monitor` | `gym.wrappers.RecordVideo` |
| `FrozenLake-v0`, `CartPole-v0` | `FrozenLake-v1`, `CartPole-v1` |
| `BipedalWalker-v2`, `LunarLanderContinuous-v2` | `BipedalWalker-v3`, `LunarLander-v3` với `continuous=True` |
| `PongNoFrameskip-v4` | cần `ale-py` (ROM đi kèm); `gym.wrappers.AtariPreprocessing` + `FrameStackObservation` thay wrapper tự viết |
| `roboschool` | `mujoco` (`Ant-v4/v5`) |
| `namedtuple(verbose=…)` | bỏ tham số `verbose` |
| `F.tanh`, `F.softmax(x)` không `dim` | `torch.tanh`, `F.softmax(x, dim=-1)` |
| `tensorboardX` | `torch.utils.tensorboard.SummaryWriter` |

Quan trọng khi chuyển: với `truncated=True` (hết thời gian), **vẫn phải bootstrap** $V(s')$; chỉ `terminated=True` mới đặt mục tiêu bằng $r$. Đây là nguồn bug phổ biến nhất của code RL cũ.

---

## 11. Cách tự kiểm tra mức độ hiểu

Nếu trả lời được những câu hỏi sau mà không nhìn tài liệu, bạn đã nắm phần đó:

- **Week 2**: Vì sao Q-learning là off-policy còn SARSA là on-policy? Điều gì xảy ra nếu ε=0 ngay từ đầu?
- **Week 3**: Vì sao cần target network? Double DQN sửa vấn đề gì? Vì sao replay buffer làm DQN không dùng được cho on-policy?
- **Week 4**: Vì sao trừ baseline không làm lệch gradient? Entropy bonus có dấu như thế nào trong loss?
- **Week 5**: Điều gì xảy ra khi $r_t>1+\epsilon$ và $\hat A_t>0$? Còn khi $\hat A_t<0$? Vai trò của λ trong GAE?
- **Week 6**: Vì sao ES chỉ cần gửi seed + reward? Rank normalization giải quyết vấn đề gì?
- **Week 7**: Vì sao chỉ thực thi hành động đầu của chuỗi tối ưu? Compounding error là gì và giảm nó thế nào?
- **Tổng quát**: khác biệt giữa `terminated` và `truncated` ảnh hưởng thế nào đến mục tiêu TD?
