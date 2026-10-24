# Kiến thức cần học để theo repo Deep Reinforcement Learning

Tài liệu này tổng hợp các kiến thức cần thiết, mức độ ưu tiên và thời gian ước lượng (cho người đã biết lập trình, học ~1–2 giờ/ngày).

## 0. Tổng quan bản đồ kiến thức

```
Toán (XS-TK, giải tích, ĐSTT)
        │
Python + NumPy ──► PyTorch ──► Deep Learning (MLP, CNN)
        │                              │
        └────────► Lý thuyết RL ◄──────┘
                      │
   MDP → DP → MC/TD → Q-learning → DQN → Policy Gradient → A2C → TRPO/PPO → ES → Model-Based
```

| Nhóm | Mức độ cần | Ước lượng thời gian |
|---|---|---|
| Toán nền tảng | Bắt buộc (mức vừa) | 2–3 tuần |
| Python + NumPy | Bắt buộc | 1 tuần (nếu đã biết thì bỏ qua) |
| Deep Learning + PyTorch | Bắt buộc | 3–4 tuần |
| Lý thuyết RL cổ điển | Cốt lõi | 2–3 tuần |
| Deep RL (DQN, PG, A2C, PPO) | Cốt lõi | 4–5 tuần |
| Chủ đề nâng cao (ES, Model-Based, ...) | Mở rộng | 2–3 tuần |

Tổng: ~3–5 tháng nếu học từ con số 0 về DL; ~8 tuần (đúng lộ trình repo) nếu đã vững Python + DL.

---

## 1. Toán nền tảng

> 📖 **Tài liệu Ôn tập Chuyên sâu & Thực nghiệm trên Mẫu:**
> Xem trọn bộ giải thích trực quan, bản chất logic và cách từng toán tử tác động lên đám mây mẫu tại **[TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.md](TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.md)** kèm kịch bản chạy thực nghiệm số học **[minh_hoa_toan_hoc_tren_mau.py](minh_hoa_toan_hoc_tren_mau.py)**.

### 1.1 Xác suất & thống kê (quan trọng nhất)
- **Biến ngẫu nhiên, phân phối**: rời rạc (Bernoulli, categorical), liên tục (Gaussian). RL dùng categorical cho hành động rời rạc và Gaussian cho hành động liên tục (xem `Week5/PPO.py`).
- **Kỳ vọng, phương sai**: mọi hàm giá trị đều là kỳ vọng của tổng phần thưởng. Phương sai cao là vấn đề chính của Policy Gradient.
- **Xác suất có điều kiện, công thức Bayes**.
- **Chuỗi Markov**: tính chất "tương lai chỉ phụ thuộc hiện tại".
- **Lấy mẫu Monte Carlo**: ước lượng kỳ vọng bằng trung bình mẫu.
- **Importance sampling**: dùng trong off-policy và PPO (tỉ lệ `π_new / π_old`).
- **Entropy, KL divergence**: dùng trong TRPO/PPO để giới hạn mức thay đổi policy.

### 1.2 Giải tích & tối ưu hóa
- Đạo hàm riêng, **gradient**, **chain rule** (nền tảng của backpropagation).
- **Gradient descent / ascent**, SGD, Adam, learning rate.
- **Log-derivative trick**: `∇θ π(a|s) = π(a|s) ∇θ log π(a|s)` — nền tảng của Policy Gradient.
- Tối ưu có ràng buộc (mức khái niệm) cho TRPO.

### 1.3 Đại số tuyến tính
- Vector, ma trận, nhân ma trận, chuẩn (norm).
- Cần ở mức đọc hiểu code và shape tensor; phần ma trận Fisher/Hessian của TRPO chỉ cần biết khái niệm.

---

## 2. Lập trình

### 2.1 Python
- OOP (class Agent, kế thừa `nn.Module`), `collections.deque` (replay buffer), list/dict comprehension, generator.
- **NumPy**: broadcasting, indexing, `np.random`, thao tác trên mảng nhiều chiều.
- **Matplotlib** để vẽ đường reward.

### 2.2 PyTorch (xem `Week3/neural_net.py`, `Week5/PPO.py`)
- Tensor, `requires_grad`, autograd.
- `nn.Module`, `nn.Linear`, `nn.Conv2d`, hàm kích hoạt.
- Vòng lặp huấn luyện: `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`.
- `torch.distributions` (Categorical, Normal), `detach()`, `no_grad()`, chuyển CPU/GPU.
- Lưu/nạp model, **TensorBoard** để theo dõi.

### 2.3 Gym / Gymnasium
```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```
Repo viết cho Gym cũ (`done` thay cho `terminated/truncated`), cần chỉnh nhẹ khi dùng Gymnasium. Cần hiểu `observation_space`, `action_space`, wrappers (xem `Week3/atari_wrappers.py`).

---

## 3. Deep Learning cần thiết
- **MLP**: dùng làm policy/value network cho CartPole, LunarLander, BipedalWalker.
- **CNN**: bắt buộc cho Week 3 (Pong từ ảnh): conv, stride, pooling, frame stacking.
- Hiểu **overfitting, normalization, khởi tạo trọng số, gradient clipping**.
- RNN: không bắt buộc cho repo này.
- Kỹ thuật ổn định: target network, batch normalization/observation normalization.

---

## 4. Lý thuyết Reinforcement Learning

### 4.1 Khái niệm cốt lõi (Week 1–2)
| Thuật ngữ | Ý nghĩa |
|---|---|
| Agent / Environment | Bộ ra quyết định / thế giới tương tác |
| State `s`, Action `a`, Reward `r` | Trạng thái, hành động, phần thưởng |
| Policy `π(a\|s)` | Chiến lược chọn hành động |
| Return `G_t` | Tổng phần thưởng chiết khấu `Σ γ^k r_{t+k}` |
| Value `V(s)`, `Q(s,a)` | Kỳ vọng return từ state / cặp (state, action) |
| Discount `γ` | Mức coi trọng tương lai |
| Exploration vs Exploitation | Thử cái mới vs dùng cái đã biết (ε-greedy) |

### 4.2 MDP và phương trình Bellman
- MDP = (S, A, P, R, γ).
- **Bellman expectation**: `V(s) = E[r + γ V(s')]`.
- **Bellman optimality**: `Q*(s,a) = E[r + γ max_a' Q*(s',a')]`.
- Đây là "công thức trung tâm" — hầu hết thuật toán RL là cách xấp xỉ nó.

### 4.3 Quy hoạch động (cần biết mô hình môi trường)
- Policy evaluation, Policy iteration, Value iteration.

### 4.4 Model-free tabular
- **Monte Carlo**: học từ episode hoàn chỉnh (variance cao, bias thấp).
- **TD(0), TD(λ)**: bootstrap, cập nhật từng bước (bias, variance thấp hơn).
- **SARSA** (on-policy) vs **Q-learning** (off-policy):
  - SARSA: `Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]`
  - Q-learning: `Q(s,a) += α [r + γ max Q(s',·) − Q(s,a)]`
- Khái niệm **on-policy / off-policy**, **bias–variance tradeoff**.

---

## 5. Deep RL theo từng thuật toán

### 5.1 DQN (Week 3)
- Thay bảng Q bằng mạng nơ-ron.
- **Experience Replay** (`buffers.py`): phá tương quan dữ liệu.
- **Target network**: ổn định mục tiêu học.
- Mở rộng (gói trong Rainbow): **Double DQN** (giảm overestimation), **Dueling** (tách V và Advantage), **Multi-step**, **Noisy Nets** (khám phá bằng nhiễu có học), Prioritized Replay, Distributional RL.
- "Deadly triad": function approximation + bootstrapping + off-policy ⇒ dễ mất ổn định.

### 5.2 Policy Gradient & Actor-Critic (Week 4)
- Mục tiêu: `J(θ) = E[Σ r]`, gradient `E[∇ log π(a|s) · G_t]` (REINFORCE).
- **Baseline** để giảm phương sai; **Advantage** `A = Q − V`.
- **Actor-Critic**: actor cập nhật policy, critic ước lượng V. A2C (đồng bộ), A3C (bất đồng bộ).
- Policy gradient hợp với **hành động liên tục**, nhưng kém sample-efficient.

### 5.3 TRPO & PPO (Week 5)
- Vấn đề: bước cập nhật quá lớn làm policy sụp đổ.
- **TRPO**: ràng buộc KL giữa policy cũ/mới (tính toán nặng).
- **PPO**: hàm mục tiêu cắt (clipped):
  `L = E[min(r_t A_t, clip(r_t, 1−ε, 1+ε) A_t)]`, với `r_t = π_new/π_old`.
- **GAE** (Generalized Advantage Estimation), nhiều epoch minibatch trên cùng dữ liệu.
- Hiện là thuật toán mặc định thực tế (kể cả RLHF cho LLM).

### 5.4 Evolution Strategies (Week 6)
- Tối ưu hộp đen: thêm nhiễu Gaussian vào tham số, đánh giá, cập nhật theo reward.
- Không cần backprop, song song hoá tuyệt vời; ưu điểm: không phụ thuộc reward thưa/trễ; nhược: tốn nhiều mẫu.
- Mirrored sampling, rank normalization, common random numbers.

### 5.5 Model-Based RL (Week 7)
- Học mô hình `s' = f(s, a)` và reward bằng NN.
- **Planning / MPC** (Model Predictive Control): mô phỏng chuỗi hành động ngẫu nhiên, chọn chuỗi tốt nhất.
- Khởi tạo bằng model-based rồi tinh chỉnh bằng model-free (MB-MF).
- Ưu: sample-efficient. Nhược: lỗi mô hình tích luỹ.

### 5.6 Nâng cao (Week 8, tuỳ chọn)
Inverse RL, imitation learning (DAgger), exploration (curiosity), transfer/meta-learning, multi-agent, AlphaGo Zero (MCTS + self-play).

---

## 6. Kỹ năng thực hành (thường bị bỏ qua nhưng rất quan trọng)
- **Debug RL**: reward có tăng không? loss, entropy, giá trị Q có phát nổ không?
- **Tuning hyperparameter**: learning rate, γ, batch size, kích thước buffer, ε-decay.
- **Reproducibility**: đặt seed, chạy nhiều seed vì RL nhiễu cao.
- **Logging**: TensorBoard / Weights & Biases.
- **Reward shaping** và hiểu hạn chế của reward thưa.

---

## 7. Lộ trình gợi ý và thời gian

| Giai đoạn | Nội dung | Thời gian | Kết quả |
|---|---|---|---|
| A | Toán (XS, gradient) + Python/NumPy | 2–3 tuần | Đọc được công thức, thao tác mảng |
| B | PyTorch + MLP/CNN, huấn luyện MNIST | 3–4 tuần | Tự viết vòng lặp train |
| C | RL cơ bản: MDP, Bellman, MC/TD, Q-learning | 2–3 tuần | Giải FrozenLake (Week 2) |
| D | DQN + biến thể | 1–2 tuần | Chạy Pong (Week 3) |
| E | Policy Gradient, A2C | 1–2 tuần | CartPole (Week 4) |
| F | PPO | 1–2 tuần | BipedalWalker (Week 5) |
| G | ES, Model-Based | 1–2 tuần | Week 6–7 |
| H | Đồ án riêng | 2–4 tuần | Dự án hoàn chỉnh |

Nếu đã biết DL cơ bản: bắt đầu từ giai đoạn C (khoảng 2 tháng).

---

## 8. Tài liệu đề xuất (theo thứ tự ưu tiên)
1. **David Silver – RL Course** (YouTube, DeepMind): lý thuyết nền tảng.
2. **Sutton & Barto – Reinforcement Learning: An Introduction** (miễn phí bản PDF): chương 1–6, 9–10, 13.
3. **CS285 – Sergey Levine (UC Berkeley)**: Deep RL hiện đại.
4. **OpenAI Spinning Up**: giải thích thuật toán ngắn gọn kèm code.
5. **Hugging Face Deep RL Course** và **CleanRL**: cài đặt hiện đại, dễ đọc.
6. Bài báo gốc: DQN (2013/2015), Rainbow, PPO (2017), ES (2017).

## 9. Lưu ý về tính cũ của repo
Repo viết ~2018–2019 (Gym cũ, Roboschool, PyTorch thế hệ đầu). Khi chạy hiện nay nên:
- Dùng **Gymnasium** thay `gym`, và `MuJoCo` thay `Roboschool` (đã ngừng phát triển).
- Cập nhật API `reset()/step()` (5 giá trị trả về).
- Dùng môi trường ảo (`venv`/`conda`) với phiên bản package tương thích.
