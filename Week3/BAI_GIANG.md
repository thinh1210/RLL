# Week 3 – Giá trị xấp xỉ bằng mạng nơ-ron: DQN và các biến thể

> Mục tiêu: hiểu **vì sao** Q-learning với mạng nơ-ron không ổn định, DQN giải quyết bằng cách nào (bằng toán và chứng minh giải tích), từng biến thể (Double, Dueling, N-step, Noisy) sửa điểm yếu gì, và đọc hiểu từng file trong thư mục này.  
> 📖 **Quy ước ký hiệu toàn khoá**: Đối chiếu toàn bộ ký hiệu giải tích, không gian và toán tử tại [BẢNG KÝ HIỆU TOÁN HỌC CHUẨN TẮC](../BANG_KY_HIEU_TOAN_HOC.md).

**Lộ trình**: §1 Từ bảng Q đến mạng Q → §2 Hàm mất mát và Chứng minh Semi-gradient → §3 Mất ổn định (Deadly Triad) → §4 Replay + Target Network → §5 Double DQN & Chứng minh Maximization Bias → §6 Dueling DQN & Chứng minh tính khả định danh → §7 N-step Learning → §8 Noisy Networks → §9 Các cải tiến khác (PER, Distributional, Rainbow) → §10 Tiền xử lý Atari và kiến trúc CNN → §11 Đọc code & Phân tích siêu tham số → §12 Ví dụ số → §13 Bài tập.

Các file: [`main.py`](main.py) · [`agent.py`](agent.py) · [`central_control.py`](central_control.py) · [`neural_net.py`](neural_net.py) · [`buffers.py`](buffers.py) · [`atari_wrappers.py`](atari_wrappers.py) · [`utils.py`](utils.py)

---

## 1. Từ bảng Q đến mạng Q

Week 2: $Q(s,a)$ là **bảng**. Với Pong, $s$ là ảnh $84\times84\times4$ ($\approx 28\,000$ điểm ảnh, mỗi điểm 256 mức) → số trạng thái không đếm được. Ta xấp xỉ bằng hàm tham số $Q(s,a;\theta)$:

$$Q(s,a;\theta)\approx Q^*(s,a)$$

* **Giải thích ký hiệu**:
  * $s$: Vector hoặc tensor quan sát trạng thái đầu vào (ví dụ ảnh $84 \times 84 \times 4$).
  * $a \in \mathcal A$: Hành động rời rạc khả dĩ.
  * $\theta \in \mathbb R^d$: Vector trọng số của mạng nơ-ron Q-Network.
  * $Q^*(s, a)$: Giá trị hành động tối ưu toàn cục thực sự.

**Kiến trúc đầu ra** (một lần forward cho mọi hành động): $s\mapsto[Q(s,a_1),\dots,Q(s,a_{|\mathcal A|})]$. Thay vì đưa $(s,a)$ vào mạng và gọi $|\mathcal A|$ lần. Chọn hành động tốt nhất là một `argmax` trên vector đầu ra. Điều này chỉ khả thi với **hành động rời rạc** (Pong: 6 hành động) — với hành động liên tục cần DDPG/actor-critic (Week 4–5).

---

## 2. Hàm mất mát và Chứng minh Semi-gradient từ MSBE

Trong lý thuyết xấp xỉ hàm, sai số Bellman bình phương trung bình (Mean Squared Bellman Error - MSBE) được định nghĩa là:
$$\operatorname{MSBE}(\theta) = \mathbb E_{(s, a) \sim \rho}\left[ \Big( Q(s, a; \theta) - (T^* Q)(s, a; \theta) \Big)^2 \right]$$
trong đó $(T^* Q)(s, a; \theta) = \mathbb E_{s'} \big[ R(s, a) + \gamma \max_{a'} Q(s', a'; \theta) \big]$.

* **Vấn đề toán học của Gradient Descent thuần túy trên MSBE**:
  Lấy đạo hàm đúng của MSBE theo $\theta$:
  $$\nabla_\theta \operatorname{MSBE}(\theta) = 2 \mathbb E_{(s, a)}\left[ \Big( Q(s, a; \theta) - (T^* Q)(s, a; \theta) \Big) \Big( \nabla_\theta Q(s, a; \theta) - \gamma \nabla_\theta (T^* Q)(s, a; \theta) \Big) \right]$$
  Số hạng $\nabla_\theta (T^* Q)$ chứa đạo hàm của kỳ vọng:
  $$\nabla_\theta (T^* Q)(s, a; \theta) = \gamma \sum_{s'} P(s' \mid s, a) \nabla_\theta \max_{a'} Q(s', a'; \theta)$$
  Để tính ước lượng không chệch của tích hai kỳ vọng này bằng lấy mẫu ngẫu nhiên (SGD), ta cần **hai mẫu độc lập** $s'_1, s'_2$ từ cùng một cặp $(s, a)$ (**Double Sampling Problem**). Điều này bất khả thi trong môi trường mô phỏng không cho phép can thiệp đặt lại trạng thái tuỳ ý (model-free).

* **Giải pháp Semi-Gradient và Target Network trong DQN**:
  DQN tách rời trọng số ở hai vế: dùng bộ trọng số riêng $\theta^-$ (Target Network) được đóng băng tạm thời để tính mục tiêu Bellman $y$:
  $$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$
  Khi đó mục tiêu $y$ trở thành **hằng số vô hướng độc lập với $\theta$** tại thời điểm tính toán:
  $$L(\theta) = \mathbb E_{(s, a, r, s') \sim \mathcal D}\left[ \big( y - Q(s, a; \theta) \big)^2 \right]$$

  > 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
  > * **Thay thế thành phần nào**: Thay thế việc lấy đạo hàm toàn phần (Full Gradient) trên hàm sai số Bellman bình phương trung bình MSBE.
  > * **Khuyết điểm của phương pháp cũ**: Đạo hàm đúng của MSBE chứa số hạng tích của hai kỳ vọng độc lập theo $s'$. Để ước lượng không chệch bằng SGD, ta bắt buộc phải lấy 2 mẫu kế tiếp độc lập $s'_1, s'_2$ từ cùng một trạng thái $s$ và hành động $a$ (**Double Sampling Problem**), điều bất khả thi trong môi trường tương tác thực tế (Model-Free).
  > * **Logic thiết kế toán học**: Đóng băng mục tiêu Bellman thành hằng số $y$, bỏ qua đạo hàm $\nabla_\theta y = 0$. Thao tác này biến bài toán từ tối ưu MSBE phức tạp thành bài toán **Hồi quy bình phương tối thiểu Semi-Gradient** tiêu chuẩn.
  > * **Hiện thực hóa trong code**: Thao tác `.detach()` ngắt đồ thị tính toán: `next_state_values.detach()` ([`central_control.py` L100](central_control.py#L100)).

* **Đạo hàm Semi-Gradient**:
  Vì $y$ không phụ thuộc $\theta$ (coi $\nabla_\theta y = 0$):
  $$\boxed{\nabla_\theta L(\theta) = -2 \, \mathbb E_{(s, a, r, s') \sim \mathcal D}\Big[ \big( y - Q(s, a; \theta) \big) \, \nabla_\theta Q(s, a; \theta) \Big]}$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Thế Mục tiêu Đóng băng $y(\theta^-)$ và Lấy Đạo hàm Hàm hợp:**
> - **Bước 1 (Hàm mất mát gốc khi chưa tách mạng Target):**
>   Nếu dùng cùng một mạng $\theta$ cho cả hai vế:
>   $$L_{\text{naive}}(\theta) = \mathbb{E}\Big[\big(r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta)\big)^2\Big]$$
>   Khi lấy đạo hàm $\nabla_\theta$, quy tắc chuỗi sẽ sinh ra $\gamma \nabla_\theta \max_{a'} Q(s', a'; \theta) - \nabla_\theta Q(s, a; \theta)$ (vừa kéo $Q(s,a)$ lên vừa đẩy $Q(s',a')$ lệch đi $\to$ mất ổn định).
> - **Bước 2 (Phép thế Target Network — Thay $\theta$ ở phần mục tiêu bằng bản sao đóng băng $\theta^-$):**
>   Đặt hàm mục tiêu $y(\theta^-) = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ và **thế $y$ vào hàm mất mát**:
>   $$L(\theta) = \mathbb{E}\Big[\big(\underbrace{y(\theta^-)}_{\text{Hằng số theo }\theta} - Q(s, a; \theta)\big)^2\Big]$$
> - **Bước 3 (Thế quy tắc đạo hàm hàm hợp $\nabla_\theta (u^2) = 2u \nabla_\theta u$ với $u(\theta) = y - Q(s,a;\theta)$):**
>   Ta tính đạo hàm của $u(\theta)$ theo $\theta$:
>   $$\nabla_\theta u(\theta) = \underbrace{\nabla_\theta y(\theta^-)}_{= \mathbf{0}\text{ (do }\theta^-\text{ cố định)}} - \nabla_\theta Q(s, a; \theta) = -\nabla_\theta Q(s, a; \theta)$$
>   Đem thế $u(\theta) = y - Q(s,a;\theta)$ và $\nabla_\theta u(\theta) = -\nabla_\theta Q(s,a;\theta)$ vào $\nabla_\theta (u^2) = 2u\nabla_\theta u$:
>   $$\nabla_\theta L(\theta) = \mathbb{E}\Big[2\big(y - Q(s,a;\theta)\big)\big(-\nabla_\theta Q(s,a;\theta)\big)\Big] = -2\,\mathbb{E}\Big[\big(y - Q(s,a;\theta)\big)\nabla_\theta Q(s,a;\theta)\Big]$$

---

## 3. Vì sao không ổn định: "Deadly Triad"

Ba yếu tố gộp lại có thể làm Q **phân kỳ**:
1. **Function approximation** (mạng nơ-ron): cập nhật một mẫu làm thay đổi Q ở mọi nơi.
2. **Bootstrapping**: mục tiêu $y$ chứa chính ước lượng Q.
3. **Off-policy**: dữ liệu không đến từ policy đang đánh giá.

Hai nguồn bất ổn cụ thể trong DQN thô:
- **Tương quan thời gian**: các mẫu liên tiếp trong một episode gần giống nhau, vi phạm giả định i.i.d. của SGD → gradient lệch, dao động.
- **Mục tiêu di động ("moving target")**: $y$ phụ thuộc $\theta$; mỗi bước cập nhật $\theta$ làm $y$ đổi theo, giống đuổi theo cái đuôi của chính mình.

DQN (Mnih et al., 2015) xử lý hai thứ này bằng **Replay Buffer** và **Target Network**.

---

## 4. Hai kỹ thuật của DQN

### 4.1 Experience Replay
Lưu mọi chuyển tiếp $(s,a,r,s',d)$ vào bộ nhớ vòng $\mathcal D$ kích thước $N$; mỗi bước học lấy **mini-batch ngẫu nhiên** (kích thước 32).

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc cập nhật trực tiếp theo chuỗi chuyển tiếp nối tiếp thời gian $(s_t, a_t, r_{t+1}, s_{t+1})$.
> * **Khuyết điểm của phương pháp cũ**: Dữ liệu thu được từ môi trường có **tương quan thời gian cực mạnh** ($s_{t+1}$ phụ thuộc chặt chẽ vào $s_t$), vi phạm nghiêm trọng giả định độc lập cùng phân phối (i.i.d) của Gradient Descent. Hơn nữa, việc học trực tiếp gây ra **Quên thảm họa (Catastrophic Forgetting)**: mạng học kỹ năng mới ở cuối game sẽ quên sạch các trạng thái ban đầu.
> * **Logic thiết kế toán học**: Lưu trữ các chuyển tiếp vào bộ đệm $\mathcal D$ và lấy mẫu ngẫu nhiên đồng đều (Uniform Random Sampling). Kỹ thuật này phá vỡ hoàn toàn tính tương quan thời gian, đưa dữ liệu về dạng i.i.d giống hệt học có giám sát và cho phép tái sử dụng mỗi chuyển tiếp nhiều lần (tăng hiệu quả dữ liệu vượt bậc).

### 4.2 Target Network
Giữ bản sao tham số $\theta^-$, cập nhật $\theta^-\leftarrow\theta$ mỗi $C$ bước (ở đây $C=1000$). Mục tiêu:
$$y=r+\gamma\,(1-d)\,\max_{a'}Q(s',a';\theta^-)$$

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế việc dùng chung một bộ trọng số $\theta$ duy nhất cho cả hàm dự đoán $Q(s, a; \theta)$ và hàm mục tiêu $y = r + \gamma \max_{a'} Q(s', a'; \theta)$.
> * **Khuyết điểm của phương pháp cũ**: **Mục tiêu di động (Moving Target Problem)**. Mỗi bước cập nhật $\theta$ nhằm kéo $Q(s, a; \theta)$ tiến lại gần $y$, nhưng vì $y$ cũng phụ thuộc $\theta$, $y$ lập tức dịch chuyển ra xa hơn. Điều này tạo ra một vòng lặp phản hồi dương (Positive Feedback Loop) khiến sai số dao động dữ dội và giá trị $Q$ bùng nổ phân kỳ ra $\pm\infty$.
> * **Logic thiết kế toán học**: Đóng băng tham số $\theta^-$ của mạng mục tiêu trong $C$ bước thời gian. Tại mỗi chu kỳ, bài toán trở thành bài toán hồi quy chuẩn mực với các nhãn mục tiêu hoàn toàn cố định, triệt tiêu phản hồi dương.

* **Giải thích ký hiệu**:
  * $r$: Phần thưởng tức thì nhận được.
  * $d \in \{0, 1\}$: Cờ báo hiệu kết thúc tập thể nghiệm (Terminal flag). Khi $d=1$, trạng thái là kết thúc nên $1-d = 0$, triệt tiêu tương lai phía sau và $y = r$.
  * $\theta^-$: Trọng số của Target Network, cố định trong $C$ bước.

> **Cảnh báo về code của repo** ([`central_control.py`](central_control.py#L50-L55)):
> - Hàm `update_target` sau khi `load_state_dict` lại gán `self.target_nn = self.moving_nn`, nên hai mạng trở thành **cùng một object** → mất tác dụng target network.
> - `done_t` được tạo nhưng **không dùng** ở [L103](central_control.py#L103), tức thiếu $(1-d)$.
>
> Hai dòng sửa đúng:
> ```python
> def update_target(self):
>     self.target_nn.load_state_dict(self.moving_nn.state_dict())
> ...
> expected = rewards_t + (self.gamma**self.n_multi_step) * next_state_values * (1 - done_t.float())
> ```

---

## 5. Double DQN – Chống đánh giá quá cao & Chứng minh Maximization Bias

### 5.1 Chứng minh toán học hiện tượng Maximization Bias (Thiên lệch cực đại hoá)
Cho $n$ biến ngẫu nhiên $X_1, X_2, \dots, X_n$ thể hiện ước lượng của các hành động bị nhiễu.

* **Bất đẳng thức Jensen**:
  Vì hàm số $f(x_1, \dots, x_n) = \max(x_1, \dots, x_n)$ là một hàm lồi (Convex Function) trên $\mathbb R^n$, theo bất đẳng thức Jensen:
  $$\boxed{\mathbb E\big[\max_i X_i\big] \ge \max_i \mathbb E[X_i]}$$

* **Ví dụ giải tích định lượng chính xác**:
  Xét hai hành động có giá trị thực đều bằng $0$ ($q_*(a_1) = q_*(a_2) = 0$). Các ước lượng thu được là độc lập và có nhiễu chuẩn:
  $$X_1 \sim \mathcal N(0, \sigma^2), \quad X_2 \sim \mathcal N(0, \sigma^2)$$
  Đặt hiệu số $Z = X_1 - X_2$. Do tổng hiệu của các biến chuẩn độc lập cũng là phân phối chuẩn:
  $$Z \sim \mathcal N(0, \sigma^2 + \sigma^2) = \mathcal N(0, 2\sigma^2)$$
  Biểu diễn toán tử $\max$ qua trị tuyệt đối:
  $$\max(X_1, X_2) = \frac{X_1 + X_2 + |X_1 - X_2|}{2} = \frac{X_1 + X_2 + |Z|}{2}$$
  Lấy kỳ vọng hai vế:
  $$\mathbb E\big[\max(X_1, X_2)\big] = \frac{\mathbb E[X_1] + \mathbb E[X_2] + \mathbb E[|Z|]}{2} = \frac{0 + 0 + \mathbb E[|Z|]}{2} = \frac{1}{2} \mathbb E[|Z|]$$
  Biến ngẫu nhiên $|Z|$ tuân theo phân phối nửa chuẩn (Half-Normal Distribution) với kỳ vọng:
  $$\mathbb E[|Z|] = \sqrt{\frac{2}{\pi}} \operatorname{std}(Z) = \sqrt{\frac{2}{\pi}} \sqrt{2}\sigma = \frac{2\sigma}{\sqrt{\pi}}$$
  Do đó:
  $$\boxed{\mathbb E\big[\max(X_1, X_2)\big] = \frac{1}{2} \cdot \frac{2\sigma}{\sqrt{\pi}} = \frac{\sigma}{\sqrt{\pi}} > 0 = \max\big(\mathbb E[X_1], \mathbb E[X_2]\big)}$$
  Với $\sigma = 1$, kỳ vọng bị thổi phồng lên $\frac{1}{\sqrt{\pi}} \approx 0.564$. Khi có nhiều hành động hơn, độ thổi phồng tăng theo bậc $\sqrt{\ln |\mathcal A|}$. Vì Q-learning dùng giá trị $\max$ này để bootstrap, sai số dương này tích luỹ qua các tầng đệ quy làm hàm $Q$ nổ ảo giác! $\square$

---

### 5.2 Giải pháp Double DQN: Tách "chọn hành động" và "đánh giá giá trị"
* **DQN thường**:
  $$y = r + \gamma Q\Big(s', \arg\max_{a'} Q(s', a'; \theta^-); \theta^-\Big) \quad (\text{Mạng target vừa chọn vừa đánh giá})$$
* **Double DQN**:
  $$\boxed{y = r + \gamma Q\Big(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\Big)}$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Từ $\max_{a'} Q$ sang Phép thế Kép (Double Estimator):**
> - **Bước 1 (Viết lại toán tử $\max_{a'} Q(s', a'; \theta^-)$ dưới dạng hàm lồng hàm):**
>   Bất kỳ phép lấy cực đại $\max_{a'} f(a')$ nào cũng thực chất là: tìm vị trí đạt cực đại $a_{\text{target}}^* = \arg\max_{a'} Q(s', a'; \theta^-)$ rồi đem thế vị trí đó vào lại chính hàm $Q(\cdot; \theta^-)$:
>   $$y_{\text{DQN}} = r + \gamma \, Q\Big(s', \underbrace{\arg\max_{a'} Q(s', a'; \theta^-)}_{a_{\text{target}}^* \text{ (do }\theta^-\text{ chọn)}}; \, \theta^-\Big)$$
> - **Bước 2 (Phép thế Double DQN — Thay hàm chọn hành động bằng Mạng Online $\theta$):**
>   Ta định nghĩa hàm chọn hành động từ mạng đang học (Online Network $\theta$):
>   $$a_{\text{online}}^*(s') = \arg\max_{a'} Q(s', a'; \theta)$$
>   Đem **thế $a_{\text{online}}^*(s')$ vào đối số thứ hai của hàm định giá $Q(s', \cdot\,; \theta^-)$** (thay cho $a_{\text{target}}^*$):
>   $$y_{\text{DDQN}} = r + \gamma \, Q\Big(s', \underbrace{a_{\text{online}}^*(s')}_{\text{Thế bằng }\arg\max_{a'} Q(s', a'; \theta)}; \, \theta^-\Big) = r + \gamma \, Q\Big(s', \arg\max_{a'} Q(s', a'; \theta); \, \theta^-\Big)$$
>   *Tác động lên mẫu:* Nếu tại mẫu $s'$, mạng $\theta$ bị nhiễu dương ở hành động $a_1$ khiến $a_{\text{online}}^* = a_1$, thì khi đem $a_1$ thế sang mạng $\theta^-$, mạng $\theta^-$ không hề bị cùng nhiễu dương đó nên trả về giá trị trung thực $Q(s', a_1; \theta^-)$, cắt đứt hoàn toàn hiệu ứng cộng hưởng sai số!

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế mục tiêu chuẩn của DQN $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$.
> * **Khuyết điểm của phương pháp cũ**: **Thiên lệch cực đại hoá (Maximization Bias)**. Theo Bất đẳng thức Jensen, $\mathbb E[\max(X_1, X_2)] \ge \max(\mathbb E[X_1], \mathbb E[X_2])$. Do mạng nơ-ron luôn có sai số xấp xỉ ngẫu nhiên, toán tử $\max$ luôn chọn lấy sai số dương lớn nhất, thổi phồng giá trị $Q$ lên $\approx \sigma/\sqrt{\pi}$, tích lũy qua các tầng bootstrap khiến agent học sai lệch hoàn toàn chất lượng hành động.
> * **Logic thiết kế toán học**: Tách rời hai vai trò: **Mạng chính $\theta$** chịu trách nhiệm chọn hành động $a^* = \arg\max_{a'} Q(s', a'; \theta)$, trong khi **Mạng target $\theta^-$** chịu trách nhiệm định giá hành động đó $Q(s', a^*; \theta^-)$.
> * **Cơ chế giải quyết**: Vì sai số ngẫu nhiên của $\theta$ và $\theta^-$ không tương quan, kỳ vọng sai số định giá triệt tiêu về 0, loại bỏ hoàn toàn hiện tượng thổi phồng giá trị $Q$.

---

## 6. Dueling Network – Tách Giá trị & Lợi thế (Identifiability Proof)

### 6.1 Khái niệm
Phân rã hàm Q thành hai thành phần trực quan:
$$Q(s, a) = V(s) + A(s, a)$$
* $V(s)$: Giá trị nội tại của trạng thái $s$ (bất kể làm gì).
* $A(s, a)$: Lợi thế riêng biệt khi chọn hành động cụ thể $a$ thay vì hành động trung bình.

### 6.2 Chứng minh bài toán không khả định danh (Unidentifiability) & Nghiệm duy nhất
Nếu chỉ định nghĩa $Q(s, a) = V(s) + A(s, a)$, hệ thống có vô số nghiệm: với mọi hằng số $c \in \mathbb R$, ta có thể biến đổi:
$$V'(s) = V(s) + c, \quad A'(s, a) = A(s, a) - c \implies V'(s) + A'(s, a) = V(s) + A(s, a) = Q(s, a)$$
Mạng nơ-ron không biết cách phân chia trọng số giữa $V$ và $A$, dẫn đến quá trình tối ưu hóa mất ổn định.

* **Chứng minh kiến trúc Dueling với phép trừ trung bình (Mean Subtraction)**:
  Wang et al. (2016) ràng buộc bằng cách trừ đi trung bình của Advantage:
  $$\boxed{Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal A|} \sum_{a' \in \mathcal A} A(s, a'; \theta, \alpha) \right)}$$
  Lấy trung bình cộng của $Q(s, a)$ trên toàn bộ các hành động $a \in \mathcal A$:
  $$\begin{aligned}
  \frac{1}{|\mathcal A|} \sum_{a \in \mathcal A} Q(s, a) &= \frac{1}{|\mathcal A|} \sum_{a \in \mathcal A} \left[ V(s) + A(s, a) - \frac{1}{|\mathcal A|} \sum_{a'} A(s, a') \right] \\
  &= V(s) + \frac{1}{|\mathcal A|} \sum_{a} A(s, a) - \frac{|\mathcal A|}{|\mathcal A|} \left( \frac{1}{|\mathcal A|} \sum_{a'} A(s, a') \right) \\
  &= V(s) + \frac{1}{|\mathcal A|} \sum_{a} A(s, a) - \frac{1}{|\mathcal A|} \sum_{a'} A(s, a') \\
  &= V(s)
  \end{aligned}$$
  Đẳng thức chứng minh rằng $V(s)$ **được xác định duy nhất** và phản ánh chính xác giá trị trung bình kỳ vọng của trạng thái $s$! $\square$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Chứng minh Phép Thế Trừ Trung Bình tự triệt tiêu độ lệch $c$:**
> - **Giả sử hai nhánh mạng bị dịch chuyển bởi một hằng số tùy ý $c$:**
>   $$\tilde{V}(s) = V(s) + c, \qquad \tilde{A}(s, a) = A(s, a) + c'$$
> - **Thế $\tilde{V}(s)$ và $\tilde{A}(s, a)$ vào công thức ghép Dueling $Q(s, a) = \tilde{V}(s) + \Big(\tilde{A}(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} \tilde{A}(s, a')\Big)$:**
>   $$\begin{aligned}
>   Q_{\text{new}}(s, a) &= \big(V(s) + c\big) + \left[ \big(A(s, a) + c'\big) - \frac{1}{|\mathcal{A}|}\sum_{a' \in \mathcal{A}} \big(A(s, a') + c'\big) \right] \\
>   &= V(s) + c + A(s, a) + \cancel{c'} - \frac{1}{|\mathcal{A}|}\sum_{a' \in \mathcal{A}} A(s, a') - \cancel{\frac{|\mathcal{A}| \cdot c'}{|\mathcal{A}|}}
>   \end{aligned}$$
>   Ta thấy mọi độ lệch hằng số $c'$ trên nhánh Advantage **bị phép thế trừ trung bình triệt tiêu tức khắc ($c' - c' = 0$)**, buộc trung bình của cụm ngoặc vuông luôn bằng $0$ và giữ cho $V(s)$ luôn neo cố định vào $\frac{1}{|\mathcal{A}|}\sum_a Q(s,a)$!

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế kiến trúc mạng Q-network nguyên khối dự đoán trực tiếp $|\mathcal A|$ giá trị $Q(s, a)$.
> * **Khuyết điểm của phương pháp cũ**: Trong nhiều trạng thái, hành động nào được chọn cũng không làm thay đổi giá trị của tình huống (ví dụ xe đang chạy an toàn trên đường cao tốc vắng). Mạng DQN nguyên khối buộc phải thăm dò qua từng hành động độc lập mới cập nhật được giá trị cho hành động đó, cực kỳ lãng phí dữ liệu.
> * **Logic thiết kế toán học**: Tách luồng tính toán thành 2 nhánh: giá trị trạng thái $V(s)$ và lợi thế $A(s, a)$. Bằng phép trừ trung bình, ta giải quyết bài toán Unidentifiability và neo $V(s)$ vào giá trị kỳ vọng.
> * **Cơ chế giải quyết**: Mỗi lần cập nhật gradient cho bất kỳ hành động nào, mạng đều cập nhật luôn $V(s)$, lan truyền thông tin giá trị trạng thái cho **toàn bộ các hành động khác cùng lúc** chỉ trong 1 bước học.

---

### 6.3 Kiến trúc trong code
Hai nhánh FC (512 đơn vị) dùng chung thân CNN: nhánh $A$ ra $|\mathcal A|$ giá trị, nhánh $V$ ra 1 giá trị ([`DuelingDQN`](neural_net.py#L58-L102)):
```python
return val + adv - torch.mean(adv, dim=1, keepdim=True)
```
Phép cộng broadcast: `val` có shape `[B,1]`, `adv` có shape `[B,|A|]`.

---

## 7. Multi-step (N-step) learning

### 7.1 Công thức & Phân tích Logic
Thay vì bootstrap sau 1 bước, cộng $n$ phần thưởng thật rồi mới bootstrap:
$$\boxed{y^{(n)}_t=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n\max_{a}Q(s_{t+n},a;\theta^-)}$$
(Code dùng $n=2$: `n_multi_step`.)

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế mục tiêu bootstrap 1 bước $r + \gamma \max Q(s')$ của DQN.
> * **Khuyết điểm của phương pháp cũ**: Trong các môi trường có **phần thưởng thưa (Sparse Rewards)** như Pong hay Montezuma's Revenge, tín hiệu phần thưởng ở bước cuối chỉ lan truyền ngược về quá khứ 1 bước mỗi lần cập nhật. Cần hàng triệu chuyển tiếp để phần thưởng lan truyền về các trạng thái khởi đầu.
> * **Logic thiết kế toán học**: Tích lũy $n$ bước phần thưởng thực nhận trước khi dùng mạng nơ-ron xấp xỉ phần đuôi.
> * **Cơ chế giải quyết**: Tăng tốc độ lan truyền tín hiệu phần thưởng lên gấp $n$ lần, đồng thời giảm sự phụ thuộc vào các ước lượng $Q$ chưa chính xác ở giai đoạn đầu (giảm bias).

### 7.2 Bias/variance
- $n$ nhỏ: bias lớn (phụ thuộc nhiều vào Q sai) nhưng phương sai thấp.
- $n$ lớn: bias nhỏ (nhiều reward thật) nhưng phương sai cao; $n\to\infty$ là Monte Carlo.
- Với reward thưa (Pong: ±1 chỉ khi ghi điểm) $n$ vừa phải giúp truyền thông tin phần thưởng nhanh hơn $n$ lần.

### 7.3 Lưu ý lý thuyết
Mục tiêu n-step dùng các hành động $a_{t+1..t+n-1}$ của policy hành vi (cũ), nên về nguyên tắc **bị bias off-policy** khi kết hợp với replay (Q-learning thuần off-policy cần hiệu chỉnh importance sampling, ví dụ Retrace/Tree-backup). Rainbow bỏ qua hiệu chỉnh này và vẫn chạy tốt trong thực tế.

---

## 8. Noisy Networks – khám phá có học được

### 8.1 Vấn đề của $\epsilon$-greedy
Khám phá bằng nhiễu hành động **độc lập theo từng bước**, không phụ thuộc trạng thái, và lịch giảm $\epsilon$ phải chỉnh tay.

### 8.2 Ý tưởng & Logic thiết kế
Thêm nhiễu vào **trọng số** của tầng tuyến tính, với biên độ nhiễu là tham số được học:
$$y=(\mu^w+\sigma^w\odot\varepsilon^w)\,x+(\mu^b+\sigma^b\odot\varepsilon^b)$$
$\mu,\sigma$ là tham số học; $\varepsilon\sim$ nhiễu chuẩn (không học).

> 💡 **Bản chất Logic & Mục đích ra đời (Thay thế/Sửa đổi cái gì?)**:
> * **Thay thế thành phần nào**: Thay thế cơ chế khám phá $\epsilon$-greedy ngẫu nhiên đều ngoài hành động.
> * **Khuyết điểm của phương pháp cũ**: $\epsilon$-greedy tạo nhiễu giật cục (dithering noise) độc lập ở mỗi bước thời gian, không có sự nhất quán theo chuỗi thời gian để giải các câu đố dài hạn (ví dụ mở khoá cửa rồi đi qua cửa).
> * **Logic thiết kế toán học**: Đưa nhiễu vào tham số mạng (Parametric Noise). Cùng một trạng thái trong suốt episode sẽ nhận được cùng một hành vi nhất quán (Consistent Exploration), và mạng tự học cách triệt tiêu $\sigma \to 0$ ở các vùng trạng thái mà nó đã nắm vững.

### 8.3 Nhiễu factorised (tiết kiệm)
Với tầng $p\to q$, thay vì $pq$ nhiễu độc lập, lấy $p+q$ biến:
$$\varepsilon^w_{ij}=f(\varepsilon_i)\,f(\varepsilon'_j),\quad f(x)=\mathrm{sgn}(x)\sqrt{|x|},\qquad\varepsilon^b_j=f(\varepsilon'_j)$$
Khởi tạo bài báo: $\mu\sim U[-1/\sqrt p,1/\sqrt p]$, $\sigma=\sigma_0/\sqrt p$ ($\sigma_0=0.5$).

### 8.4 Gradient
Tham số hoá lại (reparameterization): $\nabla_\mu y=x$, $\nabla_\sigma y=\varepsilon\odot x$ → huấn luyện bằng backprop bình thường, với $\varepsilon$ coi là hằng số trong bước đó.

> **Code của repo** ([`NoisyLinear`](neural_net.py#L9-L55)): tác giả ghi chú *"IT DOESN'T WORK"*. Nó dùng nhiễu độc lập (không factorised), $\sigma$ khởi tạo hằng 0.017, và gọi `.data.normal_()` ngay trong `forward` (in-place trên buffer), chưa hỗ trợ tách "mẫu nhiễu" khỏi "áp nhiễu" (quan trọng với target net và batch). Khi đọc, hãy hiểu ý tưởng chứ đừng coi là bản cài đặt chuẩn.

---

## 9. Các cải tiến khác (chưa cài trong repo, nhưng cần biết)

### 9.1 Prioritized Experience Replay (PER)
Lấy mẫu tỉ lệ với sai số TD: $P(i)=\dfrac{p_i^\alpha}{\sum_kp_k^\alpha},\ p_i=|\delta_i|+\epsilon$. Vì lấy mẫu lệch nên phải bù bằng **trọng số importance sampling** $w_i=\big(\tfrac1{N\,P(i)}\big)^\beta$ (chuẩn hoá theo $\max w$) nhân vào loss; $\beta$ tăng dần về 1.

### 9.2 Distributional RL (C51, QR-DQN)
Học **phân phối** của return $Z(s,a)$ thay vì kỳ vọng $Q=\mathbb E[Z]$. Phương trình Bellman phân phối: $Z(s,a)\overset d=r+\gamma Z(s',a')$. Cung cấp tín hiệu học phong phú hơn (nhiều thông tin hơn một số vô hướng).

### 9.3 Rainbow (Hessel et al., 2017)
Kết hợp sáu thành phần: **Double + Dueling + N-step + Noisy + PER + Distributional**. Ablation cho thấy PER và N-step đóng góp lớn nhất. Repo cài 4/6 thành phần (không có PER, Distributional).

---

## 10. Tiền xử lý Atari và kiến trúc CNN

### 10.1 Pipeline wrapper ([`make_env`](atari_wrappers.py#L160-L169))
| Wrapper | Tác dụng và lý do |
|---|---|
| `MaxAndSkipEnv(skip=4)` | lặp 1 hành động 4 khung, **cộng reward**, lấy **max điểm ảnh của 2 khung cuối** (khử nhấp nháy sprite Atari); giảm 4× chi phí |
| `FireResetEnv` | bấm FIRE khi reset (một số game cần để bắt đầu) |
| `WarpFrame` | RGB→xám, resize $84\times84$ (giảm chiều, $210\times160\times3\to84\times84$) |
| `ImageToPyTorch` | đổi HWC → CHW |
| `FrameStack(4)` | xếp 4 khung gần nhất để suy ra **vận tốc/hướng** (khôi phục tính Markov) |
| `ScaledFloatFrame` | chia 255 → $[0,1]$ để ổn định huấn luyện |

> **Lưu ý thứ tự**: `ImageToPyTorch` đặt **trước** `FrameStack`, và `LazyFrames` nối theo `axis=2` ([L25](atari_wrappers.py#L25)). Suy ra từ code: quan sát là $1\times84\times336$ (4 khung ghép ngang) thay vì $4\times84\times84$ (xếp theo kênh). Hãy `print(env.reset().shape)` để kiểm chứng; chuẩn của baselines là `WarpFrame → FrameStack → ImageToPyTorch`.

### 10.2 Kiến trúc Nature-DQN
Công thức kích thước đầu ra tích chập: $W_{out}=\Big\lfloor\dfrac{W_{in}-K}{S}\Big\rfloor+1$ (không padding).

| Lớp | Tham số | Đầu ra (đầu vào chuẩn 4×84×84) |
|---|---|---|
| Conv1 | 32 bộ lọc $8\times8$, stride 4 | $32\times20\times20$ ($\frac{84-8}{4}+1=20$) |
| Conv2 | 64 bộ lọc $4\times4$, stride 2 | $64\times9\times9$ ($\frac{20-4}{2}+1=9$) |
| Conv3 | 64 bộ lọc $3\times3$, stride 1 | $64\times7\times7$ ($9-3+1=7$) |
| Flatten | | $3136$ |
| FC1 + ReLU | 512 | 512 |
| FC2 | $|\mathcal A|$ (Pong: 6) | $Q(s,\cdot)$ |

`_get_conv_out` trong code tính số 3136 bằng cách cho một tensor 0 đi qua. Repo có thêm `BatchNorm2d` sau mỗi conv (bài báo không có).

---

## 11. Đọc code theo luồng chạy

```
main.py: env = make_env("PongNoFrameskip-v4"); agent = DQNAgent(...)
lặp mỗi bước môi trường:
  a  ← agent.act_eps_greedy(obs)                       # agent.py
  obs', r, done ← env.step(a)
  agent.add_env_feedback(...)   # lưu Memory vào buffer; ε ← max(ε_final, ε_start - n_iter/ε_decay)
  agent.sample_and_optimize(32) # chỉ học khi len(buffer) > buffer_start_size (10001)
        └→ buffer.sample(32)  (n-step)                 # buffers.py
        └→ central_control.optimize(batch)             # tính loss, backward, Adam
        └→ mỗi 1000 bước: update_target()
```

| Công thức | Code |
|---|---|
| $Q(s,a;\theta)$ cho hành động đã chọn | `self.moving_nn(states_t).gather(1, actions_t[:,None]).squeeze(-1)` |
| $y$ (Double hoặc thường) | [central_control.py L89-L103](central_control.py#L89-L103) |
| Loss MSE | `nn.MSELoss()(state_action_values, expected_state_action_values)` |
| Cập nhật Adam | `optim.Adam(moving_nn.parameters(), lr)` |
| $\epsilon$ tuyến tính | `max(eps_final, eps_start - n_iter/eps_decay)` |

**`gather` hoạt động ra sao**: `q = net(s)` có shape `[B,|A|]`; `q.gather(1, a[:,None])` lấy đúng phần tử $q[b,a_b]$ cho mỗi $b$ → shape `[B,1]`, rồi `squeeze` còn `[B]`. Đây là phép lấy $Q(s_b,a_b)$ **khả vi**.

### 11.1 Phân tích chuyên sâu các tham số và cơ chế tác động lên phương trình DQN

DQN kết hợp mạng nơ-ron sâu với học tăng cường, do đó các siêu tham số tương tác trực tiếp với hàm mất mát MSE và động học của gradient:

#### (1) Hệ số chiết khấu $\gamma$ (Discount Factor)
* **Xuất hiện tại**:
  $$y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-),\qquad L(\theta) = \mathbb E\Big[\big(y - Q(s, a; \theta)\big)^2\Big]$$
* **Cơ chế tác động**:
  * Gradient theo tham số mạng: $\nabla_\theta L = -2 \big(y - Q(s, a; \theta)\big) \nabla_\theta Q(s, a; \theta)$.
  * **Khi $\gamma$ quá lớn ($\gamma \to 1$)**: Trong Pong, một điểm số có thể kéo dài hàng trăm bước. Nếu $\gamma = 0.999$, giá trị $Q$ tích lũy có thể tăng vọt lên mức hàng chục đơn vị, làm biên độ mục tiêu $y$ rất lớn, dẫn tới sai số $y - Q$ lớn $\implies$ **Gradient Explosion**, phá vỡ trọng số của mạng tích chập CNN.
  * **Khi $\gamma$ quá nhỏ ($\gamma \le 0.9$)**: Tầm nhìn hiệu dụng $H_{eff} \approx \frac{1}{1-\gamma} \le 10$ bước. Khi quả bóng bay từ vợt đối phương sang vợt của agent mất khoảng 30–40 frames. Nếu $\gamma$ nhỏ, tín hiệu thưởng ghi điểm bị suy giảm theo thừa số $\gamma^{35} \approx 0.025$, khiến mạng nơ-ron không thể gán công lao (credit assignment) cho hành vi đón bóng ở khoảng cách xa.
  * **Giá trị tối ưu**: $\gamma = 0.99$ ($H_{eff} \approx 100$ bước), cân bằng giữa độ ổn định gradient và tầm nhìn chiến thuật.

---

#### (2) Chu kỳ cập nhật Target Network `n_iter_update_target` ($C$)
* **Xuất hiện tại**: Gán trọng số $\theta^- \leftarrow \theta$ mỗi $C$ bước huấn luyện.
* **Cơ chế tác động toán học**:
  * Nếu không có Target Network (tương đương $C = 1$): Mục tiêu $y(\theta) = r + \gamma \max Q(s', \cdot; \theta)$ phụ thuộc trực tiếp vào chính $\theta$. Khi optimizer cập nhật $\theta$ để kéo $Q(s, a; \theta)$ về gần $y$, thì $y$ lại nhảy sang một giá trị mới ("chasing a moving target"). Hàm mất mát không còn là hàm lồi cố định mà biến dạng liên tục $\implies$ **Dao động cực mạnh và phân kỳ (Divergence)**.
  * **Khi $C$ quá nhỏ ($C < 100$)**: Mạng đích thay đổi quá nhanh, triệt tiêu tác dụng ổn định hoá của target net.
  * **Khi $C$ quá lớn ($C > 20000$)**: Mạng đang học phải đuổi theo một mục tiêu "đóng băng" quá lâu (stale target), khiến tốc độ học bị trì trệ không cần thiết.
  * **Giá trị chuẩn**: $C = 1000$ bước (trong repo) hoặc $C = 10000$ bước (bài báo Nature). Một giải pháp thay thế hiện đại là cập nhật mềm Polyak $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ với $\tau \approx 0.005$ ở mỗi bước.

---

#### (3) Bước nhìn trước $n$ trong Multi-step Learning (`n_multi_step`)
* **Xuất hiện tại**: $y^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max Q(s_{t+n}, \cdot; \theta^-)$.
* **Cơ chế tác động toán học**:
  * **Giảm Bias**: Khi mạng $Q$ còn non nớt ở các giai đoạn đầu, giá trị dự đoán $\max Q$ chứa sai số rất lớn. Việc thay thế $n-1$ bước dự đoán bằng phần thưởng thực tế $\sum \gamma^k r$ giúp giảm thiểu sự phụ thuộc vào mạng sai $\implies$ giảm bias nghiêm trọng của phương pháp bootstrap 1-step.
  * **Tăng tốc lan truyền phần thưởng**: Trong game có phần thưởng thưa như Pong, mỗi bước $n=2$ hoặc $n=3$ giúp lan truyền tín hiệu $+1/-1$ ngược về trước nhanh hơn gấp 2–3 lần.
  * **Cái giá phải trả khi $n$ quá lớn**: Mục tiêu n-step sử dụng chuỗi hành động thực tế $a_{t+1}, \dots, a_{t+n-1}$ nằm trong bộ nhớ replay. Nhưng các hành động đó được sinh ra bởi một policy cũ (hành vi cũ). Khi $n > 5$, sự sai lệch giữa policy cũ và policy tối ưu hiện tại trở nên quá lớn, vi phạm tính hợp lệ của phương trình Bellman nếu không có trọng số hiệu chỉnh Importance Sampling.

---

#### (4) Dung lượng Replay Buffer (`buffer_capacity`) và `buffer_start_size`
* **Xuất hiện tại**: $\mathcal D = \{(s_i, a_i, r_i, s'_i, d_i)\}_{i=1}^N$, lấy mẫu ngẫu nhiên mini-batch kích thước $B = 32$.
* **Cơ chế tác động**:
  * **Phá vỡ tương quan chuỗi thời gian (Break Temporal Correlation)**: Trong game, trạng thái $s_{t+1}$ phụ thuộc gần như tuyệt đối vào $s_t$ (ảnh chỉ dịch chuyển vài pixel). Huấn luyện trực tiếp trên chuỗi tuần tự tương đương với việc huấn luyện mạng phân loại hình ảnh mà toàn bộ 1000 ảnh liên tiếp đều là một con mèo $\implies$ mạng bị Overfitting cục bộ và quên các kịch bản khác (Catastrophic Forgetting).
  * **`buffer_capacity` = 15000 (Repo)**: Khá nhỏ so với chuẩn $10^6$ của DeepMind. Nếu buffer quá nhỏ, các trải nghiệm quan trọng (như pha ghi điểm hiếm hoi) sẽ nhanh chóng bị đẩy ra ngoài trước khi mạng kịp học đủ số lần.
  * **`buffer_start_size` = 10001**: Giai đoạn thu thập ngẫu nhiên ban đầu (Warm-up). Nếu bắt đầu train ngay khi buffer chỉ có vài trăm mẫu, mạng sẽ học đi học lại một nhóm nhỏ trạng thái ban đầu, làm lệch trọng số nơ-ron ngay từ đầu.

---

#### (5) Lịch trình suy giảm $\epsilon$ (`epsilon_decay`, `epsilon_start`, `epsilon_final`)
* **Xuất hiện tại**: $\epsilon(t) = \max\Big(\epsilon_{final},\ \epsilon_{start} - \frac{t}{\text{decay}}\Big)$ với $\epsilon_{start} = 1.0, \epsilon_{final} = 0.02, \text{decay} = 10^5$.
* **Cơ chế tác động**:
  * Trong $100.000$ frames đầu tiên, $\epsilon$ giảm tuyến tính từ $100\%$ xuống $2\%$.
  * Pong có không gian hành động gồm 6 nút (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE). Giai đoạn đầu $\epsilon$ lớn giúp agent thử nghiệm mọi tổ hợp di chuyển vợt để tìm ra hành vi đỡ trúng bóng.
  * Giữ lại $\epsilon_{final} = 0.02$ thay vì $0$ để đảm bảo agent không bao giờ bị đóng băng trong một chu kỳ hành vi lặp vô tận (đặc biệt khi đối thủ máy tính thay đổi quỹ đạo đánh bóng).

---

#### (6) Tốc độ học Adam `learning_rate` ($5\times 10^{-5}$)
* **Cơ chế tác động**:
  * Mạng nơ-ron sâu trong RL khó tối ưu hơn nhiều so với học có giám sát vì dữ liệu có tính chất "non-stationary" (mục tiêu $y$ thay đổi liên tục).
  * Learning rate $5\times 10^{-5}$ là một giá trị rất nhỏ (thường trong Computer Vision là $10^{-3}$ hoặc $10^{-4}$). Nếu tăng lr lên $10^{-3}$, các bước nhảy gradient quá mạnh sẽ làm sập các bộ lọc tích chập (conv filters), khiến giá trị $Q$ phân kỳ tới vô cùng (NaN/Inf loss).

---

## 12. Ví dụ số: Double DQN

Giả sử tại $s'$: $Q(s',\cdot;\theta)=[1.0,\,1.5,\,1.2]$ (mạng học) và $Q(s',\cdot;\theta^-)=[1.1,\,0.9,\,1.4]$ (mạng đích); $r=1$, $\gamma=0.99$, $d=0$.

- **DQN thường**: $\max Q(s',\cdot;\theta^-)=1.4\Rightarrow y=1+0.99\cdot1.4=2.386$.
- **Double DQN**: mạng học chọn $a^*=\arg\max=a_2$ (1.5); mạng đích đánh giá $Q(s',a_2;\theta^-)=0.9\Rightarrow y=1+0.99\cdot0.9=1.891$.

Mục tiêu Double thấp hơn vì hai mạng "bất đồng" về hành động tốt nhất → phản ánh sự bất định, giảm lạc quan.

**N-step ($n=2$)**: $r_t=0,\ r_{t+1}=1$; $\gamma=0.99$; $Q_{max}(s_{t+2})=3$:
$$y=0+0.99\cdot1+0.99^2\cdot3=0.99+2.9403=3.9303$$
(1-step sẽ là $0+0.99\,Q_{max}(s_{t+1})$ và phải tự bootstrap phần thưởng $1$ qua Q — thông tin đến chậm hơn.)

---

## 13. Bài tập và câu hỏi kiểm tra

**Lý thuyết**
1. Vì sao gradient không lan qua mục tiêu $y$? Điều gì xảy ra nếu lan (gợi ý: "residual gradient", Baird)?
2. Chứng minh $\mathbb E[\max X_i]\ge\max\mathbb E[X_i]$ (Jensen). Tự tính lại $\mathbb E[\max(\varepsilon_1,\varepsilon_2)]=1/\sqrt\pi$.
3. Vì sao Dueling cần trừ trung bình $A$? Nếu trừ $\max_aA$ thay vì trung bình thì ý nghĩa của $V$ là gì?
4. Vì sao replay buffer bắt buộc thuật toán phải off-policy?
5. Tính $\frac{W-K}{S}+1$ cho ảnh đầu vào $84\times336$ qua 3 conv và so sánh với 3136.

**Thực hành**
1. Sửa hai lỗi ở §4.2, chạy Pong và so sánh đường `mean_reward` trước/sau (ghi lại seed).
2. Bật lần lượt `double_DQN`, `dueling`, `n_multi_step=1/2/3` và vẽ trên cùng một biểu đồ TensorBoard.
3. In `q_values.mean()` theo thời gian để quan sát overestimation giữa DQN và Double DQN.
4. Kiểm tra shape quan sát (§10.1) và đổi thứ tự wrapper; so sánh kết quả.
5. Cài **Prioritized Replay** bằng sum-tree hoặc cài NoisyLinear factorised theo §8.3.

**Công thức cần thuộc**: $L=(y-Q)^2$ · $y$ cho DQN/Double/N-step · $Q=V+A-\bar A$ · $\epsilon$ tuyến tính · $W_{out}$ conv.
