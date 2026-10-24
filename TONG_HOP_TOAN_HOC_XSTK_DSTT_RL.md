# CẨM NANG ÔN TẬP TOÁN HỌC TRỰC QUAN: XÁC SUẤT THỐNG KÊ, ĐẠI SỐ TUYẾN TÍNH & GIẢI TÍCH TRONG RL VÀ ROBOTICS

> **Mục tiêu của Tài liệu:**
> Tài liệu này được thiết kế như một **"Bản đồ Tư duy & Cơ chế Toán học"** giúp bạn ôn tập toàn diện **Xác suất Thống kê**, **Đại số Tuyến tính**, và **Giải tích Tối ưu** theo cách trực quan, logic nhất.
> 
> Thay vì chỉ liệt kê công thức khô khan, mỗi khái niệm đều trả lời trực diện **4 câu hỏi**:
> 1. **Bản chất Logic (Trực giác):** Khái niệm này sinh ra để đo lường hay giải quyết vấn đề gì?
> 2. **Cơ chế Tác động lên Mẫu dữ liệu (Action on Samples):** Khi đưa một tập mẫu dữ liệu $\{x_1, x_2, \dots, x_N\}$ qua công thức này, nó biến đổi đám mây mẫu như thế nào?
> 3. **Nó làm Nổi bật (Highlight) điều gì ẩn bên trong dữ liệu?**
> 4. **Nó nằm ở đâu trong các thuật toán RL (Tuần 1–8) và Mô phỏng Robot?**

---

## MỤC LỤC

- [BẢNG TÓM TẮT NHANH: TOÁN TỬ TÁC ĐỘNG LÊN MẪU & Ý NGHĨA NỔI BẬT](#bảng-tóm-tắt-nhanh-toán-tử-tác-động-lên-mẫu--ý-nghĩa-nổi-bật)
- [PHẦN I: XÁC SUẤT & THỐNG KÊ (PROBABILITY & STATISTICS) — NGHỆ THUẬT XỬ LÝ SỰ BẤT ĐỊNH](#phần-i-xác-suất--thống-kê-probability--statistics--nghệ-thuật-xử-lý-sự-bất-định)
  - [1. Biến ngẫu nhiên & Phân phối xác suất (Bernoulli, Categorical, Gaussian)](#1-biến-ngẫu-nhiên--phân-phối-xác-suất-bernoulli-categorical-gaussian)
  - [2. Kỳ vọng ($\mathbb{E}[X]$) & Trung bình mẫu (Sample Mean) — Bộ lọc khử nhiễu](#2-kỳ-vọng-mathbbex--trung-bình-mẫu-sample-mean--bộ-lọc-khử-nhiễu)
  - [3. Phương sai ($\operatorname{Var}$), Độ lệch chuẩn ($\sigma$) & Hiệp phương sai ($\operatorname{Cov}$) — Đo độ bung tỏa & rủi ro](#3-phương-sai-operatornamevar-độ-lệch-chuẩn-sigma--hiệp-phương-sai-operatornamecov--đo-độ-bung-tỏa--rủi-ro)
  - [4. Xác suất có điều kiện $P(A \mid B)$, Định lý Bayes & Tính chất Markov](#4-xác-suất-có-điều-kiện-pa-mid-b-định-lý-bayes--tính-chất-markov)
  - [5. Bất đẳng thức Jensen ($\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$) — Thủ phạm gây ảo tưởng giá trị (Overestimation)](#5-bất-đẳng-thức-jensen-mathbbegx-ge-gmathbbex--thủ-phạm-gây-ảo-tưởng-giá-trị-overestimation)
  - [6. Lấy mẫu Trọng số Quan trọng (Importance Sampling) — Tái chế mẫu cũ cho chính sách mới](#6-lấy-mẫu-trọng-số-quan-trọng-importance-sampling--tái-chế-mẫu-cũ-cho-chính-sách-mới)
  - [7. Lý thuyết Thông tin: Entropy ($\mathcal{H}$) & Phân kỳ KL ($D_{\text{KL}}$)](#7-lý-thuyết-thông-tin-entropy-mathcalh--phân-kỳ-kl-d_textkl)
- [PHẦN II: ĐẠI SỐ TUYẾN TÍNH (LINEAR ALGEBRA) — NGHỆ THUẬT BIẾN ĐỔI KHÔNG GIAN](#phần-ii-đại-số-tuyến-tính-linear-algebra--nghệ-thuật-biến-đổi-không-gian)
  - [1. Vector & Tích vô hướng (Dot Product $\mathbf{u} \cdot \mathbf{v}$) — Bộ đo độ tương đồng & hình chiếu](#1-vector--tích-vô-hướng-dot-product-mathbfu-cdot-mathbfv--bộ-đo-độ-tương-đồng--hình-chiếu)
  - [2. Phép nhân Ma trận ($\mathbf{y} = \mathbf{A}\mathbf{x}$) — Phép biến hình đám mây mẫu (Xoay, Co giãn, Chiếu)](#2-phép-nhân-ma-trận-mathbfy--mathbfamathbfx--phép-biến-hình-đám-mây-mẫu-xoay-co-giãn-chiếu)
  - [3. Trị riêng ($\lambda_i$) & Vector riêng ($\mathbf{v}_i$) — Tìm trục xương sống bất biến của dữ liệu](#3-trị-riêng-lambda_i--vector-riêng-mathbfv_i--tìm-trục-xương-sống-bất-biến-của-dữ-liệu)
  - [4. Các Chuẩn Đo lường ($L_1, L_2, L_\infty$, Frobenius Norm) — Cách chọn kính lúp soi lỗi trên mẫu](#4-các-chuẩn-đo-lường-l_1-l_2-l_infty-frobenius-norm--cách-chọn-kính-lúp-soi-lỗi-trên-mẫu)
  - [5. Ma trận Xác định Dương ($\mathbf{A} \succ 0$) & Dạng Toàn phương ($\mathbf{x}^T \mathbf{A} \mathbf{x}$) — Mặt bát năng lượng](#5-ma-trận-xác-định-dương-mathbfa-succ-0--dạng-toàn-phương-mathbfxt-mathbfa-mathbfx--mặt-bát-năng-lượng)
- [PHẦN III: GIẢI TÍCH ĐA BIẾN & TỐI ƯU HÓA (CALCULUS) — KẾT NỐI ĐẠI SỐ VÀ XÁC SUẤT](#phần-iii-giải-tích-đa-biến--tối-ưu-hóa-calculus--kết-nối-đại-số-và-xác-suất)
  - [1. Gradient ($\nabla f$), Ma trận Jacobian ($\mathbf{J}$) & Ma trận Hessian ($\mathbf{H}$)](#1-gradient-nabla-f-ma-trận-jacobian-mathbfj--ma-trận-hessian-mathbfh)
  - [2. Thủ thuật Đạo hàm Logarit (Log-Derivative Trick) — Biến đạo hàm của kỳ vọng thành kỳ vọng của đạo hàm](#2-thủ-thuật-đạo-hàm-logarit-log-derivative-trick--biến-đạo-hàm-của-kỳ-vọng-thành-kỳ-vọng-của-đạo-hàm)
  - [3. Khai triển Taylor & Ma trận Thông tin Fisher ($\mathbf{F}$) — Bản đồ địa hình vùng tin cậy](#3-khai-triển-taylor--ma-trận-thông-tin-fisher-mathbff--bản-đồ-địa-hình-vùng-tin-cậy)
- [PHẦN IV: BẢN ĐỒ TỔNG HỢP CÁC PHÉP THẾ HÀM KINH ĐIỂN TRONG RL (STEP-BY-STEP FUNCTION SUBSTITUTIONS)](#phần-iv-bản-đồ-tổng-hợp-các-phép-thế-hàm-kinh-điển-trong-rl-step-by-step-function-substitutions)
- [PHẦN V: CHẠY CODE THỰC NGHIỆM TRỰC QUAN TRÊN MẪU (`minh_hoa_toan_hoc_tren_mau.py`)](#phần-v-chạy-code-thực-nghiệm-trực-quan-trên-mẫu-minh_hoa_toan_hoc_tren_maupy)

---

## BẢNG TÓM TẮT NHANH: TOÁN TỬ TÁC ĐỘNG LÊN MẪU & Ý NGHĨA NỔI BẬT

| Khái niệm Toán học | Công thức cốt lõi | Cách tác động lên tập mẫu $\{x_i\}$ | Nêu bật / Bóc tách ra điều gì từ mẫu? | Vị trí xuất hiện trong RL & Robot |
| :--- | :--- | :--- | :--- | :--- |
| **Kỳ vọng / Trung bình mẫu** | $\hat{\mu} = \frac{1}{N}\sum_{i=1}^N x_i$ | Gom toàn bộ đám mây mẫu lại, triệt tiêu các dao động ngẫu nhiên trái dấu ($\pm \epsilon$). | **Trọng tâm thực sự (Tín hiệu gốc / Chân lý trung bình)** ẩn sau lớp nhiễu. | Hàm giá trị $V(s), Q(s,a)$, Monte Carlo Return. |
| **Phương sai (Variance)** | $\hat{\sigma}^2 = \frac{1}{N}\sum (x_i - \hat{\mu})^2$ | Đo khoảng cách bình phương từ từng mẫu tới tâm $\hat{\mu}$. | **Độ phân tán, mức độ rủi ro & sự bất định** của môi trường hoặc bộ ước lượng. | Độ bất định UCB1, phương sai cao của REINFORCE, tham số GAE $\lambda$. |
| **Hiệp phương sai (Covariance)** | $\Sigma_{jk} = \mathbb{E}[(x_j-\mu_j)(x_k-\mu_k)]$ | Nhân độ lệch của 2 chiều dữ liệu trên cùng 1 mẫu với nhau. | **Sự đồng biến / nghịch biến (mối liên kết ngầm)** giữa 2 biến trạng thái/hành động. | Bộ lọc Kalman (SLAM/Odometry), Evolution Strategies (CMA-ES). |
| **Định lý Bayes** | $P(s \mid z) \propto P(z \mid s) P(s)$ | Nhân trọng số của từng mẫu giả thuyết $s$ với khả năng khớp cảm biến $P(z \mid s)$. | **Thanh lọc giả thuyết:** Giữ lại mẫu khớp với thực tế, loại bỏ mẫu mâu thuẫn với cảm biến. | Định vị hạt AMCL (Nav2), Ước lượng trạng thái POMDP. |
| **Bất đẳng thức Jensen trên hàm $\max$** | $\mathbb{E}[\max_a X_a] \ge \max_a \mathbb{E}[X_a]$ | Luôn chọn mẫu có giá trị lớn nhất trong mỗi lần thử. | **Thiên vị nhiễu dương (Overestimation Bias):** Lấy nhầm nhiễu may mắn làm giá trị thật. | Giải thích tại sao Q-learning/DQN bị nổ giá trị $\to$ cần **Double DQN**. |
| **Importance Sampling** | $\mathbb{E}_P[f(x)] = \mathbb{E}_Q\!\left[\frac{P(x)}{Q(x)}f(x)\right]$ | Nhân từng mẫu cũ $x_i \sim Q$ với tỉ số xác suất $r_i = \frac{P(x_i)}{Q(x_i)}$. | **Uốn lại phân phối mẫu:** Phóng đại mẫu mà policy mới thích, dìm mẫu policy mới ghét. | Tỉ số $r_t(\theta)$ trong **PPO / TRPO** (Week 5), Off-policy RL. |
| **Shannon Entropy** | $\mathcal{H}(P) = -\sum P(x)\ln P(x)$ | Lấy trung bình của $-\ln P(x)$ (mức độ bất ngờ của từng mẫu). | **Độ phẳng / Độ đa dạng của mẫu:** Cao = khám phá đều; Thấp ($\approx 0$) = tụ về 1 điểm duy nhất. | Entropy Bonus trong **A2C** (Week 4) & **SAC** (Week 8). |
| **Phân kỳ KL ($D_{\text{KL}}$)** | $\sum P(x) \ln \frac{P(x)}{Q(x)}$ | Đo độ lệch logarit trung bình giữa 2 mật độ xác suất trên các mẫu $x \sim P$. | **Khoảng cách thay đổi hành vi:** Cho biết 2 chính sách tạo ra đám mây mẫu khác nhau cỡ nào. | Ràng buộc vùng tin cậy **TRPO/PPO** (Week 5), **DPO** (Week 8). |
| **Tích vô hướng (Dot Product)** | $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ | Chiếu vuông góc vector mẫu $\mathbf{u}$ lên trục chuẩn $\mathbf{v}$. | **Độ cùng hướng (Alignment):** Dương = cùng chiều, $0$ = vuông góc (độc lập), Âm = ngược chiều. | Mạng tuyến tính $\mathbf{w}^T\phi(s)$, Reward hướng đích $v\cos\alpha$, Attention. |
| **Trị riêng & Vector riêng** | $\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$ | Tìm những phương hướng $\mathbf{v}_i$ mà khi nhân ma trận $\mathbf{A}$ chỉ bị co giãn $\lambda_i$ lần chứ không bị xoay lệch. | **Trục chính bất biến & Tốc độ hội tụ/phân kỳ:** $\|\lambda\|<1$ co về 0 (ổn định), $\|\lambda\|>1$ nổ tung. | Trục quán tính chính $\mathbf{I}$ (Tầng 1), Phân phối dừng MDP, Độ cong Hessian. |
| **Chuẩn $L_1, L_2, L_\infty$** | $\|e\|_1, \|e\|_2, \|e\|_\infty$ | Tổng trị tuyệt đối ($L_1$), Căn tổng bình phương ($L_2$), hoặc Giá trị cực đại ($L_\infty$). | $L_1$: Kháng mẫu nhiễu dị biệt (Outliers).<br>$L_2$: Phạt nặng sai số lớn.<br>$L_\infty$: Bắt điểm tệ nhất. | Huber Loss ($L_1+L_2$ trong DQN), Chứng minh hội tụ Bellman ($L_\infty$). |

---

## PHẦN I: XÁC SUẤT & THỐNG KÊ (PROBABILITY & STATISTICS) — NGHỆ THUẬT XỬ LÝ SỰ BẤT ĐỊNH

Trong Reinforcement Learning và Robotics, **không có gì là chắc chắn 100%**: cảm biến LiDAR luôn có nhiễu, mặt sàn có chỗ trơn chỗ nhám, và chính sách của Agent cũng cần tính ngẫu nhiên để khám phá. Xác suất & Thống kê chính là bộ công cụ để **trích xuất sự thật từ những mẫu thử ngẫu nhiên**.

---

### 1. Biến ngẫu nhiên & Phân phối xác suất (Bernoulli, Categorical, Gaussian)

#### 💡 Bản chất Logic:
- Một **Biến ngẫu nhiên $X$** không phải là một con số cố định, mà là một **"chiếc hộp sinh mẫu"**. Mỗi lần bạn thò tay vào hộp (gọi hàm `.sample()`), nó trả ra một giá trị mẫu $x_i$ theo một quy luật mật độ $P(X = x)$ gọi là **Phân phối xác suất**.

#### 🔬 3 Phân phối "xương sống" trong RL tác động lên việc sinh mẫu như thế nào?

```
1. Categorical (Rời rạc - DQN/A2C Atari):       2. Gaussian N(μ, σ²) (Liên tục - PPO/SAC Robot):
   Xác suất P(a|s)                                 Mật độ p(a|s)
     ▲                                               ▲          ╭───╮  <-- σ nhỏ: Mẫu tụ sát vào μ
     │       ███                                     │         ╱  │  ╲     (Khai thác - Exploitation)
     │  ███  ███                                     │      ╭─────┼─────╮ <-- σ lớn: Mẫu trải rộng
     │  ███  ███  ███                                │    ─╯      │      ╰─   (Khám phá - Exploration)
     └───┴────┴────┴───► Hành động                   └────────────┼───────────► Mô-men / Vận tốc a
       Trái Đứng Phải                                             μ (Tâm điều khiển)
```

1. **Phân phối Bernoulli & Categorical (Softmax) — Dùng cho Hành động Rời rạc (Tuần 1, 3, 4):**
   $$P(a_k \mid s) = \frac{e^{z_k / \tau}}{\sum_{j=1}^{|\mathcal{A}|} e^{z_j / \tau}}$$
   - **Cách tác động lên mẫu:** Biến mảng điểm số thô (logits $z_k \in (-\infty, +\infty)$) thành xác suất lấy mẫu $\in (0, 1)$ có tổng bằng $1$.
   - **Vai trò của Nhiệt độ $\tau$ (Temperature):**
     - Khi $\tau \to 0$: Đẩy toàn bộ xác suất dồn vào nhánh có $z_k$ cao nhất $\implies$ Các mẫu sinh ra **giống hệt nhau 100%** (Khai thác cực đoan).
     - Khi $\tau \to \infty$: San phẳng mọi nhánh bằng nhau $\frac{1}{|\mathcal{A}|}$ $\implies$ Các mẫu sinh ra **trải đều khắp mọi lựa chọn** (Khám phá ngẫu nhiên).

2. **Phân phối Chuẩn / Gaussian $\mathcal{N}(\mu, \sigma^2)$ — Dùng cho Điều khiển Liên tục Robot & Nhiễu Cảm biến (Tuần 5, 6, 7, 8):**
   $$p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
   - **Cách sinh mẫu (Reparameterization Trick trong SAC - Tuần 8):**
     $$x = \mu + \sigma \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)$$
   - **Nó nêu bật điều gì?** Tách một mẫu hành động thành 2 phần tách biệt:
     - **$\mu(s)$ (Tâm):** Hành động tốt nhất mà mạng nơ-ron đang nghĩ tới.
     - **$\sigma(s)$ (Bán kính đám mây mẫu):** Mức độ mạnh dạn thử nghiệm xung quanh $\mu$. Khi robot đã đi vững, mạng tự giảm $\sigma \to 0$ để đi chính xác; khi gặp địa hình lạ, $\sigma$ tăng lên để dò đường.

---

### 2. Kỳ vọng ($\mathbb{E}[X]$) & Trung bình mẫu (Sample Mean) — Bộ lọc khử nhiễu

#### 💡 Bản chất Logic:
Trong RL, khi thực hiện hành động $a$ tại trạng thái $s$, lần thứ nhất bạn nhận thưởng $R_1 = +12$, lần thứ hai gặp gió cản bạn nhận $R_2 = +6$, lần thứ ba nhận $R_3 = +9$. Đâu mới là **giá trị thật** của hành động đó?
Đó chính là **Kỳ vọng toán học (Expectation)** $Q^\pi(s, a) = \mathbb{E}[G_t \mid S_t=s, A_t=a]$.

Vì ta không biết công thức xác suất $P$ của thế giới thực, ta phải ước lượng Kỳ vọng bằng **Trung bình mẫu (Monte Carlo Sample Mean)**:
$$\mathbb{E}[X] = \int x \, p(x)\,dx \approx \bar{x}_N = \frac{1}{N}\sum_{i=1}^N x_i$$

#### 🔬 Cách nó tác động lên mẫu & Định luật Số lớn (Law of Large Numbers):
Giả sử mỗi mẫu thu được bằng **Giá trị thật $\mu$ cộng với nhiễu ngẫu nhiên $\epsilon_i$** (có trung bình bằng $0$ và độ lệch chuẩn $\sigma$): $x_i = \mu + \epsilon_i$.
Khi bạn lấy trung bình $N$ mẫu:
$$\bar{x}_N = \frac{1}{N}\sum_{i=1}^N (\mu + \epsilon_i) = \mu + \underbrace{\frac{1}{N}\sum_{i=1}^N \epsilon_i}_{\text{Nhiễu trung bình } \to 0}$$
Sai số chuẩn (Standard Error) của trung bình mẫu giảm theo quy luật căn bậc hai:
$$\operatorname{SE}(\bar{x}_N) = \frac{\sigma}{\sqrt{N}}$$

> 🎯 **Nó nêu bật điều gì trên mẫu?**
> Phép lấy trung bình mẫu hoạt động như một **Bộ lọc Thông thấp (Noise Canceller)**: các nhiễu dương $+\epsilon$ và nhiễu âm $-\epsilon$ giữa các mẫu tự triệt tiêu lẫn nhau, **làm lộ ra "Trọng tâm bất biến" ($\mu$)** của đám mây dữ liệu. Muốn giảm sai số đi $10$ lần, bạn cần thu thập gấp $N = 10^2 = 100$ lần số mẫu!

---

### 3. Phương sai ($\operatorname{Var}$), Độ lệch chuẩn ($\sigma$) & Hiệp phương sai ($\operatorname{Cov}$) — Đo độ bung tỏa & rủi ro

#### 💡 Bản chất Logic:
Nếu Kỳ vọng cho biết **"Tâm đám mây mẫu nằm ở đâu"**, thì Phương sai và Hiệp phương sai cho biết **"Đám mây mẫu phình to cỡ nào và nghiêng về hướng nào"**.

$$\operatorname{Var}(X) = \mathbb{E}\big[(X - \mu)^2\big] \approx \frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2, \qquad \sigma = \sqrt{\operatorname{Var}(X)}$$

#### 🔬 Tại sao phải bình phương độ lệch $(x_i - \bar{x})^2$?
1. Nếu không bình phương, các mẫu lệch trái $(x_i - \bar{x} < 0)$ và lệch phải $(x_i - \bar{x} > 0)$ sẽ cộng lại bằng $0$!
2. Phép bình phương **phóng đại các mẫu nằm xa tâm**: một mẫu lệch gấp $3$ lần sẽ đóng góp gấp $3^2 = 9$ lần vào phương sai.

#### 🎯 Nó nêu bật điều gì trong RL? (Bài toán Đánh đổi Bias – Variance ở Tuần 2, 4, 5):
Hãy nhìn vào phương sai của tổng phần thưởng Monte Carlo $G_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-1} R_T$. Nếu các bước độc lập tương đối, phương sai cộng dồn qua từng bước thời gian:
$$\operatorname{Var}(G_t) \approx \sum_{k=0}^{T-1} \gamma^{2k} \operatorname{Var}(R_{t+k+1})$$
- **Hệ quả trực tiếp lên mẫu:** Một quỹ đạo dài $T=1000$ bước có phương sai gấp hàng trăm lần 1 bước đơn lẻ! Vì vậy mẫu $G_t$ nhảy loạn xạ từ $-500$ lên $+800$, làm đạo hàm Policy Gradient (REINFORCE) bị nhiễu nặng. Đó là lý do ta phải dùng **TD Bootstrap** (Tuần 2), **Baseline $V(s)$** (Tuần 4) và **GAE($\gamma, \lambda$)** (Tuần 5) để ép phương sai của mẫu xuống!

#### 📐 Ma trận Hiệp phương sai (Covariance Matrix $\boldsymbol{\Sigma}$):
Khi mỗi mẫu là một vector nhiều chiều $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ (ví dụ tọa độ robot $[x, y, \theta]$):
$$\boldsymbol{\Sigma} = \frac{1}{N}\sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T = \begin{bmatrix} \operatorname{Var}(x_1) & \operatorname{Cov}(x_1, x_2) \\ \operatorname{Cov}(x_2, x_1) & \operatorname{Var}(x_2) \end{bmatrix}$$
- **Phần tử đường chéo $\operatorname{Var}(x_j)$:** Độ trải rộng của mẫu dọc theo từng trục riêng lẻ.
- **Phần tử ngoài đường chéo $\operatorname{Cov}(x_1, x_2)$:** Nêu bật **mối tương quan tuyến tính** giữa 2 biến:
  - $\operatorname{Cov}(x_1, x_2) > 0$: Khi $x_1$ tăng thì $x_2$ cũng có xu hướng tăng (đám mây mẫu nghiêng chéo góc $+45^\circ$).
  - $\operatorname{Cov}(x_1, x_2) = 0$: Hai chiều dữ liệu biến thiên độc lập vuông góc nhau.

---

### 4. Xác suất có điều kiện $P(A \mid B)$, Định lý Bayes & Tính chất Markov

#### 💡 Bản chất Logic:
- **Xác suất có điều kiện $P(A \mid B)$:** Khi bạn đã biết chắc sự kiện $B$ xảy ra, bạn **vứt bỏ toàn bộ các mẫu không thuộc $B$**, và chuẩn hóa lại phần không gian mẫu còn lại.
- **Định lý Bayes (Cập nhật niềm tin khi có mẫu cảm biến mới):**
  $$\underbrace{P(\text{Trạng thái } s \mid \text{Cảm biến } z)}_{\text{Hậu nghiệm (Posterior)}} = \frac{\overbrace{P(z \mid s)}^{\text{Khả năng khớp (Likelihood)}} \cdot \overbrace{P(s)}^{\text{Tiên nghiệm (Prior)}}}{P(z)}$$
  - **Cách tác động lên mẫu (Ví dụ trong thuật toán định vị hạt AMCL của ROS 2 Nav2):**
    Ban đầu bạn rải $1000$ hạt mẫu $(x_i, y_i, \theta_i)$ khắp bản đồ (Prior $P(s)$). Khi LiDAR quét được khoảng cách tới tường là $z = 1.2\text{ m}$:
    - Hạt mẫu nào đứng cách tường $1.2\text{ m}$ trên bản đồ sẽ có $P(z \mid s_i)$ **rất cao** $\to$ được nhân trọng số lớn và **nhân bản lên**.
    - Hạt mẫu nào đứng giữa phòng trống (cách tường $5\text{ m}$) có $P(z \mid s_i) \approx 0$ $\to$ bị **loại bỏ**.
    - Sau vài bước lấy mẫu lại (Resampling), đám mây $1000$ hạt hội tụ chính xác vào đúng vị trí thật của Robot!

- **Tính chất Markov (Markov Property — Nền tảng của MDP ở Tuần 2):**
  $$P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = P(S_{t+1} \mid S_t, A_t)$$
  - **Ý nghĩa trên mẫu:** Trạng thái hiện tại $S_t$ đã **nén toàn bộ thông tin cần thiết của lịch sử quá khứ**. Nhờ vậy, khi lưu mẫu vào Replay Buffer $(s_t, a_t, r_t, s_{t+1})$, ta không cần phải lưu kèm cả nghìn bước lịch sử trước đó!
  - *(Hệ quả thực tế: Trong game Atari ở Tuần 3, 1 bức ảnh tĩnh không cho biết quả bóng đang bay sang trái hay sang phải $\to$ vi phạm tính Markov $\to$ phải xếp chồng 4 bức ảnh liên tiếp `FrameStack(4)` để khôi phục vận tốc!).*

---

### 5. Bất đẳng thức Jensen ($\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$) — Thủ phạm gây ảo tưởng giá trị (Overestimation)

#### 💡 Bản chất Logic:
Với mọi hàm lồi $g(x)$ (như hàm $x^2$, hàm $e^x$, hoặc đặc biệt là **hàm $\max(X_1, X_2, \dots, X_K)$**), trung bình của hàm luôn lớn hơn hoặc bằng hàm của trung bình:
$$\mathbb{E}\big[\max(X_1, X_2, \dots, X_K)\big] \ge \max\big(\mathbb{E}[X_1], \mathbb{E}[X_2], \dots, \mathbb{E}[X_K]\big)$$

#### 🔬 Cách toán tử $\max$ tác động sai lệch lên các mẫu có nhiễu:
Giả sử có 3 hành động $a_1, a_2, a_3$ đều có giá trị thật hoàn toàn bằng $0$: $Q^*(s, a_1) = Q^*(s, a_2) = Q^*(s, a_3) = 0$.
Nhưng vì ước lượng từ mẫu ngẫu nhiên nên mỗi lần ước lượng có nhiễu $\hat{Q}(s, a_k) \sim \mathcal{N}(0, 1)$:
- Lần thử 1: Mẫu ước lượng ra $[-0.8, +1.2, -0.3] \xrightarrow{\max} +1.2$
- Lần thử 2: Mẫu ước lượng ra $[+0.9, -1.1, +0.4] \xrightarrow{\max} +0.9$
- Lần thử 3: Mẫu ước lượng ra $[-0.2, -0.5, +1.5] \xrightarrow{\max} +1.5$

Trung bình của từng hành động vẫn là $\approx 0$, nhưng **trung bình của $\max$ lại vọt lên $+1.2 > 0$**!

> 🎯 **Nó nêu bật điều gì?**
> Toán tử $\max$ hoạt động như một **"Bộ lọc chỉ chọn những mẫu bị nhiễu dương"**. Càng có nhiều hành động $K$ và phương sai mẫu $\sigma^2$ càng lớn, độ phóng đại ảo $\mathbb{E}[\max_a \hat{Q}(s,a)] - \max_a Q^*(s,a)$ càng nghiêm trọng. Đây chính là lý do **Double DQN (Tuần 3)** phải tách mạng **Chọn hành động ($\arg\max$)** và mạng **Đánh giá mẫu ($Q_{\theta^-}$)** ra làm đôi!

---

### 6. Lấy mẫu Trọng số Quan trọng (Importance Sampling) — Tái chế mẫu cũ cho chính sách mới

#### 💡 Bản chất Logic:
Giả sử bạn muốn tính điểm thưởng trung bình của **Chính sách mới $\pi_\theta$**, nhưng trong bộ nhớ chỉ có các mẫu $(s_i, a_i, r_i)$ được thu thập từ **Chính sách cũ $\pi_{\theta_{\text{old}}}$**. Làm sao tính kỳ vọng dưới $\pi_\theta$ mà không cần chạy lại mô phỏng?

#### 📐 Công thức Toán học:
$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \sum_a \pi_\theta(a \mid s) f(a) = \sum_a \pi_{\theta_{\text{old}}}(a \mid s) \underbrace{\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}}_{r_t(\theta) \text{ (Tỉ số IS)}} f(a) = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}}\!\Big[ r_t(\theta) \cdot f(a) \Big]$$

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution):**
> - **Bước 1 (Hàm mục tiêu gốc cần tính):** Ta cần tính kỳ vọng dưới phân phối mới $\pi_\theta$:
>   $$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[f(a)] = \sum_{a} \pi_\theta(a \mid s) \cdot f(a) \quad (1)$$
> - **Bước 2 (Tạo hằng đẳng thức nhân - chia với hàm phân phối cũ $\pi_{\theta_{\text{old}}}$):** Vì $\frac{\pi_{\theta_{\text{old}}}(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} = 1$ (với mọi $a$ có $\pi_{\theta_{\text{old}}}(a \mid s) > 0$), ta viết lại hàm $\pi_\theta(a \mid s)$ dưới dạng tích của hàm cũ và hàm tỉ số $r_t(\theta)$:
>   $$\pi_\theta(a \mid s) = \pi_{\theta_{\text{old}}}(a \mid s) \cdot \underbrace{\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}}_{\text{Đặt bằng hàm } r_t(\theta)} = \pi_{\theta_{\text{old}}}(a \mid s) \cdot r_t(\theta) \quad (2)$$
> - **Bước 3 (Thế hàm $(2)$ vào vị trí của $\pi_\theta(a \mid s)$ trong $(1)$):**
>   $$J(\theta) = \sum_{a} \Big[\pi_{\theta_{\text{old}}}(a \mid s) \cdot r_t(\theta)\Big] \cdot f(a) = \sum_{a} \pi_{\theta_{\text{old}}}(a \mid s) \cdot \Big[r_t(\theta) \cdot f(a)\Big]$$
> - **Bước 4 (Thu gọn về định nghĩa hàm Kỳ vọng theo phân phối cũ):** Theo định nghĩa $\sum_a P(a) \cdot g(a) = \mathbb{E}_{a \sim P}[g(a)]$, ta thế cụm $\sum_a \pi_{\theta_{\text{old}}}(a \mid s)[\dots]$ thành toán tử kỳ vọng $\mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}}[\dots]$:
>   $$J(\theta) = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}}\!\Big[ r_t(\theta) \cdot f(a) \Big]$$
>   *Ý nghĩa:* Nhờ phép thế này, ta đã đổi hoàn toàn việc phải lấy mẫu từ hàm mới $\pi_\theta$ (chưa chạy) sang lấy trung bình trên các mẫu có sẵn của hàm cũ $\pi_{\theta_{\text{old}}}$!

#### 🔬 Cách nó tác động lên từng mẫu:
Tỉ số Importance Sampling $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ đóng vai trò là **Hệ số co giãn trọng lượng của từng mẫu**:
- Nếu mẫu hành động $a_t$ được chính sách mới $\pi_\theta$ ưu ái hơn chính sách cũ ($r_t(\theta) > 1$): Nó **phóng đại tiếng nói của mẫu đó lên** (như thể mẫu đó xuất hiện nhiều lần hơn).
- Nếu mẫu hành động $a_t$ ít được $\pi_\theta$ chọn hơn ($r_t(\theta) < 1$): Nó **dìm trọng lượng của mẫu đó xuống**.
- **Nguy cơ khi $r_t(\theta)$ quá lớn:** Nếu $\pi_{\theta_{\text{old}}}(a_t) = 0.001$ mà $\pi_\theta(a_t) = 0.8 \implies r_t = 800$. Chỉ 1 mẫu duy nhất bị nhân gấp $800$ lần sẽ làm nổ phương sai của toàn bộ batch! Đó là lý do **PPO (Tuần 5)** phải dùng hàm `clip(r_t, 1 - ε, 1 + ε)` để khóa cứng trọng số mẫu trong đoạn $[0.8, 1.2]$.

---

### 7. Lý thuyết Thông tin: Entropy ($\mathcal{H}$) & Phân kỳ KL ($D_{\text{KL}}$)

#### 7.1. Shannon Entropy $\mathcal{H}(\pi)$ — Thước đo độ phẳng của đám mây mẫu
$$\mathcal{H}(\pi(\cdot \mid s)) = -\sum_{a \in \mathcal{A}} \pi(a \mid s) \ln \pi(a \mid s) = \mathbb{E}_{a \sim \pi}\big[-\ln \pi(a \mid s)\big]$$
- **Cách tác động:** Đo lượng thông tin ("độ bất ngờ" $-\ln \pi(a\mid s)$) trung bình khi rút một mẫu từ phân phối $\pi$.
- **Nêu bật điều gì?**
  - **Entropy Cực đại ($\mathcal{H} = \ln |\mathcal{A}|$):** Khi mọi hành động có xác suất ngang nhau $\pi(a) = 1/|\mathcal{A}|$. Các mẫu rút ra đa dạng tối đa $\to$ Agent đang **tích cực khám phá**.
  - **Entropy Bằng $0$ ($\mathcal{H} = 0$):** Khi 1 hành động chiếm xác suất $100\%$ ($\pi(a^*)=1$, các nhánh khác bằng $0$). Mọi mẫu rút ra đều trùng làm một $\to$ Chính sách đã **đông cứng (Policy Collapse)**.
  - Vì vậy trong **A2C (Tuần 4)** và **SAC (Tuần 8)**, việc cộng $+\beta \mathcal{H}(\pi)$ vào hàm mục tiêu có tác dụng **"Bơm hơi đẩy phồng đám mây mẫu ra"**, ngăn không cho mạng nơ-ron hội tụ sớm vào một lối mòn cục bộ!

#### 7.2. Phân kỳ Kullback–Leibler ($D_{\text{KL}}(P \| Q)$) — Thước đo khoảng cách giữa 2 quy luật sinh mẫu
$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \ln \!\left(\frac{P(x)}{Q(x)}\right) = \mathbb{E}_{x \sim P}\big[\ln P(x) - \ln Q(x)\big] \ge 0$$
- **Tại sao không dùng khoảng cách hiệu bình phương $\|\theta_{\text{new}} - \theta_{\text{old}}\|_2^2$ mà phải dùng $D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}})$?**
  - Giả sử chính sách Gaussian có $\sigma = 0.001$ (rất nhọn): Dịch tâm $\mu$ đi một đoạn cực nhỏ $\Delta \mu = 0.01$ sẽ làm hai đám mây mẫu **tách rời nhau hoàn toàn** ($D_{\text{KL}} = 50.0$, thay đổi hành vi 100%).
  - Nhưng nếu $\sigma = 10.0$ (rất bẹt): Dịch tâm $\mu$ đi cùng một đoạn $\Delta \mu = 0.01$ thì hai đám mây mẫu **vẫn trùng khít 99.9%** ($D_{\text{KL}} = 0.0000005$).
  - 👉 **Kết luận:** $D_{\text{KL}}$ nêu bật **sự thay đổi thực sự về phân phối mẫu đầu ra**, bất chấp tham số bên trong mạng được co giãn thế nào! Đây là nền tảng của **TRPO (Tuần 5)** và **DPO (Tuần 8)**.

---

## PHẦN II: ĐẠI SỐ TUYẾN TÍNH (LINEAR ALGEBRA) — NGHỆ THUẬT BIẾN ĐỔI KHÔNG GIAN

Nếu Xác suất xử lý sự ngẫu nhiên của mẫu, thì **Đại số Tuyến tính** là ngôn ngữ để **biểu diễn hình học, xoay hệ trục tọa độ, nén đặc trưng và giải hệ phương trình động lực học** trên các mẫu dữ liệu nhiều chiều.

---

### 1. Vector & Tích vô hướng (Dot Product $\mathbf{u} \cdot \mathbf{v}$) — Bộ đo độ tương đồng & hình chiếu

#### 💡 Bản chất Logic & Công thức:
Một mẫu trạng thái của robot (gồm vị trí, vận tốc, 12 tia LiDAR) được gom thành một mũi tên trong không gian $d$ chiều gọi là **Vector** $\mathbf{x} \in \mathbb{R}^d$.
**Tích vô hướng (Inner / Dot Product)** giữa hai vector $\mathbf{u}$ và $\mathbf{v}$ được định nghĩa theo cả đại số lẫn hình học:
$$\mathbf{u}^T \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\|_2 \cdot \|\mathbf{v}\|_2 \cdot \cos(\theta)$$

```
                Vector mẫu u
                  ↗
                ╱  |
              ╱    |
            ╱ θ    |  (Chiếu vuông góc u xuống v)
          •────────┼────────► Vector mẫu chuẩn / Trọng số v
          |◄──────►|
       Độ dài hình chiếu = ||u|| cos(θ) = (u · v) / ||v||
```

#### 🔬 Cách nó tác động lên mẫu & Nêu bật điều gì?
Khi bạn lấy tích vô hướng của một vector mẫu trạng thái $\mathbf{x}$ với một vector trọng số $\mathbf{w}$ (như trong tầng `nn.Linear` hoặc hàm xấp xỉ tuyến tính $\hat{V}(s) = \mathbf{w}^T \boldsymbol{\phi}(s)$):
1. **Nếu $\mathbf{w}^T \mathbf{x} > 0$ lớn ($\cos\theta \approx 1$):** Mẫu $\mathbf{x}$ **cùng hướng (khớp mẫu / tương đồng cao)** với bộ lọc $\mathbf{w}$ $\to$ Nơ-ron kích hoạt mạnh!
2. **Nếu $\mathbf{w}^T \mathbf{x} = 0$ ($\cos\theta = 0, \theta = 90^\circ$):** Mẫu $\mathbf{x}$ **vuông góc (trực giao / không liên quan gì)** tới đặc trưng $\mathbf{w}$ $\to$ Nơ-ron bằng $0$ (bỏ qua hoàn toàn thành phần nhiễu vuông góc này!).
3. **Nếu $\mathbf{w}^T \mathbf{x} < 0$ ($\cos\theta \approx -1$):** Mẫu $\mathbf{x}$ **ngược hướng** với $\mathbf{w}$.

---

### 2. Phép nhân Ma trận ($\mathbf{y} = \mathbf{A}\mathbf{x}$) — Phép biến hình đám mây mẫu (Xoay, Co giãn, Chiếu)

#### 💡 Bản chất Logic:
Đừng nhìn một ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ như một bảng số vô hồn! Hãy nhìn nó như một **Cỗ máy Biến hình Không gian**:
- Mỗi **cột thứ $j$ của ma trận $\mathbf{A}$** chính là tọa độ mới của vector đơn vị trục thứ $j$ sau khi bị biến hình!
- Khi nhân $\mathbf{Y} = \mathbf{A}\mathbf{X}$ (với $\mathbf{X}$ là ma trận chứa $N$ mẫu), bạn đang đồng thời **xoay, kéo giãn, hoặc chiếu phẳng toàn bộ đám mây $N$ mẫu sang một hệ tọa độ mới**.

#### 🔬 3 loại Ma trận quan trọng nhất tác động lên mẫu trong RL & Robotics:

1. **Ma trận Trực giao / Ma trận Xoay ($\mathbf{R}^T \mathbf{R} = \mathbf{I}, \det(\mathbf{R}) = 1$):**
   $$\begin{bmatrix} \Delta x_{\text{body}} \\ \Delta y_{\text{body}} \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} \Delta x_{\text{world}} \\ \Delta y_{\text{world}} \end{bmatrix}$$
   - **Tác động lên mẫu:** Xoay cứng toàn bộ đám mây điểm LiDAR và điểm đích từ hệ tọa độ mặt đất (`map`/`odom`) về hệ tọa độ gắn trên mũi robot (`base_link`) mà **bảo toàn tuyệt đối khoảng cách và góc giữa các mẫu**: $\|\mathbf{R}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$.

2. **Ma trận Đường chéo ($\mathbf{D} = \operatorname{diag}(d_1, d_2, \dots, d_n)$):**
   - **Tác động lên mẫu:** Kéo giãn hoặc nén từng trục tọa độ độc lập (ví dụ chuẩn hóa Observation Normalization: chia từng chiều của mẫu cho độ lệch chuẩn $\sigma_i$ của chiều đó để đưa mọi đặc trưng về cùng thang đo $[-1, 1]$).

3. **Ma trận Trọng số Mạng Nơ-ron ($\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$):**
   - Khi $d_{\text{out}} < d_{\text{in}}$ (như Encoder CNN trong DQN Atari nén ảnh $84\times 84 = 7056$ chiều về $512$ chiều): Nó **chiếu phẳng (Project)** đám mây mẫu từ không gian nhiều chiều xuống một không gian con chứa các đặc trưng cốt lõi nhất (vị trí bóng, vận tốc vợt), lọc bỏ các điểm ảnh nền dư thừa.

---

### 3. Trị riêng ($\lambda_i$) & Vector riêng ($\mathbf{v}_i$) — Tìm trục xương sống bất biến của dữ liệu

#### 💡 Bản chất Logic:
Khi một ma trận vuông $\mathbf{A}$ tác động lên mọi vector mẫu $\mathbf{x}$ trong không gian, hầu hết các vector đều bị **vừa kéo giãn vừa bị bẻ lệch hướng**.
Tuy nhiên, có những hướng đặc biệt $\mathbf{v}_i \neq \mathbf{0}$ mà khi ma trận $\mathbf{A}$ tác động vào, vector đó **KHÔNG HỀ BỊ ĐỔI HƯỚNG**, chỉ bị phóng to hoặc thu nhỏ đúng $\lambda_i$ lần:
$$\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$$
- $\mathbf{v}_i$: Gọi là **Vector riêng (Eigenvector)** — định nghĩa **"Trục xương sống / Trục chính bất biến"** của phép biến đổi.
- $\lambda_i$: Gọi là **Trị riêng (Eigenvalue)** — định nghĩa **"Hệ số phóng đại / Sức mạnh"** dọc theo trục chính đó.

```
     Đám mây mẫu 2D có tương quan chéo:
             y ▲              ↗ Trục Vector riêng chính v_1 (Trị riêng λ_1 LỚN:
               │           •  •   Hướng dữ liệu biến thiên mạnh nhất)
               │      ↖  •  •  •
               │    v_2 •  •  •
               │       •  •
               └─────────────────► x
             (v_2 có λ_2 NHỎ: Hướng vuông góc ít biến thiên)
```

#### 🎯 Trị riêng & Vector riêng làm nổi bật 4 tính chất sống còn trong RL & Robotics:

1. **Trên Ma trận Hiệp phương sai Mẫu $\boldsymbol{\Sigma}$ (Phân tích thành phần chính PCA):**
   - Vector riêng $\mathbf{v}_{\max}$ ứng với trị riêng lớn nhất $\lambda_{\max}$ chỉ ra **phương hướng mà đám mây mẫu đang trải dài ra mạnh nhất**.
   - Trị riêng $\lambda_i = 0$ báo hiệu chiều dữ liệu đó **bị suy biến (dư thừa 100%)** vì có thể suy ra tuyến tính từ các chiều khác.

2. **Trên Ma trận Quán tính Vật lý $\mathbf{I}_{3\times 3}$ của Robot (Tầng 1 Simulation):**
   - 3 Vector riêng là **3 trục đối xứng cơ học chính** của thân robot (nếu quay quanh 3 trục này, robot quay êm mà không sinh lực văng lệch tâm).
   - 3 Trị riêng $I_1, I_2, I_3$ là mô-men quán tính chính; bắt buộc phải thỏa mãn $I_1, I_2, I_3 > 0$ và $I_1 + I_2 \ge I_3$ (như đã kiểm định trong `01_urdf_physics_inspector.py`).

3. **Trên Ma trận Chuyển trạng thái Markov $\mathbf{P}$ & Hệ Động lực học $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t$ (Tuần 2 & Tuần 7):**
   Nếu áp dụng ma trận chuyển tiếp lặp đi lặp lại $t$ bước ($\mathbf{x}_t = \mathbf{A}^t \mathbf{x}_0 = \sum c_i \lambda_i^t \mathbf{v}_i$):
   - **Nếu mọi $|\lambda_i| < 1$:** $\lambda_i^t \to 0$ khi $t \to \infty$ $\implies$ Mọi sai số và dao động đều **tắt dần về trạng thái cân bằng ổn định**.
   - **Nếu tồn tại $|\lambda_i| = 1$:** Tương ứng với **Phân phối dừng (Stationary State Distribution $\boldsymbol{\mu}^T \mathbf{P} = 1 \cdot \boldsymbol{\mu}^T$)** của chuỗi Markov!
   - **Nếu tồn tại $|\lambda_i| > 1$:** $\lambda_i^t \to \infty$ theo cấp số nhân $\implies$ **Mất ổn định số / Nổ quỹ đạo (Exploding Gradient / Unstable Rollout)**!

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Tại sao $\mathbf{x}_t = \mathbf{A}^t \mathbf{x}_0$ lại biến thành tổng lũy thừa của các Trị riêng $\lambda_i^t$?**
> - **Bước 1 (Biểu diễn mẫu ban đầu $\mathbf{x}_0$ theo cơ sở các Vector riêng):**
>   Giả sử ma trận $\mathbf{A} \in \mathbb{R}^{d\times d}$ có $d$ vector riêng độc lập tuyến tính $\{\mathbf{v}_1, \dots, \mathbf{v}_d\}$ thỏa mãn định nghĩa $\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$. Khi đó, bất kỳ trạng thái khởi đầu $\mathbf{x}_0$ nào cũng có thể phân tích thành tổ hợp tuyến tính:
>   $$\mathbf{x}_0 = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_d \mathbf{v}_d = \sum_{i=1}^d c_i \mathbf{v}_i$$
> - **Bước 2 (Thế $\mathbf{x}_0$ vào bước chuyển tiếp thứ nhất $\mathbf{x}_1 = \mathbf{A}\mathbf{x}_0$):**
>   $$\mathbf{x}_1 = \mathbf{A}\left(\sum_{i=1}^d c_i \mathbf{v}_i\right) = \sum_{i=1}^d c_i (\mathbf{A}\mathbf{v}_i)$$
> - **Bước 3 (Thế định nghĩa Vector riêng $\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$ vào biểu thức trên):**
>   Thay cụm $(\mathbf{A}\mathbf{v}_i)$ (phép nhân ma trận - vector phức tạp) bằng $(\lambda_i \mathbf{v}_i)$ (phép nhân số vô hướng đơn giản):
>   $$\mathbf{x}_1 = \sum_{i=1}^d c_i \lambda_i \mathbf{v}_i$$
> - **Bước 4 (Lặp lại phép thế $t$ lần cho $\mathbf{x}_t = \mathbf{A}^t \mathbf{x}_0$):**
>   Mỗi lần nhân thêm $\mathbf{A}$ vào $\mathbf{v}_i$ lại sinh ra thêm một thừa số $\lambda_i$ ($\mathbf{A}^t \mathbf{v}_i = \lambda_i^t \mathbf{v}_i$). Thế vào ta được:
>   $$\mathbf{x}_t = \mathbf{A}^t \left(\sum_{i=1}^d c_i \mathbf{v}_i\right) = \sum_{i=1}^d c_i (\mathbf{A}^t \mathbf{v}_i) = \sum_{i=1}^d c_i \lambda_i^t \mathbf{v}_i$$
>   *Kết luận trực quan:* Nhờ thế $\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$, phép lũy thừa ma trận $\mathbf{A}^t$ được tách thành $d$ kênh độc lập $\lambda_i^t$, giải thích tại sao chỉ cần 1 trị riêng $|\lambda_i| > 1$ là toàn bộ hệ thống phát nổ khi $t$ lớn!

4. **Số Điều kiện của Ma trận ($\kappa(\mathbf{A}) = \frac{|\lambda_{\max}|}{|\lambda_{\min}|}$ — Condition Number):**
   - Nêu bật xem bài toán tối ưu hóa dễ hay khó: Nếu $\lambda_{\max} \gg \lambda_{\min}$ (thung lũng hàm mất mát hẹp dài như khe vực), Gradient Descent thường sẽ đập qua đập lại hai bên vách núi rất chậm $\to$ phải dùng **Adam Optimizer** hoặc **Natural Gradient (Ma trận Fisher $\mathbf{F}^{-1}$ ở Tuần 5)** để nắn lại các trục có trị riêng lệch nhau về cùng tỷ lệ $1:1$!

---

### 4. Các Chuẩn Đo lường ($L_1, L_2, L_\infty$, Frobenius Norm) — Cách chọn kính lúp soi lỗi trên mẫu

Giả sử mô hình dự đoán sai lệch trên một tập gồm $N$ mẫu với vector sai số $\mathbf{e} = [e_1, e_2, \dots, e_N]^T$. Cách bạn chọn **Chuẩn (Norm $\|\mathbf{e}\|_p$)** quyết định **mẫu nào sẽ chi phối quá trình học**:

| Tên Chuẩn | Công thức Toán học | Cách đối xử với từng mẫu sai số $e_i$ | Khi nào dùng trong RL & Robotics? |
| :--- | :--- | :--- | :--- |
| **Chuẩn $L_1$<br>(Manhattan)** | $\|\mathbf{e}\|_1 = \sum_{i=1}^N \|e_i\|$ | Đạo hàm theo $e_i$ là $\operatorname{sgn}(e_i) = \pm 1$ **không phụ thuộc độ lớn của lỗi**. Một mẫu nhiễu dị biệt ($e_i = 1000$) cũng chỉ kéo gradient bằng $1$ đơn vị. | **Kháng nhiễu dị biệt (Robust to Outliers):** Kết hợp với $L_2$ tạo thành **Smooth $L_1$ / Huber Loss** trong DQN (Tuần 3) để tránh nổ gradient khi gặp mẫu TD-error bất thường. |
| **Chuẩn $L_2$<br>(Euclidean)** | $\|\mathbf{e}\|_2 = \sqrt{\sum_{i=1}^N e_i^2}$ | Đạo hàm của $\|\mathbf{e}\|_2^2$ tỉ lệ bậc nhất với chính sai số ($2 e_i$). Mẫu nào sai gấp $10$ lần sẽ kéo gradient mạnh gấp $10$ lần; khi $e_i \to 0$ thì bước nhảy tự nhỏ dần về $0$ rất êm. | **MSE Loss** của Critic $V_\phi(s)$, **Weight Decay (L2 Regularization)**, và **Gradient Norm Clipping** (`clip_grad_norm_`). |
| **Chuẩn $L_\infty$<br>(Max / Chebyshev)** | $\|\mathbf{e}\|_\infty = \max_{1 \le i \le N} \|e_i\|$ | Bỏ qua hoàn toàn $N-1$ mẫu tốt, **chỉ soi duy nhất vào 1 mẫu có sai số tồi tệ nhất!** | **Chứng minh Hội tụ Toán tử Bellman (Tuần 2):** $\|\mathcal{T}V_1 - \mathcal{T}V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$ (đảm bảo trạng thái tệ nhất trên bàn cờ cũng phải co sai số về $0$). |
| **Chuẩn Frobenius<br>(Ma trận)** | $\|\mathbf{J}\|_F = \sqrt{\sum_{i,j} J_{ij}^2}$ | Tổng bình phương mọi phần tử của ma trận (bằng căn tổng bình phương các giá trị kỳ dị $\sqrt{\sum \sigma_i^2}$). | **Tikhonov Regularization (Tuần 7):** Phạt $\|\mathbf{J}_f(s,a)\|_F^2$ để ngăn mô hình World Model phóng đại sai số qua nhiều bước rollout. |

---

### 5. Ma trận Xác định Dương ($\mathbf{A} \succ 0$) & Dạng Toàn phương ($\mathbf{x}^T \mathbf{A} \mathbf{x}$) — Mặt bát năng lượng

#### 💡 Bản chất Logic:
Biểu thức bậc hai $\mathcal{Q}(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x} = \sum_i \sum_j x_i A_{ij} x_j$ được gọi là một **Dạng toàn phương (Quadratic Form)**.
Một ma trận đối xứng $\mathbf{A}$ được gọi là **Xác định dương ($\mathbf{A} \succ 0$, Positive Definite)** nếu với mọi vector khác không $\mathbf{x} \neq \mathbf{0}$, ta luôn có:
$$\mathbf{x}^T \mathbf{A} \mathbf{x} > 0 \iff \text{Tất cả các trị riêng } \lambda_1, \lambda_2, \dots, \lambda_d > 0$$

#### 🎯 Ý nghĩa Vật lý & Tối ưu hóa cực kỳ quan trọng:
1. **Hình học:** Đồ thị của hàm $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x}$ có hình dạng một **Chiếc bát lõm hoàn hảo (Convex Bowl)** có đáy duy nhất tại $\mathbf{x} = \mathbf{0}$.
2. **Trong Vật lý Robot:**
   - Động năng tịnh tiến và quay của robot là $E_k = \frac{1}{2}\dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}}$. Vì động năng của vật đang chuyển động ($\dot{\mathbf{q}} \neq \mathbf{0}$) không bao giờ âm hay bằng $0$, **Ma trận Khối lượng – Quán tính $\mathbf{M}(\mathbf{q})$ và $\mathbf{I}_{3\times 3}$ bắt buộc phải Xác định Dương ($\succ 0$)**!
3. **Trong Lý thuyết Điều khiển (Hàm năng lượng Lyapunov $V(\mathbf{x}) = \mathbf{x}^T \mathbf{P} \mathbf{x} > 0$) và TRPO (Tuần 5):**
   - Khoảng cách vùng tin cậy KL bậc hai $\frac{1}{2}\Delta\boldsymbol{\theta}^T \mathbf{F} \Delta\boldsymbol{\theta} \le \delta$ tạo thành một hình Ellipsoid đóng kín nhờ ma trận Fisher $\mathbf{F} \succeq 0$.

---

## PHẦN III: GIẢI TÍCH ĐA BIẾN & TỐI ƯU HÓA (CALCULUS) — KẾT NỐI ĐẠI SỐ VÀ XÁC SUẤT

Giải tích là chiếc cầu nối biến các đại lượng Đại số Tuyến tính và Xác suất Thống kê thành **Quy tắc dịch chuyển tham số ($\theta_{t+1} = \theta_t + \Delta \theta$)** để máy tự học từ mẫu.

---

### 1. Gradient ($\nabla f$), Ma trận Jacobian ($\mathbf{J}$) & Ma trận Hessian ($\mathbf{H}$)

| Ký hiệu | Kích thước (Đầu vào $\mathbb{R}^n \to$ Đầu ra) | Ý nghĩa Hình học & Tác động lên mẫu |
| :--- | :--- | :--- |
| **Gradient $\nabla_\theta J(\theta)$** | $\mathbb{R}^n \to \mathbb{R}^1$<br>Vector cột $n \times 1$ | Chỉ hướng **dốc lên nhanh nhất** của hàm thưởng $J(\theta)$. Mỗi mẫu $(s_i, a_i)$ đóng góp một mũi tên nhỏ $\nabla_\theta \ln \pi_\theta(a_i \mid s_i) A_i$ kéo tham số $\theta$ về phía nó. |
| **Jacobian $\mathbf{J}_f(\mathbf{x}) = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$** | $\mathbb{R}^n \to \mathbb{R}^m$<br>Ma trận $m \times n$ | Xấp xỉ tuyến tính cục bộ: $\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}_f(\mathbf{x})\Delta\mathbf{x}$. Cho biết nếu **mẫu đầu vào lệch một chút $\Delta\mathbf{x}$ thì đầu ra bị lệch bao nhiêu**. Trong Robot, Jacobian cánh tay/bánh xe ánh xạ vận tốc khớp sang vận tốc bàn tay/thân xe: $\mathbf{v} = \mathbf{J}(\mathbf{q})\dot{\mathbf{q}}$. |
| **Hessian $\mathbf{H}_f(\theta) = \nabla^2_\theta f(\theta)$** | $\mathbb{R}^n \to \mathbb{R}^1$<br>Ma trận vuông $n \times n$ | Đo **độ cong bậc 2 (Curvature)** của mặt hàm mục tiêu. Nếu $\mathbf{H}$ có trị riêng rất lớn theo hướng nào, bề mặt hàm đang uốn cong rất gắt theo hướng đó $\to$ phải giảm bước nhảy để không bị văng khỏi đỉnh đồi! |

---

### 2. Thủ thuật Đạo hàm Logarit (Log-Derivative Trick) — Biến đạo hàm của kỳ vọng thành kỳ vọng của đạo hàm

#### 💡 Vấn đề hóc búa nhất trong RL:
Ta muốn tính đạo hàm của kỳ vọng phần thưởng $\nabla_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]$. Nhưng xác suất của quỹ đạo $p_\theta(\tau) = p(s_0)\prod \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$ chứa hàm chuyển tiếp vật lý $P(s_{t+1} \mid s_t, a_t)$ **không biết công thức và không khả vi**!

#### 📐 Phép biến đổi thần kỳ (Likelihood Ratio / Score Function Trick - Tuần 4):
Sử dụng hằng đẳng thức giải tích cơ bản $\nabla_\theta \ln g(\theta) = \frac{\nabla_\theta g(\theta)}{g(\theta)} \iff \nabla_\theta g(\theta) = g(\theta) \nabla_\theta \ln g(\theta)$:
$$\begin{aligned}
\nabla_\theta \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] &= \nabla_\theta \int p_\theta(\tau) R(\tau)\,d\tau \\
&= \int \big(\nabla_\theta p_\theta(\tau)\big) R(\tau)\,d\tau \\
&= \int p_\theta(\tau) \big(\nabla_\theta \ln p_\theta(\tau)\big) R(\tau)\,d\tau \\
&= \mathbb{E}_{\tau \sim p_\theta}\Big[ \nabla_\theta \ln p_\theta(\tau) \cdot R(\tau) \Big]
\end{aligned}$$
Và vì $\ln\big(\prod_t \pi_\theta \cdot P\big) = \sum_t \ln \pi_\theta + \sum_t \ln P$, khi lấy đạo hàm $\nabla_\theta$, toàn bộ số hạng $\nabla_\theta \ln P(s_{t+1}\mid s_t, a_t) = 0$ **biến mất hoàn toàn**!

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Hai lần thế hàm làm nên Định lý Policy Gradient:**
> - **Phép thế thứ 1 (Thế Đạo hàm bằng Đạo hàm Logarit — Log-Derivative Identity):**
>   - *Hàm gốc:* $\nabla_\theta J(\theta) = \int \big(\nabla_\theta p_\theta(\tau)\big) R(\tau)\,d\tau$
>   - *Hằng đẳng thức đem thế:* Từ công thức đạo hàm hàm hợp $\nabla_\theta \ln p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$, nhân chéo mẫu số $p_\theta(\tau)$ lên ta có:
>     $$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\nabla_\theta \ln p_\theta(\tau)$$
>   - *Thực hiện thế vào tích phân:* Thay cụm $\nabla_\theta p_\theta(\tau)$ bằng $p_\theta(\tau)\nabla_\theta \ln p_\theta(\tau)$ để xuất hiện lại mật độ xác suất $p_\theta(\tau)$ (điều kiện bắt buộc để đưa tích phân trở lại dạng Kỳ vọng $\mathbb{E}_{\tau \sim p_\theta}[\cdot]$):
>     $$\nabla_\theta J(\theta) = \int p_\theta(\tau) \Big[\nabla_\theta \ln p_\theta(\tau) R(\tau)\Big] d\tau = \mathbb{E}_{\tau \sim p_\theta}\Big[\nabla_\theta \ln p_\theta(\tau) R(\tau)\Big]$$
> - **Phép thế thứ 2 (Thế Xác suất Quỹ đạo $p_\theta(\tau)$ vào hàm $\ln(\cdot)$ để triệt tiêu Môi trường):**
>   - *Hàm đem thế:* Xác suất sinh ra quỹ đạo $\tau = (s_0, a_0, s_1, \dots, s_T)$ theo chuỗi Markov là tích:
>     $$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$$
>   - *Thực hiện thế vào $\ln p_\theta(\tau)$ (biến tích $\prod$ thành tổng $\sum$):*
>     $$\ln p_\theta(\tau) = \ln p(s_0) + \sum_{t=0}^{T-1} \ln \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \ln P(s_{t+1} \mid s_t, a_t)$$
>   - *Lấy đạo hàm $\nabla_\theta$ hai vế và thế ngược lại vào Kỳ vọng:* Vì $p(s_0)$ và $P(s_{t+1} \mid s_t, a_t)$ không phụ thuộc vào $\theta$ nên đạo hàm của chúng bằng $0$:
>     $$\nabla_\theta \ln p_\theta(\tau) = \underbrace{\nabla_\theta \ln p(s_0)}_{=0} + \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t) + \underbrace{\sum_{t=0}^{T-1} \nabla_\theta \ln P(s_{t+1} \mid s_t, a_t)}_{=0} = \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t)$$
>   - *Kết quả sau 2 lần thế:*
>     $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t)\right) R(\tau)\right]$$

> 🎯 **Tác động lên mẫu:**
> Nhờ phép lấy $\ln$ (biến tích thành tổng) và đạo hàm logarit, ta có thể **ước lượng đạo hàm chỉ bằng cách rút mẫu quỹ đạo thực tế** mà không cần biết phương trình vật lý của môi trường!

---

### 3. Khai triển Taylor & Ma trận Thông tin Fisher ($\mathbf{F}$) — Bản đồ địa hình vùng tin cậy

Khi cập nhật tham số từ $\theta_k$ sang $\theta$, khai triển Taylor bậc 2 của phân kỳ KL xung quanh $\theta_k$ là:
$$D_{\text{KL}}\big(\pi_{\theta_k} \,\|\, \pi_\theta\big) \approx \underbrace{D_{\text{KL}}(\pi_{\theta_k} \| \pi_{\theta_k})}_{=0} + \underbrace{\nabla_\theta D_{\text{KL}}\big|_{\theta=\theta_k}^T}_{=\mathbf{0}} (\theta - \theta_k) + \frac{1}{2}(\theta - \theta_k)^T \mathbf{F}(\theta_k) (\theta - \theta_k)$$
trong đó **Ma trận Thông tin Fisher (Fisher Information Matrix $\mathbf{F}$)** được tính trực tiếp từ trung bình mẫu của tích ngoài vector Score:
$$\mathbf{F}(\theta) = \mathbb{E}_{s, a \sim \pi_\theta}\Big[ \big(\nabla_\theta \ln \pi_\theta(a \mid s)\big)\big(\nabla_\theta \ln \pi_\theta(a \mid s)\big)^T \Big]$$
- **Ý nghĩa trên mẫu:** Ma trận Fisher $\mathbf{F}$ vừa là **Ma trận Hiệp phương sai của Gradient Log-Policy trên các mẫu**, vừa là **Ma trận Độ cong (Hessian) của khoảng cách KL**. Nhân nghịch đảo $\mathbf{F}^{-1} \nabla_\theta J$ (Natural Gradient trong TRPO - Tuần 5) giúp tự động **hãm phanh ở những chiều nhạy cảm** và **tăng tốc ở những chiều phẳng**, giữ cho phân phối mẫu của Agent không bao giờ bị sụp đổ!

> 🔄 **Giải phẫu các bước thế hàm (Step-by-Step Function Substitution) — Tại sao số hạng bậc 0, bậc 1 đều bằng $0$ và số hạng bậc 2 lại là Ma trận Fisher $\mathbf{F}$?**
> - **Bước 1 (Hàm gốc đem khai triển Taylor quanh điểm $\theta = \theta_k$):**
>   Đặt $g(\theta) = D_{\text{KL}}(\pi_{\theta_k} \| \pi_\theta) = \sum_a \pi_{\theta_k}(a|s) \ln \frac{\pi_{\theta_k}(a|s)}{\pi_\theta(a|s)} = \sum_a \pi_{\theta_k}(a|s)\big[\ln \pi_{\theta_k}(a|s) - \ln \pi_\theta(a|s)\big]$.
>   Khai triển Taylor bậc 2 của $g(\theta)$ quanh $\theta_k$ với bước dịch $\Delta\theta = \theta - \theta_k$:
>   $$g(\theta) \approx g(\theta_k) + \nabla_\theta g(\theta_k)^T \Delta\theta + \frac{1}{2}\Delta\theta^T \nabla_\theta^2 g(\theta_k) \Delta\theta$$
> - **Bước 2 (Thế $\theta = \theta_k$ vào số hạng bậc 0 và bậc 1):**
>   - *Số hạng bậc 0:* $g(\theta_k) = \sum_a \pi_{\theta_k}(a|s) \ln \frac{\pi_{\theta_k}(a|s)}{\pi_{\theta_k}(a|s)} = \sum_a \pi_{\theta_k}(a|s)\ln(1) = 0$.
>   - *Số hạng bậc 1:* Lấy đạo hàm $\nabla_\theta g(\theta) = -\sum_a \pi_{\theta_k}(a|s)\nabla_\theta \ln \pi_\theta(a|s) = -\sum_a \pi_{\theta_k}(a|s)\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$.
>     Khi thế $\theta = \theta_k$, phân số $\frac{\pi_{\theta_k}(a|s)}{\pi_{\theta_k}(a|s)} = 1$ triệt tiêu nhau:
>     $$\nabla_\theta g(\theta_k) = -\sum_a \nabla_\theta \pi_\theta(a|s)\Big|_{\theta=\theta_k} = -\nabla_\theta \underbrace{\left(\sum_a \pi_\theta(a|s)\right)}_{=1} = -\nabla_\theta(1) = \mathbf{0}$$
> - **Bước 3 (Thế đạo hàm bậc 2 — Hessian của KL — bằng tích ngoài của Gradient bậc 1):**
>   Lấy đạo hàm lần nữa của $\nabla_\theta g(\theta) = -\sum_a \pi_{\theta_k}(a|s)\nabla_\theta \ln \pi_\theta(a|s)$:
>   $$\nabla_\theta^2 g(\theta)\Big|_{\theta=\theta_k} = -\sum_a \pi_{\theta_k}(a|s)\nabla_\theta^2 \ln \pi_\theta(a|s)\Big|_{\theta=\theta_k}$$
>   Ta dùng quy tắc đạo hàm của một thương $\nabla_\theta^2 \ln \pi_\theta = \nabla_\theta\left(\frac{\nabla_\theta \pi_\theta}{\pi_\theta}\right) = \frac{\nabla_\theta^2 \pi_\theta}{\pi_\theta} - \left(\frac{\nabla_\theta \pi_\theta}{\pi_\theta}\right)\left(\frac{\nabla_\theta \pi_\theta}{\pi_\theta}\right)^T = \frac{\nabla_\theta^2 \pi_\theta}{\pi_\theta} - (\nabla_\theta \ln \pi_\theta)(\nabla_\theta \ln \pi_\theta)^T$.
>   Thế biểu thức này vào $\nabla_\theta^2 g(\theta_k)$, số hạng đầu $\sum_a \pi_{\theta_k} \frac{\nabla_\theta^2 \pi_{\theta_k}}{\pi_{\theta_k}} = \nabla_\theta^2 \left(\sum_a \pi_{\theta_k}\right) = \nabla_\theta^2(1) = \mathbf{0}$ bị triệt tiêu, chỉ còn lại đúng **Ma trận Fisher $\mathbf{F}(\theta_k)$**:
>   $$\nabla_\theta^2 g(\theta_k) = \mathbb{E}_{a \sim \pi_{\theta_k}}\Big[\big(\nabla_\theta \ln \pi_{\theta_k}(a|s)\big)\big(\nabla_\theta \ln \pi_{\theta_k}(a|s)\big)^T\Big] = \mathbf{F}(\theta_k)$$
>   *Ý nghĩa thực chiến:* Thay vì phải tính đạo hàm bậc 2 (Hessian) tốn bộ nhớ gấp $d^2$ lần, phép thế này cho phép ta tính độ cong chỉ bằng **tích ngoài của đạo hàm bậc 1** trên các mẫu!

---

## PHẦN IV: BẢN ĐỒ TỔNG HỢP CÁC PHÉP THẾ HÀM KINH ĐIỂN TRONG RL (STEP-BY-STEP FUNCTION SUBSTITUTIONS)

Trong toàn bộ chương trình từ Tuần 1 đến Tuần 8, mọi định lý lớn của Reinforcement Learning thực chất đều được xây dựng từ **8 phép thế hàm cốt lõi (Function Substitutions)** dưới đây. Khi đọc bất kỳ bài báo hay phương trình RL nào, bạn chỉ cần tra cứu bảng giải phẫu này để thấy rõ **"Hàm nào được rút ra và đem thế vào chỗ nào"**:

| STT | Tên Phép Thế & Tuần học | (1) Phương trình / Hàm Gốc | (2) Hàm / Định nghĩa đem thế vào | (3) Kết quả sau khi thế & Ý nghĩa thực tiễn |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Truy hồi Trung bình Mẫu**<br>*(Tuần 1 - Bandit)* | $Q_{n+1} = \frac{1}{n}\left(R_n + \sum_{i=1}^{n-1} R_i\right)$ | Rút tổng quá khứ từ định nghĩa $Q_n = \frac{1}{n-1}\sum_{i=1}^{n-1} R_i$:<br>$\implies \sum_{i=1}^{n-1} R_i = (n-1)Q_n$ | $Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$<br>**Ý nghĩa:** Xóa bỏ việc phải lưu trữ $n-1$ mẫu cũ trong bộ nhớ RAM (từ $\mathcal{O}(n)$ về $\mathcal{O}(1)$). |
| **2** | **Truy hồi Bellman từ Return $G_t$**<br>*(Tuần 2 - MDP)* | $V^\pi(s) = \mathbb{E}_\pi\big[G_t \mid S_t = s\big]$<br>và $V^\pi(s) = \sum_a \pi(a\|s) Q^\pi(s,a)$ | - Thế tách đệ quy: $G_t = R_{t+1} + \gamma G_{t+1}$<br>- Thế kỳ vọng bước sau: $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s'] = V^\pi(s')$ | $V^\pi(s) = \sum_a \pi(a\|s)\sum_{s',r} P(s',r\|s,a)\big[r + \gamma V^\pi(s')\big]$<br>**Ý nghĩa:** Không cần đợi hết ván chơi ($\infty$ bước) mới tính được giá trị trạng thái hiện tại (Bootstrapping). |
| **3** | **Khử Thiên lệch Cực đại hóa & Dueling**<br>*(Tuần 3 - Deep Q)* | - Target gốc: $Y_t = r + \gamma Q(s', a^*; \theta^-)$<br>- Dueling gốc: $Q(s,a) = V(s) + A(s,a)$ | - Thế $a^* = \arg\max_{a'} Q(s', a'; \theta)$ (mạng Online chọn) thay vì $\theta^-$<br>- Thế $A(s,a) \leftarrow A(s,a) - \frac{1}{\|\mathcal{A}\|}\sum_{a'} A(s,a')$ | - Double DQN: $Y_t = r + \gamma Q\big(s', \arg\max_{a'} Q(s',a';\theta); \theta^-\big)$<br>- Dueling: Khi $V \leftarrow V+C, A \leftarrow A-C$, hằng số $C$ tự triệt tiêu hoàn toàn! |
| **4** | **Log-Derivative & Advantage TD**<br>*(Tuần 4 - Policy Gradient)* | $\nabla_\theta J = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau$<br>với trọng số $\Psi_t = G_t - b(s_t)$ | - Thế $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\nabla_\theta \ln p_\theta(\tau)$<br>- Thế $G_t \approx r_t + \gamma V_\phi(s_{t+1})$ và $b(s_t) = V_\phi(s_t)$ | $\nabla_\theta J \approx \mathbb{E}\big[\nabla_\theta \ln \pi_\theta(a_t\|s_t) \cdot \delta_t\big]$ với $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.<br>**Ý nghĩa:** Vừa triệt tiêu hàm vật lý $P(s'\|s,a)$, vừa cắt giảm phương sai mẫu. |
| **5** | **Đối ngẫu Lagrange TRPO & Thu gọn GAE**<br>*(Tuần 5 - TRPO/PPO)* | - Cực đại hóa $\mathbf{g}^T \Delta\theta$ dưới ràng buộc $\frac{1}{2}\Delta\theta^T \mathbf{F}\Delta\theta \le \delta$<br>- GAE: $(1-\lambda)\sum_{k=1}^\infty \lambda^{k-1}\hat{A}_t^{(k)}$ | - Từ đạo hàm Lagrange $=0 \implies$ thế $\Delta\theta = \frac{1}{\nu}\mathbf{F}^{-1}\mathbf{g}$ vào lại phương trình ràng buộc KL.<br>- Thế $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1}\gamma^l \delta_{t+l}^V$ và đổi thứ tự tổng. | - Tìm ra bước nhảy đóng: $\Delta\theta^* = \sqrt{\frac{2\delta}{\mathbf{g}^T \mathbf{F}^{-1}\mathbf{g}}}\mathbf{F}^{-1}\mathbf{g}$<br>- GAE rút gọn thành đệ quy ngược: $\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma\lambda)\hat{A}_{t+1}^{\text{GAE}}$. |
| **6** | **Triệt tiêu nhiễu chẵn bằng Khai triển Taylor Đối xứng**<br>*(Tuần 6 - Evolution Strategies)* | Ước lượng đạo hàm:<br>$\hat{g} = \frac{F(\theta+\sigma\varepsilon) - F(\theta-\sigma\varepsilon)}{2\sigma}\varepsilon$ | Thế khai triển Taylor bậc 2 của $F(\theta+\sigma\varepsilon)$ và $F(\theta-\sigma\varepsilon)$ quanh $\theta$ vào tử số. | Các số hạng bậc chẵn $F(\theta)$ và $\frac{\sigma^2}{2}\varepsilon^T \mathbf{H}\varepsilon$ **tự trừ nhau bằng 0**, chỉ giữ lại đúng thành phần đạo hàm bậc nhất $\big(\varepsilon \varepsilon^T\big)\nabla F(\theta)$! |
| **7** | **Cộng dồn Sai số Rollout & Nhiễu Tikhonov**<br>*(Tuần 7 - Model-Based RL)* | - Rollout: $\hat{s}_{t+1} = \hat{s}_t + f_\theta(\hat{s}_t, a_t)$<br>- Loss có nhiễu: $\mathbb{E}_\xi\big[\|f(x+\xi) - y\|_2^2\big]$ | - Thế liên tiếp $\hat{s}_1, \hat{s}_2, \dots$ vào $\hat{s}_H$<br>- Thế $f(x+\xi) \approx f(x) + \mathbf{J}_f(x)\xi$ vào chuẩn $L_2^2$ | - $\hat{s}_H = s_0 + \sum_{t=0}^{H-1} f_\theta(\hat{s}_t, a_t)$ (đường tắt Gradient = 1)<br>- $\mathcal{L}_{\text{noisy}} \approx \|f(x)-y\|_2^2 + \sigma_\xi^2 \|\mathbf{J}_f(x)\|_F^2$ (thêm nhiễu đầu vào tương đương phạt chuẩn Jacobian!). |
| **8** | **Tái tham số hóa SAC & Khử $Z(x)$ trong DPO**<br>*(Tuần 8 - SAC & RLHF/DPO)* | - SAC: $\nabla_\theta \mathbb{E}_{a \sim \pi_\theta}[\alpha \ln \pi_\theta(a\|s) - Q_\phi(s,a)]$<br>- DPO: $P(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$ | - Thế biến hành động $a = f_\theta(\varepsilon; s) = \tanh(\mu_\theta + \sigma_\theta \odot \varepsilon)$ với $\varepsilon \sim \mathcal{N}(0,I)$<br>- Thế nghiệm tối ưu $r(x,y) = \beta \ln\frac{\pi^*(y\|x)}{\pi_{\text{ref}}(y\|x)} + \beta \ln Z(x)$ vào hiệu $r(x,y_w) - r(x,y_l)$. | - SAC: Đưa $\nabla_\theta$ chui thẳng vào trong $Q_\phi\big(s, f_\theta(\varepsilon;s)\big)$!<br>- DPO: Hằng số phân hoạch $\beta \ln Z(x)$ **tự triệt tiêu hoàn toàn** khi trừ $r(x,y_w) - r(x,y_l)$, giúp huấn luyện LLM trực tiếp không cần mạng Reward! |

---

## PHẦN V: CHẠY CODE THỰC NGHIỆM TRỰC QUAN TRÊN MẪU (`minh_hoa_toan_hoc_tren_mau.py`)

Để nhìn thấy tận mắt từng công thức trên biến đổi các con số mẫu như thế nào, tôi đã viết kèm kịch bản thực nghiệm độc lập tại **[minh_hoa_toan_hoc_tren_mau.py](file:///home/thinh/Reinforcement-Learning/minh_hoa_toan_hoc_tren_mau.py)**.

Bạn chỉ cần chạy lệnh:
```bash
python3 minh_hoa_toan_hoc_tren_mau.py
```
Chương trình sẽ chạy **6 thí nghiệm trực quan trên các tập mẫu thực tế**:
1. **Thí nghiệm 1 (Kỳ vọng & Luật Số lớn):** Quan sát trung bình mẫu khử nhiễu Gaussian theo tốc độ $\sigma/\sqrt{N}$ để hội tụ về $Q^*(s,a)$ thật.
2. **Thí nghiệm 2 (Bất đẳng thức Jensen & Maximization Bias):** Chứng kiến toán tử $\max$ bị đánh lừa bởi mẫu nhiễu dương trong Q-Learning và cách **Double Estimator** khử sạch độ chệch này về $0$.
3. **Thí nghiệm 3 (Importance Sampling & PPO Clipping):** Cách tỉ số $r_t(\theta) = \pi_{\text{new}}/\pi_{\text{old}}$ tái cân bằng trọng số trên cùng 1 tập mẫu cũ, và điều gì xảy ra khi 1 mẫu có $r_t$ quá lớn.
4. **Thí nghiệm 4 (Entropy & KL Divergence vs. Khoảng cách Euclid):** Chứng minh cùng một độ dịch tham số $\Delta \mu = 0.5$ nhưng tạo ra khoảng cách thông tin $D_{\text{KL}}$ chênh lệch gấp $10,000$ lần tùy theo độ rộng $\sigma$ của đám mây mẫu.
5. **Thí nghiệm 5 (Đại số Tuyến tính: Ma trận Hiệp phương sai, Trị riêng & Vector riêng):** Tạo một đám mây mẫu 2D nghiêng chéo của $(v, \omega)$ và dùng Trị riêng/Vector riêng tìm ra chính xác góc nghiêng trục chính và phương sai dọc theo trục đó.
6. **Thí nghiệm 6 (So sánh Chuẩn $L_1, L_2, L_\infty$ trước Mẫu Nhiễu Dị Biệt - Outlier):** Xem cách 1 mẫu lỗi duy nhất làm lệch chuẩn $L_2$ và $L_\infty$, trong khi $L_1$ / Huber giữ vững sự ổn định.
