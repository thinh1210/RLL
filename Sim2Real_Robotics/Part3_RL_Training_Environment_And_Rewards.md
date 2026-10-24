# PHẦN 3 (TẦNG 3): ĐÓNG GÓI MÔI TRƯỜNG RL, THIẾT KẾ HÀM THƯỞNG VẬT LÝ & DOMAIN RANDOMIZATION

> **Mục tiêu của Tầng 3:**
> Làm thế nào để biến Bộ giải Vật lý ở Tầng 2 thành một môi trường huấn luyện Reinforcement Learning (`Env`) mà chính sách (Policy) học ra **không bị học vẹt (overfit) vào mô phỏng**, không làm cháy động cơ và có thể mang thẳng sang robot thật?
> 
> Tài liệu này giải thích **4 nguyên tắc thiết kế sống còn ở Tầng 3**: Không gian quan sát hệ quy chiếu thân xe (Ego-centric Observation), Bộ lọc thông thấp cơ khí (Action Low-Pass Filter), Hàm thưởng có ràng buộc vật lý (Physics-Aware Reward), và Ngẫu nhiên hóa miền vật lý (Domain Randomization).

---

## 1. Thiết kế Không gian Quan sát Khả thi Thực tế (Real-World Feasible Observation Space)

### ❌ Lỗi phổ biến nhất khi tự tạo môi trường RL:
Nhiều người đưa trực tiếp tọa độ tuyệt đối của robot $(x, y)$ và tọa độ của từng chướng ngại vật $(x_{\text{obs}}, y_{\text{obs}})$ vào mạng nơ-ron.
- **Hậu quả:** Khi đem ra robot thật, không có "Thượng đế" nào cho robot biết cái ghế trước mặt có tọa độ $(x=1.42\text{m}, y=-0.85\text{m})$! Robot thật chỉ có mắt đọc LiDAR và bộ đếm bánh xe Odometry.

### ✅ Thiết kế chuẩn Sim-to-Real: Hệ quy chiếu Gắn liền Thân xe (`base_link` Ego-centric Frame)
Mọi thông tin đầu vào của mạng nơ-ron phải được quy về **góc nhìn của chính con robot tại thời điểm $t$**:

1. **Vị trí Điểm đích trong hệ tọa độ Thân xe (`base_link`):**
   Gọi $(x_t, y_t, \theta_t)$ là vị trí robot và $(x_g, y_g)$ là điểm đích. Phép quay hệ trục tọa độ một góc $-\theta_t$ cho ta tọa độ tương đối của đích so với mũi xe:
   $$\begin{bmatrix} \Delta x_{\text{body}} \\ \Delta y_{\text{body}} \end{bmatrix} = \begin{bmatrix} \cos\theta_t & \sin\theta_t \\ -\sin\theta_t & \cos\theta_t \end{bmatrix} \begin{bmatrix} x_g - x_t \\ y_g - y_t \end{bmatrix}$$
   Từ đó suy ra **Khoảng cách tới đích** $d_t$ và **Góc lệch mũi xe so với đích** $\alpha_t \in [-\pi, \pi]$:
   $$d_t = \sqrt{\Delta x_{\text{body}}^2 + \Delta y_{\text{body}}^2}, \qquad \alpha_t = \operatorname{atan2}(\Delta y_{\text{body}}, \Delta x_{\text{body}})$$
   *(Biểu diễn góc bằng cặp $(\cos\alpha_t, \sin\alpha_t)$ giúp mạng nơ-ron không bị đứt gãy liên tục tại điểm nhảy $-\pi \leftrightarrow +\pi$).*

2. **Tầm nhìn Cảm biến LiDAR chuẩn hóa:**
   Chia $360^\circ$ (hoặc góc nhìn phía trước) thành các vùng quạt (sectors), chuẩn hóa khoảng cách đo được về đoạn $[0, 1]$:
   $$\tilde{l}_i = \frac{\min(d_{\text{lidar}, i}, d_{\max})}{d_{\max}} \in [0, 1]$$

3. **Trạng thái Bản thể (Proprioception) & Lịch sử Hành động trước đó ($a_{t-1}$):**
   Đưa vận tốc hiện tại $(v_t, \omega_t)$ **cùng với lệnh điều khiển ở bước ngay trước đó** $\mathbf{a}_{t-1} = (a_{t-1}^{(v)}, a_{t-1}^{(\omega)})$ vào vector quan sát $\mathbf{o}_t$.
   - **Tại sao phải đưa $\mathbf{a}_{t-1}$ vào $\mathbf{o}_t$?**
     1. Để mạng nơ-ron biết bước trước mình vừa ra lệnh gì, từ đó tính được độ biến thiên $\Delta \mathbf{a}_t = \mathbf{a}_t - \mathbf{a}_{t-1}$ nhằm tối ưu hóa hàm phạt chống giật cục!
     2. Bằng cách so sánh lệnh đã phát ra $\mathbf{a}_{t-1}$ với vận tốc thực tế đạt được $(v_t, \omega_t)$, mạng nơ-ron ngầm suy luận được **mặt sàn hiện tại đang trơn hay nhám, xe đang nặng hay nhẹ (Implicit System Identification)**.

---

## 2. Bộ lọc Thông thấp Cơ khí (Actuator Low-Pass Filter / Action Smoothing)

Mạng nơ-ron RL có thể xuất ra chuỗi hành động nhảy vọt giữa 2 bước liên tiếp: bước $t-1$ xuất $+1.0$ (tiến tối đa), bước $t$ xuất $-1.0$ (lùi tối đa).
Ngoài đời thực, việc đảo chiều dòng điện động cơ đột ngột ở tần số $20-50\text{ Hz}$ gây ra:
- Dòng điện xung ngược gấp $2\times$ dòng kẹt trục ($I_{\text{peak}} \approx 2 I_{\text{stall}}$), gây sụt áp nguồn làm sập nguồn Raspberry Pi / Jetson hoặc cháy MOSFET cầu H.
- Vỡ răng hộp số giảm tốc do xung lực cơ học.

👉 **Giải pháp ở Tầng 3:** Chèn một **Bộ lọc Thông thấp Bậc 1 (Exponential Moving Average - EMA Filter)** giữa đầu ra $\mathbf{a}_t \in [-1, 1]^2$ của mạng nơ-ron và lệnh vận tốc thực tế $\mathbf{u}_t$:
$$\mathbf{u}_t = \beta_{\text{filter}} \cdot \mathbf{a}_t + (1 - \beta_{\text{filter}}) \cdot \mathbf{u}_{t-1}, \qquad \beta_{\text{filter}} \in (0, 1]$$
- Tần số cắt (Cutoff Frequency) của bộ lọc liên hệ với $\beta_{\text{filter}}$ và chu kỳ điều khiển $\Delta t_c$ qua công thức:
  $$f_c = \frac{\beta_{\text{filter}}}{2\pi (1 - \beta_{\text{filter}}) \Delta t_c}$$
  Mọi dao động nhiễu cao tần trên ngưỡng $f_c$ đều bị triệt tiêu trước khi chạm tới động cơ!

---

## 3. Thiết kế Hàm Phần thưởng Vật lý (Physics-Aware Reward Shaping)

Trong bài toán học thuật (Toy RL), hàm thưởng thường chỉ có:
$$R_{\text{toy}} = (d_{t-1} - d_t) + R_{\text{goal}} - R_{\text{collision}}$$
Hàm thưởng này tạo ra các chính sách **"Bang-Bang Control"** (lắc tay lái trái-phải điên cuồng với biên độ tối đa). Để chạy được trên Robot thật, hàm thưởng **bắt buộc phải cộng thêm 3 thành phần phạt vật lý**:

$$\begin{aligned}
R_t^{\text{physical}} &= \underbrace{w_{\text{prog}} (d_{t-1} - d_t)}_{\text{1. Tiến gần tới đích}} + \underbrace{w_{\text{head}} \cdot v_t \cos(\alpha_t)}_{\text{2. Đi thẳng hướng đích}} - \underbrace{w_{\text{clear}} \sum_i \max\!\left(0,\; 1 - \frac{d_{\text{lidar}, i}}{d_{\text{safe}}}\right)^2}_{\text{3. Giữ khoảng cách an toàn với vật cản}} \\
&\quad - \underbrace{w_{\text{rate}} \|\mathbf{a}_t - \mathbf{a}_{t-1}\|_2^2}_{\text{4. Phạt tốc độ bẻ lái (Action Rate)}} - \underbrace{w_{\text{jerk}} \|\mathbf{a}_t - 2\mathbf{a}_{t-1} + \mathbf{a}_{t-2}\|_2^2}_{\text{5. Phạt độ giật gia tốc (Jerk)}} - \underbrace{w_{\text{power}} (\tau_L^2 + \tau_R^2)}_{\text{6. Phạt tổn hao nhiệt động cơ } I^2 R}
\end{aligned}$$

### Giải thích bản chất Vật lý của 3 số hạng phạt mới:
1. **Số hạng 4 (`Action Rate Penalty` $-\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$):** Tương đương đạo hàm bậc 1 của lệnh điều khiển ($\dot{\mathbf{a}}$), ép mạng nơ-ron thay đổi vận tốc mượt mà theo đường cong trơn.
2. **Số hạng 5 (`Action Jerk Penalty` $-\|\mathbf{a}_t - 2\mathbf{a}_{t-1} + \mathbf{a}_{t-2}\|^2$):** Tương đương đạo hàm bậc 2 ($\ddot{\mathbf{a}}$), triệt tiêu hoàn toàn hiện tượng lắc cổ xe qua lại ở tần số Nyquist (Chattering / Limit-cycle oscillation).
3. **Số hạng 6 (`Torque / Thermal Loss Penalty` $-(\tau_L^2 + \tau_R^2)$):** Vì dòng điện $I = \tau / K_t$, công suất tỏa nhiệt làm nóng cuộn dây đồng động cơ là $P_{\text{heat}} = I^2 R_a = \frac{R_a}{K_t^2} \tau^2$. Phạt $\tau^2$ chính là tối thiểu hóa nhiệt lượng tỏa ra trên động cơ!

---

## 4. Ngẫu nhiên hóa Miền Vật lý (Domain Randomization)

Thay vì huấn luyện trên 1 con robot cố định, đầu mỗi episode (`env.reset()`), ta lấy mẫu ngẫu nhiên một "phiên bản vật lý" mới của robot:
$$\begin{aligned}
m &\sim \mathcal{U}(0.80 m_0,\; 1.25 m_0) \quad &\text{(Khối lượng xe biến thiên }-20\% \to +25\%) \\
I_{zz} &\sim \mathcal{U}(0.80 I_0,\; 1.25 I_0) \quad &\text{(Quán tính quay biến thiên)} \\
\mu_{\text{static}} &\sim \mathcal{U}(0.25,\; 1.10) \quad &\text{(Mặt sàn từ trơn ướt đến thảm nhám)} \\
\tau_{\text{stall}} &\sim \mathcal{U}(0.85 \tau_0,\; 1.15 \tau_0) \quad &\text{(Sức mạnh motor khi pin đầy vs. pin yếu)} \\
\tau_{\text{delay}} &\sim \operatorname{Uniform}\{0,\; 50\text{ ms},\; 100\text{ ms}\} \quad &\text{(Độ trễ truyền thông thay đổi)}
\end{aligned}$$

---

## 5. Mã nguồn Minh họa Thực thi (`03_train_rl_navigation_agent.py`)

File mã nguồn [03_train_rl_navigation_agent.py](file:///home/thinh/Reinforcement-Learning/Sim2Real_Robotics/code/03_train_rl_navigation_agent.py) thực hiện huấn luyện và so sánh trực tiếp **2 Chính sách Điều hướng Robot Andino**:
1. **`Naive_Toy_Policy`:** Được huấn luyện trên môi trường Động học Lý tưởng (`ideal_kinematic_mode=True`), không có bộ lọc thông thấp, không có phạt giật cục cơ khí và không có Domain Randomization.
2. **`Sim2Real_Robust_Policy`:** Được huấn luyện trên Bộ giải Động lực học Vật lý (Tầng 2), có tích hợp đầy đủ **Physics-Aware Reward + Action Low-Pass Filter + Domain Randomization**.

Cả hai bộ trọng số sau khi huấn luyện sẽ được xuất tự động ra file `Sim2Real_Robotics/code/trained_policies.json` để đưa thẳng sang **Tầng 4 (Kiểm định Thực tế Vật lý)** và **Tầng 5 (Node triển khai ROS 2)**.

🔗 **Tiếp theo:** Xem [Phần 4: Bộ 7 Bài Kiểm định Thực tế Vật lý Chỉ Bằng Code](file:///home/thinh/Reinforcement-Learning/Sim2Real_Robotics/Part4_Code_Only_Physical_Verification.md).
