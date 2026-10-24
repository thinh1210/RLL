# PHẦN 2 (TẦNG 2): BỘ GIẢI ĐỘNG LỰC HỌC VẬT LÝ, CƠ HỌC TIẾP XÚC & MÔ HÌNH HÓA CẢM BIẾN

> **Mục tiêu của Tầng 2:**
> Giải thích cơ chế toán học bên trong **Trái tim của Trình mô phỏng (Physics Engine & Sensor Engine)**.
> Tại sao một thuật toán chạy hoàn hảo trên mô phỏng đơn giản (Toy Simulation) lại thất bại thảm hại khi đưa ra robot thật? Câu trả lời nằm ở **5 hiện tượng vật lý cốt lõi ở Tầng 2**: Quán tính, Giới hạn Động cơ, Trượt ma sát Coulomb, Độ trễ phần cứng, và Sai số Cảm biến.

---

## 1. Khoảng cách Thực tế (The Sim-to-Real Gap): Mô phỏng Động học vs. Mô phỏng Động lực học

Hãy so sánh hai cấp độ mô phỏng khi bạn gửi một lệnh vận tốc $(v_{\text{cmd}}, \omega_{\text{cmd}})$ xuống robot:

| Đặc tính Vật lý | Mô phỏng Ngây thơ (Toy Kinematic Sim) | Mô phỏng Vật lý Thực tế (High-Fidelity Physical Sim) |
| :--- | :--- | :--- |
| **Đáp ứng vận tốc** | Tức thì: $v(t) = v_{\text{cmd}}$ ngay trong 1 bước lệnh (Gia tốc $a = \infty$). | Phải thắng quán tính khối lượng $m$ và quán tính quay $I_{zz}$ thông qua phương trình vi phân bậc 2. |
| **Độ trễ (Latency)** | $0\text{ ms}$. | Tín hiệu mất $15 - 40\text{ ms}$ đi qua cáp USB/CAN, mạch cầu H và bộ lọc phần cứng. |
| **Lực bám mặt sàn** | Tuyệt đối không bao giờ trượt. | Giới hạn bởi ma sát Coulomb $|F_{\text{traction}}| \le \mu N$. Nếu đạp ga mạnh trên sàn trơn, bánh xe quay tít tại chỗ (Wheel Slip)! |
| **Vị trí Robot** | Luôn biết chính xác tuyệt đối $(x, y, \theta)$. | Chỉ ước lượng qua Encoder bánh xe (bị lệch khi bánh trượt) + IMU (bị trôi góc theo thời gian) + LiDAR (có nhiễu & góc khuất). |

---

## 2. Hệ Phương trình Vi phân Động lực học Robot Vi sai Thực tế

Xét robot **Andino** có khối lượng tổng $m$, mô-men quán tính quay quanh trục thẳng đứng $I_{zz}$, bán kính bánh xe $r$, khoảng cách hai bánh $L$:

### Bước 1: Hàng đợi Độ trễ Phần cứng (Hardware Latency FIFO Queue)
Lệnh điện áp/vận tốc góc gửi tại thời điểm $t$ chỉ thực sự tác động vào động cơ sau khoảng thời gian trễ $\tau_{\text{delay}} = k \cdot \Delta t$:
$$\mathbf{u}_{\text{applied}}(t) = \mathbf{u}_{\text{cmd}}(t - \tau_{\text{delay}})$$

### Bước 2: Bộ điều khiển Vận tốc Nội tại & Đường đặc tính Động cơ DC
Mạch điều khiển động cơ tạo mô-men xoắn $\tau_{\text{raw}, i}$ ($i \in \{L, R\}$) theo sai lệch tốc độ góc bánh xe, nhưng bị chặn cứng bởi đường bao vật lý của động cơ DC:
$$\tau_{\max}(\omega_i) = \tau_{\text{stall}} \cdot \max\left(0,\; 1 - \frac{|\omega_i|}{\omega_{\text{no\_load}}}\right)$$
$$\tau_{\text{motor}, i} = \operatorname{clip}\Big(K_p(\omega_{\text{target}, i} - \omega_i),\; -\tau_{\max}(\omega_i),\; +\tau_{\max}(\omega_i)\Big)$$

### Bước 3: Tương tác Lốp xe – Mặt sàn & Hiện tượng Trượt bánh (Coulomb Friction & Wheel Slip)
Lực đẩy tiếp tuyến mong muốn tại điểm tiếp đất của bánh xe là $F_{\text{des}, i} = \frac{\tau_{\text{motor}, i} - b \cdot \omega_i}{r}$.
Tuy nhiên, mặt sàn chỉ có thể cung cấp phản lực ma sát tối đa theo định luật Coulomb:
$$F_{\max\_\text{grip}} = \mu_{\text{floor}} \cdot \frac{m g}{2}$$
- **Khi $|F_{\text{des}, i}| \le F_{\max\_\text{grip}}$ (Lăn không trượt - Pure Rolling):**
  Lực đẩy thực tế truyền vào thân xe là $F_i = F_{\text{des}, i}$, và tốc độ quay của bánh xe khớp với vận tốc đất: $\omega_{\text{wheel}, i} = v_i / r$.
- **Khi $|F_{\text{des}, i}| > F_{\max\_\text{grip}}$ (Bánh xe bị trượt - Wheel Slip / Spin):**
  Lực đẩy truyền vào thân xe bị bão hòa tại ngưỡng ma sát trượt $\mu_{\text{kinetic}}$:
  $$F_i = \operatorname{sgn}(F_{\text{des}, i}) \cdot \mu_{\text{kinetic}} \frac{m g}{2}$$
  Phần mô-men xoắn dư thừa $\Delta F = F_{\text{des}, i} - F_i$ làm trục bánh xe quay vọt lên (trượt bánh trên mặt sàn), khiến **Encoder đếm nhiều vòng quay hơn quãng đường thực tế mà xe đi được!**

### Bước 4: Phương trình Chuyển động của Thân Robot (Rigid-Body Dynamics)
Từ lực đẩy thực tế của bánh trái $F_L$ và bánh phải $F_R$, gia tốc dài $\dot{v}$ và gia tốc góc $\dot{\omega}$ của thân robot là:
$$\dot{v} = \frac{F_R + F_L - c_v \cdot v}{m}, \qquad \dot{\omega} = \frac{\frac{L}{2}(F_R - F_L) - c_\omega \cdot \omega}{I_{zz}}$$
Và đạo hàm tọa độ trong hệ quy chiếu mặt đất (World Frame):
$$\dot{x} = v \cos\theta, \qquad \dot{y} = v \sin\theta, \qquad \dot{\theta} = \omega$$

---

## 3. Các Phương pháp Giải tích Số (Numerical Integrators): Symplectic Euler vs. RK4

Để giải hệ phương trình vi phân $\dot{\mathbf{s}} = f(\mathbf{s}, \mathbf{u})$ theo từng bước $\Delta t$:

### 3.1. Explicit Euler (Euler Hiện — KHÔNG NÊN DÙNG trong Vật lý)
$$\begin{aligned}
x_{t+\Delta t} &= x_t + v_t \cos\theta_t \cdot \Delta t \\
v_{t+\Delta t} &= v_t + a_t \cdot \Delta t
\end{aligned}$$
- **Lỗ hổng:** Cập nhật vị trí bằng vận tốc cũ $v_t$ trước khi cập nhật vận tốc mới. Đối với hệ dao động (như va chạm đàn hồi), Explicit Euler làm tổng cơ năng tăng lên theo cấp số nhân $E_{k+1} = (1 + \omega_n^2 \Delta t^2) E_k \to \infty$ (tự sinh ra năng lượng từ hư không!).

### 3.2. Semi-Implicit / Symplectic Euler (Tiêu chuẩn trong Gazebo, गेम Engine & Isaac Gym)
$$\begin{aligned}
v_{t+\Delta t} &= v_t + a_t \cdot \Delta t \quad &\text{(1. Cập nhật Vận tốc trước)} \\
x_{t+\Delta t} &= x_t + v_{t+\Delta t} \cos\theta_{t+\Delta t} \cdot \Delta t \quad &\text{(2. Dùng Vận tốc MỚI để cập nhật Vị trí)}
\end{aligned}$$
- **Ưu điểm:** Bảo toàn thể tích không gian pha (Symplectic manifold), giữ cho cơ năng dao động ổn định quanh giá trị thực mà chỉ tốn 1 lần tính lực mỗi bước.

### 3.3. Runge-Kutta Bậc 4 (RK4 — Tiêu chuẩn chính xác cao trong MuJoCo & Kiểm định Kỹ thuật)
Lấy mẫu đạo hàm tại 4 điểm trung gian trong khoảng $[t, t+\Delta t]$:
$$\mathbf{k}_1 = f(\mathbf{s}_t), \quad \mathbf{k}_2 = f\!\left(\mathbf{s}_t + \frac{\Delta t}{2}\mathbf{k}_1\right), \quad \mathbf{k}_3 = f\!\left(\mathbf{s}_t + \frac{\Delta t}{2}\mathbf{k}_2\right), \quad \mathbf{k}_4 = f(\mathbf{s}_t + \Delta t\mathbf{k}_3)$$
$$\mathbf{s}_{t+\Delta t} = \mathbf{s}_t + \frac{\Delta t}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big)$$
- **Ưu điểm:** Sai số cục bộ bậc $O(\Delta t^5)$. Dùng làm **"Chuẩn tham chiếu vật lý" (Ground-Truth Physics Reference)** để kiểm định chéo Sim-to-Sim!

---

## 4. Toán học Mô phỏng Cảm biến Thực tế (Realistic Sensor Modeling)

### 4.1. Cảm biến LiDAR 2D (Ray-casting Geometry)
Cảm biến LiDAR gắn trên robot phát ra $N_{\text{rays}}$ tia laser (ví dụ $12$ hoặc $360$ tia) ở các góc $\alpha_i = \theta_{\text{robot}} + \beta_i$.
Để tính khoảng cách tia laser chạm vào một chướng ngại vật hình tròn tâm $\mathbf{c} = (x_c, y_c)$ bán kính $R$, trình mô phỏng giải phương trình bậc 2 của tia $\mathbf{p}(d) = \mathbf{p}_{\text{robot}} + d \cdot \hat{\mathbf{u}}(\alpha_i)$:
$$\|\mathbf{p}_{\text{robot}} + d \cdot \hat{\mathbf{u}} - \mathbf{c}\|^2 = R^2 \implies d^2 + 2(\mathbf{f} \cdot \hat{\mathbf{u}})d + (\|\mathbf{f}\|^2 - R^2) = 0 \quad (\text{với }\mathbf{f} = \mathbf{p}_{\text{robot}} - \mathbf{c})$$
Nghiệm dương nhỏ nhất $d^* = -(\mathbf{f}\cdot\hat{\mathbf{u}}) - \sqrt{(\mathbf{f}\cdot\hat{\mathbf{u}})^2 - (\|\mathbf{f}\|^2 - R^2)}$ chính là khoảng cách vật lý thực. Sau đó cộng thêm nhiễu quang học:
$$d_{\text{measured}} = \operatorname{clip}\big(d^* + \mathcal{N}(0, \sigma_{\text{lidar}}^2),\; d_{\min},\; d_{\max}\big)$$

### 4.2. Cảm biến Wheel Encoder & Sự hình thành Sai số Odometry (Odometry Drift)
Encoder không đo được tọa độ mặt đất $(x, y)$, mà chỉ đếm số xung góc quay trục bánh xe (lượng tử hóa theo $N_{\text{CPR}}$ xung/vòng):
$$\Delta \phi_{\text{enc}} = \frac{2\pi}{N_{\text{CPR}}} \operatorname{round}\!\left(\frac{\omega_{\text{wheel}} \cdot \Delta t \cdot N_{\text{CPR}}}{2\pi}\right)$$
Khi bánh xe bị trượt trên sàn ($\omega_{\text{wheel}} \neq v_{\text{ground}}/r$), tọa độ Odometry $(x_{\text{odom}}, y_{\text{odom}}, \theta_{\text{odom}})$ sẽ **tích lũy sai số lệch dần khỏi tọa độ thật $(x_{\text{true}}, y_{\text{true}}, \theta_{\text{true}})$**.

---

## 5. Mã nguồn Minh họa Thực thi (`02_physics_engine_from_scratch.py`)

File mã nguồn [02_physics_engine_from_scratch.py](file:///home/thinh/Reinforcement-Learning/Sim2Real_Robotics/code/02_physics_engine_from_scratch.py) cài đặt trọn vẹn một bộ giải vật lý 2 cấp độ (Symplectic Euler & RK4) kèm cảm biến thực tế cho robot Andino. Khi chạy trực tiếp, chương trình sẽ đặt cùng một chuỗi lệnh điều khiển lên 3 môi trường:
1. **Mô phỏng Động học Ngây thơ (Toy Kinematic Sim)**
2. **Mô phỏng Vật lý Thực tế trên Sàn Khô ($\mu = 0.85$, có quán tính + độ trễ $20\text{ ms}$)**
3. **Mô phỏng Vật lý Thực tế trên Sàn Trơn ($\mu = 0.20$, xuất hiện trượt bánh xe & lệch Odometry)**

🔗 **Tiếp theo:** Xem [Phần 3: Đóng gói Môi trường RL, Thiết kế Hàm thưởng Vật lý & Huấn luyện Agent](file:///home/thinh/Reinforcement-Learning/Sim2Real_Robotics/Part3_RL_Training_Environment_And_Rewards.md).
