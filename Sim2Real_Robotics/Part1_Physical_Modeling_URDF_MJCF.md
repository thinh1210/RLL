# PHẦN 1 (TẦNG 1): MÔ TẢ CƠ HỌC, CẤU HÌNH VẬT LÝ URDF/MJCF & KIỂM ĐỊNH TÍNH HỢP LỆ VẬT LÝ

> **Mục tiêu của Tầng 1:**
> Trước khi trình mô phỏng chạy bất kỳ dòng code nào, bạn phải khai báo cho máy tính biết "Con robot này có hình dáng ra sao, nặng bao nhiêu, trọng tâm ở đâu và động cơ mạnh cỡ nào" thông qua file cấu hình **URDF / SDF (Gazebo)** hoặc **MJCF XML (MuJoCo / Isaac Lab)**.
> 
> Tài liệu này giải thích **bản chất toán - lý của từng thông số cấu hình ở Tầng 1** và cách **dùng code để phát hiện ngay lập tức một file cấu hình có hợp lệ trong thế giới thực hay không**.

---

## 1. Cấu trúc Phân cấp Cơ học của Robot: Chuỗi Động học (Kinematic Tree)

Một con robot trong trình mô phỏng được biểu diễn dưới dạng một **Đồ thị Hình cây (Tree Graph)**:
- **Nút (Nodes) = `<link>`:** Các vật rắn không biến dạng (khung xe `base_link`, bánh trái `left_wheel`, bánh phải `right_wheel`, cụm cảm biến `laser_link`).
- **Cạnh có hướng (Directed Edges) = `<joint>`:** Ràng buộc chuyển động tương đối giữa Link cha (`parent`) và Link con (`child`).

### Phép biến đổi Không gian 3D ($SE(3)$) trong thẻ `<origin>`
Trong mỗi thẻ `<joint>` hoặc `<inertial>`, dòng cấu hình:
```xml
<origin xyz="x y z" rpy="φ θ ψ"/>
```
tương ứng với một **Ma trận Biến đổi Đồng nhất $4 \times 4$ (Homogeneous Transformation Matrix)** $\mathbf{T} \in SE(3)$:
$$\mathbf{T} = \begin{bmatrix} \mathbf{R}(\phi, \theta, \psi) & \mathbf{p} \\ \mathbf{0}_{1\times 3} & 1 \end{bmatrix}, \qquad \mathbf{p} = \begin{bmatrix} x \\ y \\ z \end{bmatrix} \in \mathbb{R}^3$$
trong đó $\mathbf{R}(\phi, \theta, \psi) = \mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi) \in SO(3)$ là ma trận xoay từ góc Euler:
- $\phi$ (**Roll**): Góc nghiêng quanh trục $X$ (trục dọc hướng tới trước).
- $\theta$ (**Pitch**): Góc chúc/ngửa quanh trục $Y$ (trục ngang hướng sang trái).
- $\psi$ (**Yaw**): Góc quay hướng quanh trục $Z$ (trục thẳng đứng hướng lên trời).

---

## 2. Ba Điều kiện Bất khả xâm phạm để Ma trận Quán tính `<inertial>` Tồn tại Ngoài Đời Thực

Khi kỹ sư chỉnh tay các thông số `<mass>` và `<inertia>` trong file URDF, có những bộ số trông rất bình thường nhưng lại **vi phạm định luật vật lý của vũ trụ** (không có vật thể thực nào có thể sở hữu bộ số đó), khiến trình mô phỏng tính ra kết quả sai lệch hoặc phát nổ.

Để một vật rắn tồn tại được trong thế giới vật lý (Physical Realizability), nó **bắt buộc phải thỏa mãn 3 điều kiện toán học** sau:

### Điều kiện 1: Khối lượng Dương thực sự
$$m > 0$$

### Điều kiện 2: Ma trận Quán tính Đối xứng Xác định Dương ($\mathbf{I} \succ 0$)
Động năng quay của vật rắn với vận tốc góc $\boldsymbol{\omega} \neq \mathbf{0}$ luôn phải dương:
$$E_{\text{rot}} = \frac{1}{2}\boldsymbol{\omega}^T \mathbf{I} \boldsymbol{\omega} > 0 \iff \text{Mọi trị riêng chính } I_1, I_2, I_3 > 0$$

### Điều kiện 3: Bất đẳng thức Tam giác của Mô-men Quán tính (Triangle Inequality)
> 💡 **Định lý Vật lý Quan trọng:** Tổng của 2 mô-men quán tính chính bất kỳ luôn phải lớn hơn hoặc bằng mô-men quán tính chính thứ 3:
> $$I_1 + I_2 \ge I_3, \qquad I_2 + I_3 \ge I_1, \qquad I_3 + I_1 \ge I_2$$

#### 📐 Chứng minh Toán học:
Theo định nghĩa tích phân khối lượng của mô-men quán tính quanh 3 trục chính đi qua trọng tâm:
$$\begin{aligned}
I_1 &= \int_{\mathcal{V}} (y^2 + z^2)\,\rho(\mathbf{r})\,dV \\
I_2 &= \int_{\mathcal{V}} (x^2 + z^2)\,\rho(\mathbf{r})\,dV \\
I_3 &= \int_{\mathcal{V}} (x^2 + y^2)\,\rho(\mathbf{r})\,dV
\end{aligned}$$
Cộng $I_1$ và $I_2$ lại, ta có:
$$I_1 + I_2 = \int_{\mathcal{V}} (x^2 + y^2)\,\rho(\mathbf{r})\,dV + 2\int_{\mathcal{V}} z^2\,\rho(\mathbf{r})\,dV = I_3 + 2\int_{\mathcal{V}} z^2\,\rho(\mathbf{r})\,dV$$
Vì mật độ khối lượng $\rho(\mathbf{r}) > 0$ và $z^2 \ge 0$, số hạng $2\int z^2 dm \ge 0$ luôn không âm (chỉ bằng $0$ khi vật thể là một tấm phẳng mỏng tuyệt đối có $z = 0$). Do đó:
$$I_1 + I_2 \ge I_3 \quad \blacksquare$$

> ⚠️ **Hệ quả thực tế:** Nếu bạn gõ tay `ixx="0.01" iyy="0.01" izz="0.05"`, vì $0.01 + 0.01 = 0.02 < 0.05$, cấu hình này tương đương với một vật thể có mật độ khối lượng **âm** ($\int z^2 dm < 0$)! Khi robot xoay tròn trên sàn và va chạm nhẹ, phương trình Euler $\boldsymbol{\alpha} = \mathbf{I}^{-1}(\boldsymbol{\tau} - \boldsymbol{\omega} \times \mathbf{I}\boldsymbol{\omega})$ sẽ sinh ra hiệu ứng con quay hồi chuyển ngược tự kích, làm robot lắc điên cuồng!

---

## 3. Đa giác Chân đế (Support Polygon) & Giới hạn Chống lật Khi Phanh/Tăng tốc

Làm sao biết chỉ qua file cấu hình rằng robot có bị chổng bánh lật úp khi tăng tốc hoặc phanh gấp hay không?

```
                    ┌─────────────────────────┐
                    │       Thân Robot        │
                    │    • Trọng tâm (CoM)    │ ───► Lực quán tính F_q = m * a
                    │    │       Độ cao h_cm  │
                    └────┼────────────────────┘
                         │ Trọng lực P = m * g
              ╭───────╮  ▼                      ╭───────╮
              │Bánh Đỡ│◄─── Khoảng cách d_f ────│BánhChính│
              ╰───•───╯                         ╰───•───╯
  ════════════════╧═════════════════════════════════╧══════════════
                   Điểm tựa lật A                Điểm tiếp đất B
```

Xét robot có trọng tâm ở độ cao $h_{\text{cm}}$ so với mặt đất, và khoảng cách theo chiều dọc từ hình chiếu trọng tâm tới điểm tiếp đất của bánh trước/sau gần nhất là $d_{\min}$.
Khi robot phanh gấp hoặc tăng tốc với gia tốc $a$, mô-men gây lật quanh điểm tựa là $\tau_{\text{lật}} = m \cdot |a| \cdot h_{\text{cm}}$, còn mô-men giữ thăng bằng của trọng lực là $\tau_{\text{giữ}} = m \cdot g \cdot d_{\min}$.

Để robot **không bị nhấc bánh / lật nhào** ($\tau_{\text{lật}} < \tau_{\text{giữ}}$), gia tốc cực đại cho phép là:
$$a_{\max\_\text{stable}} = g \cdot \frac{d_{\min}}{h_{\text{cm}}}$$

---

## 4. Đặc tính Vật lý của Động cơ Điện (DC / BLDC Motor): Sự khác biệt giữa URDF và Thực tế

Trong file URDF, giới hạn của động cơ chỉ được khai báo bằng 2 con số độc lập:
```xml
<limit effort="1.5" velocity="20.0"/>
```
Điều này tạo ra một **vùng làm việc hình chữ nhật lý tưởng**: ở tốc độ tối đa $\omega = 20\text{ rad/s}$, simulator vẫn cho phép động cơ xuất ra $100\%$ mô-men xoắn $\tau = 1.5\text{ N}\cdot\text{m}$!

**Nhưng ngoài đời thực thì sao?**
Theo định luật Kirchhoff và định luật cảm ứng điện từ Faraday trên cuộn dây phần ứng động cơ điện:
$$V_{\text{supply}} \ge I \cdot R_a + K_e \cdot \omega$$
Vì mô-men xoắn tỷ lệ với dòng điện ($\tau = K_t \cdot I \implies I = \tau / K_t$), suy ra **mô-men xoắn cực đại thực tế giảm tuyến tính khi tốc độ quay tăng lên** (do sức phản điện động Back-EMF $K_e \omega$ chống lại điện áp nguồn):
$$\tau_{\max}(\omega) = \tau_{\text{stall}} \left( 1 - \frac{|\omega|}{\omega_{\text{no\_load}}} \right)$$

```
Mô-men xoắn τ (N.m)
  ▲
  │
τ_stall ┼───────────────────────────────┐  ◄── Giới hạn ảo trong URDF chuẩn (Hình chữ nhật)
  │     │ ╲                             │      (Vùng tam giác phía trên bên phải là
  │     │   ╲  Đường đặc tính vật lý    │       VÙNG BẤT KHẢ THI VẬT LÝ!)
  │     │     ╲  thực tế của Motor DC   │
  │     │       ╲                       │
  │     │         ╲  Vùng làm việc thực │
  │     │           ╲                   │
  0 ────┴─────────────╲─────────────────┴──────► Vận tốc góc ω (rad/s)
                      ω_no_load
```

> 💡 **Kết luận quan trọng để kiểm định bằng Code:**
> Nếu Policy RL trong mô phỏng liên tục ra lệnh đòi hỏi cả $(\omega \approx \omega_{\max})$ lẫn $(\tau \approx \tau_{\max})$ nằm ở góc trên bên phải của đồ thị, ngoài đời thực động cơ sẽ **không thể tạo ra lực đó**, dẫn đến robot thực bị hụt lực và sụp đổ!

---

## 5. Mã nguồn Minh họa Thực thi: Bộ Kiểm tra Vật lý Cấu hình Robot (`01_urdf_physics_inspector.py`)

File mã nguồn [01_urdf_physics_inspector.py](file:///home/thinh/Reinforcement-Learning/Sim2Real_Robotics/code/01_urdf_physics_inspector.py) tự động thực hiện **5 bài kiểm tra vật lý ở Tầng 1** đối với cấu hình của robot **Andino** (và so sánh với một cấu hình bị gõ sai thông số):
1. Kiểm tra tính xác định dương của Tensor quán tính ($\mathbf{I} \succ 0$).
2. Kiểm tra Bất đẳng thức Tam giác Quán tính ($I_1 + I_2 \ge I_3$).
3. Tính toán Đa giác chân đế, gia tốc chống lật cực đại $a_{\max\_\text{stable}}$ và góc dốc tối đa không lật $\theta_{\max} = \arctan(d_{\min} / h_{\text{cm}})$.
4. Tính toán điều kiện ổn định bước thời gian tích phân $\Delta t_{\max}$ từ độ cứng tiếp xúc $k_p$.
5. Kiểm tra đường đặc tính Cơ - Điện của động cơ (Stall Torque, No-load Speed, Max Mechanical Power).

🔗 **Tiếp theo:** Sau khi đảm bảo thông số tĩnh của Robot hợp lệ ở Tầng 1, hãy chuyển sang [Phần 2: Bộ giải Động lực học & Mô hình hóa Cảm biến](file:///home/thinh/Reinforcement-Learning/Sim2Real_Robotics/Part2_Physics_Engine_And_Sensors.md).
