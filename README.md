# Hệ Thống Học Tăng Cường Sâu (Deep Reinforcement Learning) & Sim-to-Real Robotics

> **Tài liệu học tập, chứng minh toán học từng bước (Step-by-Step Function Substitutions), phân tích mã nguồn và kiểm định vật lý Sim-to-Real Robotics.**  
> 🌐 **Xem trực tiếp trên trình duyệt Local:** Mở tệp **[`index.html`](index.html)** bằng bất kỳ trình duyệt nào (Chrome, Firefox, Edge, Brave) để đọc toàn bộ hệ thống bài giảng với công thức toán học LaTeX/KaTeX, mục lục tự động, tìm kiếm nhanh và giao diện Sáng/Tối.

---

## 🌐 1. Cách Xem Toàn Bộ Tài Liệu Bằng HTML Trên Trình Duyệt Local

Toàn bộ các tệp kiến thức Markdown (`.md`) trong kho lưu trữ này đã được biên dịch sẵn sang các tệp **`.html` tương ứng tại đúng vị trí thư mục** (và trang cổng chính tại **[`index.html`](index.html)**).

- **Cách 1 (Mở trực tiếp tệp tin):** Nhấp đúp vào **[`index.html`](index.html)** hoặc mở đường dẫn:
  ```bash
  xdg-open /home/thinh/Reinforcement-Learning/index.html
  ```
- **Cách 2 (Chạy máy chủ HTTP cục bộ nếu muốn):**
  ```bash
  cd /home/thinh/Reinforcement-Learning
  python3 -m http.server 8080
  # Sau đó truy cập: http://localhost:8080/index.html
  ```
- **Cách cập nhật lại toàn bộ HTML sau khi chỉnh sửa bất kỳ file `.md` nào:**
  ```bash
  python3 build_html_docs.py
  ```

---

## 📐 2. Nền Tảng Toán Học, Ký Hiệu & Sổ Tay Tra Cứu Toàn Khóa

| STT | Tài liệu | Bản Markdown (`.md`) | Bản Trình duyệt (`.html`) | Nội dung cốt lõi |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Bảng Ký Hiệu Toán Học Chuẩn Tắc** | [BANG_KY_HIEU_TOAN_HOC.md](BANG_KY_HIEU_TOAN_HOC.md) | [BANG_KY_HIEU_TOAN_HOC.html](BANG_KY_HIEU_TOAN_HOC.html) | Chuẩn hóa toàn bộ ký hiệu không gian ($\mathcal{S}, \mathcal{A}$), toán tử Bellman ($\mathcal{T}^\pi, \mathcal{T}^*$), phân phối, đạo hàm và chỉ số xuyên suốt 8 tuần. |
| **2** | **Tổng Hợp Toán Học: XSTK, ĐSTT & Giải Tích** | [TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.md](TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.md) | [TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.html](TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.html) | Giải thích trực quan cách từng toán tử tác động lên đám mây mẫu dữ liệu, nêu bật tính chất ẩn và **Bản đồ 8 phép thế hàm kinh điển trong RL**. Kèm code thực nghiệm [minh_hoa_toan_hoc_tren_mau.py](minh_hoa_toan_hoc_tren_mau.py). |
| **3** | **Cẩm Nang Toàn Diện Về Tham Số RL** | [CAM_NANG_THAM_SO_RL.md](CAM_NANG_THAM_SO_RL.md) | [CAM_NANG_THAM_SO_RL.html](CAM_NANG_THAM_SO_RL.html) | Giải mã cơ chế tác động giải tích, khoảng giá trị chuẩn và bảng chẩn đoán bệnh (Triệu chứng $\to$ Nguyên nhân toán học $\to$ Cách chỉnh tham số). |
| **4** | **Lộ Trình Kiến Thức Cần Học** | [KIEN_THUC_CAN_HOC.md](KIEN_THUC_CAN_HOC.md) | [KIEN_THUC_CAN_HOC.html](KIEN_THUC_CAN_HOC.html) | Bản đồ kỹ năng từ Toán nền tảng, Python/NumPy, PyTorch đến Lý thuyết RL cổ điển và Deep RL. |
| **5** | **Tổng Hợp Kiến Thức & Đối Chiếu Code 8 Tuần** | [KIEN_THUC_CHI_TIET_TUNG_TUAN.md](KIEN_THUC_CHI_TIET_TUNG_TUAN.md) | [KIEN_THUC_CHI_TIET_TUNG_TUAN.html](KIEN_THUC_CHI_TIET_TUNG_TUAN.html) | Tóm tắt công thức trọng tâm, phân loại thuật toán, bảng tổng hợp lỗi (Bug) trong code mẫu và cách nâng cấp lên thư viện Gymnasium/PyTorch mới. |
| **6** | **Hướng Dẫn Môi Trường Ảo (`.venv`), Chạy Code RL & ROS 2** | [HUONG_DAN_MOI_TRUONG_VA_CHAY_CODE.md](HUONG_DAN_MOI_TRUONG_VA_CHAY_CODE.md) | [HUONG_DAN_MOI_TRUONG_VA_CHAY_CODE.html](HUONG_DAN_MOI_TRUONG_VA_CHAY_CODE.html) | Hướng dẫn kích hoạt `.venv`, chọn Jupyter Kernel `rl-ros-env`, chạy toàn bộ `.ipynb`, `.py` và triển khai Node ROS 2 trên Docker `robotics_essentials_ros2`. |

---

## 🎓 3. Hệ Thống Bài Giảng Chuyên Sâu 8 Tuần (Lý Thuyết, Thế Hàm Từng Bước & Giải Phẫu Code)

| Tuần | Chủ đề Bài giảng | Bản Markdown (`.md`) | Bản Trình duyệt (`.html`) | Mã nguồn & Thực hành |
| :--- | :--- | :--- | :--- | :--- |
| **Tuần 1** | **Giới thiệu RL, Multi-Armed Bandits (UCB1) & Pong from Pixels** | [Week1/BAI_GIANG.md](Week1/BAI_GIANG.md) | [Week1/BAI_GIANG.html](Week1/BAI_GIANG.html) | Lý thuyết nền tảng & giải tích đạo hàm Log-Bernoulli |
| **Tuần 2** | **MDP, Phương trình Bellman, Định lý Co Banach, MC, TD & Q-Learning** | [Week2/BAI_GIANG.md](Week2/BAI_GIANG.md) | [Week2/BAI_GIANG.html](Week2/BAI_GIANG.html) | [Week2/frozenlake_Qlearning.ipynb](Week2/frozenlake_Qlearning.ipynb) |
| **Tuần 3** | **Xấp xỉ Hàm Giá trị & Các Biến thể Deep Q-Network (DQN, Double, Dueling, Multi-step, NoisyNet)** | [Week3/BAI_GIANG.md](Week3/BAI_GIANG.md) | [Week3/BAI_GIANG.html](Week3/BAI_GIANG.html) | [Week3/main.py](Week3/main.py), [central_control.py](Week3/central_control.py), [neural_net.py](Week3/neural_net.py) |
| **Tuần 4** | **Định lý Policy Gradient, REINFORCE, Giảm Phương sai & Advantage Actor-Critic (A2C)** | [Week4/BAI_GIANG.md](Week4/BAI_GIANG.md) | [Week4/BAI_GIANG.html](Week4/BAI_GIANG.html) | [Week4/PolicyGradient.ipynb](Week4/PolicyGradient.ipynb), [Week4/A2C.ipynb](Week4/A2C.ipynb) |
| **Tuần 5** | **Policy Gradient Nâng cao: Vùng Tin cậy TRPO, PPO-Clip & Ước lượng Lợi thế Tổng quát (GAE)** | [Week5/BAI_GIANG.md](Week5/BAI_GIANG.md) | [Week5/BAI_GIANG.html](Week5/BAI_GIANG.html) | [Week5/PPO.py](Week5/PPO.py) |
| **Tuần 6** | **Tối ưu Hộp đen: Evolution Strategies (ES), Mirrored Sampling & Genetic Algorithms (GA)** | [Week6/BAI_GIANG.md](Week6/BAI_GIANG.md) | [Week6/BAI_GIANG.html](Week6/BAI_GIANG.html) | [Week6/ES.py](Week6/ES.py) |
| **Tuần 7** | **Model-Based RL: Học Động học Phần dư, Chính quy hóa Tikhonov, Simulation Lemma & MPC** | [Week7/BAI_GIANG.md](Week7/BAI_GIANG.md) | [Week7/BAI_GIANG.html](Week7/BAI_GIANG.html) | [Week7/model_based.py](Week7/model_based.py) |
| **Tuần 8** | **Chủ đề Hiện đại (SOTA): MaxEnt RL & SAC, IRL/GAIL, RND, AlphaGo Zero, OpenAI Five, RLHF & DPO** | [Week8/BAI_GIANG.md](Week8/BAI_GIANG.md) | [Week8/BAI_GIANG.html](Week8/BAI_GIANG.html) | Tổng hợp kiến trúc SOTA & Đồ án Capstone |

---

## 🤖 4. Học Phần Simulation Vật Lý & Kiến Trúc 7 Tầng Sim-to-Real Robotics

| Hạng mục | Bản Markdown (`.md`) | Bản Trình duyệt (`.html`) | Kịch bản Python Thực nghiệm (Chạy độc lập) |
| :--- | :--- | :--- | :--- |
| **Cẩm nang Tổng quan Simulation & Vật lý Robot** | [HUONG_DAN_SIMULATION_VA_VAT_LY_ROBOT.md](HUONG_DAN_SIMULATION_VA_VAT_LY_ROBOT.md) | [HUONG_DAN_SIMULATION_VA_VAT_LY_ROBOT.html](HUONG_DAN_SIMULATION_VA_VAT_LY_ROBOT.html) | — |
| **Tổng quan Kiến trúc 7 Tầng Sim-to-Real** | [Sim2Real_Robotics/README.md](Sim2Real_Robotics/README.md) | [Sim2Real_Robotics/README.html](Sim2Real_Robotics/README.html) | — |
| **Phần 1 (Tầng 1): Mô tả Cơ học, URDF/MJCF & Kiểm định Quán tính** | [Sim2Real_Robotics/Part1_Physical_Modeling_URDF_MJCF.md](Sim2Real_Robotics/Part1_Physical_Modeling_URDF_MJCF.md) | [Sim2Real_Robotics/Part1_Physical_Modeling_URDF_MJCF.html](Sim2Real_Robotics/Part1_Physical_Modeling_URDF_MJCF.html) | [01_urdf_physics_inspector.py](Sim2Real_Robotics/code/01_urdf_physics_inspector.py) |
| **Phần 2 (Tầng 2): Bộ giải Động lực học (Euler/RK4), Ma sát & Cảm biến** | [Sim2Real_Robotics/Part2_Physics_Engine_And_Sensors.md](Sim2Real_Robotics/Part2_Physics_Engine_And_Sensors.md) | [Sim2Real_Robotics/Part2_Physics_Engine_And_Sensors.html](Sim2Real_Robotics/Part2_Physics_Engine_And_Sensors.html) | [02_physics_engine_from_scratch.py](Sim2Real_Robotics/code/02_physics_engine_from_scratch.py) |
| **Phần 3 (Tầng 3): Đóng gói Môi trường RL, Hàm thưởng Vật lý & Domain Randomization** | [Sim2Real_Robotics/Part3_RL_Training_Environment_And_Rewards.md](Sim2Real_Robotics/Part3_RL_Training_Environment_And_Rewards.md) | [Sim2Real_Robotics/Part3_RL_Training_Environment_And_Rewards.html](Sim2Real_Robotics/Part3_RL_Training_Environment_And_Rewards.html) | [03_train_rl_navigation_agent.py](Sim2Real_Robotics/code/03_train_rl_navigation_agent.py) |
| **Phần 4 (Tầng 4): Bộ 7 Bài Kiểm định Thực tế Vật lý Chỉ Bằng Code** | [Sim2Real_Robotics/Part4_Code_Only_Physical_Verification.md](Sim2Real_Robotics/Part4_Code_Only_Physical_Verification.md) | [Sim2Real_Robotics/Part4_Code_Only_Physical_Verification.html](Sim2Real_Robotics/Part4_Code_Only_Physical_Verification.html) | [04_physical_reality_verification_suite.py](Sim2Real_Robotics/code/04_physical_reality_verification_suite.py) |
| **Phần 5 (Tầng 5–7): Cầu nối ROS 2, Lá chắn An toàn & Triển khai Gazebo/Robot Thật** | [Sim2Real_Robotics/Part5_ROS2_Gazebo_And_Real_Robot_Bridge.md](Sim2Real_Robotics/Part5_ROS2_Gazebo_And_Real_Robot_Bridge.md) | [Sim2Real_Robotics/Part5_ROS2_Gazebo_And_Real_Robot_Bridge.html](Sim2Real_Robotics/Part5_ROS2_Gazebo_And_Real_Robot_Bridge.html) | [05_ros2_rl_deployment_node.py](Sim2Real_Robotics/code/05_ros2_rl_deployment_node.py) |
