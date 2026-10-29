# Hướng Dẫn Môi Trường Ảo (`.venv`), Chạy Code RL (`.py` / `.ipynb`) & Triển Khai ROS 2

> **Tài liệu này lưu lại toàn bộ cấu hình môi trường thực thi đã được cài đặt sẵn trong kho lưu trữ `/home/thinh/Reinforcement-Learning`, giúp bạn chạy ngay lập tức mọi kịch bản Python (`.py`), Jupyter Notebook (`.ipynb`) và Node điều khiển Robot ROS 2.**

---

## 1. Tổng Quan Môi Trường Đã Cài Đặt Sẵn

Tại thư mục gốc `/home/thinh/Reinforcement-Learning`, hệ thống đã được thiết lập sẵn:

1. **Môi trường ảo Python (`.venv/`)**:
   - Phiên bản: **Python 3.14.4** (`/home/thinh/Reinforcement-Learning/.venv/bin/python`).
   - Các thư viện cốt lõi đã cài đặt (danh sách đầy đủ trong [`requirements.txt`](requirements.txt)):
     - **Deep RL & Tính toán khoa học**: `torch` (`2.15.0+cpu`), `numpy` (`2.5.4`), `scipy` (`1.18.1`), `scikit-learn` (`1.9.1`).
     - **Môi trường mô phỏng RL**: `gymnasium` (`1.4.0`), `pygame-ce` (`2.5.8`), `opencv-python-headless` (`5.0.0`).
     - **Trực quan hóa & Theo dõi huấn luyện**: `matplotlib` (`3.11.2`), `tensorboardX` (`2.6.5`), `tqdm`.
     - **Jupyter & Tài liệu**: `jupyter`, `notebook`, `jupyterlab`, `ipykernel`, `ipywidgets`, `markdown`.
2. **Jupyter Kernel đăng ký sẵn (`rl-ros-env`)**:
   - Tên hiển thị: **`Python 3.14 (RL & ROS Sim2Real .venv)`**.
   - Đường dẫn cấu hình: `~/.local/share/jupyter/kernels/rl-ros-env`.
   - Đã gắn mặc định vào cả 3 file `.ipynb` trong `Week2/` và `Week4/`.
3. **Lớp tương thích Classic Gym & Roboschool ([`setup_rl_env.py`](setup_rl_env.py))**:
   - Cài đặt trực tiếp tại `.venv/lib/python3.14/site-packages/gym/__init__.py` và `roboschool.py`.
   - Tự động chuyển đổi hàm `env.reset()` (từ tuple `(obs, info)` của Gymnasium sang `obs`) và `env.step(action)` (từ 5-tuple sang 4-tuple `(obs, reward, done, info)`).
   - Tự động ánh xạ các tên môi trường cổ điển:
     - `FrozenLake-v0` $\to$ `FrozenLake-v1`
     - `Taxi-v2` $\to$ `Taxi-v3`
     - `CartPole-v0` $\to$ `CartPole-v1`
     - `PongNoFrameskip-v4` $\to$ Môi trường giả lập Atari Pong RGB $(210 \times 160 \times 3)$ tích hợp sẵn.
     - `BipedalWalker-v2`, `LunarLanderContinuous-v2`, `RoboschoolAnt-v1` $\to$ Môi trường động lực học điều khiển liên tục tích hợp sẵn.

---

## 2. Kích Hoạt Môi Trường Ảo Hàng Ngày

Mỗi khi mở một cửa sổ Terminal mới, bạn di chuyển vào thư mục dự án và kích hoạt `.venv`:

```bash
cd /home/thinh/Reinforcement-Learning
source .venv/bin/activate
```

Sau khi kích hoạt, dấu nhắc lệnh sẽ có tiền tố `(.venv)` và lệnh `python` / `pip` / `jupyter` sẽ tự động trỏ vào môi trường ảo.  
*(Nếu không muốn chạy `source .venv/bin/activate`, bạn có thể gọi trực tiếp `.venv/bin/python <tên_file.py>`).*

---

## 3. Hướng Dẫn Chạy Các File Jupyter Notebook (`.ipynb`)

Kho lưu trữ có **3 bài thực hành trên Jupyter Notebook**:
- [`Week2/frozenlake_Qlearning.ipynb`](Week2/frozenlake_Qlearning.ipynb) — Thuật toán Tabular Q-Learning trên lưới băng `FrozenLake-v0`.
- [`Week4/PolicyGradient.ipynb`](Week4/PolicyGradient.ipynb) — Thuật toán REINFORCE với Baseline trên `CartPole-v0`.
- [`Week4/A2C.ipynb`](Week4/A2C.ipynb) — Thuật toán Advantage Actor-Critic (A2C) đa môi trường song song ($N=40$) trên `CartPole-v0`.

### Cách 1: Chạy trực tiếp trong VS Code / Antigravity IDE
1. Mở file `.ipynb` bất kỳ trong trình soạn thảo.
2. Ở góc trên bên phải của cửa sổ Notebook, kiểm tra Kernel đang chọn là **`Python 3.14 (RL & ROS Sim2Real .venv)`** (hoặc chọn đường dẫn `/home/thinh/Reinforcement-Learning/.venv/bin/python`).
3. Nhấn **Run All** (hoặc `Shift + Enter` từng ô).

### Cách 2: Mở giao diện Jupyter Notebook / JupyterLab trên trình duyệt
```bash
cd /home/thinh/Reinforcement-Learning
.venv/bin/jupyter lab
# Hoặc giao diện cổ điển:
.venv/bin/jupyter notebook
```

---

## 4. Hướng Dẫn Chạy Các File Python (`.py`) Theo Từng Phần

### 4.1. Thực nghiệm Toán học (Xác suất Thống kê & Đại số Tuyến tính trên Mẫu)
Chạy 10 kịch bản kiểm chứng cách các toán tử toán học tác động trực tiếp lên đám mây mẫu dữ liệu:
```bash
.venv/bin/python minh_hoa_toan_hoc_tren_mau.py
```

### 4.2. Các Thuật toán Deep RL (Tuần 3, Tuần 5, Tuần 6, Tuần 7)

| Tuần | Thuật toán | Lệnh chạy trong Terminal | Ghi chú cấu hình nhanh |
| :--- | :--- | :--- | :--- |
| **Week 3** | **DQN / Double DQN / Dueling DQN / Multi-step** | `cd Week3 && ../.venv/bin/python main.py` | Chỉnh từ điển `DQN_HYPERPARAMS` trong [`Week3/main.py`](Week3/main.py) (`double_DQN: True`, `dueling: True`). |
| **Week 5** | **PPO-Clip + Chuỗi GAE($\gamma, \lambda$)** | `.venv/bin/python Week5/PPO.py` | Lưu log vào `content/runs/` và lưu trọng số tốt nhất vào `checkpoints/`. |
| **Week 6** | **Evolution Strategies (Mirrored Sampling đa tiến trình)** | `.venv/bin/python Week6/ES.py` | Sử dụng `torch.multiprocessing` với `MAX_WORKERS = 4`. |
| **Week 7** | **Model-Based RL + Random Shooting MPC** | `.venv/bin/python Week7/model_based.py` | Tự động dùng GPU nếu có hoặc chuyển về CPU (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`). |

---

## 5. Hướng Dẫn Chạy Chuỗi 7 Tầng Sim-to-Real Robotics & ROS 2

### 5.1. Chạy trực tiếp trên máy tính bằng `.venv` (Không cần bật Docker/Gazebo)
Toàn bộ 5 kịch bản kiểm định vật lý và mô phỏng luồng bản tin ROS 2 đều chạy độc lập 100% trên `.venv`:

```bash
# Tầng 1: Kiểm tra thông số vật lý URDF (khối lượng, ma trận quán tính, bất đẳng thức tam giác)
.venv/bin/python Sim2Real_Robotics/code/01_urdf_physics_inspector.py

# Tầng 2: Giải phương trình vi phân động lực học (So sánh RK4 vs Euler, giới hạn trượt ma sát & LiDAR Ray-casting)
.venv/bin/python Sim2Real_Robotics/code/02_physics_engine_from_scratch.py

# Tầng 3: Huấn luyện RL Policy điều hướng (So sánh Naive Policy vs Domain Randomization Policy)
# -> Tự động xuất file trọng số Sim2Real_Robotics/code/trained_policies.json
.venv/bin/python Sim2Real_Robotics/code/03_train_rl_navigation_agent.py

# Tầng 4: Bộ 7 Bài Kiểm Định Thực Tế Vật Lý Chỉ Bằng Code (Code-Only Physical Verification Suite)
.venv/bin/python Sim2Real_Robotics/code/04_physical_reality_verification_suite.py

# Tầng 5–6–7: Giả lập vòng lặp Node ROS 2 (/scan + /odom -> Policy -> Safety Shield -> /cmd_vel ở 20 Hz)
.venv/bin/python Sim2Real_Robotics/code/05_ros2_rl_deployment_node.py --standalone-test
```

### 5.2. Chạy Node RL trực tiếp trong Docker ROS 2 Humble & Gazebo
Trên máy của bạn đã có sẵn container **`robotics_essentials_ros2`** (chứa ROS 2 Humble & Gazebo). Để kết nối bộ điều khiển RL vào ROS 2 thật:

```bash
# Bước 1: Khởi động container ROS 2 có sẵn trên máy
docker start robotics_essentials_ros2

# Bước 2: Sao chép Node triển khai RL và file trọng số đã huấn luyện vào trong container
docker cp Sim2Real_Robotics/code/05_ros2_rl_deployment_node.py robotics_essentials_ros2:/home/user/
docker cp Sim2Real_Robotics/code/trained_policies.json robotics_essentials_ros2:/home/user/

# Bước 3: Chạy Node ROS 2 thực tế (lắng nghe /scan, /odom và xuất lệnh /cmd_vel)
docker exec -it robotics_essentials_ros2 bash -c "source /opt/ros/humble/setup.bash && python3 /home/user/05_ros2_rl_deployment_node.py --goal-x 1.9 --goal-y 0.25"
```

---

## 6. Cách Cập Nhật Lại Hệ Thống Trang Web HTML & Phục Hồi Môi Trường

### 6.1. Biên dịch lại toàn bộ Markdown sang HTML (kèm 10 Phòng thí nghiệm mô phỏng)
Mỗi khi bạn chỉnh sửa nội dung các file `.md` hoặc [`interactive_labs.py`](interactive_labs.py), hãy chạy:
```bash
.venv/bin/python build_html_docs.py
```

### 6.2. Khôi phục lại `.venv` từ đầu (nếu chuyển sang máy tính mới)
Nếu bạn clone kho lưu trữ này sang một máy tính khác, chỉ cần chạy 3 lệnh sau để tái tạo toàn bộ môi trường:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python setup_rl_env.py
```
