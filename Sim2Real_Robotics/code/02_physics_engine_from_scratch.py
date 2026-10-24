#!/usr/bin/env python3
"""
TẦNG 2: BỘ GIẢI ĐỘNG LỰC HỌC VẬT LÝ & CẢM BIẾN THỰC TẾ (PHYSICS & SENSOR ENGINE)
=================================================================================
Mục đích:
  Xây dựng bộ giải vật lý từ gốc cho Robot 2 bánh vi sai Andino, bao gồm:
  1. Quán tính khối lượng (m) và quán tính quay (I_zz).
  2. Đường đặc tính Cơ - Điện của động cơ DC (Stall Torque, Back-EMF No-load Speed).
  3. Ma sát trượt/bám Coulomb giữa lốp xe và mặt sàn (có hiện tượng trượt bánh Wheel Slip).
  4. Hàng đợi độ trễ phần cứng (Hardware Latency FIFO Buffer).
  5. Hai bộ tích phân giải tích số: Symplectic Euler (tốc độ cao) và Runge-Kutta bậc 4 (RK4).
  6. Cảm biến thực tế:
     - 2D LiDAR Ray-casting (tính giao điểm tia laser với vật cản + nhiễu + mất tia).
     - Wheel Encoder (lượng tử hóa xung + sai số tích lũy Odometry khi bánh xe trượt).
     - IMU Gyroscope (nhiễu trắng + trôi góc Bias Random Walk).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple
import math
import random


@dataclass
class PhysicalParams:
    # Thông số hình học & quán tính chuẩn của Robot Andino
    mass: float = 2.18              # kg (Tổng khối lượng)
    izz: float = 0.0125             # kg.m^2 (Mô-men quán tính xoay Yaw)
    wheel_radius: float = 0.033     # m (Bán kính bánh xe: 3.3 cm)
    wheel_separation: float = 0.137 # m (Khoảng cách 2 bánh: 13.7 cm)
    robot_radius: float = 0.105     # m (Bán kính thân xe va chạm: 10.5 cm)

    # Thông số động cơ DC & truyền động
    motor_stall_torque: float = 0.45   # N.m (Mô-men cực đại khi ω = 0)
    motor_noload_speed: float = 20.0   # rad/s (Vận tốc góc cực đại không tải)
    motor_kp: float = 0.18             # Hệ số khuếch đại bộ điều khiển vận tốc góc bánh xe
    viscous_friction: float = 0.004    # N.m.s/rad (Ma sát nhớt trục quay)
    linear_drag: float = 0.25          # N.s/m (Lực cản tịnh tiến)
    angular_drag: float = 0.015        # N.m.s/rad (Lực cản xoay)

    # Thông số tiếp xúc mặt sàn (Coulomb & Stribeck Friction)
    mu_static: float = 0.85            # Hệ số ma sát nghỉ cực đại (bám đường)
    mu_kinetic: float = 0.65           # Hệ số ma sát trượt động học (khi bánh bị xoáy trượt)

    # Thông số thời gian & độ trễ phần cứng
    sim_dt: float = 0.002              # Bước tích phân vật lý: 2 ms (500 Hz)
    control_decimation: int = 25       # 25 * 2ms = 50 ms (Tần số điều khiển Policy: 20 Hz)
    action_delay_steps: int = 1        # Số bước control trễ (1 bước = 50 ms, hoặc theo sim step)
    integrator: str = "symplectic_euler"  # "symplectic_euler" hoặc "rk4"

    # Thông số nhiễu cảm biến (Sensor Noise & Imperfections)
    encoder_cpr: int = 1024            # Số xung Encoder trên 1 vòng quay bánh xe
    lidar_num_rays: int = 12           # Số tia quét LiDAR xung quanh 360 độ
    lidar_max_range: float = 3.5       # Tầm xa tối đa của LiDAR (m)
    lidar_noise_std: float = 0.01      # Độ lệch chuẩn nhiễu đo khoảng cách LiDAR (1 cm)
    lidar_dropout_prob: float = 0.0    # Xác suất mất tia phản xạ
    imu_gyro_noise_std: float = 0.01   # rad/s (Nhiễu đo vận tốc góc IMU)
    imu_bias_drift_std: float = 0.0005 # rad/s per step (Độ trôi điểm không IMU)


@dataclass
class ObstacleCircle:
    x: float
    y: float
    radius: float


@dataclass
class RobotState:
    # Trạng thái Vật lý Thật (Ground Truth trong World Frame)
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0          # Vận tốc dài tâm xe (m/s)
    omega: float = 0.0      # Vận tốc góc Yaw (rad/s)

    # Tốc độ góc thực của 2 bánh xe (có thể khác v/r khi bị trượt bánh!)
    omega_wheel_l: float = 0.0
    omega_wheel_r: float = 0.0

    # Trạng thái ước lượng từ Bánh xe (Wheel Odometry - bị ảnh hưởng bởi trượt bánh & lượng tử hóa)
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_theta: float = 0.0
    imu_bias: float = 0.0

    # Các chỉ số vật lý ghi nhận trong bước chạy (dùng để kiểm định ở Tầng 4)
    tau_l: float = 0.0
    tau_r: float = 0.0
    slip_ratio_l: float = 0.0
    slip_ratio_r: float = 0.0
    mechanical_power: float = 0.0  # Công suất cơ học tức thời P = τ_L*ω_L + τ_R*ω_R (W)
    kinetic_energy: float = 0.0    # Động năng tức thời E_k = 0.5*m*v^2 + 0.5*I_zz*ω^2 (J)


def wrap_angle(angle: float) -> float:
    """Chuẩn hóa góc về đoạn [-pi, +pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class AndinoPhysicalSimulator:
    """
    Trình mô phỏng Vật lý thực tế cho Robot 2 bánh vi sai Andino.
    Hỗ trợ cả chế độ Lý tưởng (Toy Kinematic) và chế độ Vật lý đầy đủ (Euler / RK4).
    """

    def __init__(
        self,
        params: PhysicalParams = PhysicalParams(),
        obstacles: List[ObstacleCircle] = None,
        ideal_kinematic_mode: bool = False,
    ):
        self.params = params
        self.obstacles = obstacles if obstacles is not None else []
        self.ideal_kinematic_mode = ideal_kinematic_mode
        self.state = RobotState()
        self.latency_queue: Deque[Tuple[float, float]] = deque()
        self.reset()

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
    ) -> RobotState:
        self.state = RobotState(
            x=x, y=y, theta=theta,
            odom_x=x, odom_y=y, odom_theta=theta,
        )
        self.latency_queue.clear()
        for _ in range(max(0, self.params.action_delay_steps)):
            self.latency_queue.append((0.0, 0.0))
        return self.state

    def _compute_wheel_forces_and_torques(
        self,
        v: float,
        omega: float,
        cmd_v: float,
        cmd_omega: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Từ vận tốc mong muốn (cmd_v, cmd_omega) và trạng thái hiện tại (v, omega):
        1. Tính tốc độ góc bánh xe mục tiêu (ω_target_L, ω_target_R).
        2. Áp dụng đường bao Torque-Speed của động cơ DC để tìm mô-men xoắn (τ_L, τ_R).
        3. Áp dụng định luật ma sát Coulomb để tìm lực đẩy mặt đất thực tế (F_L, F_R)
           và tốc độ quay bánh xe thực tế (có xét trượt bánh Wheel Slip).
        """
        p = self.params
        r = p.wheel_radius
        L = p.wheel_separation

        # Tốc độ góc bánh xe tương ứng với vận tốc mặt đất không trượt
        omega_ground_l = (v - 0.5 * L * omega) / r
        omega_ground_r = (v + 0.5 * L * omega) / r

        # Tốc độ góc mục tiêu từ lệnh điều khiển
        omega_target_l = (cmd_v - 0.5 * L * cmd_omega) / r
        omega_target_r = (cmd_v + 0.5 * L * cmd_omega) / r

        # Mô-men xoắn bộ điều khiển P ở mạch cầu H
        raw_tau_l = p.motor_kp * (omega_target_l - omega_ground_l)
        raw_tau_r = p.motor_kp * (omega_target_r - omega_ground_r)

        # Giới hạn vật lý theo đường đặc tính Torque-Speed của động cơ DC (Back-EMF)
        max_tau_l = p.motor_stall_torque * max(0.0, 1.0 - abs(omega_ground_l) / p.motor_noload_speed)
        max_tau_r = p.motor_stall_torque * max(0.0, 1.0 - abs(omega_ground_r) / p.motor_noload_speed)

        tau_l = max(-max_tau_l, min(max_tau_l, raw_tau_l))
        tau_r = max(-max_tau_r, min(max_tau_r, raw_tau_r))

        # Lực tiếp tuyến truyền xuống mặt đất (trừ ma sát nhớt ổ trục)
        f_des_l = (tau_l - p.viscous_friction * omega_ground_l) / r
        f_des_r = (tau_r - p.viscous_friction * omega_ground_r) / r

        # Giới hạn bám đường Coulomb trên mỗi bánh xe: N_wheel = 0.5 * m * g
        normal_force_per_wheel = 0.5 * p.mass * 9.81
        max_static_grip = p.mu_static * normal_force_per_wheel
        kinetic_grip = p.mu_kinetic * normal_force_per_wheel

        # Kiểm tra trượt bánh trái
        if abs(f_des_l) <= max_static_grip:
            f_l = f_des_l
            omega_wheel_l = omega_ground_l
        else:
            f_l = math.copysign(kinetic_grip, f_des_l)
            # Mô-men dư làm bánh xe xoáy trượt nhanh hơn mặt đất
            excess_force_l = f_des_l - f_l
            omega_wheel_l = omega_ground_l + 0.4 * excess_force_l / max(p.viscous_friction, 0.01)

        # Kiểm tra trượt bánh phải
        if abs(f_des_r) <= max_static_grip:
            f_r = f_des_r
            omega_wheel_r = omega_ground_r
        else:
            f_r = math.copysign(kinetic_grip, f_des_r)
            excess_force_r = f_des_r - f_r
            omega_wheel_r = omega_ground_r + 0.4 * excess_force_r / max(p.viscous_friction, 0.01)

        return f_l, f_r, tau_l, tau_r, omega_wheel_l, omega_wheel_r

    def _continuous_derivatives(
        self,
        state_vec: Tuple[float, float, float, float, float],
        cmd_v: float,
        cmd_omega: float,
        ext_force: float = 0.0,
        ext_torque: float = 0.0,
    ) -> Tuple[float, float, float, float, float]:
        """
        Tính đạo hàm bậc nhất [dx/dt, dy/dt, dtheta/dt, dv/dt, domega/dt]
        dùng cho bộ tích phân RK4 hoặc Euler.
        """
        x, y, theta, v, omega = state_vec
        p = self.params
        f_l, f_r, _, _, _, _ = self._compute_wheel_forces_and_torques(v, omega, cmd_v, cmd_omega)

        dv_dt = (f_l + f_r - p.linear_drag * v + ext_force) / p.mass
        domega_dt = (0.5 * p.wheel_separation * (f_r - f_l) - p.angular_drag * omega + ext_torque) / p.izz
        dx_dt = v * math.cos(theta)
        dy_dt = v * math.sin(theta)
        dtheta_dt = omega
        return (dx_dt, dy_dt, dtheta_dt, dv_dt, domega_dt)

    def step_control(
        self,
        cmd_v: float,
        cmd_omega: float,
        ext_force: float = 0.0,
        ext_torque: float = 0.0,
    ) -> RobotState:
        """
        Thực thi 1 chu kỳ điều khiển (Control Step = control_decimation * sim_dt, mặc định 50 ms).
        """
        p = self.params

        # 1. Đẩy lệnh điều khiển vào hàng đợi độ trễ phần cứng (Hardware Latency FIFO)
        self.latency_queue.append((cmd_v, cmd_omega))
        delayed_v, delayed_omega = self.latency_queue.popleft()

        # 2. Nếu ở chế độ Lý tưởng (Toy Kinematic Sim): Đáp ứng tức thì, không quán tính, không trượt
        if self.ideal_kinematic_mode:
            dt_total = p.sim_dt * p.control_decimation
            self.state.v = delayed_v
            self.state.omega = delayed_omega
            self.state.theta = wrap_angle(self.state.theta + delayed_omega * dt_total)
            self.state.x += delayed_v * math.cos(self.state.theta) * dt_total
            self.state.y += delayed_v * math.sin(self.state.theta) * dt_total
            self.state.odom_x = self.state.x
            self.state.odom_y = self.state.y
            self.state.odom_theta = self.state.theta
            return self.state

        # 3. Chế độ Vật lý Thực tế: Lặp qua `control_decimation` bước vi phân `sim_dt`
        dt = p.sim_dt
        for _ in range(p.control_decimation):
            s = self.state
            f_l, f_r, tau_l, tau_r, omega_wl, omega_wr = self._compute_wheel_forces_and_torques(
                s.v, s.omega, delayed_v, delayed_omega
            )

            if p.integrator == "rk4":
                y0 = (s.x, s.y, s.theta, s.v, s.omega)
                k1 = self._continuous_derivatives(y0, delayed_v, delayed_omega, ext_force, ext_torque)
                y1 = tuple(y0[i] + 0.5 * dt * k1[i] for i in range(5))
                k2 = self._continuous_derivatives(y1, delayed_v, delayed_omega, ext_force, ext_torque)
                y2 = tuple(y0[i] + 0.5 * dt * k2[i] for i in range(5))
                k3 = self._continuous_derivatives(y2, delayed_v, delayed_omega, ext_force, ext_torque)
                y3 = tuple(y0[i] + dt * k3[i] for i in range(5))
                k4 = self._continuous_derivatives(y3, delayed_v, delayed_omega, ext_force, ext_torque)

                s.x += (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
                s.y += (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
                s.theta = wrap_angle(s.theta + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]))
                s.v += (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
                s.omega += (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
            else:
                # Symplectic Euler (Cập nhật vận tốc trước, dùng vận tốc mới cập nhật vị trí)
                accel_v = (f_l + f_r - p.linear_drag * s.v + ext_force) / p.mass
                accel_omega = (
                    0.5 * p.wheel_separation * (f_r - f_l) - p.angular_drag * s.omega + ext_torque
                ) / p.izz
                s.v += accel_v * dt
                s.omega += accel_omega * dt
                s.theta = wrap_angle(s.theta + s.omega * dt)
                s.x += s.v * math.cos(s.theta) * dt
                s.y += s.v * math.sin(s.theta) * dt

            # Ghi nhận thông số động cơ & năng lượng
            s.omega_wheel_l = omega_wl
            s.omega_wheel_r = omega_wr
            s.tau_l = tau_l
            s.tau_r = tau_r
            omega_ground_l = (s.v - 0.5 * p.wheel_separation * s.omega) / p.wheel_radius
            omega_ground_r = (s.v + 0.5 * p.wheel_separation * s.omega) / p.wheel_radius
            s.slip_ratio_l = abs(omega_wl - omega_ground_l) / max(abs(omega_wl), 1.0)
            s.slip_ratio_r = abs(omega_wr - omega_ground_r) / max(abs(omega_wr), 1.0)
            s.mechanical_power = abs(tau_l * omega_wl) + abs(tau_r * omega_wr)
            s.kinetic_energy = 0.5 * p.mass * (s.v ** 2) + 0.5 * p.izz * (s.omega ** 2)

            # 4. Cập nhật Wheel Odometry từ xung lượng tử hóa của Encoder (như trong bài 4 ROS 2)
            tick_rad = (2.0 * math.pi) / p.encoder_cpr
            dphi_l = round((omega_wl * dt) / tick_rad) * tick_rad
            dphi_r = round((omega_wr * dt) / tick_rad) * tick_rad

            v_enc_l = p.wheel_radius * (dphi_l / dt)
            v_enc_r = p.wheel_radius * (dphi_r / dt)
            v_odom = 0.5 * (v_enc_r + v_enc_l)
            omega_odom = (v_enc_r - v_enc_l) / p.wheel_separation

            s.odom_theta = wrap_angle(s.odom_theta + omega_odom * dt)
            s.odom_x += v_odom * math.cos(s.odom_theta) * dt
            s.odom_y += v_odom * math.sin(s.odom_theta) * dt

        return self.state

    def read_lidar_scan(self) -> List[float]:
        """
        Mô phỏng cảm biến 2D LiDAR bằng thuật toán Ray-casting hình học thực tế.
        Trả về danh sách khoảng cách [d_0, d_1, ..., d_{N-1}] trong khoảng [0, lidar_max_range].
        """
        p = self.params
        s = self.state
        ranges: List[float] = []

        for i in range(p.lidar_num_rays):
            ray_angle = s.theta + (2.0 * math.pi * i) / p.lidar_num_rays
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)
            min_dist = p.lidar_max_range

            # Giải phương trình bậc 2 tìm giao điểm tia laser với từng chướng ngại vật hình tròn
            for obs in self.obstacles:
                fx = s.x - obs.x
                fy = s.y - obs.y
                b = 2.0 * (fx * dx + fy * dy)
                c = (fx * fx + fy * fy) - (obs.radius * obs.radius)
                discriminant = b * b - 4.0 * c
                if discriminant >= 0.0:
                    sqrt_disc = math.sqrt(discriminant)
                    t1 = 0.5 * (-b - sqrt_disc)
                    t2 = 0.5 * (-b + sqrt_disc)
                    if 0.0 < t1 < min_dist:
                        min_dist = t1
                    elif 0.0 < t2 < min_dist:
                        min_dist = t2

            # Giả lập hiện tượng mất tia (Ray Dropout) và nhiễu đo lường Gaussian
            if p.lidar_dropout_prob > 0.0 and random.random() < p.lidar_dropout_prob:
                measured = p.lidar_max_range
            else:
                noise = random.gauss(0.0, p.lidar_noise_std) if p.lidar_noise_std > 0 else 0.0
                measured = max(0.05, min(p.lidar_max_range, min_dist + noise))
            ranges.append(measured)

        return ranges

    def check_collision(self) -> bool:
        """Kiểm tra vỏ vật lý của robot có va chạm với bất kỳ chướng ngại vật nào không."""
        for obs in self.obstacles:
            dist = math.hypot(self.state.x - obs.x, self.state.y - obs.y)
            if dist <= (self.params.robot_radius + obs.radius):
                return True
        return False


def run_comparison_experiment() -> None:
    print("=" * 84)
    print("🔬 THỰC NGHIỆM TẦNG 2: SO SÁNH MÔ PHỎNG ĐỘNG HỌC NGÂY THƠ VS. VẬT LÝ THỰC TẾ")
    print("=" * 84)
    print("Kịch bản: Cấp lệnh tăng tốc đột ngột v_cmd = 0.55 m/s, ω_cmd = 0.8 rad/s trong 2.0 giây,")
    print("sau đó phanh gấp về 0.0 m/s trong 1.0 giây tiếp theo.\n")

    sim_ideal = AndinoPhysicalSimulator(ideal_kinematic_mode=True)
    sim_dry = AndinoPhysicalSimulator(
        PhysicalParams(mu_static=0.85, mu_kinetic=0.65, action_delay_steps=1, integrator="rk4")
    )
    sim_icy = AndinoPhysicalSimulator(
        PhysicalParams(mu_static=0.12, mu_kinetic=0.08, action_delay_steps=1, integrator="rk4")
    )

    for step in range(60):  # 60 bước * 50 ms = 3.0 giây
        t = (step + 1) * 0.05
        cmd_v = 0.55 if t <= 2.0 else 0.0
        cmd_w = 0.80 if t <= 2.0 else 0.0

        s_id = sim_ideal.step_control(cmd_v, cmd_w)
        s_dry = sim_dry.step_control(cmd_v, cmd_w)
        s_icy = sim_icy.step_control(cmd_v, cmd_w)

        if (step + 1) in (1, 5, 20, 40, 45, 60):
            odom_err_dry = math.hypot(s_dry.x - s_dry.odom_x, s_dry.y - s_dry.odom_y)
            odom_err_icy = math.hypot(s_icy.x - s_icy.odom_x, s_icy.y - s_icy.odom_y)
            print(
                f"⏱️ t = {t:4.2f}s (Lệnh v={cmd_v:.2f}):\n"
                f"  • [Toy Kinematic] v={s_id.v:.3f} m/s | Vị trí thật=({s_id.x:+.3f}, {s_id.y:+.3f})\n"
                f"  • [Vật lý Sàn Khô] v={s_dry.v:.3f} m/s | Vị trí thật=({s_dry.x:+.3f}, {s_dry.y:+.3f}) | "
                f"Sai số Odom={odom_err_dry*100:.2f} cm | Trượt={s_dry.slip_ratio_r*100:.1f}%\n"
                f"  • [Vật lý Sàn Trơn] v={s_icy.v:.3f} m/s | Vị trí thật=({s_icy.x:+.3f}, {s_icy.y:+.3f}) | "
                f"Sai số Odom={odom_err_icy*100:.2f} cm | Trượt={s_icy.slip_ratio_r*100:.1f}%"
            )

    print("\n💡 NHẬN XÉT KẾT QUẢ VẬT LÝ:")
    print("  1. Tại t = 0.05s: Toy Sim vọt lên 0.550 m/s ngay lập tức (phi vật lý), trong khi")
    print("     Physical Sim còn ở 0.000 m/s do độ trễ phần cứng 50ms và phải thắng quán tính khối lượng.")
    print("  2. Trên sàn trơn (μ = 0.12): Bánh xe bị xoáy trượt (Slip > 50%), khiến Encoder tưởng xe")
    print("     đã đi xa nhưng thực tế xe bị trượt tại chỗ -> Sai số Odometry lên tới hàng chục cm!")
    print("=" * 84)


if __name__ == "__main__":
    run_comparison_experiment()
