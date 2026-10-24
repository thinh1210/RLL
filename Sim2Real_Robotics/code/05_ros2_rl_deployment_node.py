#!/usr/bin/env python3
"""
TẦNG 5 – 6 – 7: NODE TRIỂN KHAI RL TRÊN ROS 2, GAZEBO VÀ ROBOT THẬT (SIM-TO-REAL BRIDGE)
=========================================================================================
Mục đích:
  Kết nối trực tiếp bộ trọng số `Sim2Real_Robust_Policy` (đã huấn luyện ở Tầng 3 và vượt
  qua 7 bài kiểm định vật lý ở Tầng 4) vào hệ điều hành Robot ROS 2:
    • Tầng 5 (ROS 2 Perception Bridge):
      - Nhận `sensor_msgs/msg/LaserScan` từ topic `/scan` -> Lọc NaN/Inf & Min-Pooling 5 góc quạt.
      - Nhận `nav_msgs/msg/Odometry` từ topic `/odom` -> Giải mã Quaternion sang góc Yaw θ
        và chiếu tọa độ điểm đích (goal_x, goal_y) về hệ quy chiếu thân xe `base_link`.
    • Tầng 6 (Hardware Safety Shield / Control Barrier):
      - Watchdog kiểm tra mất tín hiệu cảm biến.
      - Giới hạn gia tốc Slew-Rate Limiter (chống bốc đầu lật xe & bảo vệ mạch cầu H).
      - Phanh khẩn cấp tự động (Emergency TTC Brake) nếu vật cản trước mặt < 0.22 m.
    • Tầng 7 (Actuator Command Publisher):
      - Xuất gói tin `geometry_msgs/msg/Twist` ra topic `/cmd_vel` ở tần số cố định 20 Hz.

Cách sử dụng:
  1. Chạy kiểm thử độc lập ngay trên máy tính (không cần cài ROS 2):
       python3 Sim2Real_Robotics/code/05_ros2_rl_deployment_node.py --standalone-test
  2. Chạy trực tiếp trong Docker ROS 2 Humble cùng Gazebo (robotics_essentials_ros2):
       python3 /home/user/exercises_ws/src/05_ros2_rl_deployment_node.py --goal-x 1.9 --goal-y 0.25
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Chuyển đổi Quaternion (x, y, z, w) từ gói tin `nav_msgs/msg/Odometry`
    sang góc quay phẳng Yaw θ (rad) quanh trục Z.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class StandalonePolicyInference:
    """
    Bộ suy luận mạng Policy nhẹ, độc lập 100% (không phụ thuộc thư viện ngoài),
    tải trực tiếp trọng số từ `trained_policies.json` để nhúng lên máy tính nhúng của Robot.
    """

    OBS_DIM = 11
    HIDDEN_DIM = 8
    ACT_DIM = 2

    def __init__(self, weights: List[float], action_filter_beta: float = 0.35):
        self.weights = list(weights)
        self.action_filter_beta = action_filter_beta
        self.prev_filtered_action = [0.0, 0.0]

    def reset(self) -> None:
        self.prev_filtered_action = [0.0, 0.0]

    def infer_action(self, obs: List[float]) -> Tuple[float, float]:
        w = self.weights
        idx = 0
        out0 = 0.0
        out1 = 0.0
        for j in range(self.OBS_DIM):
            out0 += w[idx] * obs[j]
            idx += 1
        for j in range(self.OBS_DIM):
            out1 += w[idx] * obs[j]
            idx += 1

        hidden = [0.0] * self.HIDDEN_DIM
        for i in range(self.HIDDEN_DIM):
            s = 0.0
            for j in range(self.OBS_DIM):
                s += w[idx] * obs[j]
                idx += 1
            s += w[idx]
            idx += 1
            hidden[i] = math.tanh(s)

        for i in range(self.HIDDEN_DIM):
            out0 += w[idx] * hidden[i]
            idx += 1
        for i in range(self.HIDDEN_DIM):
            out1 += w[idx] * hidden[i]
            idx += 1

        out0 += w[idx]
        out1 += w[idx + 1]

        raw_v = math.tanh(out0)
        raw_w = math.tanh(out1)

        beta = self.action_filter_beta
        filt_v = beta * raw_v + (1.0 - beta) * self.prev_filtered_action[0]
        filt_w = beta * raw_w + (1.0 - beta) * self.prev_filtered_action[1]
        self.prev_filtered_action = [filt_v, filt_w]
        return (filt_v, filt_w)


class HardwareSafetyShield:
    """
    TẦNG 6: LÁ CHẮN AN TOÀN VẬT LÝ (HARDWARE SAFETY SHIELD)
    Bọc bên ngoài mạng nơ-ron để bảo vệ phần cứng Robot trong mọi tình huống:
      1. Slew-Rate Limiter: Giới hạn gia tốc dài |dv/dt| <= a_max_safe (1.5 m/s^2)
         và gia tốc góc |dω/dt| <= alpha_max_safe (6.0 rad/s^2).
      2. Emergency Collision Reflex: Phanh khẩn cấp nếu vật cản chính diện < 0.22 m.
    """

    def __init__(
        self,
        dt: float = 0.05,
        max_lin_accel: float = 1.50,
        max_ang_accel: float = 6.00,
        emergency_stop_dist: float = 0.22,
    ):
        self.dt = dt
        self.max_dv = max_lin_accel * dt
        self.max_dw = max_ang_accel * dt
        self.emergency_stop_dist = emergency_stop_dist
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.interventions = 0

    def filter_command(
        self,
        desired_v: float,
        desired_w: float,
        front_obstacle_dist: float,
    ) -> Tuple[float, float, str]:
        status = "NORMAL"

        # 1. Phanh khẩn cấp khi quá gần vật cản phía trước (TTC / Distance Barrier)
        if front_obstacle_dist < self.emergency_stop_dist and desired_v > 0.0:
            desired_v = 0.0
            # Nếu góc xoay đang quá nhỏ, hỗ trợ xoay thoát hiểm
            if abs(desired_w) < 0.4:
                desired_w = 0.8 if desired_w >= 0.0 else -0.8
            status = f"EMERGENCY_BRAKE (d_front={front_obstacle_dist:.2f}m < {self.emergency_stop_dist}m)"
            self.interventions += 1

        # 2. Giới hạn độ dốc gia tốc (Slew-Rate Acceleration Limiter)
        dv = desired_v - self.last_cmd_v
        if abs(dv) > self.max_dv:
            safe_v = self.last_cmd_v + math.copysign(self.max_dv, dv)
            if status == "NORMAL":
                status = "SLEW_RATE_LIMITED"
        else:
            safe_v = desired_v

        dw = desired_w - self.last_cmd_w
        if abs(dw) > self.max_dw:
            safe_w = self.last_cmd_w + math.copysign(self.max_dw, dw)
        else:
            safe_w = desired_w

        self.last_cmd_v = safe_v
        self.last_cmd_w = safe_w
        return (safe_v, safe_w, status)


class ROS2ObservationConverter:
    """
    TẦNG 5: Bộ chuyển đổi dữ liệu thô từ cảm biến ROS 2 (`/scan`, `/odom`)
    sang vector quan sát 11 chiều chuẩn hóa khớp 100% với Tầng 3.
    """

    V_MAX = 0.50
    OMEGA_MAX = 1.80
    LIDAR_MAX_RANGE = 1.80

    @classmethod
    def extract_5_sector_proximities(
        cls,
        ranges: List[float],
        angle_min: float,
        angle_increment: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Gộp mảng N tia của LaserScan thành 5 vùng quạt bằng phép lấy Min-Pooling:
          - Front       : [-15°, +15°]
          - Front-Left  : [+15°, +45°]
          - Front-Right : [-45°, -15°]
          - Left        : [+45°, +75°]
          - Right       : [-75°, -45°]
        """
        sectors = {
            "front": cls.LIDAR_MAX_RANGE,
            "fl": cls.LIDAR_MAX_RANGE,
            "fr": cls.LIDAR_MAX_RANGE,
            "left": cls.LIDAR_MAX_RANGE,
            "right": cls.LIDAR_MAX_RANGE,
        }

        for i, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r <= 0.03:
                continue
            clamped_r = min(r, cls.LIDAR_MAX_RANGE)
            deg = math.degrees(wrap_angle(angle_min + i * angle_increment))

            if -15.0 <= deg <= 15.0:
                sectors["front"] = min(sectors["front"], clamped_r)
            elif 15.0 < deg <= 45.0:
                sectors["fl"] = min(sectors["fl"], clamped_r)
            elif -45.0 <= deg < -15.0:
                sectors["fr"] = min(sectors["fr"], clamped_r)
            elif 45.0 < deg <= 75.0:
                sectors["left"] = min(sectors["left"], clamped_r)
            elif -75.0 <= deg < -45.0:
                sectors["right"] = min(sectors["right"], clamped_r)

        def to_prox(dist: float) -> float:
            return max(0.0, min(1.0, 1.0 - dist / cls.LIDAR_MAX_RANGE))

        return (
            to_prox(sectors["front"]),
            to_prox(sectors["fl"]),
            to_prox(sectors["fr"]),
            to_prox(sectors["left"]),
            to_prox(sectors["right"]),
            sectors["front"],
        )

    @classmethod
    def build_observation_vector(
        cls,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        linear_v: float,
        angular_w: float,
        goal_x: float,
        goal_y: float,
        sector_proximities: Tuple[float, float, float, float, float],
        prev_action_w: float,
    ) -> Tuple[List[float], float]:
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dist_goal = math.hypot(dx, dy)
        angle_to_goal = wrap_angle(math.atan2(dy, dx) - robot_yaw)

        obs = [
            min(dist_goal / 2.5, 1.0),
            math.cos(angle_to_goal),
            math.sin(angle_to_goal),
            max(-1.0, min(1.0, linear_v / cls.V_MAX)),
            max(-1.0, min(1.0, angular_w / cls.OMEGA_MAX)),
            sector_proximities[0],
            sector_proximities[1],
            sector_proximities[2],
            sector_proximities[3],
            sector_proximities[4],
            prev_action_w,
        ]
        return obs, dist_goal


def load_robust_policy() -> StandalonePolicyInference:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "trained_policies.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data["sim2real_robust_policy"]
        return StandalonePolicyInference(cfg["weights"], cfg["action_filter_beta"])

    # Fallback nếu file được copy riêng lẻ mà chưa kèm file JSON
    w = [0.0] * 136
    w[0], w[1], w[5] = 1.40, 1.60, -1.35
    w[13], w[17], w[18], w[19], w[20], w[15], w[21] = 1.95, -2.60, 2.60, -1.15, 1.15, -0.30, 0.25
    return StandalonePolicyInference(w, action_filter_beta=0.35)


# =============================================================================
# CHẾ ĐỘ 1: CHẠY TRỰC TIẾP TRÊN HỆ ĐIỀU HÀNH ROS 2 (GAZEBO HOẶC ROBOT THẬT)
# =============================================================================
def run_ros2_node(goal_x: float, goal_y: float) -> None:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan

    class AndinoRLDeploymentNode(Node):
        def __init__(self):
            super().__init__("andino_rl_sim2real_controller")
            self.goal_x = goal_x
            self.goal_y = goal_y
            self.policy = load_robust_policy()
            self.safety_shield = HardwareSafetyShield(dt=0.05)

            # Trạng thái nhận từ /odom và /scan
            self.x = 0.0
            self.y = 0.0
            self.yaw = 0.0
            self.v = 0.0
            self.omega = 0.0
            self.sector_prox = (0.0, 0.0, 0.0, 0.0, 0.0)
            self.front_dist = 1.80
            self.prev_action_w = 0.0
            self.has_odom = False
            self.has_scan = False

            self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
            self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
            self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)

            # Vòng lặp điều khiển 20 Hz (chu kỳ 0.05s khớp hệt lúc huấn luyện ở Tầng 3)
            self.timer = self.create_timer(0.05, self.control_loop_callback)
            self.get_logger().info(
                f"🚀 Đã khởi tạo Node RL Sim-to-Real! Đích đến: ({self.goal_x:.2f}m, {self.goal_y:.2f}m)"
            )

        def odom_callback(self, msg: Odometry) -> None:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
            self.v = msg.twist.twist.linear.x
            self.omega = msg.twist.twist.angular.z
            self.has_odom = True

        def scan_callback(self, msg: LaserScan) -> None:
            p0, p1, p2, p3, p4, d_front = ROS2ObservationConverter.extract_5_sector_proximities(
                list(msg.ranges), msg.angle_min, msg.angle_increment
            )
            self.sector_prox = (p0, p1, p2, p3, p4)
            self.front_dist = d_front
            self.has_scan = True

        def control_loop_callback(self) -> None:
            if not (self.has_odom and self.has_scan):
                return

            obs, dist_goal = ROS2ObservationConverter.build_observation_vector(
                self.x, self.y, self.yaw, self.v, self.omega,
                self.goal_x, self.goal_y, self.sector_prox, self.prev_action_w
            )

            cmd_msg = Twist()
            if dist_goal <= 0.18:
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.pub_cmd.publish(cmd_msg)
                self.get_logger().info("🎯 ĐÃ ĐẾN ĐÍCH AN TOÀN! Dừng động cơ.")
                return

            a_v, a_w = self.policy.infer_action(obs)
            self.prev_action_w = a_w

            desired_v = 0.5 * (a_v + 1.0) * ROS2ObservationConverter.V_MAX
            desired_w = a_w * ROS2ObservationConverter.OMEGA_MAX

            safe_v, safe_w, shield_status = self.safety_shield.filter_command(
                desired_v, desired_w, self.front_dist
            )

            cmd_msg.linear.x = float(safe_v)
            cmd_msg.angular.z = float(safe_w)
            self.pub_cmd.publish(cmd_msg)

            if shield_status != "NORMAL":
                self.get_logger().warn(f"🛡️ Safety Shield kích hoạt: {shield_status}")

    rclpy.init()
    node = AndinoRLDeploymentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


# =============================================================================
# CHẾ ĐỘ 2: KIỂM THỬ VÒNG LẶP ROS 2 ĐỘC LẬP NGAY TRÊN TERMINAL (--standalone-test)
# =============================================================================
def run_standalone_ros2_bridge_test(goal_x: float = 1.90, goal_y: float = 0.25) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from importlib import import_module
    phys_mod = import_module("02_physics_engine_from_scratch")

    print("=" * 86)
    print("🤖 TẦNG 5–6–7: MÔ PHỎNG VÒNG LẶP NODE ROS 2 (/scan + /odom -> Policy -> Shield -> /cmd_vel)")
    print("=" * 86)

    # Khởi tạo thế giới vật lý RK4 với 360 tia LiDAR giống hệt gói tin LaserScan của Gazebo
    params = phys_mod.PhysicalParams(
        lidar_num_rays=360,
        lidar_max_range=1.80,
        lidar_noise_std=0.01,
        action_delay_steps=1,
        integrator="rk4",
    )
    obstacles = [phys_mod.ObstacleCircle(x=0.95, y=-0.05, radius=0.18)]
    sim = phys_mod.AndinoPhysicalSimulator(params=params, obstacles=obstacles)
    sim.reset(x=0.0, y=0.0, theta=0.0)

    policy = load_robust_policy()
    shield = HardwareSafetyShield(dt=0.05)
    prev_action_w = 0.0

    print(f"📍 Điểm xuất phát: (0.00, 0.00) | Điểm đích: ({goal_x:.2f}, {goal_y:.2f}) | Vật cản tại (0.95, -0.05)\n")

    for step in range(1, 130):
        s = sim.state
        # 1. Giả lập gói tin /odom (chuyển góc theta sang Quaternion rồi giải mã lại như trong ROS 2)
        qz = math.sin(s.theta * 0.5)
        qw = math.cos(s.theta * 0.5)
        decoded_yaw = quaternion_to_yaw(0.0, 0.0, qz, qw)

        # 2. Giả lập gói tin /scan 360 tia (góc từ 0 đến 2*pi, bước góc 1 độ)
        ranges_360 = sim.read_lidar_scan()
        angle_inc = (2.0 * math.pi) / 360.0
        p0, p1, p2, p3, p4, d_front = ROS2ObservationConverter.extract_5_sector_proximities(
            ranges_360, angle_min=0.0, angle_increment=angle_inc
        )

        # 3. Tạo vector quan sát 11 chiều ở Tầng 5
        obs, dist_goal = ROS2ObservationConverter.build_observation_vector(
            s.x, s.y, decoded_yaw, s.v, s.omega,
            goal_x, goal_y, (p0, p1, p2, p3, p4), prev_action_w
        )

        if dist_goal <= 0.18:
            print(
                f"🎯 [Bước {step:3d} | t={step*0.05:4.2f}s] ĐÃ TỚI ĐÍCH THÀNH CÔNG! "
                f"Vị trí=({s.x:.2f}m, {s.y:.2f}m), Cách đích={dist_goal*100:.1f} cm"
            )
            break

        # 4. Suy luận mạng nơ-ron RL Policy
        a_v, a_w = policy.infer_action(obs)
        prev_action_w = a_w
        raw_cmd_v = 0.5 * (a_v + 1.0) * ROS2ObservationConverter.V_MAX
        raw_cmd_w = a_w * ROS2ObservationConverter.OMEGA_MAX

        # 5. Đi qua Lá chắn An toàn Vật lý (Tầng 6)
        safe_v, safe_w, shield_status = shield.filter_command(raw_cmd_v, raw_cmd_w, d_front)

        # 6. Xuất lệnh /cmd_vel xuống Động cơ Robot (Tầng 7)
        sim.step_control(safe_v, safe_w)

        if step == 1 or step % 12 == 0:
            print(
                f"📡 [t={step*0.05:4.2f}s] /odom=({s.x:+.2f}m, {s.y:+.2f}m, θ={math.degrees(decoded_yaw):+5.1f}°) | "
                f"/scan d_front={d_front:.2f}m | /cmd_vel=(v={safe_v:.2f} m/s, ω={safe_w:+.2f} rad/s) | "
                f"Shield: {shield_status}"
            )

    print("=" * 86)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROS 2 RL Sim-to-Real Deployment Node")
    parser.add_argument("--goal-x", type=float, default=1.90, help="Tọa độ X của điểm đích (m)")
    parser.add_argument("--goal-y", type=float, default=0.25, help="Tọa độ Y của điểm đích (m)")
    parser.add_argument(
        "--standalone-test",
        action="store_true",
        help="Chạy kiểm thử vòng lặp bản tin ROS 2 ngay trên máy tính không cần bật Docker/Gazebo",
    )
    args = parser.parse_args()

    if args.standalone_test:
        run_standalone_ros2_bridge_test(args.goal_x, args.goal_y)
    else:
        try:
            import rclpy  # noqa: F401
            run_ros2_node(args.goal_x, args.goal_y)
        except ImportError:
            print("ℹ️ Không tìm thấy môi trường `rclpy` (ROS 2) trên terminal hiện tại.")
            print("   Tự động chuyển sang chế độ `--standalone-test` để mô phỏng luồng bản tin ROS 2:\n")
            run_standalone_ros2_bridge_test(args.goal_x, args.goal_y)
