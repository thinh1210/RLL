#!/usr/bin/env python3
"""
TẦNG 1: CÔNG CỤ KIỂM ĐỊNH VẬT LÝ CỦA CẤU HÌNH ROBOT (URDF / MJCF PHYSICS INSPECTOR)
====================================================================================
Mục đích:
  Chỉ dùng code để kiểm tra xem các thông số cấu hình của một Robot (Khối lượng,
  Trọng tâm, Tensor quán tính 3x3, Kích thước hình học, Độ cứng tiếp xúc, Động cơ)
  có thỏa mãn các định luật Vật lý của thế giới thực hay không, trước khi đưa vào
  Gazebo / MuJoCo / Isaac Lab.

Các bài kiểm tra thực hiện:
  1. Kiểm tra Khối lượng dương (m > 0).
  2. Kiểm tra Ma trận Quán tính đối xứng xác định dương (Eigenvalues I_1, I_2, I_3 > 0).
  3. Kiểm tra Bất đẳng thức Tam giác Vật lý của Mô-men quán tính (I_i + I_j >= I_k).
  4. Kiểm tra Đa giác chân đế (Support Polygon), Gia tốc phanh chống lật tối đa
     (a_tip_over) và Góc leo dốc tối đa không lật.
  5. Kiểm tra Điều kiện Ổn định Số CFL (max_step_size dt_max) theo độ cứng mặt sàn kp.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class LinkInertialConfig:
    name: str
    mass: float  # kg
    com_xyz: Tuple[float, float, float]  # m (tọa độ trọng tâm)
    ixx: float  # kg.m^2
    iyy: float  # kg.m^2
    izz: float  # kg.m^2
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    def principal_moments(self) -> Tuple[float, float, float]:
        """
        Tính chính xác 3 trị riêng (Principal Moments of Inertia I_1 <= I_2 <= I_3)
        của ma trận đối xứng 3x3 bằng thuật toán lượng giác giải tích (Oliver Smith, 1961).
        """
        a11, a22, a33 = self.ixx, self.iyy, self.izz
        a12, a13, a23 = self.ixy, self.ixz, self.iyz

        p1 = a12**2 + a13**2 + a23**2
        if p1 < 1e-18:
            # Ma trận đã ở dạng đường chéo
            eigs = sorted([a11, a22, a33])
            return (eigs[0], eigs[1], eigs[2])

        q = (a11 + a22 + a33) / 3.0
        p2 = (a11 - q)**2 + (a22 - q)**2 + (a33 - q)**2 + 2.0 * p1
        p = math.sqrt(p2 / 6.0)

        # Ma trận B = (1 / p) * (A - q * I)
        b11, b22, b33 = (a11 - q) / p, (a22 - q) / p, (a33 - q) / p
        b12, b13, b23 = a12 / p, a13 / p, a23 / p

        det_b = (
            b11 * (b22 * b33 - b23**2)
            - b12 * (b12 * b33 - b13 * b23)
            + b13 * (b12 * b23 - b13 * b22)
        )
        r = max(-1.0, min(1.0, det_b / 2.0))
        phi = math.acos(r) / 3.0

        eig3 = q + 2.0 * p * math.cos(phi)
        eig1 = q + 2.0 * p * math.cos(phi + (2.0 * math.pi / 3.0))
        eig2 = 3.0 * q - eig1 - eig3
        eigs = sorted([eig1, eig2, eig3])
        return (eigs[0], eigs[1], eigs[2])


@dataclass
class RobotPhysicalConfig:
    robot_name: str
    links: List[LinkInertialConfig]
    wheel_radius: float        # m (Andino: 0.033 m)
    wheel_separation: float    # m (Andino: 0.137 m)
    front_support_x: float     # m (vị trí tiếp đất bánh trước)
    rear_support_x: float      # m (vị trí tiếp đất bánh tự lựa phía sau)
    contact_kp: float          # N/m (độ cứng tiếp xúc mặt sàn)
    contact_kd: float          # N.s/m (giảm chấn tiếp xúc)
    sim_dt: float              # s (bước tích phân thời gian của simulator)
    motor_stall_torque: float  # N.m (mô-men xoắn cực đại khi kẹt trục)
    motor_noload_speed: float  # rad/s (tốc độ góc tối đa không tải)


def compute_equivalent_bounding_ellipsoid(
    mass: float, principal_moments: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Từ 3 mô-men quán tính chính (I_1, I_2, I_3) và khối lượng m,
    khôi phục bán trục (a, b, c) của khối Ellipsoid đồng nhất tương đương:
      I_1 = (1/5) m (b^2 + c^2), v.v.
      => a = sqrt(2.5 / m * (I_2 + I_3 - I_1))
    Nếu biểu thức trong căn âm -> Vật thể KHÔNG THỂ TỒN TẠI trong không gian 3D!
    """
    I1, I2, I3 = principal_moments
    val_a = 2.5 / mass * (I2 + I3 - I1)
    val_b = 2.5 / mass * (I1 + I3 - I2)
    val_c = 2.5 / mass * (I1 + I2 - I3)
    if min(val_a, val_b, val_c) < 0:
        return (float("nan"), float("nan"), float("nan"))
    return (math.sqrt(val_a), math.sqrt(val_b), math.sqrt(val_c))


def audit_robot_physics(config: RobotPhysicalConfig) -> bool:
    print("=" * 80)
    print(f"🔍 BÁO CÁO KIỂM ĐỊNH VẬT LÝ CẤU HÌNH ROBOT: {config.robot_name}")
    print("=" * 80)

    all_passed = True
    total_mass = 0.0
    weighted_com = [0.0, 0.0, 0.0]

    # 1. KIỂM TRA TỪNG LINK (MASS, POSITIVE DEFINITENESS & TRIANGLE INEQUALITY)
    print("\n[1] KIỂM TRA KHỐI LƯỢNG & MA TRẬN QUÁN TÍNH TỪNG LINK (<inertial>):")
    for link in config.links:
        I1, I2, I3 = link.principal_moments()

        mass_ok = link.mass > 0.0
        pos_def_ok = (I1 > 1e-9) and (I2 > 1e-9) and (I3 > 1e-9)
        # Bất đẳng thức tam giác: Tổng 2 trị riêng nhỏ nhất phải >= trị riêng lớn nhất
        triangle_margin = (I1 + I2) - I3
        triangle_ok = triangle_margin >= -1e-9

        status = "✅ HỢP LỆ" if (mass_ok and pos_def_ok and triangle_ok) else "❌ VÔ LÝ VẬT LÝ"
        if not (mass_ok and pos_def_ok and triangle_ok):
            all_passed = False

        print(f"  • Link '{link.name:<14}': m = {link.mass:6.3f} kg | "
              f"Trị riêng I = [{I1:.5f}, {I2:.5f}, {I3:.5f}] kg.m² -> {status}")

        if not pos_def_ok:
            print("      ⚠️ LỖI: Ma trận quán tính có trị riêng <= 0! Simulator sẽ chia cho 0 (NaN).")
        elif not triangle_ok:
            print(f"      ⚠️ LỖI VI PHẠM BẤT ĐẲNG THỨC TAM GIÁC: I_1 + I_2 ({I1+I2:.5f}) < I_3 ({I3:.5f})!")
            print("         Không tồn tại phân bố mật độ khối lượng dương nào tạo ra quán tính này.")
        else:
            semi_axes = compute_equivalent_bounding_ellipsoid(link.mass, (I1, I2, I3))
            print(f"      ↳ Bán trục Ellipsoid đồng nhất tương đương: "
                  f"[{semi_axes[0]*100:.1f} cm, {semi_axes[1]*100:.1f} cm, {semi_axes[2]*100:.1f} cm]")

        if mass_ok:
            total_mass += link.mass
            for idx in range(3):
                weighted_com[idx] += link.mass * link.com_xyz[idx]

    overall_com = [c / max(total_mass, 1e-9) for c in weighted_com]

    # 2. KIỂM TRA ĐA GIÁC CHÂN ĐẾ & GIỚI HẠN LẬT (SUPPORT POLYGON & TIP-OVER LIMIT)
    print("\n[2] KIỂM TRA TRỌNG TÂM TỔNG HỢP & ĐỘ ỔN ĐỊNH CHỐNG LẬT (SUPPORT POLYGON):")
    x_com, y_com, z_com = overall_com
    x_min = min(config.rear_support_x, config.front_support_x)
    x_max = max(config.rear_support_x, config.front_support_x)
    y_max = config.wheel_separation / 2.0

    inside_x = x_min < x_com < x_max
    inside_y = -y_max < y_com < y_max
    static_stable = inside_x and inside_y and (z_com > 0)

    d_front = abs(x_max - x_com)
    d_rear = abs(x_com - x_min)
    d_min = min(d_front, d_rear)
    g = 9.81
    a_tip_over = g * (d_min / max(z_com, 1e-6))
    max_slope_deg = math.degrees(math.atan2(d_min, max(z_com, 1e-6)))

    print(f"  • Tổng khối lượng Robot        : {total_mass:.3f} kg")
    print(f"  • Tọa độ Trọng tâm tổng (CoM)  : x = {x_com*100:+.2f} cm, y = {y_com*100:+.2f} cm, z = {z_com*100:.2f} cm")
    print(f"  • Đa giác tiếp đất dọc trục X  : [{x_min*100:+.1f} cm, {x_max*100:+.1f} cm]")
    if static_stable:
        print("  • Trạng thái cân bằng tĩnh     : ✅ Nằm vững trong đa giác chân đế")
        print(f"  • Gia tốc phanh/tăng tốc tối đa trước khi chổng bánh lật: a_max = {a_tip_over:.2f} m/s² ({a_tip_over/g:.2f} G)")
        print(f"  • Góc dốc tối đa không bị lật ngửa                      : θ_max = {max_slope_deg:.1f}°")
    else:
        all_passed = False
        print("  • Trạng thái cân bằng tĩnh     : ❌ TRỌNG TÂM RƠI RA NGOÀI ĐA GIÁC CHÂN ĐẾ! Robot sẽ tự lật úp.")

    # 3. KIỂM TRA BƯỚC THỜI GIAN TÍCH PHÂN SỐ (NUMERICAL STABILITY CFL CONDITION)
    print("\n[3] KIỂM TRA ỔN ĐỊNH GIẢI TÍCH SỐ CỦA BỘ GIẢI VA CHẠM (CONTACT SOLVER STABILITY):")
    min_link_mass = min(link.mass for link in config.links if link.mass > 0)
    omega_n = math.sqrt(config.contact_kp / min_link_mass)  # Tần số góc dao động riêng (rad/s)
    dt_critical_euler = 2.0 / omega_n
    zeta = config.contact_kd / (2.0 * math.sqrt(config.contact_kp * min_link_mass))

    dt_ok = config.sim_dt <= dt_critical_euler
    if not dt_ok:
        all_passed = False

    print(f"  • Độ cứng mặt sàn (kp)         : {config.contact_kp:.1f} N/m | Giảm chấn (kd): {config.contact_kd:.1f} N.s/m")
    print(f"  • Tỉ số tắt dần va chạm (ζ)    : {zeta:.2f} ({'Quá nảy (Underdamped)' if zeta < 0.7 else 'Êm ái (Well-damped)'})")
    print(f"  • Bước thời gian tới hạn CFL   : dt_crit = {dt_critical_euler*1000:.3f} ms | Bước hiện tại: sim_dt = {config.sim_dt*1000:.3f} ms")
    if dt_ok:
        print("  • Đánh giá tích phân số        : ✅ Ổn định (sim_dt <= dt_crit)")
    else:
        print("  • Đánh giá tích phân số        : ❌ MẤT ỔN ĐỊNH SỐ (sim_dt > dt_crit)! Robot sẽ tự rung lắc hoặc nổ tung!")

    # 4. KIỂM TRA ĐỘNG LỰC HỌC ĐỘNG CƠ (ACTUATOR TORQUE-SPEED-POWER ENVELOPE)
    print("\n[4] ĐẶC TÍNH CƠ - ĐIỆN CỦA ĐỘNG CƠ TRUYỀN ĐỘNG (DC MOTOR ENVELOPE):")
    v_max_linear = config.wheel_radius * config.motor_noload_speed
    f_max_traction = (2.0 * config.motor_stall_torque) / config.wheel_radius
    a_max_motor = f_max_traction / total_mass
    p_mech_max_per_motor = 0.25 * config.motor_stall_torque * config.motor_noload_speed

    print(f"  • Vận tốc dài tối đa lý thuyết : v_max = {v_max_linear:.2f} m/s ({v_max_linear*3.6:.1f} km/h)")
    print(f"  • Gia tốc đề-pa tối đa (2 bánh): a_motor = {a_max_motor:.2f} m/s²")
    print(f"  • Công suất cơ học đỉnh / motor: P_max = {p_mech_max_per_motor:.2f} W (tại ω = {0.5*config.motor_noload_speed:.1f} rad/s)")
    if static_stable and a_max_motor > a_tip_over:
        print(f"  ⚠️ CẢNH BÁO: Gia tốc cực đại của motor ({a_max_motor:.2f} m/s²) > Gia tốc lật ({a_tip_over:.2f} m/s²).")
        print("     Cần giới hạn gia tốc (Slew-rate limiter) trong code điều khiển để tránh bốc đầu xe!")

    print("\n" + "=" * 80)
    print(f"🏁 KẾT LUẬN TỔNG THỂ CHO '{config.robot_name}': "
          f"{'✅ ĐẠT CHUẨN VẬT LÝ THỰC TẾ (PASS)' if all_passed else '❌ PHÁT HIỆN LỖI CẤU HÌNH VẬT LÝ (FAIL)'}")
    print("=" * 80 + "\n")
    return all_passed


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # TRƯỜNG HỢP 1: CẤU HÌNH CHUẨN VẬT LÝ CỦA ROBOT ANDINO (TRONG KHOÁ HỌC ROS 2)
    # -------------------------------------------------------------------------
    andino_valid_config = RobotPhysicalConfig(
        robot_name="Andino Differential Drive Robot (Chuẩn Vật Lý)",
        links=[
            LinkInertialConfig(
                name="base_link",
                mass=1.80,
                com_xyz=(-0.03, 0.0, 0.055),
                ixx=0.0045, iyy=0.0082, izz=0.0110
            ),
            LinkInertialConfig(
                name="left_wheel",
                mass=0.15,
                com_xyz=(0.0, 0.0685, 0.033),
                ixx=0.00005, iyy=0.00005, izz=0.00008
            ),
            LinkInertialConfig(
                name="right_wheel",
                mass=0.15,
                com_xyz=(0.0, -0.0685, 0.033),
                ixx=0.00005, iyy=0.00005, izz=0.00008
            ),
            LinkInertialConfig(
                name="caster_wheel",
                mass=0.08,
                com_xyz=(-0.12, 0.0, 0.015),
                ixx=0.00001, iyy=0.00001, izz=0.00001
            ),
        ],
        wheel_radius=0.033,       # Bán kính bánh xe 3.3 cm (khớp với bài 4 Robot Odometry)
        wheel_separation=0.137,   # Khoảng cách 2 bánh 13.7 cm (khớp với bài 4 Robot Odometry)
        front_support_x=0.0,      # Trục 2 bánh chính tại x = 0.0 m
        rear_support_x=-0.12,     # Bánh tự lựa phía sau tại x = -0.12 m
        contact_kp=50000.0,       # Độ cứng tiếp xúc hợp lý cho lốp cao su nhỏ
        contact_kd=150.0,         # Hệ số giảm chấn êm ái
        sim_dt=0.001,             # Bước tích phân 1 ms (1000 Hz)
        motor_stall_torque=0.35,  # Mô-men xoắn tối đa 0.35 N.m mỗi bánh
        motor_noload_speed=18.0,  # Tốc độ tối đa 18 rad/s (~0.6 m/s)
    )

    audit_robot_physics(andino_valid_config)

    # -------------------------------------------------------------------------
    # TRƯỜNG HỢP 2: CẤU HÌNH BỊ KỸ SƯ GÕ TAY SAI QUY LUẬT VẬT LÝ (PHỔ BIẾN)
    # -------------------------------------------------------------------------
    broken_robot_config = RobotPhysicalConfig(
        robot_name="Robot Cấu Hình Lỗi (Chỉnh tay sai Inertia, CoM & Step Size)",
        links=[
            LinkInertialConfig(
                name="base_link",
                mass=2.00,
                com_xyz=(0.05, 0.0, 0.15),  # Trọng tâm nhô ra phía trước trục bánh chính!
                ixx=0.002, iyy=0.002, izz=0.015  # Vi phạm bất đẳng thức tam giác: 0.002 + 0.002 < 0.015
            ),
            LinkInertialConfig(
                name="left_wheel",
                mass=0.05,
                com_xyz=(0.0, 0.0685, 0.033),
                ixx=0.00005, iyy=0.00005, izz=0.00008
            ),
        ],
        wheel_radius=0.033,
        wheel_separation=0.137,
        front_support_x=0.0,
        rear_support_x=-0.12,
        contact_kp=1000000.0,     # Độ cứng quá cao (10^6 N/m) cho bánh xe chỉ nặng 50g
        contact_kd=1.0,           # Thiếu giảm chấn
        sim_dt=0.020,             # Bước tích phân quá thô (20 ms) -> Vi phạm CFL!
        motor_stall_torque=1.50,
        motor_noload_speed=25.0,
    )

    audit_robot_physics(broken_robot_config)
