#!/usr/bin/env python3
"""
TẦNG 4: BỘ 7 BÀI KIỂM ĐỊNH THỰC TẾ VẬT LÝ CHỈ BẰNG CODE (CODE-ONLY REALITY VERIFICATION)
========================================================================================
Mục đích:
  Trả lời câu hỏi thực chiến: "Khi chưa có robot thật trong tay, chỉ với code làm sao
  để biết mô phỏng có tuân thủ định luật vật lý không, và Policy RL có chạy sống sót
  ngoài thế giới thực hay không?"

7 Bài Kiểm Định Định Lượng:
  1. [Simulator Audit] Kiểm định Bảo toàn Năng lượng & Tính Thụ động Nhiệt động lực học.
  2. [Motor & Thermal Audit] Kiểm định Đường bao Torque-Speed & Tải nhiệt hiệu dụng RMS.
  3. [Spectral & Jerk Audit] Phân tích Phổ Tần số Fourier (DFT/FFT) tìm rung lắc cao tần (Chattering).
  4. [Latency Margin Test] Kiểm định Sức chịu đựng Độ trễ Phần cứng (0ms -> 50ms -> 100ms).
  5. [Sim-to-Sim Cross-Validation] Huấn luyện trên Euler (500Hz), kiểm định trên RK4 (2000Hz) + Lệch bánh xe.
  6. [OOD Physical Stress Test] Sàn trơn ướt (μ=0.28) + Quá tải khối lượng (+30%) + Xung lực đẩy ngang.
  7. [Sensor Degradation Test] Nhiễu LiDAR tăng gấp 3x + Mất 10% tia phản xạ (Ray Dropout).
"""

from typing import Dict, List, Tuple
import json
import math
import os
import random
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from importlib import import_module
phys_mod = import_module("02_physics_engine_from_scratch")
train_mod = import_module("03_train_rl_navigation_agent")

AndinoPhysicalSimulator = phys_mod.AndinoPhysicalSimulator
PhysicalParams = phys_mod.PhysicalParams
AndinoRLNavigationEnv = train_mod.AndinoRLNavigationEnv
PolicyNetwork = train_mod.PolicyNetwork
build_warmstart_weights = train_mod.build_warmstart_weights


def compute_high_frequency_energy_ratio(signal: List[float], fs: float = 20.0, cutoff_hz: float = 3.0) -> float:
    """
    Tính Biến đổi Fourier Rời rạc (DFT) của chuỗi lệnh điều khiển u[n] (tần số lấy mẫu fs = 20 Hz)
    và trả về Tỷ lệ Năng lượng Cao tần (HFER %) nằm trên ngưỡng cutoff_hz (3.0 Hz).
    """
    n_samples = len(signal)
    if n_samples < 4:
        return 0.0

    # Khử thành phần một chiều (DC mean)
    mean_val = sum(signal) / n_samples
    centered = [x - mean_val for x in signal]

    total_ac_energy = 0.0
    high_freq_energy = 0.0

    # Xét các tần số dương k = 1 .. N // 2
    for k in range(1, (n_samples // 2) + 1):
        freq_hz = (k * fs) / n_samples
        re = 0.0
        im = 0.0
        for n, val in enumerate(centered):
            angle = (2.0 * math.pi * k * n) / n_samples
            re += val * math.cos(angle)
            im -= val * math.sin(angle)
        mag_sq = re * re + im * im
        total_ac_energy += mag_sq
        if freq_hz > cutoff_hz:
            high_freq_energy += mag_sq

    if total_ac_energy < 1e-12:
        return 0.0
    return (high_freq_energy / total_ac_energy) * 100.0


def test_1_simulator_thermodynamic_passivity() -> bool:
    """
    TEST 1: Kiểm định Bản thân Trình Mô phỏng có tuân thủ Định luật Bảo toàn Năng lượng
    và Tính Thụ động (Passivity) khi ngắt điện động cơ hay không.
    """
    print("\n[TEST 1] KIỂM ĐỊNH ĐỊNH LUẬT BẢO TOÀN NĂNG LƯỢNG & TÍNH THỤ ĐỘNG CỦA SIMULATOR:")
    sim = AndinoPhysicalSimulator(PhysicalParams(integrator="rk4", action_delay_steps=0))
    sim.reset()

    # Giai đoạn A: Cấp điện tăng tốc trong 1.5s, đo Tổng công cơ học W_motor so với Động năng E_k
    work_motor = 0.0
    dt_control = sim.params.sim_dt * sim.params.control_decimation
    for _ in range(30):
        s = sim.step_control(0.45, 0.5)
        work_motor += s.mechanical_power * dt_control

    ek_peak = sim.state.kinetic_energy
    energy_conservation_ok = ek_peak <= (work_motor + 1e-5)

    # Giai đoạn B: Ngắt hoàn toàn mô-men xoắn (tau_stall = 0 -> Coast-down tự do),
    # kiểm tra dE_k/dt <= 0 ở mọi bước
    sim.params.motor_stall_torque = 0.0
    monotonic_decay_ok = True
    prev_ek = ek_peak
    for _ in range(25):
        s = sim.step_control(0.0, 0.0)
        if s.kinetic_energy > prev_ek + 1e-9:
            monotonic_decay_ok = False
        prev_ek = s.kinetic_energy

    passed = energy_conservation_ok and monotonic_decay_ok
    print(f"  • Tổng công cơ học cung cấp (W_motor) : {work_motor:.4f} J")
    print(f"  • Động năng đỉnh của Robot (E_k_peak) : {ek_peak:.4f} J (Năng lượng ma sát tiêu hao: {work_motor - ek_peak:.4f} J)")
    print(f"  • Kiểm tra ΔE_k <= W_motor            : {'✅ HỢP LỆ' if energy_conservation_ok else '❌ VI PHẠM (Sinh năng lượng ảo)'}")
    print(f"  • Kiểm tra tắt dần khi ngắt điện      : E_k giảm từ {ek_peak:.4f} J -> {prev_ek:.4f} J "
          f"({'✅ Đơn điệu giảm (dE_k/dt <= 0)' if monotonic_decay_ok else '❌ Tăng năng lượng phi vật lý'})")
    return passed


def run_policy_audit(policy_name: str, policy: PolicyNetwork, bang_bang_dither: float = 0.0) -> Dict[str, bool]:
    print("\n" + "-" * 86)
    print(f"🤖 ĐANG SÁT HẠCH VẬT LÝ POLICY: '{policy_name}'")
    print("-" * 86)

    results: Dict[str, bool] = {}

    # -------------------------------------------------------------------------
    # TEST 2 & TEST 3: KIỂM ĐỊNH ĐỘNG CƠ, ĐỘ GIẬT (JERK) & PHỔ TẦN SỐ FFT
    # -------------------------------------------------------------------------
    env_std = AndinoRLNavigationEnv(
        ideal_kinematic_mode=False,
        domain_randomization=False,
        physics_aware_reward=True,
    )
    omega_cmds: List[float] = []
    torques_sq: List[float] = []
    rate_sqs: List[float] = []
    jerk_sqs: List[float] = []
    max_torque_seen = 0.0
    std_successes = 0

    for ep in range(4):
        obs = env_std.reset(scenario_idx=ep)
        policy.reset_state()
        done = False
        step_idx = 0
        while not done:
            dither = bang_bang_dither * (1.0 if (step_idx % 2 == 0) else -1.0)
            act = policy.act(obs, high_freq_dither=dither)
            if ep == 0:
                omega_cmds.append(act[1])
            obs, _, done, info = env_std.step(act)
            s = env_std.sim.state
            max_torque_seen = max(max_torque_seen, abs(s.tau_l), abs(s.tau_r))
            torques_sq.append(0.5 * (s.tau_l ** 2 + s.tau_r ** 2))
            rate_sqs.append(info["action_rate_sq"])
            jerk_sqs.append(info["action_jerk_sq"])
            step_idx += 1
            if done and info["reached_goal"]:
                std_successes += 1

    rms_torque = math.sqrt(sum(torques_sq) / max(1, len(torques_sq)))
    rated_continuous_torque = 0.55 * env_std.base_params.motor_stall_torque  # 0.2475 N.m
    motor_ok = (max_torque_seen <= env_std.base_params.motor_stall_torque + 1e-6) and (
        rms_torque <= rated_continuous_torque
    )
    results["Test2_Motor_Thermal"] = motor_ok

    mean_rate_sq = sum(rate_sqs) / max(1, len(rate_sqs))
    mean_jerk_sq = sum(jerk_sqs) / max(1, len(jerk_sqs))
    hfer_pct = compute_high_frequency_energy_ratio(omega_cmds, fs=20.0, cutoff_hz=3.0)
    spectral_ok = (hfer_pct < 15.0) and (mean_rate_sq < 0.08)
    results["Test3_Spectral_Smoothness"] = spectral_ok

    print(f"  [Test 2 - Đường bao Cơ-Điện & Nhiệt Motor]:")
    print(f"    • Mô-men đỉnh |τ|_max = {max_torque_seen:.3f} N.m (Giới hạn kẹt trục: {env_std.base_params.motor_stall_torque:.3f} N.m)")
    print(f"    • Mô-men nhiệt hiệu dụng τ_RMS = {rms_torque:.3f} N.m (Giới hạn dài hạn: {rated_continuous_torque:.3f} N.m) "
          f"-> {'✅ PASS' if motor_ok else '❌ FAIL (Quá nhiệt motor)'}")

    print(f"  [Test 3 - Độ mượt Cơ khí & Phổ Tần số Fourier (FFT Chattering)]:")
    print(f"    • Độ biến thiên lệnh E[||Δa||²] = {mean_rate_sq:.4f} (Ngưỡng < 0.0800)")
    print(f"    • Độ giật gia tốc E[||Δ²a||²]   = {mean_jerk_sq:.4f}")
    print(f"    • Tỷ lệ năng lượng rung cao tần (>3Hz) HFER = {hfer_pct:5.1f}% (Ngưỡng < 15.0%) "
          f"-> {'✅ PASS (Điều khiển êm)' if spectral_ok else '❌ FAIL (Rung lắc Bang-Bang phá hộp số!)'}")

    # -------------------------------------------------------------------------
    # TEST 4: KIỂM ĐỊNH BIÊN ĐỘ TRỄ PHẦN CỨNG (HARDWARE LATENCY MARGIN: 100 ms)
    # -------------------------------------------------------------------------
    params_latency = PhysicalParams(
        lidar_num_rays=12,
        lidar_max_range=1.8,
        action_delay_steps=2,  # 2 * 50ms = 100 ms độ trễ phần cứng!
        integrator="symplectic_euler",
    )
    env_lat = AndinoRLNavigationEnv(custom_params=params_latency)
    _, lat_succ_rate, _ = train_mod.evaluate_policy(
        policy, env_lat, num_episodes=4, bang_bang_dither=bang_bang_dither
    )
    latency_ok = lat_succ_rate >= 0.75
    results["Test4_Latency_100ms"] = latency_ok
    print(f"  [Test 4 - Stress-Test Độ trễ Phần cứng 100 ms]:")
    print(f"    • Tỉ lệ tới đích an toàn dưới độ trễ 100 ms = {lat_succ_rate*100:5.1f}% (Ngưỡng >= 75%) "
          f"-> {'✅ PASS' if latency_ok else '❌ FAIL (Mất ổn định do trễ pha)'}")

    # -------------------------------------------------------------------------
    # TEST 5: KIỂM ĐỊNH CHÉO SIM-TO-SIM (EULER 500Hz -> RK4 2000Hz + LỆCH BÁNH XE)
    # -------------------------------------------------------------------------
    params_rk4 = PhysicalParams(
        lidar_num_rays=12,
        lidar_max_range=1.8,
        sim_dt=0.0005,          # Bước tích phân siêu mịn 0.5 ms (2000 Hz)
        control_decimation=100, # 100 * 0.5 ms = 50 ms (20 Hz)
        integrator="rk4",       # Đổi sang bộ giải vi phân Runge-Kutta bậc 4
        wheel_radius=0.033 * 0.97,  # Sai lệch bán kính lốp xe thực tế -3%
        action_delay_steps=1,
    )
    env_rk4 = AndinoRLNavigationEnv(custom_params=params_rk4)
    _, rk4_succ_rate, _ = train_mod.evaluate_policy(
        policy, env_rk4, num_episodes=4, bang_bang_dither=bang_bang_dither
    )
    sim2sim_ok = rk4_succ_rate >= 0.75
    results["Test5_Sim2Sim_RK4"] = sim2sim_ok
    print(f"  [Test 5 - Kiểm định Chéo Sim-to-Sim (Euler 500Hz -> RK4 2000Hz + Lốp mòn 3%)]:")
    print(f"    • Tỉ lệ thành công trên Bộ giải tham chiếu RK4 = {rk4_succ_rate*100:5.1f}% (Ngưỡng >= 75%) "
          f"-> {'✅ PASS' if sim2sim_ok else '❌ FAIL (Overfit vào sai số Euler)'}")

    # -------------------------------------------------------------------------
    # TEST 6: KIỂM ĐỊNH NGOÀI PHÂN PHỐI (SÀN TRƠN μ=0.28 + CHỞ NẶNG +30% + GIÓ ĐẨY)
    # -------------------------------------------------------------------------
    params_ood = PhysicalParams(
        mass=2.18 * 1.30,       # Nặng thêm 30% (2.83 kg)
        izz=0.0125 * 1.30,
        mu_static=0.28,         # Sàn gạch ướt trơn trượt
        mu_kinetic=0.20,
        lidar_num_rays=12,
        lidar_max_range=1.8,
        action_delay_steps=1,
        integrator="rk4",
    )
    env_ood = AndinoRLNavigationEnv(custom_params=params_ood)
    ood_succ = 0
    for ep in range(4):
        obs = env_ood.reset(scenario_idx=ep)
        policy.reset_state()
        done = False
        step_idx = 0
        while not done:
            dither = bang_bang_dither * (1.0 if (step_idx % 2 == 0) else -1.0)
            act = policy.act(obs, high_freq_dither=dither)
            # Giữa đường (bước 25..30), tác dụng một xung lực đẩy ngang/xoay (External Disturbance)
            ext_tau = 0.04 if (25 <= step_idx <= 30) else 0.0
            obs, _, done, info = env_ood.step(act, ext_torque=ext_tau)
            step_idx += 1
            if done and info["reached_goal"]:
                ood_succ += 1

    ood_rate = ood_succ / 4.0
    ood_ok = ood_rate >= 0.75
    results["Test6_OOD_Slippery_Heavy"] = ood_ok
    print(f"  [Test 6 - Stress-Test Ngoài Phân phối (Sàn ướt μ=0.28 + Quá tải +30% + Ngoại lực xô đẩy)]:")
    print(f"    • Tỉ lệ tới đích an toàn = {ood_rate*100:5.1f}% (Ngưỡng >= 75%) "
          f"-> {'✅ PASS' if ood_ok else '❌ FAIL (Trượt bánh đâm vật cản)'}")

    # -------------------------------------------------------------------------
    # TEST 7: KIỂM ĐỊNH SUY GIẢM CẢM BIẾN (NHIỄU LIDAR 3cm + MẤT 10% TIA PHẢN XẠ)
    # -------------------------------------------------------------------------
    params_sensor = PhysicalParams(
        lidar_num_rays=12,
        lidar_max_range=1.8,
        lidar_noise_std=0.03,      # Nhiễu 3 cm (gấp 3 lần bình thường)
        lidar_dropout_prob=0.10,   # Rớt 10% tia laser
        action_delay_steps=1,
        integrator="rk4",
    )
    env_sens = AndinoRLNavigationEnv(custom_params=params_sensor)
    _, sens_rate, _ = train_mod.evaluate_policy(
        policy, env_sens, num_episodes=4, bang_bang_dither=bang_bang_dither
    )
    sensor_ok = sens_rate >= 0.75
    results["Test7_Sensor_Corruption"] = sensor_ok
    print(f"  [Test 7 - Stress-Test Suy giảm Cảm biến (Nhiễu LiDAR 3cm + Mất 10% tia)]:")
    print(f"    • Tỉ lệ tới đích an toàn = {sens_rate*100:5.1f}% (Ngưỡng >= 75%) "
          f"-> {'✅ PASS' if sensor_ok else '❌ FAIL (Nhạy cảm với nhiễu cảm biến)'}")

    return results


def main() -> None:
    print("=" * 86)
    print("🛡️ TẦNG 4: BỘ 7 BÀI KIỂM ĐỊNH THỰC TẾ VẬT LÝ BẰNG CODE (SIM-TO-REAL REALITY AUDIT)")
    print("=" * 86)

    # 1. Kiểm định bản thân Simulator trước
    sim_pass = test_1_simulator_thermodynamic_passivity()

    # 2. Tải trọng số 2 Policy đã huấn luyện ở Tầng 3 (hoặc khởi tạo nếu chưa chạy file 03)
    weights_path = os.path.join(CURRENT_DIR, "trained_policies.json")
    if os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        naive_pol = PolicyNetwork(
            data["naive_toy_policy"]["weights"],
            action_filter_beta=data["naive_toy_policy"]["action_filter_beta"],
        )
        naive_dither = data["naive_toy_policy"].get("bang_bang_dither", 0.55)

        robust_pol = PolicyNetwork(
            data["sim2real_robust_policy"]["weights"],
            action_filter_beta=data["sim2real_robust_policy"]["action_filter_beta"],
        )
        robust_dither = data["sim2real_robust_policy"].get("bang_bang_dither", 0.0)
    else:
        naive_pol = PolicyNetwork(build_warmstart_weights(True), action_filter_beta=1.0)
        naive_dither = 0.55
        robust_pol = PolicyNetwork(build_warmstart_weights(False), action_filter_beta=0.35)
        robust_dither = 0.0

    # 3. Chạy kiểm định cho cả 2 Policy
    res_naive = run_policy_audit("Policy A: Naive_Toy_Policy (Train trên Toy Sim)", naive_pol, naive_dither)
    res_robust = run_policy_audit("Policy B: Sim2Real_Robust_Policy (Train với Physics + DR)", robust_pol, robust_dither)

    # 4. Tổng kết Bảng So sánh Quyết định Triển khai (Deployment Gatekeeper Summary)
    print("\n" + "=" * 86)
    print("📊 BẢNG TỔNG KẾT PHÊ DUYỆT TRIỂN KHAI LÊN ROBOT THẬT (SIM-TO-REAL GATEKEEPER)")
    print("=" * 86)
    print(f"  • Test 1 (Tính đúng đắn Nhiệt động lực học của Simulator) : {'✅ PASS' if sim_pass else '❌ FAIL'}")
    for key in res_naive:
        s_n = "✅ PASS" if res_naive[key] else "❌ FAIL"
        s_r = "✅ PASS" if res_robust[key] else "❌ FAIL"
        print(f"  • {key:<32} : Naive Toy = {s_n}  |  Sim2Real Robust = {s_r}")

    naive_all = sim_pass and all(res_naive.values())
    robust_all = sim_pass and all(res_robust.values())
    print("-" * 86)
    print(f"  👉 QUYẾT ĐỊNH CHO 'Naive_Toy_Policy'       : {'🟢 CHO PHÉP NHÚNG' if naive_all else '🔴 CẤM NHÚNG XUỐNG ROBOT THẬT (Nguy cơ tai nạn/hỏng phần cứng)'}")
    print(f"  👉 QUYẾT ĐỊNH CHO 'Sim2Real_Robust_Policy' : {'🟢 ĐẠT CHUẨN 7/7 - SẴN SÀNG NHÚNG XUỐNG ROS 2 & ROBOT THẬT!' if robust_all else '🔴 CẦN TINH CHỈNH THÊM'}")
    print("=" * 86)


if __name__ == "__main__":
    main()
