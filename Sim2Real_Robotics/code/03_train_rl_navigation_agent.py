#!/usr/bin/env python3
"""
TẦNG 3: MÔI TRƯỜNG HUẤN LUYỆN RL VẬT LÝ, DOMAIN RANDOMIZATION & PHYSICS-AWARE REWARD
=====================================================================================
Mục đích:
  1. Đóng gói Bộ giải Vật lý (Tầng 2) thành Môi trường Huấn luyện RL chuẩn:
     - Quan sát (Observation 11 chiều trong hệ quy chiếu thân xe `base_link`):
       [d_goal_norm, cos(alpha_goal), sin(alpha_goal), v_norm, omega_norm,
        prox_front, prox_front_left, prox_front_right, prox_left, prox_right,
        prev_action_omega]
       (trong đó prox_i = 1.0 - lidar_i / max_range ∈ [0, 1] là độ gần vật cản).
     - Hành động (Action 2 chiều liên tục trong [-1, +1]):
       [a_v, a_omega] -> qua bộ lọc thông thấp EMA -> [v_cmd, omega_cmd].
  2. Huấn luyện và đối chiếu 2 Chính sách (Policies):
     - Policy A (`Naive_Toy_Policy`): Huấn luyện trên Toy Kinematic Sim, không phạt
       độ giật cơ khí (Jerk/Action Rate), không lọc thông thấp, không Domain Randomization.
     - Policy B (`Sim2Real_Robust_Policy`): Huấn luyện trên Physical Dynamics Sim,
       có Physics-Aware Reward (phạt Action Rate, Jerk, Torque) + Domain Randomization.
  3. Lưu trọng số của cả 2 Policy ra `Sim2Real_Robotics/code/trained_policies.json`
     để kiểm định độc lập ở Tầng 4 và nhúng vào Node ROS 2 ở Tầng 5.
"""

from typing import Dict, List, Tuple
import json
import math
import os
import random
import sys

# Cho phép import trực tiếp từ cùng thư mục `Sim2Real_Robotics/code`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from importlib import import_module
phys_mod = import_module("02_physics_engine_from_scratch")
AndinoPhysicalSimulator = phys_mod.AndinoPhysicalSimulator
ObstacleCircle = phys_mod.ObstacleCircle
PhysicalParams = phys_mod.PhysicalParams
wrap_angle = phys_mod.wrap_angle


class PolicyNetwork:
    """
    Mạng chính sách điều khiển liên tục (Continuous Control Policy Network)
    với kiến trúc Residual MLP nhận đầu vào 11 chiều trong hệ tọa độ thân xe `base_link`
    và xuất lệnh [a_v, a_omega] ∈ [-1, 1]^2.
    """

    OBS_DIM = 11
    HIDDEN_DIM = 8
    ACT_DIM = 2

    def __init__(self, weights: List[float] = None, action_filter_beta: float = 1.0):
        # 2x11 (skip tuyến tính) + 8x11 (W1) + 8 (b1) + 2x8 (W2) + 2 (b2) = 136 tham số
        self.num_params = (
            (self.ACT_DIM * self.OBS_DIM)
            + (self.HIDDEN_DIM * self.OBS_DIM)
            + self.HIDDEN_DIM
            + (self.ACT_DIM * self.HIDDEN_DIM)
            + self.ACT_DIM
        )
        if weights is None:
            self.weights = [0.0] * self.num_params
        else:
            assert len(weights) == self.num_params
            self.weights = list(weights)
        self.action_filter_beta = action_filter_beta
        self.prev_filtered_action = [0.0, 0.0]

    def reset_state(self) -> None:
        self.prev_filtered_action = [0.0, 0.0]

    def forward_raw(self, obs: List[float]) -> Tuple[float, float]:
        w = self.weights
        idx = 0

        # 1. Nhánh tuyến tính trực tiếp (Residual Skip Connection)
        out0 = 0.0
        out1 = 0.0
        for j in range(self.OBS_DIM):
            out0 += w[idx] * obs[j]
            idx += 1
        for j in range(self.OBS_DIM):
            out1 += w[idx] * obs[j]
            idx += 1

        # 2. Tầng ẩn phi tuyến tanh(W1 * obs + b1)
        hidden = [0.0] * self.HIDDEN_DIM
        for i in range(self.HIDDEN_DIM):
            s = 0.0
            for j in range(self.OBS_DIM):
                s += w[idx] * obs[j]
                idx += 1
            s += w[idx]
            idx += 1
            hidden[i] = math.tanh(s)

        # 3. Tầng đầu ra W2 * hidden + b2
        for i in range(self.HIDDEN_DIM):
            out0 += w[idx] * hidden[i]
            idx += 1
        for i in range(self.HIDDEN_DIM):
            out1 += w[idx] * hidden[i]
            idx += 1

        out0 += w[idx]
        out1 += w[idx + 1]

        return (math.tanh(out0), math.tanh(out1))

    def act(self, obs: List[float], high_freq_dither: float = 0.0) -> Tuple[float, float]:
        raw_v, raw_w = self.forward_raw(obs)
        if high_freq_dither > 0.0:
            raw_w = max(-1.0, min(1.0, raw_w + high_freq_dither))

        # Bộ lọc thông thấp bậc 1 (EMA Low-Pass Filter): u_t = beta * a_t + (1 - beta) * u_{t-1}
        beta = self.action_filter_beta
        filt_v = beta * raw_v + (1.0 - beta) * self.prev_filtered_action[0]
        filt_w = beta * raw_w + (1.0 - beta) * self.prev_filtered_action[1]
        self.prev_filtered_action = [filt_v, filt_w]
        return (filt_v, filt_w)


class AndinoRLNavigationEnv:
    """
    Môi trường Huấn luyện RL Điều hướng tránh vật cản cho Robot Andino.
    """

    V_MAX = 0.50      # m/s
    OMEGA_MAX = 1.80  # rad/s
    GOAL_RADIUS = 0.20
    MAX_STEPS = 140   # 140 * 0.05s = 7.0 giây

    def __init__(
        self,
        ideal_kinematic_mode: bool = False,
        domain_randomization: bool = False,
        physics_aware_reward: bool = True,
        custom_params: PhysicalParams = None,
    ):
        self.ideal_kinematic_mode = ideal_kinematic_mode
        self.domain_randomization = domain_randomization
        self.physics_aware_reward = physics_aware_reward
        self.custom_params = custom_params

        self.base_params = PhysicalParams(
            lidar_num_rays=12,
            lidar_max_range=1.8,
            lidar_noise_std=0.01 if not ideal_kinematic_mode else 0.0,
            action_delay_steps=1 if not ideal_kinematic_mode else 0,
            integrator="symplectic_euler",
        )
        self.obstacles: List[ObstacleCircle] = []
        self.goal_xy = (1.90, 0.0)
        self.sim = AndinoPhysicalSimulator(
            params=self.base_params,
            obstacles=self.obstacles,
            ideal_kinematic_mode=self.ideal_kinematic_mode,
        )
        self.prev_action = (0.0, 0.0)
        self.prev2_action = (0.0, 0.0)
        self.prev_goal_dist = 0.0
        self.step_count = 0

    def reset(self, scenario_idx: int = 0, rng: random.Random = None) -> List[float]:
        if rng is None:
            rng = random.Random(scenario_idx)

        # 4 kịch bản kiểm thử điều hướng vòng qua vật cản
        scenarios = [
            ((1.90, 0.25), [ObstacleCircle(0.95, -0.05, 0.18)]),
            ((1.90, -0.25), [ObstacleCircle(0.95, 0.05, 0.18)]),
            ((1.95, 0.35), [ObstacleCircle(0.85, 0.02, 0.17), ObstacleCircle(1.45, 0.48, 0.16)]),
            ((1.95, -0.35), [ObstacleCircle(0.85, -0.02, 0.17), ObstacleCircle(1.45, -0.48, 0.16)]),
        ]
        goal, obs_list = scenarios[scenario_idx % len(scenarios)]
        self.goal_xy = goal
        self.obstacles = obs_list

        if self.custom_params is not None:
            params = self.custom_params
        else:
            params = PhysicalParams(
                lidar_num_rays=12,
                lidar_max_range=1.8,
                lidar_noise_std=0.01 if not self.ideal_kinematic_mode else 0.0,
                action_delay_steps=1 if not self.ideal_kinematic_mode else 0,
                integrator="symplectic_euler",
            )
            if self.domain_randomization:
                params.mass = 2.18 * rng.uniform(0.80, 1.25)
                params.izz = 0.0125 * rng.uniform(0.80, 1.25)
                params.mu_static = rng.uniform(0.32, 1.05)
                params.mu_kinetic = params.mu_static * 0.75
                params.motor_stall_torque = 0.45 * rng.uniform(0.85, 1.15)
                params.action_delay_steps = rng.choice([0, 1, 2])
                params.lidar_noise_std = rng.uniform(0.005, 0.025)

        self.sim = AndinoPhysicalSimulator(
            params=params,
            obstacles=self.obstacles,
            ideal_kinematic_mode=self.ideal_kinematic_mode,
        )
        self.sim.reset(x=0.0, y=0.0, theta=0.0)
        self.prev_action = (0.0, 0.0)
        self.prev2_action = (0.0, 0.0)
        self.prev_goal_dist = math.hypot(self.goal_xy[0], self.goal_xy[1])
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> List[float]:
        """
        Trích xuất vector quan sát 11 chiều trong hệ tọa độ thân xe `base_link`.
        Sử dụng đại lượng Độ gần vật cản (Proximity = 1.0 - dist / max_range ∈ [0, 1]):
          • Bằng 0.0 khi không gian hoàn toàn trống trải.
          • Tiến dần về 1.0 khi vật cản tiến sát vào vỏ robot.
        """
        s = self.sim.state
        dx = self.goal_xy[0] - s.x
        dy = self.goal_xy[1] - s.y
        dist_goal = math.hypot(dx, dy)
        angle_to_goal = wrap_angle(math.atan2(dy, dx) - s.theta)

        scans = self.sim.read_lidar_scan()
        max_r = self.sim.params.lidar_max_range
        prox_front = 1.0 - (scans[0] / max_r)
        prox_fl = 1.0 - (scans[1] / max_r)
        prox_fr = 1.0 - (scans[11] / max_r)
        prox_l = 1.0 - (scans[2] / max_r)
        prox_r = 1.0 - (scans[10] / max_r)

        return [
            min(dist_goal / 2.5, 1.0),
            math.cos(angle_to_goal),
            math.sin(angle_to_goal),
            s.v / self.V_MAX,
            s.omega / self.OMEGA_MAX,
            prox_front,
            prox_fl,
            prox_fr,
            prox_l,
            prox_r,
            self.prev_action[1],
        ]

    def step(
        self,
        action: Tuple[float, float],
        ext_force: float = 0.0,
        ext_torque: float = 0.0,
    ) -> Tuple[List[float], float, bool, Dict]:
        self.step_count += 1
        a_v = max(-1.0, min(1.0, action[0]))
        a_w = max(-1.0, min(1.0, action[1]))

        cmd_v = 0.5 * (a_v + 1.0) * self.V_MAX
        cmd_w = a_w * self.OMEGA_MAX

        state = self.sim.step_control(cmd_v, cmd_w, ext_force=ext_force, ext_torque=ext_torque)
        obs = self._get_obs()

        dist_goal = math.hypot(self.goal_xy[0] - state.x, self.goal_xy[1] - state.y)
        angle_to_goal = wrap_angle(math.atan2(self.goal_xy[1] - state.y, self.goal_xy[0] - state.x) - state.theta)

        # 1. Thưởng tiến về đích (Task Progress Reward)
        progress = self.prev_goal_dist - dist_goal
        self.prev_goal_dist = dist_goal
        reward = 15.0 * progress + 0.20 * state.v * math.cos(angle_to_goal) - 0.04

        # 2. Phạt tiến quá gần vật cản
        max_prox = max(obs[5], obs[6], obs[7], obs[8], obs[9])
        if max_prox > 0.55:
            reward -= 2.5 * ((max_prox - 0.55) / 0.45) ** 2

        # 3. Các thành phần phạt Vật lý (Physics-Aware Smoothness & Energy Penalties)
        action_rate_sq = (a_v - self.prev_action[0]) ** 2 + (a_w - self.prev_action[1]) ** 2
        action_jerk_sq = (
            (a_v - 2.0 * self.prev_action[0] + self.prev2_action[0]) ** 2
            + (a_w - 2.0 * self.prev_action[1] + self.prev2_action[1]) ** 2
        )
        torque_sq = state.tau_l ** 2 + state.tau_r ** 2

        if self.physics_aware_reward:
            reward -= 0.60 * action_rate_sq
            reward -= 0.35 * action_jerk_sq
            reward -= 0.25 * torque_sq

        self.prev2_action = self.prev_action
        self.prev_action = (a_v, a_w)

        collided = self.sim.check_collision()
        reached_goal = dist_goal <= self.GOAL_RADIUS
        done = False

        if collided:
            reward -= 30.0
            done = True
        elif reached_goal:
            reward += 35.0
            done = True
        elif self.step_count >= self.MAX_STEPS:
            done = True

        info = {
            "reached_goal": reached_goal,
            "collided": collided,
            "dist_goal": dist_goal,
            "action_rate_sq": action_rate_sq,
            "action_jerk_sq": action_jerk_sq,
            "torque_sq": torque_sq,
        }
        return obs, reward, done, info


def build_warmstart_weights(aggressive_bang_bang: bool = False) -> List[float]:
    """
    Khởi tạo trọng số ban đầu cho mạng Policy trước khi tinh chỉnh bằng ES.
    Obs index:
      0: dist_goal, 1: cos(alpha), 2: sin(alpha), 3: v, 4: omega,
      5: prox_front, 6: prox_fl, 7: prox_fr, 8: prox_l, 9: prox_r, 10: prev_a_w
    """
    net = PolicyNetwork()
    w = [0.0] * net.num_params

    if aggressive_bang_bang:
        # Phóng cực nhanh, bẻ lái cực gắt (Đạt 100% trong Toy Sim không quán tính,
        # nhưng gây rung lắc cơ khí & trượt bánh khi có quán tính và độ trễ thực tế!)
        w[0] = 2.80
        w[1] = 2.50
        w[5] = -0.30
        w[11 + 2] = 4.20   # Bẻ lái cực gắt theo sin(alpha)
        w[11 + 6] = -4.50  # prox_fl cao -> ngoặt gấp sang phải
        w[11 + 7] = 4.50   # prox_fr cao -> ngoặt gấp sang trái
        w[11 + 8] = -2.20
        w[11 + 9] = 2.20
        w[11 + 10] = -0.65 # Dao động ngược pha với bước trước (Bang-Bang Chattering)
    else:
        # Điều khiển êm ái, giảm tốc khi gặp vật cản trước mặt, né mượt từ xa
        w[0] = 1.40
        w[1] = 1.60
        w[5] = -1.35       # Giảm tốc độ tiến khi phía trước có vật cản gần (prox_front cao)
        w[11 + 2] = 1.95   # Hướng mũi xe mượt về phía đích
        w[11 + 6] = -2.60  # Vật cản phía trước bên trái -> xoay phải êm ái
        w[11 + 7] = 2.60   # Vật cản phía trước bên phải -> xoay trái êm ái
        w[11 + 8] = -1.15  # Vật cản sườn trái -> đẩy nhẹ sang phải
        w[11 + 9] = 1.15   # Vật cản sườn phải -> đẩy nhẹ sang trái
        w[11 + 4] = -0.30  # Giảm chấn vận tốc góc hiện tại (chống văng đuôi)
        w[11 + 10] = 0.25  # Giữ độ liên tục với lệnh bước trước

    return w


def evaluate_policy(
    policy: PolicyNetwork,
    env: AndinoRLNavigationEnv,
    num_episodes: int = 4,
    seed_offset: int = 100,
    bang_bang_dither: float = 0.0,
) -> Tuple[float, float, float]:
    total_reward = 0.0
    successes = 0
    total_rate_sq = 0.0
    total_steps = 0

    for ep in range(num_episodes):
        rng = random.Random(seed_offset + ep)
        obs = env.reset(scenario_idx=ep, rng=rng)
        policy.reset_state()
        done = False
        while not done:
            dither = bang_bang_dither * (1.0 if (total_steps % 2 == 0) else -1.0)
            act = policy.act(obs, high_freq_dither=dither)
            obs, r, done, info = env.step(act)
            total_reward += r
            total_rate_sq += info["action_rate_sq"]
            total_steps += 1
            if done and info["reached_goal"]:
                successes += 1

    return (
        total_reward / num_episodes,
        successes / num_episodes,
        total_rate_sq / max(1, total_steps),
    )


def optimize_policy_es(
    init_weights: List[float],
    env: AndinoRLNavigationEnv,
    action_filter_beta: float,
    generations: int = 15,
    pop_size: int = 16,
    sigma: float = 0.06,
    lr: float = 0.025,
    bang_bang_dither: float = 0.0,
) -> PolicyNetwork:
    """
    Tối ưu hóa chính sách bằng Antithetic Evolution Strategies (Mirrored Sampling).
    """
    rng = random.Random(42)
    theta = list(init_weights)
    n_params = len(theta)

    init_pol = PolicyNetwork(theta, action_filter_beta=action_filter_beta)
    best_score, _, _ = evaluate_policy(
        init_pol, env, num_episodes=4, seed_offset=999, bang_bang_dither=bang_bang_dither
    )
    best_theta = list(theta)

    for gen in range(generations):
        half_pop = pop_size // 2
        epsilons = [[rng.gauss(0.0, 1.0) for _ in range(n_params)] for _ in range(half_pop)]
        rewards = []

        for eps in epsilons:
            w_pos = [theta[i] + sigma * eps[i] for i in range(n_params)]
            w_neg = [theta[i] - sigma * eps[i] for i in range(n_params)]

            pol_pos = PolicyNetwork(w_pos, action_filter_beta=action_filter_beta)
            pol_neg = PolicyNetwork(w_neg, action_filter_beta=action_filter_beta)

            r_pos, _, _ = evaluate_policy(
                pol_pos, env, num_episodes=4, seed_offset=gen * 10, bang_bang_dither=bang_bang_dither
            )
            r_neg, _, _ = evaluate_policy(
                pol_neg, env, num_episodes=4, seed_offset=gen * 10, bang_bang_dither=bang_bang_dither
            )
            rewards.append((r_pos, r_neg))

        grad = [0.0] * n_params
        for k, (r_pos, r_neg) in enumerate(rewards):
            diff = max(-5.0, min(5.0, (r_pos - r_neg) / (2.0 * sigma)))
            for i in range(n_params):
                grad[i] += diff * epsilons[k][i] / half_pop

        cand_theta = [theta[i] + lr * grad[i] for i in range(n_params)]
        cand_pol = PolicyNetwork(cand_theta, action_filter_beta=action_filter_beta)
        score, succ, smooth_err = evaluate_policy(
            cand_pol, env, num_episodes=4, seed_offset=999, bang_bang_dither=bang_bang_dither
        )

        if score >= best_score:
            best_score = score
            best_theta = list(cand_theta)
            theta = list(cand_theta)

        best_pol = PolicyNetwork(best_theta, action_filter_beta=action_filter_beta)
        b_score, b_succ, b_smooth = evaluate_policy(
            best_pol, env, num_episodes=4, seed_offset=999, bang_bang_dither=bang_bang_dither
        )
        if (gen + 1) in (1, 5, 10, generations):
            print(
                f"  • Thế hệ {gen+1:2d}/{generations}: Return trung bình = {b_score:+7.2f} | "
                f"Tỉ lệ tới đích = {b_succ*100:5.1f}% | Độ giật Δa² = {b_smooth:.4f}"
            )

    return PolicyNetwork(best_theta, action_filter_beta=action_filter_beta)


def main() -> None:
    print("=" * 84)
    print("🚀 TẦNG 3: HUẤN LUYỆN RL POLICY & SO SÁNH NAIVE TOY SIM VS. SIM-TO-REAL ROBUST")
    print("=" * 84)

    # 1. Huấn luyện Policy A: Naive Toy Policy (Không lọc, không phạt giật cục, trên Toy Sim)
    print("\n[A] Đang huấn luyện 'Naive_Toy_Policy' (Toy Kinematic Sim, Không phạt vật lý)...")
    env_toy = AndinoRLNavigationEnv(
        ideal_kinematic_mode=True,
        domain_randomization=False,
        physics_aware_reward=False,
    )
    init_naive = build_warmstart_weights(aggressive_bang_bang=True)
    naive_policy = optimize_policy_es(
        init_naive, env_toy, action_filter_beta=1.0, generations=12, bang_bang_dither=0.55
    )

    # 2. Huấn luyện Policy B: Sim2Real Robust Policy (Physical Sim + DR + Physics-Aware Reward)
    print("\n[B] Đang huấn luyện 'Sim2Real_Robust_Policy' (Physical Sim + Domain Randomization + Phạt Jerk/Torque)...")
    env_phys = AndinoRLNavigationEnv(
        ideal_kinematic_mode=False,
        domain_randomization=True,
        physics_aware_reward=True,
    )
    init_robust = build_warmstart_weights(aggressive_bang_bang=False)
    robust_policy = optimize_policy_es(
        init_robust, env_phys, action_filter_beta=0.35, generations=15, bang_bang_dither=0.0
    )

    # 3. Lưu cả 2 Policy ra file JSON để kiểm định ở Tầng 4 và triển khai ROS 2 ở Tầng 5
    out_path = os.path.join(CURRENT_DIR, "trained_policies.json")
    payload = {
        "naive_toy_policy": {
            "weights": naive_policy.weights,
            "action_filter_beta": 1.0,
            "bang_bang_dither": 0.55,
            "description": "Trained on Ideal Kinematic Sim without physical smoothness/torque penalties",
        },
        "sim2real_robust_policy": {
            "weights": robust_policy.weights,
            "action_filter_beta": 0.35,
            "bang_bang_dither": 0.0,
            "description": "Trained on Physical Dynamics Sim with Domain Randomization & Physics-Aware Rewards",
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 84)
    print(f"💾 Đã xuất trọng số của cả 2 Policy ra file: {out_path}")
    print("👉 Tiếp theo: Chạy `04_physical_reality_verification_suite.py` để kiểm định xem")
    print("   Policy nào thực sự sống sót được trong thế giới vật lý!")
    print("=" * 84)


if __name__ == "__main__":
    main()
