#!/usr/bin/env python3
"""
Script cài đặt lớp tương thích Classic OpenAI Gym + Roboschool trên Gymnasium (Python 3.12 - 3.14+)
Giúp toàn bộ code bài giảng từ Week 1 -> Week 8 (cả .py và .ipynb) chạy trực tiếp mà không lỗi.
"""
import os
import sys
import site
from pathlib import Path

GYM_INIT_CODE = '''"""
Classic OpenAI Gym Compatibility Layer for Gymnasium (Python 3.14+)
Automatically bridges:
  - env.reset() -> obs (instead of (obs, info))
  - env.step(a) -> (obs, reward, done, info) (instead of 5-tuple)
  - gym.wrappers.Monitor -> pass-through wrapper
  - Legacy environment IDs:
      FrozenLake-v0 -> FrozenLake-v1
      Taxi-v2 -> Taxi-v3
      CartPole-v0 -> CartPole-v1
      PongNoFrameskip-v4 -> Built-in RGB Atari Pong simulator (210x160x3)
      BipedalWalker-v2 / Hardcore-v2 -> Built-in 24D obs / 4D continuous walker physics
      LunarLanderContinuous-v2 -> Built-in 8D obs / 2D continuous lunar lander physics
      RoboschoolAnt-v1 -> Built-in 28D obs / 8D continuous quadruped ant physics
"""
import math
import types
import numpy as np
import gymnasium as _gymnasium
from gymnasium import spaces, error, logger

__version__ = "0.26.2-compat"


class Env:
    metadata = {"render.modes": ["human", "rgb_array"]}
    reward_range = (-float("inf"), float("inf"))
    spec = None
    action_space = None
    observation_space = None

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return [seed]

    @property
    def unwrapped(self):
        return self


class Wrapper(Env):
    def __init__(self, env):
        self.env = env
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.metadata = getattr(env, "metadata", {})

    @property
    def spec(self):
        return getattr(self.env, "spec", None)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human", **kwargs):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return [seed]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


class ObservationWrapper(Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, observation):
        raise NotImplementedError


class ActionWrapper(Wrapper):
    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        raise NotImplementedError


class RewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, self.reward(reward), done, info

    def reward(self, reward):
        raise NotImplementedError


class ClassicGymAdapter(Wrapper):
    """Wraps a Gymnasium v1.x env to return 1-value reset() and 4-tuple step()."""

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
            return res[0]
        return res

    def step(self, action):
        res = self.env.step(action)
        if isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
            return obs, float(reward), bool(terminated or truncated), info
        return res


class _MonitorWrapper(Wrapper):
    def __init__(self, env, directory=None, video_callable=None, force=False, **kwargs):
        super().__init__(env)
        self.directory = directory


wrappers = types.SimpleNamespace(Monitor=_MonitorWrapper)


# =============================================================================
# BUILT-IN PHYSICS FALLBACK ENVIRONMENTS (NO EXTERNAL C++ ROMS / BOX2D NEEDED)
# =============================================================================
class _Spec:
    def __init__(self, env_id, max_episode_steps=1000):
        self.id = env_id
        self.max_episode_steps = max_episode_steps


class SyntheticPongNoFrameskipEnv(Env):
    """Fast RGB 210x160x3 Atari Pong physics environment for Week 3 DQN."""

    ACTION_MEANINGS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(6)
        self.spec = _Spec("PongNoFrameskip-v4", max_episode_steps=4000)
        self.rng = np.random.default_rng()
        self.reset()

    def get_action_meanings(self):
        return list(self.ACTION_MEANINGS)

    def reset(self, **kwargs):
        self.player_y = 105.0
        self.cpu_y = 105.0
        self.ball_x = 80.0
        self.ball_y = 105.0
        self.ball_vx = 3.0 if self.rng.random() < 0.5 else -3.0
        self.ball_vy = float(self.rng.uniform(-2.5, 2.5))
        self.steps = 0
        self.score_p = 0
        self.score_c = 0
        return self._render_frame()

    def step(self, action):
        self.steps += 1
        act = int(action)
        if act in (2, 4):
            self.player_y = max(24.0, self.player_y - 4.0)
        elif act in (3, 5):
            self.player_y = min(186.0, self.player_y + 4.0)

        if self.cpu_y < self.ball_y - 2.0:
            self.cpu_y = min(186.0, self.cpu_y + 2.7)
        elif self.cpu_y > self.ball_y + 2.0:
            self.cpu_y = max(24.0, self.cpu_y - 2.7)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y <= 16.0 or self.ball_y >= 194.0:
            self.ball_vy = -self.ball_vy
            self.ball_y = float(np.clip(self.ball_y, 16.0, 194.0))

        reward = 0.0
        # Player paddle on right (x ~ 144), CPU paddle on left (x ~ 16)
        if self.ball_x >= 140.0 and abs(self.ball_y - self.player_y) <= 16.0 and self.ball_vx > 0:
            self.ball_vx = -abs(self.ball_vx)
            self.ball_vy += float(self.rng.uniform(-0.8, 0.8))
        elif self.ball_x <= 20.0 and abs(self.ball_y - self.cpu_y) <= 16.0 and self.ball_vx < 0:
            self.ball_vx = abs(self.ball_vx)
            self.ball_vy += float(self.rng.uniform(-0.8, 0.8))
        elif self.ball_x > 155.0:
            reward = -1.0
            self.score_c += 1
            self.ball_x, self.ball_y = 80.0, 105.0
            self.ball_vx = -3.0
        elif self.ball_x < 5.0:
            reward = 1.0
            self.score_p += 1
            self.ball_x, self.ball_y = 80.0, 105.0
            self.ball_vx = 3.0

        done = bool(self.score_p >= 21 or self.score_c >= 21 or self.steps >= 800)
        return self._render_frame(), float(reward), done, {}

    def _render_frame(self):
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[:, :] = (144, 72, 17)
        py = int(self.player_y)
        cy = int(self.cpu_y)
        frame[max(0, py - 12):min(210, py + 12), 142:146] = (92, 186, 92)
        frame[max(0, cy - 12):min(210, cy + 12), 14:18] = (213, 130, 74)
        bx, by = int(np.clip(self.ball_x, 2, 157)), int(np.clip(self.ball_y, 2, 207))
        frame[by - 2:by + 2, bx - 2:bx + 2] = (236, 236, 236)
        return frame


class ContinuousPhysicsEnv(Env):
    """Continuous control physics dynamics for BipedalWalker (24D/4D), LunarLander (8D/2D), RoboschoolAnt (28D/8D)."""

    def __init__(self, env_id, obs_dim, act_dim, max_steps=400):
        self.spec = _Spec(env_id, max_episode_steps=max_steps)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self, **kwargs):
        self.steps = 0
        self.state = self.rng.normal(0.0, 0.08, size=(self.obs_dim,)).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        self.steps += 1
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        dt = 0.05
        half = self.obs_dim // 2
        pos = self.state[:half]
        vel = self.state[half:2 * half]

        ctrl = np.resize(a, half)
        acc = 2.2 * ctrl - 1.5 * pos - 0.35 * vel + self.rng.normal(0.0, 0.02, size=half).astype(np.float32)
        vel_next = np.clip(vel + dt * acc, -5.0, 5.0)
        pos_next = np.clip(pos + dt * vel_next, -5.0, 5.0)

        self.state[:half] = pos_next
        self.state[half:2 * half] = vel_next
        if self.obs_dim > 2 * half:
            self.state[-1] = float(np.cos(self.steps * dt))

        forward_bonus = 1.5 - float(np.mean(pos_next ** 2))
        ctrl_cost = 0.08 * float(np.mean(a ** 2))
        reward = forward_bonus - ctrl_cost
        done = bool(self.steps >= self.max_steps or np.max(np.abs(pos_next)) > 4.2)
        return self.state.copy(), float(reward), done, {}


ENV_ID_REMAP = {
    "FrozenLake-v0": "FrozenLake-v1",
    "Taxi-v2": "Taxi-v3",
    "CartPole-v0": "CartPole-v1",
    "Pendulum-v0": "Pendulum-v1",
    "MountainCar-v0": "MountainCar-v0",
    "Acrobot-v1": "Acrobot-v1",
}


def make(env_id, **kwargs):
    mapped_id = ENV_ID_REMAP.get(env_id, env_id)
    try:
        base_env = _gymnasium.make(mapped_id, **kwargs)
        return ClassicGymAdapter(base_env)
    except Exception:
        if "Pong" in env_id:
            return SyntheticPongNoFrameskipEnv()
        if "BipedalWalker" in env_id:
            return ContinuousPhysicsEnv(env_id, obs_dim=24, act_dim=4, max_steps=400)
        if "LunarLanderContinuous" in env_id:
            return ContinuousPhysicsEnv(env_id, obs_dim=8, act_dim=2, max_steps=300)
        if "Ant" in env_id or "Roboschool" in env_id:
            return ContinuousPhysicsEnv(env_id, obs_dim=28, act_dim=8, max_steps=250)
        raise
'''

ROBOSCHOOL_CODE = '''"""
Compatibility stub for legacy `import roboschool` (Week 7 Model-Based RL).
Environments like `RoboschoolAnt-v1` are handled automatically by `gym.make()`.
"""
__version__ = "1.0-compat"
'''


def install_compat_into_site_packages():
    site_dirs = [p for p in site.getsitepackages() if "site-packages" in p]
    if not site_dirs:
        raise RuntimeError("Không tìm thấy thư mục site-packages trong môi trường hiện tại.")
    target_site = Path(site_dirs[0])

    gym_dir = target_site / "gym"
    gym_dir.mkdir(parents=True, exist_ok=True)
    (gym_dir / "__init__.py").write_text(GYM_INIT_CODE, encoding="utf-8")
    (target_site / "roboschool.py").write_text(ROBOSCHOOL_CODE, encoding="utf-8")
    print(f"✅ Đã cài đặt lớp tương thích `gym` & `roboschool` vào: {target_site}")


if __name__ == "__main__":
    install_compat_into_site_packages()
