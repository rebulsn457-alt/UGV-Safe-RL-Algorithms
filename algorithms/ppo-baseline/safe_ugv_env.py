"""Lightweight safe UGV path-planning environment.

This environment mirrors the interface expected from the later Gazebo wrapper:
30-D observation = 24 lidar rays + 6 auxiliary values, 2-D continuous action =
linear velocity and yaw rate, and safety cost exposed through ``info["cost"]``.
It is intentionally simple enough to train on CPU before ROS/Gazebo is ready.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import register
except ImportError:  # pragma: no cover - compatibility with old gym stacks
    import gym
    from gym import spaces
    from gym.envs.registration import register


@dataclass
class UGVConfig:
    world_size: float = 10.0
    max_steps: int = 250
    dt: float = 0.1
    robot_radius: float = 0.35
    goal_radius: float = 0.45
    max_speed: float = 1.5
    max_yaw_rate: float = 1.2
    lidar_range: float = 4.0
    lidar_rays: int = 24
    obstacle_count: int = 8
    obstacle_radius_min: float = 0.25
    obstacle_radius_max: float = 0.75
    safety_margin: float = 0.25
    goal_bonus: float = 120.0
    collision_penalty: float = 120.0
    timeout_penalty: float = 15.0
    progress_scale: float = 8.0
    time_penalty: float = 0.03
    near_obstacle_penalty: float = 1.5
    smoothness_penalty: float = 0.05
    seed: int | None = None


class SafeUGVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        self.cfg = UGVConfig(**kwargs)
        low = np.concatenate(
            [
                np.zeros(self.cfg.lidar_rays, dtype=np.float32),
                np.array([0.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.ones(self.cfg.lidar_rays, dtype=np.float32),
                np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.cfg.max_yaw_rate], dtype=np.float32),
            high=np.array([self.cfg.max_speed, self.cfg.max_yaw_rate], dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(self.cfg.seed)
        self.robot = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles = np.zeros((0, 3), dtype=np.float32)
        self.steps = 0
        self.path_length = 0.0
        self.smoothness = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_goal_dist = 0.0
        self.last_lidar = np.ones(self.cfg.lidar_rays, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.path_length = 0.0
        self.smoothness = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.obstacles = self._sample_obstacles()
        self.robot[:2] = self._sample_free_point()
        self.robot[2] = float(self._rng.uniform(-math.pi, math.pi))
        self.goal = self._sample_free_point(min_dist_from=self.robot[:2], min_dist=5.0)
        self.last_goal_dist = self._goal_distance()
        obs = self._get_obs()
        return obs, self._info(0.0, "reset")

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_pos = self.robot[:2].copy()
        prev_goal_dist = self._goal_distance()

        speed, yaw_rate = float(action[0]), float(action[1])
        self.robot[2] = self._wrap_angle(self.robot[2] + yaw_rate * self.cfg.dt)
        direction = np.array([math.cos(self.robot[2]), math.sin(self.robot[2])], dtype=np.float32)
        self.robot[:2] = self.robot[:2] + direction * speed * self.cfg.dt
        self.steps += 1

        move_dist = float(np.linalg.norm(self.robot[:2] - prev_pos))
        self.path_length += move_dist
        action_delta = float(np.linalg.norm(action - self.last_action))
        self.smoothness += action_delta
        self.last_action = action

        goal_dist = self._goal_distance()
        min_clearance = self._min_clearance()
        collision = self._is_collision(min_clearance)
        success = goal_dist <= self.cfg.goal_radius
        timeout = self.steps >= self.cfg.max_steps

        progress = prev_goal_dist - goal_dist
        near_penalty = max(0.0, self.cfg.safety_margin - min_clearance)
        reward = (
            self.cfg.progress_scale * progress
            - self.cfg.time_penalty
            - self.cfg.near_obstacle_penalty * near_penalty
            - self.cfg.smoothness_penalty * action_delta
        )
        event = "running"
        terminated = False
        truncated = False
        cost = 0.0

        if success:
            reward += self.cfg.goal_bonus
            terminated = True
            event = "success"
        elif collision:
            reward -= self.cfg.collision_penalty
            terminated = True
            event = "collision"
            cost = 1.0
        elif timeout:
            reward -= self.cfg.timeout_penalty
            truncated = True
            event = "timeout"

        if not collision and near_penalty > 0.0:
            cost = min(1.0, near_penalty / max(self.cfg.safety_margin, 1e-6))

        self.last_goal_dist = goal_dist
        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, self._info(cost, event)

    def _sample_obstacles(self):
        obstacles = []
        attempts = 0
        while len(obstacles) < self.cfg.obstacle_count and attempts < 1000:
            attempts += 1
            radius = float(self._rng.uniform(self.cfg.obstacle_radius_min, self.cfg.obstacle_radius_max))
            margin = radius + self.cfg.robot_radius + 0.4
            xy = self._rng.uniform(-self.cfg.world_size / 2 + margin, self.cfg.world_size / 2 - margin, size=2)
            if all(np.linalg.norm(xy - np.array(o[:2])) > radius + o[2] + 0.8 for o in obstacles):
                obstacles.append((float(xy[0]), float(xy[1]), radius))
        return np.asarray(obstacles, dtype=np.float32)

    def _sample_free_point(self, min_dist_from=None, min_dist=0.0):
        for _ in range(2000):
            xy = self._rng.uniform(-self.cfg.world_size / 2 + 0.8, self.cfg.world_size / 2 - 0.8, size=2)
            if min_dist_from is not None and np.linalg.norm(xy - min_dist_from) < min_dist:
                continue
            if self._point_clearance(xy) > self.cfg.robot_radius + self.cfg.safety_margin + 0.2:
                return xy.astype(np.float32)
        # Deterministic fallback avoids rare reset failures in dense maps.
        return np.array([-self.cfg.world_size * 0.35, -self.cfg.world_size * 0.35], dtype=np.float32)

    def _point_clearance(self, xy):
        world_clearance = self.cfg.world_size / 2 - max(abs(float(xy[0])), abs(float(xy[1])))
        if len(self.obstacles) == 0:
            return world_clearance
        dists = np.linalg.norm(self.obstacles[:, :2] - xy, axis=1) - self.obstacles[:, 2]
        return float(min(world_clearance, np.min(dists)))

    def _min_clearance(self):
        return self._point_clearance(self.robot[:2]) - self.cfg.robot_radius

    def _is_collision(self, min_clearance):
        outside = np.any(np.abs(self.robot[:2]) >= self.cfg.world_size / 2 - self.cfg.robot_radius)
        return bool(outside or min_clearance <= 0.0)

    def _goal_distance(self):
        return float(np.linalg.norm(self.goal - self.robot[:2]))

    def _get_obs(self):
        lidar = self._lidar_scan()
        goal_vec = self.goal - self.robot[:2]
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_angle = self._wrap_angle(math.atan2(goal_vec[1], goal_vec[0]) - self.robot[2])
        aux = np.array(
            [
                np.clip(goal_dist / (self.cfg.world_size * math.sqrt(2.0)), 0.0, 1.0),
                math.sin(goal_angle),
                math.cos(goal_angle),
                (self.last_action[0] / max(self.cfg.max_speed, 1e-6)) * 2.0 - 1.0,
                self.last_action[1] / max(self.cfg.max_yaw_rate, 1e-6),
                float(np.min(lidar)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([lidar.astype(np.float32), aux]).astype(np.float32)

    def _lidar_scan(self):
        ray_angles = self.robot[2] + np.linspace(-math.pi, math.pi, self.cfg.lidar_rays, endpoint=False)
        readings = np.full(self.cfg.lidar_rays, self.cfg.lidar_range, dtype=np.float32)
        for i, angle in enumerate(ray_angles):
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            readings[i] = min(readings[i], self._ray_box_distance(direction))
            for obs_x, obs_y, obs_r in self.obstacles:
                readings[i] = min(readings[i], self._ray_circle_distance(direction, np.array([obs_x, obs_y]), obs_r))
        self.last_lidar = np.clip(readings / self.cfg.lidar_range, 0.0, 1.0).astype(np.float32)
        return self.last_lidar

    def _ray_box_distance(self, direction):
        distances = []
        half = self.cfg.world_size / 2 - self.cfg.robot_radius
        for axis in range(2):
            if abs(direction[axis]) < 1e-6:
                continue
            for bound in (-half, half):
                t = (bound - self.robot[axis]) / direction[axis]
                if t > 0:
                    point = self.robot[:2] + direction * t
                    other = 1 - axis
                    if -half <= point[other] <= half:
                        distances.append(float(t))
        return min(distances) if distances else self.cfg.lidar_range

    def _ray_circle_distance(self, direction, center, radius):
        oc = self.robot[:2] - center
        effective_radius = radius + self.cfg.robot_radius
        b = 2.0 * float(np.dot(oc, direction))
        c = float(np.dot(oc, oc) - effective_radius * effective_radius)
        disc = b * b - 4.0 * c
        if disc < 0:
            return self.cfg.lidar_range
        sqrt_disc = math.sqrt(disc)
        candidates = [(-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0]
        hits = [t for t in candidates if t > 0.0]
        return min(hits) if hits else self.cfg.lidar_range

    def _info(self, cost, event):
        return {
            "cost": float(cost),
            "event": event,
            "success": event == "success",
            "collision": event == "collision",
            "timeout": event == "timeout",
            "path_length": float(self.path_length),
            "smoothness": float(self.smoothness),
            "goal_distance": float(self.last_goal_dist),
            "min_lidar": float(np.min(self.last_lidar)),
        }

    @staticmethod
    def _wrap_angle(angle):
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


try:
    register(
        id="SafeUGV-v0",
        entry_point="safe_ugv_env:SafeUGVEnv",
        max_episode_steps=250,
    )
except Exception:
    # Re-importing this module inside VS Code should not fail because the id is
    # already registered.
    pass
