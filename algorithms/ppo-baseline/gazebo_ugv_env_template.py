"""Template for the future ROS/Gazebo UGV Gym wrapper.

This file is not imported by default and does not require ROS. It documents the
algorithm-side contract that the simulation/interface group should implement on
the senior's Gazebo VM:

- observation: 30-D vector = 24 lidar rays + 6 auxiliary states
- action: [linear_velocity, angular_velocity]
- info["cost"]: safety cost used by PPO-Lagrangian/CPO
- info["event"]: one of success/collision/timeout/running

When ROS is available, replace the TODO blocks with rospy/rclpy publishers and
subscribers for /scan, /odom, /cmd_vel, /gazebo/reset_world, and
/gazebo/set_model_state.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class GazeboUGVEnvTemplate(gym.Env):
    def __init__(
        self,
        max_speed=2.0,
        max_yaw_rate=1.2,
        lidar_rays=24,
        lidar_range=6.0,
        max_steps=250,
        control_dt=0.1,
        goal_radius=0.5,
        robot_radius=0.38,
    ):
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.max_steps = max_steps
        self.control_dt = control_dt
        self.goal_radius = goal_radius
        self.robot_radius = robot_radius
        self.steps = 0

        self.observation_space = spaces.Box(
            low=np.concatenate([np.zeros(lidar_rays), np.array([0, -1, -1, -1, -1, 0])]).astype(np.float32),
            high=np.ones(lidar_rays + 6, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -max_yaw_rate], dtype=np.float32),
            high=np.array([max_speed, max_yaw_rate], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        # TODO(interface group): call /gazebo/reset_world and set robot/goal states.
        observation = self._build_observation()
        return observation, {"event": "reset", "cost": 0.0}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        linear_velocity, angular_velocity = float(action[0]), float(action[1])

        # TODO(interface group): publish geometry_msgs/Twist to /cmd_vel.
        # TODO(interface group): wait control_dt, then read latest /scan and /odom.
        self.steps += 1

        observation = self._build_observation()
        success = self._is_success()
        collision = self._is_collision()
        timeout = self.steps >= self.max_steps
        cost = 1.0 if collision else self._near_obstacle_cost()
        reward = self._compute_reward(success, collision, timeout)

        terminated = bool(success or collision)
        truncated = bool(timeout and not terminated)
        event = "success" if success else "collision" if collision else "timeout" if timeout else "running"
        info = {
            "cost": float(cost),
            "event": event,
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "path_length": 0.0,  # TODO: accumulate odom-based path length.
            "smoothness": 0.0,  # TODO: accumulate action deltas.
            "min_lidar": 0.0,  # TODO: min downsampled lidar distance / lidar_range.
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
        return observation, reward, terminated, truncated, info

    def _build_observation(self):
        # TODO(interface group): downsample /scan to 24 normalized rays.
        lidar = np.ones(self.lidar_rays, dtype=np.float32)
        # TODO(interface group): fill distance-to-goal, bearing sin/cos,
        # normalized linear/yaw velocity, and min lidar.
        aux = np.array([1.0, 0.0, 1.0, -1.0, 0.0, float(np.min(lidar))], dtype=np.float32)
        return np.concatenate([lidar, aux]).astype(np.float32)

    def _is_success(self):
        # TODO(interface group): true when robot-goal distance <= goal_radius.
        return False

    def _is_collision(self):
        # TODO(interface group): true when bumper/contact state or min lidar is
        # smaller than robot_radius.
        return False

    def _near_obstacle_cost(self):
        # TODO(interface group): soft cost when min lidar is inside safety margin.
        return 0.0

    def _compute_reward(self, success, collision, timeout):
        # TODO(algorithm group): keep consistent with SafeUGVEnv reward shaping.
        if success:
            return 120.0
        if collision:
            return -120.0
        if timeout:
            return -15.0
        return -0.03
