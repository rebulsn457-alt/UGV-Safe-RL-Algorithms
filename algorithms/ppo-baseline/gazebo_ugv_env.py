"""ROS1/Gazebo Gym wrapper for UGV safe path planning.

Expected interface from the project group:

- observation: 30-D = 24-D /scan lidar + 6-D vehicle/goal state
- action: (v, w) = linear velocity + angular velocity
- ROS input: /scan, /odom
- ROS output: /cmd_vel
- algorithm pipeline: scan + odom -> state -> policy -> safety -> cmd_vel -> Gazebo

This module imports ROS packages lazily, so it can live in this repository even
when ROS/Gazebo is not installed on the algorithm VM.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import register
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces
    from gym.envs.registration import register

from safety_shield import LidarSafetyShield


@dataclass
class GazeboUGVConfig:
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    cmd_vel_topic: str = "/cmd_vel"
    reset_world_service: str = "/gazebo/reset_world"
    lidar_rays: int = 24
    lidar_range: float = 6.0
    world_size: float = 18.0
    max_steps: int = 250
    control_dt: float = 0.1
    robot_radius: float = 0.38
    safety_margin: float = 0.45
    goal_radius: float = 0.5
    max_speed: float = 2.0
    max_yaw_rate: float = 1.2
    goal_x: float = 6.0
    goal_y: float = 0.0
    randomize_goal: bool = False
    goal_x_min: float = 6.0
    goal_x_max: float = 6.0
    goal_y_min: float = 0.0
    goal_y_max: float = 0.0
    use_safety_shield: bool = True
    shield_warning_distance: float = 0.16
    shield_stop_distance: float = 0.08


class GazeboUGVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        self.cfg = GazeboUGVConfig(**kwargs)
        self._load_ros()
        self._init_ros_node()

        self.observation_space = spaces.Box(
            low=np.concatenate([np.zeros(self.cfg.lidar_rays), np.array([0, -1, -1, -1, -1, 0])]).astype(np.float32),
            high=np.ones(self.cfg.lidar_rays + 6, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.cfg.max_yaw_rate], dtype=np.float32),
            high=np.array([self.cfg.max_speed, self.cfg.max_yaw_rate], dtype=np.float32),
            dtype=np.float32,
        )
        self.cmd_pub = self.rospy.Publisher(self.cfg.cmd_vel_topic, self.Twist, queue_size=1)
        self.scan_sub = self.rospy.Subscriber(self.cfg.scan_topic, self.LaserScan, self._scan_cb, queue_size=1)
        self.odom_sub = self.rospy.Subscriber(self.cfg.odom_topic, self.Odometry, self._odom_cb, queue_size=1)
        self.reset_world = self.rospy.ServiceProxy(self.cfg.reset_world_service, self.Empty)
        self.shield = (
            LidarSafetyShield(
                lidar_rays=self.cfg.lidar_rays,
                warning_distance=self.cfg.shield_warning_distance,
                stop_distance=self.cfg.shield_stop_distance,
                max_speed=self.cfg.max_speed,
                max_yaw_rate=self.cfg.max_yaw_rate,
            )
            if self.cfg.use_safety_shield
            else None
        )

        self.latest_scan = None
        self.latest_odom = None
        self._rng = np.random.default_rng()
        self.goal = np.array([self.cfg.goal_x, self.cfg.goal_y], dtype=np.float32)
        self.steps = 0
        self.path_length = 0.0
        self.smoothness = 0.0
        self.last_xy = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_goal_dist = 0.0
        self.last_obs = np.ones(self.cfg.lidar_rays + 6, dtype=np.float32)
        self._wait_for_first_messages()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._select_goal(options)
        self._publish_cmd(0.0, 0.0)
        try:
            self.rospy.wait_for_service(self.cfg.reset_world_service, timeout=3.0)
            self.reset_world()
        except Exception as exc:
            self.rospy.logwarn(f"reset_world failed; continuing with current world: {exc}")
        time.sleep(0.3)
        self.steps = 0
        self.path_length = 0.0
        self.smoothness = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        obs = self._build_observation()
        xy, _ = self._pose_xy_yaw()
        self.last_xy = xy
        self.last_goal_dist = self._goal_distance(xy)
        return obs, self._info(cost=0.0, event="reset")

    def step(self, action):
        raw_action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        obs_before = self._build_observation()
        shield_info = {}
        if self.shield is not None:
            safe_action, shield_info = self.shield.filter_action(raw_action, obs_before)
        else:
            safe_action = raw_action

        prev_xy, _ = self._pose_xy_yaw()
        prev_goal_dist = self._goal_distance(prev_xy)
        self._publish_cmd(float(safe_action[0]), float(safe_action[1]))
        time.sleep(self.cfg.control_dt)

        self.steps += 1
        obs = self._build_observation()
        xy, _ = self._pose_xy_yaw()
        move_dist = float(np.linalg.norm(xy - prev_xy))
        self.path_length += move_dist
        self.smoothness += float(np.linalg.norm(safe_action - self.last_action))
        self.last_action = safe_action
        self.last_xy = xy

        goal_dist = self._goal_distance(xy)
        min_lidar_norm = float(np.min(obs[: self.cfg.lidar_rays]))
        min_lidar_m = min_lidar_norm * self.cfg.lidar_range
        success = goal_dist <= self.cfg.goal_radius
        collision = min_lidar_m <= self.cfg.robot_radius
        timeout = self.steps >= self.cfg.max_steps
        near = max(0.0, self.cfg.safety_margin - min_lidar_m)
        cost = 1.0 if collision else min(1.0, near / max(self.cfg.safety_margin, 1e-6))

        progress = prev_goal_dist - goal_dist
        reward = 8.0 * progress - 0.03 - 1.5 * near - 0.05 * float(np.linalg.norm(safe_action - raw_action))
        event = "running"
        terminated = False
        truncated = False
        if success:
            reward += 120.0
            terminated = True
            event = "success"
            self._publish_cmd(0.0, 0.0)
        elif collision:
            reward -= 120.0
            terminated = True
            event = "collision"
            self._publish_cmd(0.0, 0.0)
        elif timeout:
            reward -= 15.0
            truncated = True
            event = "timeout"
            self._publish_cmd(0.0, 0.0)

        self.last_goal_dist = goal_dist
        info = self._info(cost=cost, event=event)
        info.update(shield_info)
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if hasattr(self, "cmd_pub"):
            self._publish_cmd(0.0, 0.0)

    def _load_ros(self):
        try:
            import rospy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan
            from std_srvs.srv import Empty
        except ImportError as exc:
            raise ImportError(
                "GazeboUGVEnv requires ROS1 Python packages. Use SafeUGV-v0 on the algorithm VM, "
                "and run GazeboUGV-v0 on the ROS/Gazebo VM."
            ) from exc
        self.rospy = rospy
        self.Twist = Twist
        self.Odometry = Odometry
        self.LaserScan = LaserScan
        self.Empty = Empty

    def _init_ros_node(self):
        if not self.rospy.core.is_initialized():
            self.rospy.init_node("safe_rl_gazebo_ugv_env", anonymous=True, disable_signals=True)

    def _wait_for_first_messages(self):
        deadline = time.time() + 10.0
        while not self.rospy.is_shutdown() and time.time() < deadline:
            if self.latest_scan is not None and self.latest_odom is not None:
                return
            time.sleep(0.05)
        raise TimeoutError(f"Did not receive both {self.cfg.scan_topic} and {self.cfg.odom_topic} within 10 seconds.")

    def _scan_cb(self, msg):
        self.latest_scan = msg

    def _odom_cb(self, msg):
        self.latest_odom = msg

    def _publish_cmd(self, linear, angular):
        msg = self.Twist()
        msg.linear.x = float(np.clip(linear, 0.0, self.cfg.max_speed))
        msg.angular.z = float(np.clip(angular, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate))
        self.cmd_pub.publish(msg)

    def _select_goal(self, options=None):
        options = options or {}
        if "goal_x" in options and "goal_y" in options:
            self.goal = np.array([float(options["goal_x"]), float(options["goal_y"])], dtype=np.float32)
            return
        if self.cfg.randomize_goal:
            x_min, x_max = sorted((float(self.cfg.goal_x_min), float(self.cfg.goal_x_max)))
            y_min, y_max = sorted((float(self.cfg.goal_y_min), float(self.cfg.goal_y_max)))
            self.goal = np.array(
                [
                    float(self._rng.uniform(x_min, x_max)),
                    float(self._rng.uniform(y_min, y_max)),
                ],
                dtype=np.float32,
            )
            return
        self.goal = np.array([self.cfg.goal_x, self.cfg.goal_y], dtype=np.float32)

    def _build_observation(self):
        lidar = self._downsample_scan()
        xy, yaw = self._pose_xy_yaw()
        goal_vec = self.goal - xy
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_angle = self._wrap_angle(math.atan2(goal_vec[1], goal_vec[0]) - yaw)
        linear_v, yaw_rate = self._twist()
        aux = np.array(
            [
                np.clip(goal_dist / (self.cfg.world_size * math.sqrt(2.0)), 0.0, 1.0),
                math.sin(goal_angle),
                math.cos(goal_angle),
                np.clip(linear_v / max(self.cfg.max_speed, 1e-6), -1.0, 1.0),
                np.clip(yaw_rate / max(self.cfg.max_yaw_rate, 1e-6), -1.0, 1.0),
                float(np.min(lidar)),
            ],
            dtype=np.float32,
        )
        self.last_obs = np.concatenate([lidar, aux]).astype(np.float32)
        return self.last_obs

    def _downsample_scan(self):
        if self.latest_scan is None:
            return np.ones(self.cfg.lidar_rays, dtype=np.float32)
        msg = self.latest_scan
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.cfg.lidar_range, posinf=self.cfg.lidar_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.cfg.lidar_range)
        if ranges.size == 0:
            return np.ones(self.cfg.lidar_rays, dtype=np.float32)
        angle_increment = float(getattr(msg, "angle_increment", 0.0) or 0.0)
        if abs(angle_increment) > 1e-8:
            source_angles = float(getattr(msg, "angle_min", -math.pi)) + np.arange(ranges.size) * angle_increment
            target_angles = np.linspace(-math.pi, math.pi, self.cfg.lidar_rays, endpoint=False)
            lidar = np.full(self.cfg.lidar_rays, self.cfg.lidar_range, dtype=np.float32)
            max_angle_error = max(abs(angle_increment) * 1.5, math.pi / max(self.cfg.lidar_rays, 1))
            for i, angle in enumerate(target_angles):
                diffs = np.abs(self._angle_diff(source_angles, angle))
                nearest = int(np.argmin(diffs))
                if diffs[nearest] <= max_angle_error:
                    lidar[i] = ranges[nearest]
        else:
            bins = np.array_split(ranges, self.cfg.lidar_rays)
            lidar = np.asarray([np.min(chunk) if len(chunk) else self.cfg.lidar_range for chunk in bins], dtype=np.float32)
        return np.clip(lidar / self.cfg.lidar_range, 0.0, 1.0)

    def _pose_xy_yaw(self):
        if self.latest_odom is None:
            return np.zeros(2, dtype=np.float32), 0.0
        pose = self.latest_odom.pose.pose
        xy = np.array([pose.position.x, pose.position.y], dtype=np.float32)
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return xy, float(yaw)

    def _twist(self):
        if self.latest_odom is None:
            return 0.0, 0.0
        twist = self.latest_odom.twist.twist
        return float(twist.linear.x), float(twist.angular.z)

    def _goal_distance(self, xy):
        return float(np.linalg.norm(self.goal - xy))

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
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "min_lidar": float(np.min(self.last_obs[: self.cfg.lidar_rays])),
            "steps": int(self.steps),
        }

    @staticmethod
    def _wrap_angle(angle):
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def _angle_diff(a, b):
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi


try:
    register(id="GazeboUGV-v0", entry_point="gazebo_ugv_env:GazeboUGVEnv", max_episode_steps=250)
except Exception:
    pass
