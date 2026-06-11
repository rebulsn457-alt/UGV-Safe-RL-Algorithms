"""Action safety shield shared by local UGV and Gazebo wrappers.

The shield is deliberately lightweight: it does not replace the policy, it only
clips unsafe velocity commands before they are sent to the environment or
``/cmd_vel``. This matches the project pipeline:

scan + odom -> state -> policy -> safety -> cmd_vel -> Gazebo.
"""

from __future__ import annotations

import numpy as np


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class LidarSafetyShield:
    def __init__(
        self,
        lidar_rays=24,
        warning_distance=0.08,
        stop_distance=0.04,
        max_speed=1.5,
        max_yaw_rate=1.2,
        front_fov_ratio=0.25,
    ):
        self.lidar_rays = int(lidar_rays)
        self.warning_distance = float(warning_distance)
        self.stop_distance = float(stop_distance)
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.front_fov_ratio = float(front_fov_ratio)

    def filter_action(self, action, observation):
        action = np.asarray(action, dtype=np.float32).copy()
        lidar = np.asarray(observation[: self.lidar_rays], dtype=np.float32)
        if lidar.size == 0:
            return action, {"shield_active": False, "shield_reason": "no_lidar"}

        front = self._front_sector(lidar)
        front_min = float(np.min(front))
        if front_min >= self.warning_distance:
            return action, {
                "shield_active": False,
                "shield_reason": "clear",
                "front_min_lidar": front_min,
            }

        left_mean = self._mean_sector(lidar, 0.0, np.pi / 2.0)
        right_mean = self._mean_sector(lidar, -np.pi / 2.0, 0.0)
        # Positive yaw_rate turns left in our env convention. Turn toward the
        # side with more clearance.
        turn_sign = 1.0 if left_mean >= right_mean else -1.0

        if front_min <= self.stop_distance:
            safe_speed = 0.0
            reason = "stop"
        else:
            ratio = (front_min - self.stop_distance) / max(self.warning_distance - self.stop_distance, 1e-6)
            safe_speed = max(0.0, min(float(action[0]), self.max_speed * ratio))
            reason = "slow_down"

        safe_yaw = turn_sign * min(self.max_yaw_rate, max(abs(float(action[1])), 0.45 * self.max_yaw_rate))
        action[0] = safe_speed
        action[1] = safe_yaw
        return action, {
            "shield_active": True,
            "shield_reason": reason,
            "front_min_lidar": front_min,
            "left_lidar_mean": left_mean,
            "right_lidar_mean": right_mean,
        }

    def _front_sector(self, lidar):
        angles = self._ray_angles()
        half_fov = np.pi * self.front_fov_ratio
        mask = np.abs(angles) <= half_fov
        if not np.any(mask):
            mask[np.argmin(np.abs(angles))] = True
        return lidar[mask]

    def _mean_sector(self, lidar, min_angle, max_angle):
        angles = self._ray_angles()
        mask = (angles >= min_angle) & (angles <= max_angle)
        if not np.any(mask):
            return float(np.mean(lidar))
        return float(np.mean(lidar[mask]))

    def _ray_angles(self):
        # SafeUGVEnv and GazeboUGVEnv expose scans in a normalized convention:
        # 24 rays spanning [-pi, pi), so the forward ray is near the center.
        return _wrap_angle(np.linspace(-np.pi, np.pi, self.lidar_rays, endpoint=False))
