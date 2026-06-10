"""Action safety shield shared by local UGV and Gazebo wrappers.

The shield is deliberately lightweight: it does not replace the policy, it only
clips unsafe velocity commands before they are sent to the environment or
``/cmd_vel``. This matches the project pipeline:

scan + odom -> state -> policy -> safety -> cmd_vel -> Gazebo.
"""

from __future__ import annotations

import numpy as np


class LidarSafetyShield:
    def __init__(
        self,
        lidar_rays=24,
        warning_distance=0.32,
        stop_distance=0.18,
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

        left_mean = float(np.mean(lidar[: self.lidar_rays // 2]))
        right_mean = float(np.mean(lidar[self.lidar_rays // 2 :]))
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
        # In SafeUGVEnv the 24 rays start at -pi relative to heading and wrap
        # around. The front sector is therefore split across both ends.
        count = max(1, int(round(self.lidar_rays * self.front_fov_ratio / 2.0)))
        return np.concatenate([lidar[:count], lidar[-count:]])
