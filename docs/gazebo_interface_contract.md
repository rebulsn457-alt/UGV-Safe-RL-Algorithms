# Gazebo-RL Interface Contract

更新时间：2026-06-11

本文件用于对齐算法组、仿真组和算法-仿真连接组。当前确认接口为：

```text
obs: 30维
action: (v, w)
ROS input: /scan, /odom
ROS output: /cmd_vel
pipeline: scan + odom -> state -> policy -> safety -> cmd_vel -> Gazebo
```

## 1. 状态空间 obs

状态向量长度固定为 30：

- `obs[0:24]`：24 维激光雷达距离。
- `obs[24]`：目标距离归一化值。
- `obs[25]`：目标相对方位角的 `sin`。
- `obs[26]`：目标相对方位角的 `cos`。
- `obs[27]`：当前线速度归一化值。
- `obs[28]`：当前角速度归一化值。
- `obs[29]`：最近障碍距离归一化值，也就是 `min(obs[0:24])`。

激光雷达距离建议统一归一化到 `[0, 1]`：

```text
normalized_lidar = clip(raw_lidar_meter / lidar_range, 0, 1)
```

24 维雷达建议统一成 `[-pi, pi)` 的相对角度顺序，也就是正前方在数组中间附近。`gazebo_ugv_env.py` 会优先根据 `LaserScan.angle_min` 和 `angle_increment` 对 `/scan` 重采样，避免不同雷达驱动的数组起点不一致。

## 2. 动作空间 action

动作是二维连续控制：

```text
action[0] = v = 线速度
action[1] = w = 角速度
```

建议默认边界：

```text
v in [0.0, 2.0] m/s
w in [-1.2, 1.2] rad/s
```

算法输出动作后，先经过 safety shield，再发布到 `/cmd_vel`。

## 3. ROS topic

输入：

- `/scan`：`sensor_msgs/LaserScan`，用于构造 24 维雷达状态和最近障碍距离。
- `/odom`：`nav_msgs/Odometry`，用于构造车辆位置、航向、速度，并计算目标距离/方位。

输出：

- `/cmd_vel`：`geometry_msgs/Twist`，其中 `linear.x = v`，`angular.z = w`。

可选服务：

- `/gazebo/reset_world`：每个 episode reset 时调用。
- `/gazebo/set_model_state`：如果仿真组后续需要随机初始位置和目标点，再加入。

## 4. Gym wrapper 约定

算法只依赖 Gym/Gymnasium 风格接口：

```python
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(action)
```

`info` 必须至少包含：

```python
{
    "cost": 0.0,                 # 安全代价，碰撞为 1，过近障碍为软 cost
    "event": "running",          # running/success/collision/timeout
    "success": False,
    "collision": False,
    "timeout": False,
    "path_length": 0.0,
    "smoothness": 0.0,
    "min_lidar": 1.0,
}
```

PPO 只记录 `cost`；PPO-Lagrangian/CPO 会用 `cost` 参与优化。

## 5. Safety shield

当前仓库提供 `safety_shield.py`：

- 前方雷达距离安全：直接放行动作。
- 进入 warning distance：降低线速度。
- 进入 stop distance：线速度置 0，并向更空的一侧转向。
- shield 的前方扇区按雷达相对角度 `0` 附近计算；左/右避障按 `[-pi/2, 0]` 和 `[0, pi/2]` 的前半平面比较。

默认训练不强制开启 shield，避免 PPO log probability 和实际动作不一致；评估、部署、Gazebo 演示时建议开启：

```bash
python evaluate_policy.py \
  --env-module safe_ugv_env \
  --env-id SafeUGV-v0 \
  --episodes 50 \
  --hidden-dim 128 \
  --use-safety-shield
```

本地 `SafeUGV-v0` 评估推荐轻量阈值：`warning=0.08`、`stop=0.04`。Gazebo 真实 `/scan` 建议先从更保守的归一化阈值 `warning=0.16`、`stop=0.08` 开始，再根据 `lidar_range` 和小车半径调小或调大。

## 6. 当前仓库对应文件

- `algorithms/ppo-baseline/safe_ugv_env.py`：无 ROS 本地训练环境。
- `algorithms/ppo-baseline/gazebo_ugv_env.py`：ROS1/Gazebo wrapper。
- `algorithms/ppo-baseline/gazebo_ugv_env_template.py`：更轻量的接口模板。
- `algorithms/ppo-baseline/safety_shield.py`：policy 到 `/cmd_vel` 之间的安全过滤。
- `algorithms/ppo-baseline/ppo_training.py`：PPO baseline。
- `algorithms/ppo-baseline/ppo_lagrangian_training.py`：PPO-Lagrangian。
- `algorithms/ppo-baseline/evaluate_policy.py`：自动成功率/碰撞率/cost 评估。

## 7. 中期前最低验收

算法组：

- `SafeUGV-v0` 上能跑 PPO 和 PPO-Lagrangian。
- 自动生成 `success_rate`、`collision_rate`、`mean_cost`、`path_length`、`smoothness`。
- 至少提供 1 张训练日志截图和 1 个 `eval_metrics.csv`。

仿真/建模组：

- Gazebo 中小车模型能启动。
- `/scan`、`/odom`、`/cmd_vel` topic 可见。
- 手动发布 `/cmd_vel` 时小车能运动。

接口组：

- 在学长 Gazebo 虚拟机中尝试导入 `gazebo_ugv_env.py`。
- 能完成 `env.reset()` 和一次 `env.step(action)`。
- `step()` 返回 30 维 obs 和包含 `cost/event` 的 info。
