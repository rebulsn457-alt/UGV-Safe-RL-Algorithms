# Gazebo PPO 多目标泛化推进流程

更新时间：2026-06-13

本流程用于把当前“固定目标点 PPO 已跑通”的结果，推进到中期审核更有说服力的“多目标泛化评估”。当前已知结果是：`(2.0, 0.0)`、`(1.8, 0.0)`、`(2.2, 0.0)` 通过，`(1.5, 0.5)` 超时。因此下一步重点不是继续证明单点成功，而是让训练阶段覆盖一组目标点。

## 1. 更新代码

在学长物理机上进入仓库后执行：

```bash
git pull
cd algorithms/ppo-baseline
ls ppo_training.py evaluate_goal_grid.py gazebo_ugv_env.py
```

如果是直接拷贝覆盖仓库文件，也至少确认上述三个文件存在，且旧入口 `ppo._training.py` 不再作为主入口使用。

## 2. 启动 Gazebo 和 ROS 环境

先启动 Gazebo 小车场景，并在算法终端确认：

```bash
source /opt/ros/noetic/setup.bash
source ~/桌面/mdcar/devel/setup.bash
cd ~/桌面/UGV-Safe-RL-Algorithms/algorithms/ppo-baseline
rostopic list | grep -E "/scan|/odom|/cmd_vel"
```

正常应看到：

```text
/scan
/odom
/cmd_vel
```

## 3. 先跑当前模型的网格评估

先不要训练，直接用已有模型验证当前短板：

```bash
python3 evaluate_goal_grid.py \
  --env-module gazebo_ugv_env \
  --env-id GazeboUGV-v0 \
  --model-path models/best_ppo_actor_GazeboUGV-v0_20260613083156.pth \
  --goal-x-values 1.8,2.0,2.2 \
  --goal-y-values -0.3,0.0,0.3 \
  --episodes-per-goal 3 \
  --max-steps 300 \
  --env-kwargs '{"scan_topic":"/scan","odom_topic":"/odom","cmd_vel_topic":"/cmd_vel","lidar_range":6.0,"max_speed":0.25,"max_yaw_rate":0.60,"robot_radius":0.20,"safety_margin":0.35,"shield_warning_distance":0.10,"shield_stop_distance":0.04,"use_safety_shield":true}'
```

输出会保存在：

```text
eval_logs/goal_grid_<时间>/grid_summary.csv
eval_logs/goal_grid_<时间>/grid_summary.json
eval_logs/goal_grid_<时间>/episodes.csv
```

这一步的作用是形成基线：当前模型在哪些目标点能通过，哪些目标点超时。

## 4. 第一阶段：前向目标随机化训练

先只随机 `x`，不随机 `y`，让策略保留已经学会的直线到达能力，同时覆盖近邻距离变化。

```bash
python3 ppo_training.py \
  --env-module gazebo_ugv_env \
  --env-id GazeboUGV-v0 \
  --episodes 300 \
  --max-steps 300 \
  --steps-per-update 1024 \
  --batch-size 64 \
  --eval-interval 20 \
  --eval-episodes 5 \
  --reward-scale 1.0 \
  --entropy-coef 0.01 \
  --target-kl 0.03 \
  --init-model-path models/best_ppo_actor_GazeboUGV-v0_20260613083156.pth \
  --run-name GazeboUGV-v0_goal_x_curriculum_20260613 \
  --env-kwargs '{"scan_topic":"/scan","odom_topic":"/odom","cmd_vel_topic":"/cmd_vel","lidar_range":6.0,"max_speed":0.25,"max_yaw_rate":0.60,"goal_x":2.0,"goal_y":0.0,"randomize_goal":true,"goal_x_min":1.8,"goal_x_max":2.2,"goal_y_min":0.0,"goal_y_max":0.0,"robot_radius":0.20,"safety_margin":0.35,"shield_warning_distance":0.10,"shield_stop_distance":0.04,"use_safety_shield":true}'
```

跑完后记录终端最后的 `best eval`，并保存 `logs/GazeboUGV-v0_goal_x_curriculum_20260613/eval_metrics.csv`。

## 5. 第二阶段：小范围横向目标随机化训练

如果第一阶段网格评估稳定，再把 `y` 加到 `[-0.3, 0.3]`：

```bash
python3 ppo_training.py \
  --env-module gazebo_ugv_env \
  --env-id GazeboUGV-v0 \
  --episodes 500 \
  --max-steps 300 \
  --steps-per-update 1024 \
  --batch-size 64 \
  --eval-interval 20 \
  --eval-episodes 5 \
  --reward-scale 1.0 \
  --entropy-coef 0.015 \
  --target-kl 0.03 \
  --init-model-path models/best_ppo_actor_GazeboUGV-v0_goal_x_curriculum_20260613.pth \
  --run-name GazeboUGV-v0_goal_xy_curriculum_20260613 \
  --env-kwargs '{"scan_topic":"/scan","odom_topic":"/odom","cmd_vel_topic":"/cmd_vel","lidar_range":6.0,"max_speed":0.25,"max_yaw_rate":0.60,"goal_x":2.0,"goal_y":0.0,"randomize_goal":true,"goal_x_min":1.8,"goal_x_max":2.2,"goal_y_min":-0.3,"goal_y_max":0.3,"robot_radius":0.20,"safety_margin":0.35,"shield_warning_distance":0.10,"shield_stop_distance":0.04,"use_safety_shield":true}'
```

如果第二阶段一开始明显退化，可以先把 `goal_y_min/max` 收窄到 `-0.15,0.15`，成功率恢复后再扩大。

## 6. 训练后统一网格评估

每次训练后都用同一套网格评估命令，便于和之前证据对比：

```bash
python3 evaluate_goal_grid.py \
  --env-module gazebo_ugv_env \
  --env-id GazeboUGV-v0 \
  --model-path models/best_ppo_actor_GazeboUGV-v0_goal_xy_curriculum_20260613.pth \
  --goal-x-values 1.8,2.0,2.2 \
  --goal-y-values -0.3,0.0,0.3 \
  --episodes-per-goal 5 \
  --max-steps 300 \
  --env-kwargs '{"scan_topic":"/scan","odom_topic":"/odom","cmd_vel_topic":"/cmd_vel","lidar_range":6.0,"max_speed":0.25,"max_yaw_rate":0.60,"robot_radius":0.20,"safety_margin":0.35,"shield_warning_distance":0.10,"shield_stop_distance":0.04,"use_safety_shield":true}'
```

中期材料优先使用 `grid_summary.csv`，它比单条 reward 曲线更能说明泛化性。

## 7. 阶段验收标准

建议按以下标准判断是否进入下一步：

- 固定目标 `(2.0, 0.0)`：`success_rate >= 0.8`，`collision_rate = 0.0`。
- 前向目标 `(1.8, 0.0)`、`(2.2, 0.0)`：保持通过。
- 横向小范围目标：至少多数网格点 `success_rate >= 0.6`，且 `collision_rate = 0.0`。
- 如果横向点全部 timeout，说明仍需扩大训练随机性或增加训练 episodes，而不是直接切 CPO。

## 8. 再接 PPO-Lagrangian/CPO

当 PPO 的多目标网格评估稳定后，再把同样的 `env_kwargs` 给 `ppo_lagrangian_training.py`。PPO-Lagrangian/CPO 的汇报重点应是：

- 与 PPO 使用同一套 Gazebo wrapper。
- 使用同一套目标网格评估。
- 在成功率接近时，比较 `collision_rate`、`mean_cost` 和 `timeout_rate`。

这样中期汇报逻辑会更完整：先证明 PPO 能到达，再证明安全约束算法能降低安全代价。
