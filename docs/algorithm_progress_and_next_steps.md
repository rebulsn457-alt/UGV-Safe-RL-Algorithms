# 算法组当前进展与下一步

更新时间：2026-06-13

## 1. 当前已完成

算法仓库已经从单一 `Pendulum-v1` PPO baseline，推进到无人车安全路径规划的最小算法闭环：

- 本地无人车环境：`SafeUGV-v0`
  - 30 维状态：24 维雷达 + 6 维车辆/目标状态。
  - 2 维动作：`(v, w)`，线速度 + 角速度。
  - 自动返回 `success`、`collision`、`timeout`、`cost`、`path_length`、`smoothness`。
- PPO baseline：
  - 支持 UGV 30 维状态和 2 维连续动作。
  - 自动输出成功率、碰撞率、安全代价等指标。
- PPO-Lagrangian：
  - 增加 reward critic、cost critic 和动态 lambda。
  - 安全代价已经进入优化闭环。
- Safety shield：
  - 对应接口图中的 `policy -> safety -> cmd_vel`。
  - 可在评估/部署阶段用雷达距离过滤危险动作。
- Gazebo wrapper：
  - `GazeboUGV-v0` 已按 `/scan`、`/odom`、`/cmd_vel` 写好 ROS1/Gazebo 接口。
  - 算法 VM 没有 ROS 时用 `SafeUGV-v0` 训练；学长 Gazebo VM 有 ROS 后再测试 wrapper。
- 实验脚本：
  - `run_safe_ugv_experiments.py` 支持 PPO/PPO-Lagrangian、多 seed、多车辆配置实验。

## 2. 当前验证结果

本地 CPU 验证，`SafeUGV-v0`，base 配置，seed 0：

| 算法 | episodes | success_rate | collision_rate | mean_cost | eval_mean_reward | 说明 |
|---|---:|---:|---:|---:|---:|---|
| PPO | 300 | 0.80 | 0.00 | 3.12 | 125.21 | 到达能力更强，能作为 baseline 展示 |
| PPO-Lagrangian | 300 | 0.40 | 0.00 | 1.70 | 67.77 | 安全代价更低，体现安全约束方向 |

结论：当前代码已经能自动产生中期需要的核心指标，不再只是看 reward 曲线。

本地 CPU 三随机种子验证，`SafeUGV-v0`，base 配置，seed 0/1/2，300 episodes：

| 算法 | seeds | 平均成功率 | 平均碰撞率 | 平均超时率 | 平均 cost | 平均 eval reward | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| PPO | 3 | 0.6333 | 0.0000 | 0.3667 | 7.9833 | 98.9162 | 当前到达能力更好，适合作为 baseline 主展示 |
| PPO-Lagrangian | 3 | 0.3667 | 0.0333 | 0.6000 | 3.3135 | 58.6752 | 安全代价更低，但到达率仍需继续调参 |

2026-06-11 虚拟机 Ubuntu CPU 复验，`SafeUGV-v0`，PPO-Lagrangian soft 约束，seed 0，400 episodes：

| 设置 | episodes/eval | success_rate | collision_rate | timeout_rate | mean_cost | mean_reward | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| 训练中最后一轮评估 | 400 / 10 | 0.80 | 0.00 | 0.20 | 18.77 | 126.42 | `cost_limit=8.0`、`lambda_lr=0.005`、`entropy_coef=0.02` |
| 独立评估，不开 shield | 100 | 0.64 | 0.14 | 0.22 | 6.53 | 84.79 | 策略本身已能到达，但仍有碰撞 |
| 独立评估，修正后 light shield | 100 | 0.69 | 0.00 | 0.31 | 9.68 | 105.66 | `warning=0.08`、`stop=0.04`，碰撞归零且到达率不下降 |

同一 soft 约束默认参数，base 配置，seed 0/1/2，400 episodes 汇总：

| 算法 | seeds | 平均成功率 | 平均碰撞率 | 平均超时率 | 平均 cost | 平均 eval reward | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| PPO-Lagrangian soft | 3 | 0.6000 | 0.0000 | 0.4000 | 12.0716 | 96.4551 | 相比旧硬约束，到达率明显提升，碰撞保持为 0 |

中期建议解释：PPO 目前可作为稳定 baseline；PPO-Lagrangian 已经证明 cost 能进入优化闭环。旧硬约束参数容易学成“安全但不前进”，当前推荐使用 soft 约束参数，并在部署/演示评估时打开修正后的轻量 shield。

2026-06-13 Gazebo 实机联调结果，`GazeboUGV-v0`，Ubuntu 20.04 + ROS Noetic + Gazebo11：

| 评估项目 | goal_count | episodes_per_goal | total_episodes | mean_success_rate | mean_collision_rate | mean_timeout_rate | goal_pass_rate | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PPO-Gazebo 目标网格评估 | 9 | 5 | 45 | 1.00 | 0.00 | 0.00 | 1.00 | `goal_x=1.8/2.0/2.2`，`goal_y=-0.3/0.0/0.3` 全部通过 |

该结果对应物理机目录 `eval_logs/goal_grid_20260613153420`，摘要已整理到 `docs/evidence/gazebo_ppo_20260613/eval_logs/goal_grid_20260613153420/`。当前可证明 PPO 已完成 Gazebo 闭环、topic 交互、模型保存加载、独立评估和小范围多目标泛化验证。下一阶段要提升到更接近毕设最终水准，需要继续做 Gazebo PPO-Lagrangian/CPO 对比，并扩大初始位姿、目标点和障碍布局随机化。

## 3. 目标

优先级从高到低：

1. 固定并备份 `goal_grid_20260613153420`、成功模型和运行命令，作为中期 PPO-Gazebo baseline 证据。
2. 在同一 Gazebo/CMDP 接口上训练 PPO-Lagrangian，并对比 `success_rate`、`collision_rate`、`mean_cost`。
3. 若时间足够，扩大目标网格到更大的 `goal_x/goal_y` 范围，并加入随机起点。
4. 后续接入 CPO，复用 `info["cost"]`，不重新改环境接口。

## 4. 推荐运行命令

快速确认环境：

```bash
git pull
cd algorithms/ppo-baseline
python -c "import torch; print(torch.__version__)"
```

基础实验：

```bash
python run_safe_ugv_experiments.py \
  --algorithms ppo,ppolag \
  --configs base \
  --seeds 0,1,2 \
  --episodes 600
```

泛化实验：

```bash
python run_safe_ugv_experiments.py \
  --algorithms ppo,ppolag \
  --configs base,small_slow,fast_wide,dense \
  --seeds 0,1,2 \
  --episodes 600
```

安全屏蔽评估：

```bash
python evaluate_policy.py \
  --env-module safe_ugv_env \
  --env-id SafeUGV-v0 \
  --episodes 50 \
  --hidden-dim 128 \
  --use-safety-shield
```

如果在 Gazebo 中使用真实 `/scan`，建议先用更保守阈值：

```bash
python evaluate_policy.py \
  --env-module gazebo_ugv_env \
  --env-id GazeboUGV-v0 \
  --episodes 50 \
  --hidden-dim 128 \
  --use-safety-shield \
  --shield-warning-distance 0.16 \
  --shield-stop-distance 0.08
```

## 5. 中期汇报

- “算法组已完成 PPO baseline 从连续控制基准任务到无人车 Gazebo 路径规划任务的迁移。”
- “当前环境采用 30 维状态和 `(v,w)` 连续动作，已与 `/scan`、`/odom`、`/cmd_vel` 形成闭环。”
- “已完成 9 个目标点、45 次 Gazebo 独立评估，PPO 策略达到 100% 成功率、0% 碰撞率和 0% 超时率。”
- “已完成 CMDP 建模，统一了 reward 与 safety cost 接口，为 PPO-Lagrangian/CPO 对比提供基础。”



当前更稳妥的定位是：算法侧已经完成 PPO-Gazebo 闭环验证、小范围目标泛化评估、CMDP 建模和 PPO-Lagrangian/CPO 接口准备；下一步进入安全约束算法在 Gazebo 中的系统对比。
