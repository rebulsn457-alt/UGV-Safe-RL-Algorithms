# 算法组当前进展与下一步

更新时间：2026-06-10

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

中期建议解释：PPO 目前更容易学到到达策略；PPO-Lagrangian 已经证明 cost 能进入优化闭环，但需要继续调 `cost_limit`、`lambda_lr`、reward/cost scale 和 shield 策略，使安全性与到达率更平衡。

## 3. 周五前建议目标

优先级从高到低：

1. 跑 600 episodes，seed 0，PPO 和 PPO-Lagrangian 各一组。
2. 跑 600 episodes，seed 0/1/2，形成均值表。
3. 如果时间够，再跑 `small_slow`、`fast_wide`、`dense` 三种配置，展示泛化设计。
4. 在学长 Gazebo 虚拟机中验证 `/scan`、`/odom`、`/cmd_vel` 是否可用。
5. 如果 Gazebo topic 可用，测试 `GazeboUGV-v0.reset()` 和 `GazeboUGV-v0.step(action)`。

## 4. 推荐运行命令

快速确认环境：

```bash
git pull
cd algorithms/ppo-baseline
python -c "import torch; print(torch.__version__)"
```

今晚/明天基础实验：

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

## 5. 中期汇报建议表述

可以说：

- “算法组已完成 PPO baseline 从连续控制基准任务到无人车风格任务的迁移。”
- “当前环境采用 30 维状态和 `(v,w)` 连续动作，与 Gazebo 接口保持一致。”
- “已实现 PPO-Lagrangian，使安全代价参与优化，并自动统计成功率、碰撞率和 cost。”
- “已预留 ROS/Gazebo wrapper，后续在学长 Gazebo 虚拟机中验证 `/scan`、`/odom`、`/cmd_vel` 闭环。”

不要说：

- “CPO 已完全完成。”
- “Gazebo 强化学习训练已经跑通。”
- “学长论文结果就是本项目最终结果。”

当前更稳妥的定位是：算法侧已经完成本地可训练闭环、安全约束算法雏形和 Gazebo 接口准备，下一步进入真实 Gazebo 联调。
