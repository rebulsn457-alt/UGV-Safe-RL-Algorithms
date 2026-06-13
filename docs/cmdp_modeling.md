# 无人车安全路径规划 CMDP 建模说明

更新时间：2026-06-13

本文档用于说明算法组如何把无人车安全路径规划任务建模为受约束马尔可夫决策过程（Constrained Markov Decision Process, CMDP），并把当前仓库中的 PPO、PPO-Lagrangian、后续 CPO 与该建模统一起来。

当前项目已经完成 PPO 与 Gazebo 的闭环验证：基于 `/scan`、`/odom`、`/cmd_vel`，在 `GazeboUGV-v0` 中完成 9 个目标点、每点 5 个 episode、共 45 次独立评估，结果为成功率 100%、碰撞率 0%、超时率 0%。CMDP 建模的作用，是把这套已经跑通的 Gazebo 闭环进一步升级为“安全约束强化学习”问题，使 PPO-Lagrangian 和 CPO 能够在同一套环境接口上比较。

## 1. CMDP 建模与算法工作

建模组和仿真组主要负责车辆模型、场景、传感器、ROS topic 和 Gazebo 运行环境；算法组负责把这些接口抽象成强化学习问题。具体到 CMDP，算法组要完成：

- 定义状态空间 `S`：把 `/scan`、`/odom` 和目标信息整理成固定维度观测。
- 定义动作空间 `A`：把策略输出映射成小车线速度和角速度。
- 定义奖励函数 `R`：鼓励到达目标、减少耗时、路径更平滑。
- 定义安全代价 `C`：把碰撞、过近障碍、越界等风险变成可优化的 cost。
- 定义约束上限 `d`：给 PPO-Lagrangian/CPO 一个明确的安全约束目标。
- 提供统一接口：让 PPO、PPO-Lagrangian、CPO 都能通过 `env.step(action)` 读取 `reward` 和 `info["cost"]`。

因此，CMDP 建模不是单纯写理论，而是把“无人车安全路径规划”转成后续算法可训练、可评估、可对比的形式。

## 2. CMDP 形式化定义

当前无人车任务可以写成：

```text
M = (S, A, P, R, C, gamma_R, gamma_C, d)
```

各部分含义如下：

| 符号 | 项目中含义 | 当前实现 |
|---|---|---|
| `S` | 状态空间 | 30 维观测，24 维激光雷达 + 6 维车辆/目标状态 |
| `A` | 动作空间 | 二维连续动作 `(v, w)` |
| `P` | 状态转移 | 本地 `SafeUGV-v0` 或真实 Gazebo/ROS 仿真环境 |
| `R` | 任务奖励 | 到达目标、接近目标、时间惩罚、平滑惩罚、碰撞惩罚 |
| `C` | 安全代价 | 碰撞 cost、近障碍 soft cost |
| `gamma_R` | reward 折扣因子 | PPO/PPO-Lagrangian 默认 `0.99` |
| `gamma_C` | cost 折扣因子 | PPO-Lagrangian 默认 `0.99` |
| `d` | 安全约束上限 | PPO-Lagrangian 中的 `cost_limit` |

优化目标为：

```text
maximize    E_pi[sum_t gamma_R^t R(s_t, a_t)]
subject to  E_pi[sum_t gamma_C^t C(s_t, a_t)] <= d
```

通俗解释：策略不仅要学会“到达目标”，还要满足“安全代价不能超过阈值”。

## 3. 状态空间 S

状态向量长度固定为 30，与学长给出的 Gazebo 接口一致：

```text
obs: 30维 = 24维激光雷达 + 6维车辆/目标状态
```

具体定义如下：

| 索引 | 含义 | 来源 | 说明 |
|---:|---|---|---|
| `obs[0:24]` | 24 维激光雷达距离 | `/scan` | 归一化到 `[0, 1]`，越小表示障碍越近 |
| `obs[24]` | 目标距离 | `/odom` + 目标点 | 按世界尺度归一化 |
| `obs[25]` | 目标相对角 `sin` | `/odom` + 目标点 | 避免角度跳变 |
| `obs[26]` | 目标相对角 `cos` | `/odom` + 目标点 | 避免角度跳变 |
| `obs[27]` | 当前线速度 | `/odom` | 按最大线速度归一化 |
| `obs[28]` | 当前角速度 | `/odom` | 按最大角速度归一化 |
| `obs[29]` | 最近障碍距离 | `/scan` | `min(obs[0:24])` |

当前对应代码：

```text
algorithms/ppo-baseline/safe_ugv_env.py
algorithms/ppo-baseline/gazebo_ugv_env.py
```

本地环境和 Gazebo 环境保持同一语义，区别只是状态来源不同：本地环境由简化几何仿真生成，Gazebo 环境由真实 ROS topic 生成。

## 4. 动作空间 A

动作是二维连续控制：

```text
action = (v, w)
v = linear velocity
w = angular velocity
```

在 ROS/Gazebo 中发布为：

```text
/cmd_vel:
  linear.x  = v
  angular.z = w
```

当前 Gazebo 联调使用的较稳动作范围为：

| 动作 | 含义 | 当前建议范围 |
|---|---|---:|
| `v` | 线速度 | `[0.0, 0.25]` |
| `w` | 角速度 | `[-0.60, 0.60]` |

说明：仓库默认值中保留了更宽的速度范围，但在学长 Gazebo 实机联调时，小车和场景尺度较小，使用 `max_speed=0.25`、`max_yaw_rate=0.60` 更稳定。

## 5. 状态转移 P

CMDP 中的 `P(s_{t+1} | s_t, a_t)` 不需要算法组手写公式，当前由环境负责：

- 本地训练时：`SafeUGV-v0` 根据简化车辆运动学和障碍物几何关系产生下一状态。
- Gazebo 联调时：Gazebo 物理仿真、车辆模型、传感器插件和 ROS topic 共同产生下一状态。

算法侧只依赖标准 Gym/Gymnasium 接口：

```python
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(action)
cost = info.get("cost", 0.0)
```

这就是后续 PPO、PPO-Lagrangian、CPO 能共用同一个 wrapper 的原因。

## 6. 奖励函数 R

当前 reward 主要表达“任务完成能力”，不是安全约束本身。

在 `safe_ugv_env.py` 和 `gazebo_ugv_env.py` 中，reward 由以下部分组成：

| 项 | 作用 | 当前含义 |
|---|---|---|
| 目标进度奖励 | 鼓励向目标前进 | `prev_goal_dist - goal_dist` |
| 时间惩罚 | 鼓励更快完成 | 每步给小负值 |
| 近障碍惩罚 | 让策略远离障碍 | 离障碍过近时扣 reward |
| 动作修正惩罚 | 减少 safety shield 介入 | shield 修改动作越多，扣分越多 |
| 成功奖励 | 鼓励到达目标 | 到达目标给大正奖励 |
| 碰撞惩罚 | 避免危险终止 | 碰撞给大负奖励 |
| 超时惩罚 | 避免原地不动 | 超时给负奖励 |

Gazebo wrapper 中的核心形式为：

```text
progress = prev_goal_dist - goal_dist
near = max(0, safety_margin - min_lidar_m)

reward =
  8.0 * progress
  - 0.03
  - 1.5 * near
  - 0.05 * ||safe_action - raw_action||
```

事件奖励：

```text
success:   reward += 120
collision: reward -= 120
timeout:   reward -= 15
```

## 7. 安全代价 C

CMDP 的关键在于 cost 单独定义，不能只靠 reward 惩罚来“间接安全”。

当前安全代价统一放在：

```python
info["cost"]
```

Gazebo wrapper 中的定义为：

```text
min_lidar_m = min(obs[0:24]) * lidar_range
collision = min_lidar_m <= robot_radius
near = max(0, safety_margin - min_lidar_m)

if collision:
    cost = 1.0
else:
    cost = min(1.0, near / safety_margin)
```

含义如下：

| 情况 | cost |
|---|---:|
| 碰撞 | `1.0` |
| 未碰撞但进入安全边界 | `(safety_margin - min_lidar_m) / safety_margin` |
| 远离障碍 | `0.0` |

这个定义的好处是：

- PPO baseline 可以记录 cost，用于评估安全性。
- PPO-Lagrangian 可以用 cost critic 学习安全风险。
- CPO 可以直接把 cost 作为约束函数，不需要重新改环境。

## 8. Safety Shield 与 CMDP 的关系

Safety shield 不是 CMDP 本身，但它是部署和演示阶段的安全保护层。

当前链路为：

```text
scan + odom -> state -> policy -> safety shield -> cmd_vel -> Gazebo
```

在算法含义上：

- `policy` 输出原始动作 `raw_action`。
- `safety_shield.py` 根据激光雷达最近距离修正危险动作。
- Gazebo 执行安全修正后的 `safe_action`。
- reward 中记录 action 被修正的惩罚。
- info 中记录 cost、collision、success、timeout 等指标。

中期汇报时可以这样说：

```text
训练阶段使用 CMDP cost 约束描述安全风险；部署/演示阶段叠加轻量 safety shield，保证 Gazebo 实机联调时不会因策略探索或泛化误差造成频繁碰撞。
```

## 9. PPO、PPO-Lagrangian、CPO 的对应关系

| 算法 | 是否使用 reward | 是否使用 cost | 当前状态 | 说明 |
|---|---:|---:|---|---|
| PPO | 是 | 只记录，不参与优化 | 已完成 Gazebo 闭环验证 | 当前最稳定 baseline |
| PPO-Lagrangian | 是 | 是 | 本地环境已实现，Gazebo 可复用接口 | 用动态 lambda 惩罚超约束 cost |
| CPO | 是 | 是 | 后续接入 | 显式限制策略更新后的 cost 变化 |

PPO-Lagrangian 当前对应代码：

```text
algorithms/ppo-baseline/ppo_lagrangian_agent.py
algorithms/ppo-baseline/ppo_lagrangian_training.py
```

核心思想：

```text
policy_advantage = reward_advantage - lambda * cost_advantage
lambda = max(0, lambda + lambda_lr * (episode_cost - cost_limit))
```

CPO 后续接入时，不需要重新定义环境，只需要继续使用：

```python
cost = info.get("cost", 0.0)
```

然后在优化器中加入 trust region 和 cost constraint。

## 10. 当前验证证据

### 10.1 本地 SafeUGV-v0 证据

证据目录：

```text
docs/evidence/algorithm_completion_20260611/
```

已有结果说明 PPO-Lagrangian 的 cost 已经进入训练闭环：

| 实验 | 成功率 | 碰撞率 | 平均 reward | 说明 |
|---|---:|---:|---:|---|
| PPO-Lagrangian soft，seed 0，400 episodes 最后一轮 | 0.80 | 0.00 | 126.42 | 训练中评估结果 |
| PPO-Lagrangian soft，seed 0/1/2，400 episodes 平均 | 0.60 | 0.00 | 96.46 | 多随机种子结果 |
| 同一 checkpoint，无 safety shield，100 episodes | 0.64 | 0.14 | 84.79 | 策略本身仍有碰撞 |
| 同一 checkpoint，light shield，100 episodes | 0.69 | 0.00 | 105.66 | 碰撞归零且成功率不下降 |

### 10.2 GazeboUGV-v0 证据

物理机实测环境：

```text
Ubuntu 20.04 + ROS Noetic + Gazebo11
ROS input:  /scan, /odom
ROS output: /cmd_vel
```

最新目标网格评估：

```text
9 个目标点
每个目标点 5 个 episode
共 45 次 Gazebo 独立评估
success_rate = 1.00
collision_rate = 0.00
timeout_rate = 0.00
```

物理机保存路径：

```text
/home/ubuntu/桌面/UGV-Safe-RL-Algorithms_physical_merged_20260613_140915/UGV-Safe-RL-Algorithms/algorithms/ppo-baseline/eval_logs/goal_grid_20260613153420
```

对应中期汇报表述：

```text
算法组已完成 PPO-Gazebo 闭环与小范围目标泛化验证。基于 /scan、/odom、/cmd_vel，选取 9 个目标点组成目标网格，每个目标点测试 5 个 episode，共 45 次独立评估。结果显示 PPO 策略在目标网格内达到 100% 成功率、0% 碰撞率、0% 超时率，说明当前 Gazebo wrapper、模型加载、策略推理、安全过滤和自动评估流程已跑通。
```

## 11. 与建模组、仿真组、算法仿真连接组的接口需求

为了让 CMDP 建模在 Gazebo 中长期稳定使用，需要其他小组继续保证：

| 需求 | 负责方向 | 算法侧用途 |
|---|---|---|
| `/scan` 可用 | 仿真/传感器 | 构造 24 维激光雷达状态、计算最近障碍和 cost |
| `/odom` 可用 | 仿真/车辆模型 | 构造位置、航向、速度和目标相对状态 |
| `/cmd_vel` 可控 | 仿真/控制接口 | 执行 PPO/CPO 输出的 `(v, w)` |
| 车辆半径 | 建模组 | 判断 collision 和 safety margin |
| 最大线速度/角速度 | 建模组/仿真组 | 设置 action bounds |
| 激光雷达量程 | 仿真组 | 状态归一化、safety shield 阈值 |
| reset 机制 | 仿真连接组 | 多 episode 自动评估和训练 |
| 起点/目标点设置 | 仿真连接组 | 多目标泛化训练和目标网格评估 |

## 12. 下一步推进路线

CMDP 建模当前已经可以作为中期材料的一部分。接下来建议按下面顺序推进：

1. 固定 PPO-Gazebo 证据  
   保留 `goal_grid_20260613153420`、成功模型和运行命令，作为 baseline 完成证据。

2. 在同一 CMDP 接口上训练 Gazebo PPO-Lagrangian  
   使用 `ppo_lagrangian_training.py`，环境换成 `gazebo_ugv_env`，让 `info["cost"]` 参与优化。

3. 对 PPO 与 PPO-Lagrangian 做同场景对比  
   比较 `success_rate`、`collision_rate`、`mean_cost`、`path_length`、`smoothness`。

4. 接入 CPO  
   CPO 不需要重新定义状态/动作/reward/cost，只需要在优化器层面实现约束策略更新。

5. 扩大泛化范围  
   从当前 9 目标点小范围网格，扩展到随机目标点、随机起点、不同障碍布局。

## 13. 当前 CMDP 工作完成度判断

目前可以认定完成：

- CMDP 状态空间定义。
- CMDP 动作空间定义。
- reward 与 cost 分离。
- Gazebo wrapper 中 cost 接口落地。
- PPO baseline 在 Gazebo 中跑通并形成证据。
- PPO-Lagrangian 本地版本已实现，可直接复用 cost 接口。
- CPO 接入所需的环境侧接口已经预留。

尚未完全完成：

- Gazebo 中 PPO-Lagrangian 的正式训练结果。
- CPO 算法本体复现。
- 大范围随机起点、随机目标、随机障碍的系统泛化评估。

因此，中期审核中建议定位为：

```text
算法组已完成无人车路径规划任务的 CMDP 建模，并基于该建模实现了 PPO baseline、PPO-Lagrangian 安全约束接口和 Gazebo 闭环验证。当前 PPO-Gazebo 已在 9 目标点、45 次评估中达到 100% 成功率、0% 碰撞率，下一阶段将在同一 CMDP 接口上推进 PPO-Lagrangian 与 CPO 的安全约束对比实验。
```

## 14. 参考来源

- Eitan Altman, *Constrained Markov Decision Processes*, 1999.
- Joshua Achiam et al., "Constrained Policy Optimization", ICML 2017.
- John Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
- 参考 CMDP 建模、30 维状态、`/scan` `/odom` `/cmd_vel` 数据流与 Gazebo 桥接思路；本仓库代码为本项目算法组独立整理、复现、调参和联调后的实现。
