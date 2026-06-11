# 无人车安全路径规划 CMDP 建模说明

更新时间：2026-06-11

本文档用于说明算法组如何把无人车安全路径规划任务建模为受约束马尔可夫决策过程（Constrained Markov Decision Process, CMDP），并说明当前仓库中的 PPO、PPO-Lagrangian、后续 CPO 与该建模之间的对应关系。

## 1.解释 CMDP 算法组工作

普通 MDP 只关心最大化期望回报，容易把“尽快到达目标”和“避免碰撞”混在同一个 reward 里。无人车安全路径规划需要同时表达两件事：

- 任务目标：尽快、平滑地到达目标点。
- 安全约束：限制碰撞、越界、过近障碍等高风险行为。

CMDP 正好把这两类目标拆开：策略仍然最大化任务回报，同时显式约束安全代价。因此，CMDP 建模、cost 定义、PPO-Lagrangian 训练、CPO 接口预留都属于算法组可以完成并展示的工作。

## 2. CMDP 五元组定义

当前项目可写为：

```text
M = (S, A, P, R, C, gamma, d)
```

- `S`：状态空间，当前为 30 维观测。
- `A`：动作空间，当前为二维连续动作 `(v, w)`。
- `P`：状态转移，由本地 `SafeUGV-v0` 或 Gazebo 仿真环境给出。
- `R`：任务奖励，用于鼓励靠近目标、成功到达，并惩罚时间消耗和动作抖动。
- `C`：安全代价，用于刻画碰撞、越界、离障碍过近等风险。
- `gamma`：奖励折扣因子，当前默认 `0.99`。
- `d`：安全约束上限，例如 PPO-Lagrangian 中的 `cost_limit`。

算法目标是：

```text
maximize    E_pi[sum_t gamma^t R(s_t, a_t)]
subject to  E_pi[sum_t gamma^t C(s_t, a_t)] <= d
```

## 3. 状态空间 S

状态向量长度固定为 30：

| 索引 | 含义 | 说明 |
|---:|---|---|
| `obs[0:24]` | 24 维激光雷达 | 归一化到 `[0, 1]`，越小表示障碍越近 |
| `obs[24]` | 目标距离 | 按世界尺度归一化 |
| `obs[25]` | 目标相对角 `sin` | 避免角度跳变 |
| `obs[26]` | 目标相对角 `cos` | 避免角度跳变 |
| `obs[27]` | 当前线速度 | 归一化速度状态 |
| `obs[28]` | 当前角速度 | 归一化角速度状态 |
| `obs[29]` | 最近障碍距离 | `min(obs[0:24])` |

该定义已在 `safe_ugv_env.py` 和 `gazebo_ugv_env.py` 中对齐。Gazebo 侧由 `/scan` 和 `/odom` 构造同语义观测。

## 4. 动作空间 A

动作是二维连续控制：

| 动作 | 含义 | 本地默认范围 | Gazebo 默认范围 |
|---|---|---:|---:|
| `action[0]` | 线速度 `v` | `[0.0, 1.5]` | `[0.0, 2.0]` |
| `action[1]` | 角速度 `w` | `[-1.2, 1.2]` | `[-1.2, 1.2]` |

在 Gazebo 中，动作最终映射为：

```text
linear.x = v
angular.z = w
```

## 5. 奖励 R 与安全代价 C

当前本地环境的 reward 由以下部分组成：

- 目标进度奖励：上一时刻目标距离减去当前目标距离。
- 时间惩罚：鼓励更快完成。
- 近障惩罚：离障碍过近时降低 reward。
- 平滑惩罚：动作变化过大时降低 reward。
- 成功奖励：到达目标点时给正奖励。
- 碰撞惩罚：发生碰撞时给大负奖励。

安全 cost 独立返回在 `info["cost"]`：

- 发生碰撞：`cost = 1.0`。
- 未碰撞但进入安全边界：`cost` 按接近程度从 0 到 1 连续增加。
- 远离障碍：`cost = 0.0`。

这样做的好处是：PPO 可以记录 cost 作为安全指标，PPO-Lagrangian/CPO 可以直接把 cost 纳入约束优化。

## 6. PPO-Lagrangian 与 CPO 的关系

当前仓库已经实现 PPO-Lagrangian：

- reward critic 学习任务回报。
- cost critic 学习安全代价。
- Lagrange multiplier `lambda` 根据 cost 是否超过约束动态调整。
- 优化时使用 `reward_advantage - lambda * cost_advantage` 作为约束后的策略改进方向。

CPO 后续接入时仍使用同一个 CMDP 建模入口：

```python
next_state, reward, terminated, truncated, info = env.step(action)
cost = info.get("cost", 0.0)
```

区别在于：PPO-Lagrangian 用拉格朗日惩罚近似处理约束，CPO 会显式限制策略更新后的代价变化，使更新更保守、更贴近理论约束。

## 7. 当前验证证据

当前证据文件位于：

```text
docs/evidence/algorithm_completion_20260611/
```

关键结果：

| 实验 | 成功率 | 碰撞率 | 平均 reward | 说明 |
|---|---:|---:|---:|---|
| PPO-Lagrangian soft，seed 0，400 episodes 最后一轮 | 0.80 | 0.00 | 126.42 | 训练中评估结果 |
| PPO-Lagrangian soft，seed 0/1/2，400 episodes 平均 | 0.60 | 0.00 | 96.46 | 多随机种子结果 |
| 同一 checkpoint，无 safety shield，100 episodes | 0.64 | 0.14 | 84.79 | 策略本身仍有碰撞 |
| 同一 checkpoint，默认 light shield，100 episodes | 0.69 | 0.00 | 105.66 | 碰撞归零且成功率不下降 |

## 8. 对接建模组与仿真组的需求

为了把当前算法闭环迁移到 Gazebo，需要其他小组保证：

- `/scan` 可用，并提供正确的 `angle_min`、`angle_increment` 和 `ranges`。
- `/odom` 可用，并能读取车辆位置、姿态和速度。
- `/cmd_vel` 可控制小车线速度和角速度。
- reset 机制可用，至少支持 `/gazebo/reset_world`，后续最好支持重置起点和目标点。
- 车辆模型半径、最大速度、最大角速度、激光雷达量程需要提供给算法组，用于设置 action bounds 和 safety shield 阈值。

算法组已经提供 `gazebo_ugv_env.py` 作为 ROS1/Gazebo wrapper。联调时优先验证 `env.reset()`、`env.step(action)` 能否返回 30 维 `obs` 和包含 `cost/event` 的 `info`。

## 9. 参考来源

- Eitan Altman, *Constrained Markov Decision Processes*, 1999.
- Joshua Achiam et al., “Constrained Policy Optimization”, ICML 2017.
- John Schulman et al., “Proximal Policy Optimization Algorithms”, 2017.
- 张津硕学长论文：参考其 CMDP 建模、30 维状态、`/scan` `/odom` `/cmd_vel` 数据流与 Gazebo 桥接思路；本仓库代码为本项目算法组独立整理、复现和调参后的实现。
