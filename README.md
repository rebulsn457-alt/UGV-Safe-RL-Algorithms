# UGV-Safe-RL-Algorithms

无人车安全路径规划强化学习算法库。当前重点是把 PPO baseline 从单一 `Pendulum-v1` 实验改造成可迁移到不同小车与 Gazebo 环境的连续控制训练入口，并为后续 CPO/TRPO 等安全约束算法预留接口。

## 当前 PPO 泛化改造

主要代码：

- `algorithms/ppo-baseline/ppo_agent.py`
- `algorithms/ppo-baseline/ppo_training.py`
- `algorithms/ppo-baseline/safe_ugv_env.py`
- `algorithms/ppo-baseline/safety_shield.py`
- `algorithms/ppo-baseline/gazebo_ugv_env.py`
- `algorithms/ppo-baseline/ppo_lagrangian_agent.py`
- `algorithms/ppo-baseline/ppo_lagrangian_training.py`
- `algorithms/ppo-baseline/evaluate_policy.py`
- `algorithms/ppo-baseline/evaluate_goal_grid.py`
- `algorithms/ppo-baseline/run_safe_ugv_experiments.py`
- `algorithms/ppo-baseline/gazebo_ugv_env_template.py`
- `docs/cmdp_modeling.md`
- `docs/gazebo_interface_contract.md`
- `docs/gazebo_generalization_workflow.md`
- `docs/evidence/algorithm_completion_20260611/`
- `docs/evidence/gazebo_ppo_20260613/`

已完成的泛化优化：

- 自动读取环境 `observation_space` 与 `action_space`，不再硬编码 Pendulum 的状态维度、动作维度和 `[-2, 2]` 动作范围。
- 在归一化动作空间中训练 Gaussian policy，再映射到环境真实连续动作上下界；buffer 保存采样动作本身，避免旧版 `clamp` 后 log probability 不一致导致 PPO ratio 偏移。
- PPO 超参数全部改为命令行配置，包括学习率、`gamma`、GAE lambda、clip、entropy、target KL、batch size、rollout 长度等。
- 按 `steps-per-update` 收集 rollout，而不是按固定 episode 数更新，更适合 Gazebo 中可变 episode 长度。
- 加入 value-function clipping、advantage normalization、target KL 日志和 deterministic evaluation，便于判断策略是否真的学到稳定控制。
- 兼容 `gymnasium` 和旧版 `gym`，便于 Ubuntu + VSCode + Gazebo 工作流。
- 支持 observation normalization 和 reward normalization，适配不同车辆质量、速度量纲、传感器量纲变化。
- 保存训练配置、TensorBoard 日志、reward 曲线、最佳模型权重；如果启用观测归一化，也会保存对应 normalizer 参数。
- 新增 `SafeUGV-v0` 本地无人车路径规划环境：30 维观测、2 维动作、成功/碰撞/超时事件、安全 cost、路径长度和平滑度统计。
- 新增 `GazeboUGV-v0` ROS1/Gazebo wrapper：按 `/scan`、`/odom` 输入和 `/cmd_vel` 输出组织 Gym 接口。
- 新增 safety shield：对应 `scan + odom -> state -> policy -> safety -> cmd_vel -> Gazebo` 中的 safety 层。
- 新增 PPO-Lagrangian 最小实现：reward critic + cost critic + 动态拉格朗日乘子，用于安全约束强化学习中期展示和后续 CPO 对接。
- 新增 CMDP 建模说明：明确状态、动作、奖励、安全代价、约束目标以及与 PPO-Lagrangian/CPO 的关系。
- 新增独立评估脚本，自动输出 success rate、collision rate、timeout rate、mean cost、path length、smoothness 等指标。
- 新增证据包：保存虚拟机训练 CSV、独立评估 JSON、训练日志摘要和结果图表，便于组内复核。

## Ubuntu/VSCode 快速运行

在虚拟机 Ubuntu 项目目录中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd algorithms/ppo-baseline
python -c "import torch; print(torch.__version__)"
python ppo_training.py --env-id Pendulum-v1 --episodes 400
```

本地验证结果：

- `Pendulum-v1`，400 episodes，默认参数：deterministic eval 从约 `-1023` 提升到约 `-86`。
- `Pendulum-v1`，800 episodes，默认参数：最后 20 个训练 episode 均值约 `-146`，deterministic eval 约 `-98`。

训练时的 `train_mean` 是随机采样策略的回报，会比确定性评估低；判断是否收敛优先看 `eval_mean`。

## 本地 UGV 安全路径规划训练

在还没有 ROS/Gazebo 的情况下，算法组先用 `SafeUGV-v0` 做无人车风格任务闭环。该环境接口思路：

- 观测：30 维，24 维模拟激光雷达 + 6 维辅助状态。
- 动作：2 维连续动作，线速度和角速度。
- 安全代价：通过 `info["cost"]` 返回，碰撞为 1，距离障碍过近时返回软 cost。
- 事件：通过 `info["event"]`、`success`、`collision`、`timeout` 自动统计。

快速 smoke test：

```bash
cd algorithms/ppo-baseline
python ppo_training.py \
  --env-module safe_ugv_env \
  --env-id SafeUGV-v0 \
  --episodes 80 \
  --steps-per-update 1024 \
  --eval-interval 10 \
  --eval-episodes 5 \
  --reward-scale 0.02 \
  --normalize-obs \
  --run-name safe_ugv_ppo_smoke
```

更长训练：

```bash
python ppo_training.py \
  --env-module safe_ugv_env \
  --env-id SafeUGV-v0 \
  --episodes 600 \
  --steps-per-update 2048 \
  --batch-size 128 \
  --hidden-dim 128 \
  --eval-interval 20 \
  --eval-episodes 10 \
  --reward-scale 0.02 \
  --normalize-obs \
  --run-name safe_ugv_ppo_600
```

PPO-Lagrangian 安全约束训练：

```bash
python ppo_lagrangian_training.py \
  --episodes 600 \
  --steps-per-update 2048 \
  --batch-size 128 \
  --eval-interval 20 \
  --eval-episodes 10 \
  --run-name safe_ugv_ppolag_600
```

当前默认 PPO-Lagrangian 参数已经按虚拟机 CPU 实测调成偏稳的软约束组合：
`cost_limit=8.0`、`lambda_lr=0.005`、`entropy_coef=0.02`。旧的更硬约束组合容易出现“安全但不前进”的 timeout 策略。

训练过程中打印：

- `success`：自动成功率。
- `collision`：自动碰撞率。
- `cost` / `eval_cost`：安全代价。
- `lambda`：PPO-Lagrangian 的动态拉格朗日乘子。

训练日志保存到 `algorithms/ppo-baseline/logs/<run_name>/`， `eval_metrics.csv` 直接放进中期材料。

Codex 本地 CPU smoke/probe 验证结果：

- PPO，`SafeUGV-v0`，200 episodes：`eval_mean` 从约 `-120` 提升到约 `71`，`success_rate` 达到 `0.60`，`collision_rate` 降到 `0.20`。
- PPO-Lagrangian，`SafeUGV-v0`，200 episodes：后期 `collision_rate` 降到 `0.10`，`mean_cost` 最低约 `0.31`，说明安全代价已经进入优化闭环。
- PPO，`SafeUGV-v0`，300 episodes：最后一轮 `success_rate=0.80`，`collision_rate=0.00`，`eval_mean_reward≈125.21`。
- PPO-Lagrangian，`SafeUGV-v0`，300 episodes：最后一轮 `success_rate=0.40`，`collision_rate=0.00`，`mean_cost≈1.70`。
- 三随机种子，`SafeUGV-v0`，base 配置，300 episodes：PPO 平均 `success_rate=0.6333`、`collision_rate=0.0000`；PPO-Lagrangian 平均 `success_rate=0.3667`、`collision_rate=0.0333`、`mean_cost=3.3135`。
- 虚拟机 Ubuntu CPU，PPO-Lagrangian soft 约束，400 episodes：最后一轮 `eval_mean_reward≈126.42`、`success_rate=0.80`、`collision_rate=0.00`。
- 同一 checkpoint 100 episodes 独立评估：不开 shield 时 `success_rate=0.64`、`collision_rate=0.14`；修正雷达角度后的轻量 shield（`warning=0.08`、`stop=0.04`）为 `success_rate=0.69`、`collision_rate=0.00`。
- 虚拟机 Ubuntu CPU，PPO-Lagrangian soft 约束，base 配置，seed 0/1/2，400 episodes：平均 `success_rate=0.6000`、`collision_rate=0.0000`、`eval_mean_reward≈96.46`。

说明：200-300 episodes 只是代码链路和趋势验证。正式中期材料建议跑 600-1000 episodes，并至少用 3 个随机种子汇总平均值。

## Gazebo 实机联调证据（2026-06-13）

本仓库已加入 2026-06-13 在 Ubuntu 20.04 + ROS Noetic + Gazebo11 实机环境中的 PPO-Gazebo 闭环证据，路径为 `docs/evidence/gazebo_ppo_20260613/`。这组证据包含训练日志、独立评估 JSON/CSV、最佳 actor 权重和汇总表。

训练设置：`GazeboUGV-v0`，观测 30 维，动作 `(v, w)`，ROS topic 为 `/scan`、`/odom`、`/cmd_vel`。PPO 训练 200 episodes 后生成模型 `best_ppo_actor_GazeboUGV-v0_20260613083156.pth`，训练中最佳评估约 `129.06`。

最新目标网格评估使用 `evaluate_goal_grid.py`，目标范围为 `goal_x in {1.8, 2.0, 2.2}`、`goal_y in {-0.3, 0.0, 0.3}`，每个目标点 5 个 episode，共 45 次 Gazebo 独立评估：

| 评估范围 | goal_count | episodes_per_goal | total_episodes | mean_success_rate | mean_collision_rate | mean_timeout_rate | goal_pass_rate | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `x=1.8/2.0/2.2, y=-0.3/0.0/0.3` | 9 | 5 | 45 | 1.00 | 0.00 | 0.00 | 1.00 | 小范围多目标泛化通过 |

历史单点探测中，`(1.5, 0.5)` 曾出现超时，说明早期模型横向泛化不足；随后通过目标网格评估确认当前小范围目标集合已经全部通过。最新证据摘要见 `docs/evidence/gazebo_ppo_20260613/eval_logs/goal_grid_20260613153420/`。

当前结论：PPO 已经完成 Gazebo 闭环、模型保存、模型加载、topic 交互和独立评估；在 9 目标点、45 次评估中达到 `success_rate=1.00`、`collision_rate=0.00`、`timeout_rate=0.00`。下一步要达到更接近毕设最终水准，需要在同一 CMDP cost 接口上推进 Gazebo PPO-Lagrangian/CPO 对比，并继续扩大初始位姿、目标点和障碍布局的随机化范围。

仓库已加入下一阶段泛化工具：

- `gazebo_ugv_env.py` 支持 `randomize_goal`、`goal_x_min/max`、`goal_y_min/max`，训练时可随机目标点。
- `evaluate_goal_grid.py` 支持一次性评估多个 `(goal_x, goal_y)`，输出 `grid_summary.csv/json`，用于中期材料展示泛化性。
- 详细操作流程见 `docs/gazebo_generalization_workflow.md`。

## 自动评估脚本

训练完成后，用确定性策略评估：

```bash
python evaluate_policy.py \
  --env-module safe_ugv_env \
  --env-id SafeUGV-v0 \
  --episodes 50 \
  --hidden-dim 128
```

如果要先看随机策略 baseline：

```bash
python evaluate_policy.py \
  --env-module safe_ugv_env \
  --env-id SafeUGV-v0 \
  --episodes 50 \
  --random-policy
```

评估结果会保存到 `algorithms/ppo-baseline/eval_logs/<time>/summary.json` 和 `episodes.csv`。

部署/演示阶段可以打开安全屏蔽层：

```bash
python evaluate_policy.py \
  --env-module safe_ugv_env \
  --env-id SafeUGV-v0 \
  --episodes 50 \
  --hidden-dim 128 \
  --use-safety-shield
```

安全屏蔽层对应项目接口图中的 `policy -> safety -> cmd_vel`。默认训练不强制开启 shield，因为 PPO 的 log probability 与被 shield 修改后的实际动作会有一定偏差；正式 Gazebo 演示和安全对比评估时建议开启。

`SafeUGV-v0` 本地评估默认使用轻量阈值 `warning=0.08`、`stop=0.04`。如果接到真实 Gazebo `/scan`，建议从更保守的 `warning=0.16`、`stop=0.08` 开始，再根据小车半径和雷达量程微调。

## 多 seed / 多配置实验

中期前至少跑一组基础对照：

```bash
python run_safe_ugv_experiments.py \
  --algorithms ppo,ppolag \
  --configs base \
  --seeds 0,1,2 \
  --episodes 600
```

时间充足，再跑车辆/场景泛化：

```bash
python run_safe_ugv_experiments.py \
  --algorithms ppo,ppolag \
  --configs base,small_slow,fast_wide,dense \
  --seeds 0,1,2 \
  --episodes 600
```

脚本会把每次训练的最后一轮评估指标汇总到 `logs/safe_ugv_experiment_summary_<time>.csv`，方便直接放进中期材料。

如果 UGV/Gazebo 环境 reward 已经在合理范围，例如单步大致 `[-5, 5]`，改成：

```bash
python ppo_training.py --env-id YourGazeboUGVEnv-v0 --reward-scale 1.0 --normalize-obs --normalize-reward
```

## 对接其他小车/Gazebo 的接口要求

环境需要提供 Gym/Gymnasium 风格接口：

- `reset()` 返回一维状态向量，或 `(state, info)`。
- `step(action)` 返回 `(next_state, reward, terminated, truncated, info)`，旧 Gym 的 `(next_state, reward, done, info)` 也兼容。
- `observation_space.shape` 必须是一维，例如 `(state_dim,)`。
- `action_space` 必须是连续 `Box`，并且 `low/high` 是有限值。
- `info["cost"]` 建议始终存在，PPO 可以只记录它，PPO-Lagrangian/CPO 会用它参与优化。
- `info["success"]`、`info["collision"]`、`info["timeout"]`、`info["path_length"]`、`info["smoothness"]` 建议用于自动评估。

`gazebo_ugv_env.py` 是按当前确认接口写的 ROS1/Gazebo wrapper；`gazebo_ugv_env_template.py` 是更轻量的接口模板。建模/仿真组需要在学长 Gazebo 虚拟机里保证 `/scan`、`/odom`、`/cmd_vel`、reset 相关服务可用；算法组只依赖 Gym wrapper，不直接把 PPO 写死到 ROS 细节里。

当前确认的 Gazebo-RL 接口详见 `docs/gazebo_interface_contract.md`：

- `obs = 30维 = 24维激光 + 6维车辆/目标状态`
- `action = (v, w) = 线速度 + 角速度`
- 输入：`/scan`、`/odom`
- 输出：`/cmd_vel`
- 链路：`scan + odom -> state -> policy -> safety -> cmd_vel -> Gazebo`

不同小车建议优先配置这些项：

```bash
python ppo_training.py \
  --env-module your_gazebo_env_package \
  --env-id YourGazeboUGVEnv-v0 \
  --episodes 1000 \
  --steps-per-update 2048 \
  --batch-size 128 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --entropy-coef 0.01 \
  --target-kl 0.02 \
  --reward-scale 1.0 \
  --normalize-reward
```

如果状态量纲差异很大，例如同时包含位置、速度、角速度、激光距离，可以再加 `--normalize-obs`。

## 后续 CPO 接入建议

CPO 需要环境额外返回或计算安全代价 `cost`，例如碰撞、越界、距离障碍物过近、速度/角速度超限。建议后续把 `info["cost"]` 作为统一入口：

```python
next_state, reward, terminated, truncated, info = env.step(action)
cost = info.get("cost", 0.0)
```

这样 PPO、PPO-Lagrangian、CPO 可以共用同一个 Gazebo wrapper，算法侧只切换优化器和约束处理。

当前优先级：

1. 用 `SafeUGV-v0` 跑出 PPO 自动评估结果。
2. 用 `ppo_lagrangian_training.py` 跑出 cost/lambda/collision 指标。
3. 等 Gazebo 可用后，在学长虚拟机中测试 `gazebo_ugv_env.py` 的 `reset()` 和 `step(action)`。
4. CPO 作为后续对照算法继续复现，不建议在 Gazebo 未打通前强行把 CPO 作为唯一主线。
