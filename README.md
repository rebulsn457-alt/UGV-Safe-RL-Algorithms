# UGV-Safe-RL-Algorithms

无人车安全路径规划强化学习算法库。当前重点是把 PPO baseline 从单一 `Pendulum-v1` 实验改造成可迁移到不同小车与 Gazebo 环境的连续控制训练入口，并为后续 CPO/TRPO 等安全约束算法预留接口。

## 当前 PPO 泛化改造

主要代码：

- `algorithms/ppo-baseline/ppo_agent.py`
- `algorithms/ppo-baseline/ppo._training.py`

已完成的泛化优化：

- 自动读取环境 `observation_space` 与 `action_space`，不再硬编码 Pendulum 的状态维度、动作维度和 `[-2, 2]` 动作范围。
- 在归一化动作空间中训练 Gaussian policy，再映射到环境真实连续动作上下界；buffer 保存采样动作本身，避免旧版 `clamp` 后 log probability 不一致导致 PPO ratio 偏移。
- PPO 超参数全部改为命令行配置，包括学习率、`gamma`、GAE lambda、clip、entropy、target KL、batch size、rollout 长度等。
- 按 `steps-per-update` 收集 rollout，而不是按固定 episode 数更新，更适合 Gazebo 中可变 episode 长度。
- 加入 value-function clipping、advantage normalization、target KL 日志和 deterministic evaluation，便于判断策略是否真的学到稳定控制。
- 兼容 `gymnasium` 和旧版 `gym`，便于 Ubuntu + VSCode + Gazebo 工作流。
- 支持 observation normalization 和 reward normalization，适配不同车辆质量、速度量纲、传感器量纲变化。
- 保存训练配置、TensorBoard 日志、reward 曲线、最佳模型权重；如果启用观测归一化，也会保存对应 normalizer 参数。

## Ubuntu/VSCode 快速运行

建议在虚拟机 Ubuntu 项目目录中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd algorithms/ppo-baseline
python ppo._training.py --env-id Pendulum-v1 --episodes 400
```

本地验证结果：

- `Pendulum-v1`，400 episodes，默认参数：deterministic eval 从约 `-1023` 提升到约 `-86`。
- `Pendulum-v1`，800 episodes，默认参数：最后 20 个训练 episode 均值约 `-146`，deterministic eval 约 `-98`。

训练时的 `train_mean` 是随机采样策略的回报，会比确定性评估低；判断是否收敛优先看 `eval_mean`。

如果你的 UGV/Gazebo 环境 reward 已经在合理范围，例如单步大致 `[-5, 5]`，建议改成：

```bash
python ppo._training.py --env-id YourGazeboUGVEnv-v0 --reward-scale 1.0 --normalize-obs --normalize-reward
```

## 对接其他小车/Gazebo 的接口要求

环境需要提供 Gym/Gymnasium 风格接口：

- `reset()` 返回一维状态向量，或 `(state, info)`。
- `step(action)` 返回 `(next_state, reward, terminated, truncated, info)`，旧 Gym 的 `(next_state, reward, done, info)` 也兼容。
- `observation_space.shape` 必须是一维，例如 `(state_dim,)`。
- `action_space` 必须是连续 `Box`，并且 `low/high` 是有限值。

不同小车建议优先配置这些项：

```bash
python ppo._training.py \
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
