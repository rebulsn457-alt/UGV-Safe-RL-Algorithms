# 算法完成证据包

日期：2026-06-11

本目录保存算法组当前可复核的实验数据与图表，用于支撑“PPO 泛化、PPO-Lagrangian/CMDP 安全约束、安全屏蔽层”已经完成阶段性交付。

## 文件说明

| 文件 | 作用 |
|---|---|
| `ppolag_soft_seed0_training_eval_metrics.csv` | PPO-Lagrangian soft 约束 seed 0 训练过程中的每轮评估指标 |
| `ppolag_soft_multiseed_summary.csv` | seed 0/1/2，400 episodes 的最终汇总 |
| `eval_no_shield_summary.json` | 同一 checkpoint 不启用 safety shield 的 100 回合独立评估 |
| `eval_default_fixed_shield_summary.json` | 同一 checkpoint 启用修正后默认 light shield 的 100 回合独立评估 |
| `computed_summary.json` | 根据 CSV/JSON 计算出的汇总均值 |
| `ppolag_soft_multiseed_stdout_tail.txt` | 虚拟机训练日志末尾输出，可核验训练命令和最终结果 |
| `ppolag_soft_seed0_training_curve.png` | seed 0 训练曲线图 |
| `shield_independent_eval_comparison.png` | 不开 shield 与默认 light shield 的 100 回合评估对比图 |
| `vm_runtime_verification_20260611.txt` | 记录虚拟机 Ubuntu 环境、commit、依赖版本、语法检查和短训练 smoke test，证明算法入口在 VM 中可运行 |

## 核心结论

- PPO-Lagrangian soft 约束，seed 0，400 episodes 最后一轮：`success_rate=0.80`，`collision_rate=0.00`，`eval_mean_reward≈126.42`。
- PPO-Lagrangian soft 约束，seed 0/1/2，400 episodes 平均：`success_rate=0.6000`，`collision_rate=0.0000`，`eval_mean_reward≈96.46`。
- 默认 light shield 评估：在 100 回合独立评估中将 `collision_rate` 从 `0.14` 压到 `0.00`，同时 `success_rate` 从 `0.64` 提升到 `0.69`。

这些文件是提交算法完成情况时的直接证据，不依赖单纯文字描述。

补充说明：本证据包中的训练/评估数据来源于 Ubuntu 虚拟机路径 `/home/karwen/ppo_project/algorithms/ppo-baseline`。Windows 本机仅用于整理图表、Word 文档和 GitHub 推送，不作为最终收敛结果来源。
