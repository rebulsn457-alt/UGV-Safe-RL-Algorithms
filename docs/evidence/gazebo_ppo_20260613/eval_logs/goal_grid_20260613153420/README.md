# Gazebo PPO Goal Grid Evaluation 20260613153420

This folder records the latest compact summary transcribed from the physical Ubuntu/ROS/Gazebo machine output and `goal结果.docx`.

Physical-machine source path:

```text
/home/ubuntu/桌面/UGV-Safe-RL-Algorithms_physical_merged_20260613_140915/UGV-Safe-RL-Algorithms/algorithms/ppo-baseline/eval_logs/goal_grid_20260613153420
```

Evaluation setup:

- Environment: `GazeboUGV-v0`
- ROS inputs: `/scan`, `/odom`
- ROS output: `/cmd_vel`
- Model: `models/best_ppo_actor_GazeboUGV-v0_20260613083156.pth`
- Normalizer: `none`
- Goal grid: `goal_x in {1.8, 2.0, 2.2}`, `goal_y in {-0.3, 0.0, 0.3}`
- Episodes per goal: 5
- Total Gazebo episodes: 45

Result:

- Goal pass rate: 1.00
- Mean success rate: 1.00
- Mean collision rate: 0.00
- Mean timeout rate: 0.00

The raw per-episode files remain on the physical machine and can be copied into this folder later if needed.
