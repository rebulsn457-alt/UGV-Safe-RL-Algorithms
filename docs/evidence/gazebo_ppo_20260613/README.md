# Gazebo PPO Evidence 2026-06-13

This folder records the PPO-Gazebo closed-loop evidence produced on the physical Ubuntu/ROS/Gazebo machine on 2026-06-13.

## Contents

- `logs/GazeboUGV-v0_20260613083156/`: PPO training config, episode rewards, and evaluation metrics.
- `models/best_ppo_actor_GazeboUGV-v0_20260613083156.pth`: best actor checkpoint from the Gazebo PPO run.
- `eval_logs/20260613092339/`: deterministic evaluation on goal `(2.0, 0.0)`.
- `eval_logs/20260613094230/`: deterministic evaluation on shifted goal `(1.5, 0.5)`.
- `eval_logs/20260613094639/`: deterministic evaluation on local forward goal `(1.8, 0.0)`.
- `eval_logs/20260613095113/`: deterministic evaluation on local forward goal `(2.2, 0.0)`.
- `gazebo_ppo_eval_summary.csv`: compact table for reports.

## Run Summary

Environment:

- Ubuntu 20.04
- ROS Noetic
- Gazebo 11
- Gym wrapper: `GazeboUGV-v0`
- Observation: 30 dimensions
- Action: `(v, w)`
- ROS inputs: `/scan`, `/odom`
- ROS output: `/cmd_vel`

Training:

- PPO trained for 200 episodes.
- Best evaluation reward: about `129.0642`.
- Best model: `models/best_ppo_actor_GazeboUGV-v0_20260613083156.pth`.

## Evaluation Results

| run_id | goal | episodes | success_rate | collision_rate | timeout_rate | mean_reward | mean_cost | mean_path_length | mean_smoothness | mean_steps | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260613092339 | `(2.0, 0.0)` | 5 | 1.00 | 0.00 | 0.00 | 129.04 | 0.1166 | 1.4968 | 1.4240 | 80.0 | trained target passed |
| 20260613094230 | `(1.5, 0.5)` | 5 | 0.00 | 0.00 | 1.00 | -38.52 | 0.1011 | 4.8804 | 1.7226 | 250.0 | lateral target timed out |
| 20260613094639 | `(1.8, 0.0)` | 5 | 1.00 | 0.00 | 0.00 | 127.81 | 0.1397 | 1.3025 | 1.3421 | 70.0 | local forward generalization passed |
| 20260613095113 | `(2.2, 0.0)` | 5 | 1.00 | 0.00 | 0.00 | 130.39 | 0.1342 | 1.7073 | 1.4800 | 89.6 | local forward generalization passed |

## Conclusion

The current PPO code has completed a minimal Gazebo closed loop: `/scan` and `/odom` are converted into a 30-dimensional state, the policy outputs `(v, w)`, safety filtering is available, and `/cmd_vel` drives the Gazebo vehicle. The trained PPO checkpoint succeeds on the original Gazebo target and nearby forward targets with zero collision in these evaluations.

The lateral target `(1.5, 0.5)` timed out. This means the current checkpoint is not yet a fully generalized navigation policy. The next algorithm step is multi-goal randomized Gazebo training, followed by PPO-Lagrangian/CPO comparison using the same `cost` interface.
