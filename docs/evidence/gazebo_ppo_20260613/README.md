# Gazebo PPO Evidence 2026-06-13

This folder records the PPO-Gazebo closed-loop evidence produced on the physical Ubuntu/ROS/Gazebo machine on 2026-06-13.

## Contents

- `logs/GazeboUGV-v0_20260613083156/`: PPO training config, episode rewards, and evaluation metrics.
- `models/best_ppo_actor_GazeboUGV-v0_20260613083156.pth`: best actor checkpoint from the Gazebo PPO run.
- `eval_logs/20260613092339/`: deterministic evaluation on goal `(2.0, 0.0)`.
- `eval_logs/20260613094230/`: early deterministic probe on shifted goal `(1.5, 0.5)`.
- `eval_logs/20260613094639/`: deterministic evaluation on local forward goal `(1.8, 0.0)`.
- `eval_logs/20260613095113/`: deterministic evaluation on local forward goal `(2.2, 0.0)`.
- `eval_logs/goal_grid_20260613153420/`: latest 9-goal grid evaluation summary, 45 Gazebo episodes total.
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

Latest grid evaluation:

| run_id | goal_x_values | goal_y_values | goal_count | episodes_per_goal | total_episodes | mean_success_rate | mean_collision_rate | mean_timeout_rate | goal_pass_rate | conclusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| goal_grid_20260613153420 | `1.8,2.0,2.2` | `-0.3,0.0,0.3` | 9 | 5 | 45 | 1.00 | 0.00 | 0.00 | 1.00 | local multi-goal grid passed |

Earlier single-goal probes:

| run_id | goal | episodes | success_rate | collision_rate | timeout_rate | mean_reward | mean_cost | mean_path_length | mean_smoothness | mean_steps | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260613092339 | `(2.0, 0.0)` | 5 | 1.00 | 0.00 | 0.00 | 129.04 | 0.1166 | 1.4968 | 1.4240 | 80.0 | trained target passed |
| 20260613094230 | `(1.5, 0.5)` | 5 | 0.00 | 0.00 | 1.00 | -38.52 | 0.1011 | 4.8804 | 1.7226 | 250.0 | lateral target timed out |
| 20260613094639 | `(1.8, 0.0)` | 5 | 1.00 | 0.00 | 0.00 | 127.81 | 0.1397 | 1.3025 | 1.3421 | 70.0 | local forward generalization passed |
| 20260613095113 | `(2.2, 0.0)` | 5 | 1.00 | 0.00 | 0.00 | 130.39 | 0.1342 | 1.7073 | 1.4800 | 89.6 | local forward generalization passed |

## Conclusion

The current PPO code has completed a Gazebo closed loop: `/scan` and `/odom` are converted into a 30-dimensional state, the policy outputs `(v, w)`, safety filtering is available, and `/cmd_vel` drives the Gazebo vehicle. The trained PPO checkpoint now passes the 9-goal local grid evaluation with 45/45 successful episodes, zero collisions, and zero timeouts.

The early lateral target probe `(1.5, 0.5)` timed out, so the project should not claim full large-range navigation generalization yet. The current supported claim is stronger and more precise: PPO-Gazebo closed-loop control and small-range multi-goal generalization have been verified. The next algorithm step is PPO-Lagrangian/CPO comparison using the same CMDP `cost` interface, followed by wider goal, start-pose, and obstacle randomization.
