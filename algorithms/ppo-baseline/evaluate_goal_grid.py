"""Evaluate a policy over a grid of Gazebo goal points.

This script is intended for the project midterm evidence stage. It turns a
single-policy check into a small generalization table, e.g. goals around
``(2.0, 0.0)`` instead of only the original training target.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from evaluate_policy import (
    latest_model,
    latest_normalizer,
    load_actor,
    make_env,
    make_safety_shield,
    matching_normalizer,
    normalize_state,
    policy_action,
    reset_env,
    set_seed,
    step_env,
)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on a goal grid.")
    parser.add_argument("--env-id", default="GazeboUGV-v0")
    parser.add_argument("--env-module", default="gazebo_ugv_env")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--goal-x-values", default="1.8,2.0,2.2")
    parser.add_argument("--goal-y-values", default="0.0")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--normalizer-path", default="")
    parser.add_argument("--episodes-per-goal", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 infers hidden size from checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--use-safety-shield", action="store_true")
    parser.add_argument("--shield-warning-distance", type=float, default=0.10)
    parser.add_argument("--shield-stop-distance", type=float, default=0.04)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--collision-threshold", type=float, default=0.0)
    parser.add_argument("--out-dir", default="eval_logs")
    return parser.parse_args()


def load_normalizer(path):
    if not path or not os.path.exists(path):
        return None
    data = np.load(path)
    return {"mean": data["obs_mean"], "var": data["obs_var"]}


def make_goal_env_kwargs(base_kwargs, goal_x, goal_y):
    env_kwargs = dict(base_kwargs)
    env_kwargs["goal_x"] = float(goal_x)
    env_kwargs["goal_y"] = float(goal_y)
    env_kwargs["randomize_goal"] = False
    return env_kwargs


def evaluate_one_goal(args, actor, normalizer, device, goal_x, goal_y, base_kwargs):
    env_kwargs = make_goal_env_kwargs(base_kwargs, goal_x, goal_y)
    env = make_env(args.env_id, args.env_module, env_kwargs)
    shield = make_safety_shield(args, env)
    rows = []
    for ep in range(args.episodes_per_goal):
        state = reset_env(env, seed=args.seed + int(goal_x * 1000) + int((goal_y + 10.0) * 1000) + ep)
        episode_reward = 0.0
        episode_cost = 0.0
        last_info = {}
        steps = 0
        while args.max_steps <= 0 or steps < args.max_steps:
            if args.random_policy:
                action = env.action_space.sample()
            else:
                norm_state = normalize_state(state, normalizer)
                action = policy_action(actor, norm_state, env.action_space.low, env.action_space.high, device)
            if shield is not None:
                action, _ = shield.filter_action(action, state)
            state, reward, terminated, truncated, info = step_env(env, action)
            episode_reward += reward
            episode_cost += float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0
            last_info = info if isinstance(info, dict) else {}
            steps += 1
            if terminated or truncated:
                break
        rows.append(
            {
                "goal_x": float(goal_x),
                "goal_y": float(goal_y),
                "episode": ep,
                "reward": float(episode_reward),
                "cost": float(episode_cost),
                "success": int(bool(last_info.get("success", False))),
                "collision": int(bool(last_info.get("collision", False))),
                "timeout": int(bool(last_info.get("timeout", False))),
                "path_length": float(last_info.get("path_length", 0.0)),
                "smoothness": float(last_info.get("smoothness", 0.0)),
                "steps": int(steps),
                "event": str(last_info.get("event", "")),
            }
        )
    env.close()
    return rows


def summarize(goal_rows, success_threshold, collision_threshold):
    n = max(1, len(goal_rows))
    success_rate = float(sum(row["success"] for row in goal_rows) / n)
    collision_rate = float(sum(row["collision"] for row in goal_rows) / n)
    timeout_rate = float(sum(row["timeout"] for row in goal_rows) / n)
    return {
        "goal_x": goal_rows[0]["goal_x"],
        "goal_y": goal_rows[0]["goal_y"],
        "episodes": len(goal_rows),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "mean_reward": float(np.mean([row["reward"] for row in goal_rows])),
        "mean_cost": float(np.mean([row["cost"] for row in goal_rows])),
        "mean_path_length": float(np.mean([row["path_length"] for row in goal_rows])),
        "mean_smoothness": float(np.mean([row["smoothness"] for row in goal_rows])),
        "mean_steps": float(np.mean([row["steps"] for row in goal_rows])),
        "passed": bool(success_rate >= success_threshold and collision_rate <= collision_threshold),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    base_kwargs = json.loads(args.env_kwargs)
    goal_x_values = parse_float_list(args.goal_x_values)
    goal_y_values = parse_float_list(args.goal_y_values)
    current_path = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "" if args.random_policy else (args.model_path or latest_model(current_path))
    normalizer_path = args.normalizer_path or matching_normalizer(model_path) or latest_normalizer(current_path)
    normalizer = load_normalizer(normalizer_path)

    actor = None
    if not args.random_policy:
        first_kwargs = make_goal_env_kwargs(base_kwargs, goal_x_values[0], goal_y_values[0])
        probe_env = make_env(args.env_id, args.env_module, first_kwargs)
        actor, _ = load_actor(
            model_path,
            probe_env.observation_space.shape[0],
            probe_env.action_space.shape[0],
            args.hidden_dim,
            device,
        )
        probe_env.close()

    all_rows = []
    summary_rows = []
    for goal_y in goal_y_values:
        for goal_x in goal_x_values:
            goal_rows = evaluate_one_goal(args, actor, normalizer, device, goal_x, goal_y, base_kwargs)
            all_rows.extend(goal_rows)
            summary = summarize(goal_rows, args.success_threshold, args.collision_threshold)
            summary_rows.append(summary)
            print(
                f"goal=({goal_x:.3f},{goal_y:.3f}) "
                f"success={summary['success_rate']:.2f} "
                f"collision={summary['collision_rate']:.2f} "
                f"timeout={summary['timeout_rate']:.2f} "
                f"reward={summary['mean_reward']:.2f} "
                f"passed={int(summary['passed'])}"
            )

    pass_rate = float(np.mean([row["passed"] for row in summary_rows])) if summary_rows else 0.0
    aggregate = {
        "env_id": args.env_id,
        "env_kwargs": base_kwargs,
        "model_path": model_path or "random-policy",
        "normalizer_path": normalizer_path,
        "episodes_per_goal": args.episodes_per_goal,
        "goal_x_values": goal_x_values,
        "goal_y_values": goal_y_values,
        "success_threshold": args.success_threshold,
        "collision_threshold": args.collision_threshold,
        "goal_count": len(summary_rows),
        "goal_pass_rate": pass_rate,
        "mean_success_rate": float(np.mean([row["success_rate"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_collision_rate": float(np.mean([row["collision_rate"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_timeout_rate": float(np.mean([row["timeout_rate"] for row in summary_rows])) if summary_rows else 0.0,
        "goals": summary_rows,
    }

    out_dir = os.path.join(current_path, args.out_dir, f"goal_grid_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "grid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "grid_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    with open(os.path.join(out_dir, "episodes.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        if all_rows:
            writer.writeheader()
            writer.writerows(all_rows)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
