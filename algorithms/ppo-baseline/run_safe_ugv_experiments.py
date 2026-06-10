"""Run and summarize SafeUGV PPO/PPO-Lagrangian experiments.

Examples:

    python run_safe_ugv_experiments.py --algorithms ppo,ppolag --seeds 0,1,2 --episodes 600
    python run_safe_ugv_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ENV_CONFIGS = {
    "base": {},
    "small_slow": {"robot_radius": 0.28, "max_speed": 1.1, "max_yaw_rate": 1.0},
    "fast_wide": {"robot_radius": 0.42, "max_speed": 1.8, "max_yaw_rate": 1.3},
    "dense": {"obstacle_count": 12, "safety_margin": 0.32},
}


SUMMARY_COLUMNS = [
    "algorithm",
    "config",
    "seed",
    "run_name",
    "episode",
    "total_steps",
    "train_reward",
    "train_cost",
    "eval_mean_reward",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_cost",
    "mean_path_length",
    "mean_smoothness",
    "mean_steps",
    "lambda",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run SafeUGV algorithm comparison experiments.")
    parser.add_argument("--algorithms", default="ppo,ppolag", help="Comma list: ppo, ppolag.")
    parser.add_argument("--configs", default="base", help=f"Comma list from {','.join(ENV_CONFIGS)} or all.")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1,2.")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def command_for(algo, config_name, env_kwargs, seed, args, run_name):
    common = [
        sys.executable,
        "ppo._training.py" if algo == "ppo" else "ppo_lagrangian_training.py",
        "--env-module",
        "safe_ugv_env",
        "--env-id",
        "SafeUGV-v0",
        "--env-kwargs",
        json.dumps(env_kwargs, separators=(",", ":")),
        "--episodes",
        str(args.episodes),
        "--steps-per-update",
        str(args.steps_per_update),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]
    if algo == "ppo":
        common.extend(["--reward-scale", "0.02", "--normalize-obs"])
    return common


def read_last_metrics(log_dir, algo, config_name, seed, run_name):
    metrics_path = Path(log_dir) / run_name / "eval_metrics.csv"
    if not metrics_path.exists():
        return {
            "algorithm": algo,
            "config": config_name,
            "seed": seed,
            "run_name": run_name,
        }
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "algorithm": algo,
            "config": config_name,
            "seed": seed,
            "run_name": run_name,
        }
    row = rows[-1]
    if algo == "ppo":
        mapped = {
            "episode": row.get("episode", ""),
            "total_steps": row.get("total_steps", ""),
            "train_reward": row.get("train_mean", ""),
            "train_cost": "",
            "eval_mean_reward": row.get("eval_mean_reward", ""),
            "success_rate": row.get("success_rate", ""),
            "collision_rate": row.get("collision_rate", ""),
            "timeout_rate": row.get("timeout_rate", ""),
            "mean_cost": row.get("mean_cost", ""),
            "mean_path_length": row.get("mean_path_length", ""),
            "mean_smoothness": row.get("mean_smoothness", ""),
            "mean_steps": row.get("mean_steps", ""),
            "lambda": "",
        }
    else:
        mapped = {
            "episode": row.get("episode", ""),
            "total_steps": row.get("total_steps", ""),
            "train_reward": row.get("train_reward", ""),
            "train_cost": row.get("train_cost", ""),
            "eval_mean_reward": row.get("eval_mean_reward", ""),
            "success_rate": row.get("success_rate", ""),
            "collision_rate": row.get("collision_rate", ""),
            "timeout_rate": row.get("timeout_rate", ""),
            "mean_cost": row.get("mean_cost", ""),
            "mean_path_length": row.get("mean_path_length", ""),
            "mean_smoothness": row.get("mean_smoothness", ""),
            "mean_steps": row.get("mean_steps", ""),
            "lambda": row.get("lambda", ""),
        }
    mapped.update({"algorithm": algo, "config": config_name, "seed": seed, "run_name": run_name})
    return mapped


def write_summary(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})


def main():
    args = parse_args()
    algorithms = split_csv(args.algorithms)
    configs = list(ENV_CONFIGS) if args.configs == "all" else split_csv(args.configs)
    seeds = [int(seed) for seed in split_csv(args.seeds)]
    timestamp = time.strftime("%Y%m%d%H%M%S")
    current_dir = Path(__file__).resolve().parent
    summary_path = args.summary_path or str(current_dir / "logs" / f"safe_ugv_experiment_summary_{timestamp}.csv")
    rows = []

    for config_name in configs:
        if config_name not in ENV_CONFIGS:
            raise ValueError(f"Unknown config {config_name}. Choose from {list(ENV_CONFIGS)} or all.")
        for algo in algorithms:
            if algo not in {"ppo", "ppolag"}:
                raise ValueError("algorithms must contain only ppo and/or ppolag")
            for seed in seeds:
                run_name = f"safeugv_{algo}_{config_name}_seed{seed}_{timestamp}"
                cmd = command_for(algo, config_name, ENV_CONFIGS[config_name], seed, args, run_name)
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, cwd=current_dir, check=True)
                    rows.append(read_last_metrics(current_dir / "logs", algo, config_name, seed, run_name))

    if not args.dry_run:
        write_summary(summary_path, rows)
        print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
