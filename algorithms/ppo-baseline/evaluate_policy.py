import argparse
import glob
import importlib
import json
import os
import random
import time

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

from ppo_agent import Actor

try:
    from safety_shield import LidarSafetyShield
except ImportError:
    LidarSafetyShield = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO/PPO-Lagrangian policies with UGV safety metrics.")
    parser.add_argument("--env-id", default="SafeUGV-v0")
    parser.add_argument("--env-module", default="safe_ugv_env")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--model-path", default="", help="Path to .pth checkpoint. Empty means latest model in ./models.")
    parser.add_argument("--normalizer-path", default="", help="Optional obs_normalizer_*.npz path.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--random-policy", action="store_true", help="Evaluate random actions as a baseline.")
    parser.add_argument("--use-safety-shield", action="store_true", help="Filter actions before env.step for deployment-style evaluation.")
    parser.add_argument("--shield-warning-distance", type=float, default=0.08)
    parser.add_argument("--shield-stop-distance", type=float, default=0.04)
    parser.add_argument("--out-dir", default="eval_logs")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id, env_module="", env_kwargs=None):
    if env_module:
        importlib.import_module(env_module)
    return gym.make(env_id, **(env_kwargs or {}))


def make_safety_shield(args, env):
    if not args.use_safety_shield:
        return None
    if LidarSafetyShield is None:
        raise ImportError("safety_shield.py is required when --use-safety-shield is enabled.")
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    lidar_rays = min(24, int(env.observation_space.shape[0]))
    return LidarSafetyShield(
        lidar_rays=lidar_rays,
        warning_distance=args.shield_warning_distance,
        stop_distance=args.shield_stop_distance,
        max_speed=float(action_high[0]),
        max_yaw_rate=float(action_high[1]),
    )


def reset_env(env, seed=None):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
    else:
        next_state, reward, done, info = result
        terminated, truncated = done, False
    return next_state, float(reward), bool(terminated), bool(truncated), info


def latest_model(current_path):
    candidates = []
    for pattern in ("best_ppo_lagrangian_*.pth", "best_ppo_actor_*.pth", "best_ppo_actor.pth"):
        candidates.extend(glob.glob(os.path.join(current_path, "models", pattern)))
    if not candidates:
        raise FileNotFoundError("No model found. Pass --model-path or train a policy first.")
    return max(candidates, key=os.path.getmtime)


def latest_normalizer(current_path):
    candidates = glob.glob(os.path.join(current_path, "models", "obs_normalizer_*.npz"))
    return max(candidates, key=os.path.getmtime) if candidates else ""


def matching_normalizer(model_path):
    if not model_path:
        return ""
    model_dir = os.path.dirname(model_path)
    name = os.path.basename(model_path)
    for prefix in ("best_ppo_lagrangian_", "best_ppo_actor_"):
        if name.startswith(prefix) and name.endswith(".pth"):
            run_name = name[len(prefix):-4]
            candidate = os.path.join(model_dir, f"obs_normalizer_{run_name}.npz")
            if os.path.exists(candidate):
                return candidate
    return ""


def load_actor(model_path, state_dim, action_dim, hidden_dim, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, checkpoint


def normalize_state(state, normalizer):
    state = np.asarray(state, dtype=np.float32)
    if normalizer is None:
        return state
    normalized = (state - normalizer["mean"]) / np.sqrt(normalizer["var"] + 1e-8)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32)


@torch.no_grad()
def policy_action(actor, state, action_low, action_high, device):
    s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mean, _ = actor(s)
    norm_action = torch.clamp(mean, -1.0, 1.0)
    low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
    action = norm_action * ((high - low) / 2.0) + ((high + low) / 2.0)
    return action.cpu().numpy()[0]


def summarize(rows):
    n = max(1, len(rows))
    return {
        "episodes": len(rows),
        "success_rate": float(sum(r["success"] for r in rows) / n),
        "collision_rate": float(sum(r["collision"] for r in rows) / n),
        "timeout_rate": float(sum(r["timeout"] for r in rows) / n),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_cost": float(np.mean([r["cost"] for r in rows])),
        "mean_path_length": float(np.mean([r["path_length"] for r in rows])),
        "mean_smoothness": float(np.mean([r["smoothness"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    env_kwargs = json.loads(args.env_kwargs)
    current_path = os.path.dirname(os.path.realpath(__file__))
    env = make_env(args.env_id, args.env_module, env_kwargs)
    shield = make_safety_shield(args, env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "" if args.random_policy else (args.model_path or latest_model(current_path))
    normalizer_path = args.normalizer_path or matching_normalizer(model_path) or latest_normalizer(current_path)
    normalizer = None
    if normalizer_path and os.path.exists(normalizer_path):
        data = np.load(normalizer_path)
        normalizer = {"mean": data["obs_mean"], "var": data["obs_var"]}

    actor = None
    if not args.random_policy:
        actor, _ = load_actor(model_path, env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim, device)

    rows = []
    for ep in range(args.episodes):
        state = reset_env(env, seed=args.seed + ep)
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
            shield_info = {}
            if shield is not None:
                action, shield_info = shield.filter_action(action, state)
            state, reward, terminated, truncated, info = step_env(env, action)
            episode_reward += reward
            episode_cost += float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0
            last_info = info if isinstance(info, dict) else {}
            steps += 1
            if terminated or truncated:
                break
        rows.append(
            {
                "episode": ep,
                "reward": float(episode_reward),
                "cost": float(episode_cost),
                "success": int(bool(last_info.get("success", False))),
                "collision": int(bool(last_info.get("collision", False))),
                "timeout": int(bool(last_info.get("timeout", False))),
                "path_length": float(last_info.get("path_length", 0.0)),
                "smoothness": float(last_info.get("smoothness", 0.0)),
                "steps": steps,
                "event": str(last_info.get("event", "")),
                "shield_active": int(bool(shield_info.get("shield_active", False))),
            }
        )
    env.close()

    summary = summarize(rows)
    summary["env_id"] = args.env_id
    summary["env_kwargs"] = env_kwargs
    summary["model_path"] = model_path or "random-policy"
    summary["normalizer_path"] = normalizer_path
    summary["use_safety_shield"] = bool(args.use_safety_shield)
    out_dir = os.path.join(current_path, args.out_dir, time.strftime("%Y%m%d%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "episodes.csv"), "w", encoding="utf-8") as f:
        f.write("episode,reward,cost,success,collision,timeout,path_length,smoothness,steps,event,shield_active\n")
        for row in rows:
            f.write(
                f"{row['episode']},{row['reward']:.6f},{row['cost']:.6f},{row['success']},"
                f"{row['collision']},{row['timeout']},{row['path_length']:.6f},"
                f"{row['smoothness']:.6f},{row['steps']},{row['event']},{row['shield_active']}\n"
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
