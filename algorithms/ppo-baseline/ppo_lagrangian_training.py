import argparse
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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None

from ppo_lagrangian_agent import PPOLagrangianAgent


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x.reshape((1,) + self.mean.shape)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, x, clip=10.0):
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO-Lagrangian trainer for safe continuous UGV environments.")
    parser.add_argument("--env-id", default="SafeUGV-v0")
    parser.add_argument("--env-module", default="safe_ugv_env")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-cost-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cost-gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--cost-value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--log-std-init", type=float, default=-0.2)
    parser.add_argument("--reward-scale", type=float, default=0.02)
    parser.add_argument("--cost-limit", type=float, default=5.0)
    parser.add_argument("--lambda-lr", type=float, default=0.02)
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--normalize-obs", action="store_true", default=True)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id, env_module="", env_kwargs=None):
    if env_module:
        importlib.import_module(env_module)
    env = gym.make(env_id, **(env_kwargs or {}))
    if not hasattr(env.action_space, "low") or not hasattr(env.action_space, "high"):
        raise TypeError("PPO-Lagrangian expects a continuous Box action space.")
    return env


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


def preprocess_state(state, obs_rms, update_stats):
    state = np.asarray(state, dtype=np.float32)
    if obs_rms is None:
        return state
    if update_stats:
        obs_rms.update(state)
    return obs_rms.normalize(state)


def evaluate(agent, env_id, env_module, env_kwargs, episodes, seed, max_steps, obs_rms):
    env = make_env(env_id, env_module, env_kwargs)
    rewards, costs, path_lengths, smoothness, steps_list = [], [], [], [], []
    success = collision = timeout = 0
    for ep in range(episodes):
        state = reset_env(env, seed=seed + 20000 + ep)
        episode_reward = 0.0
        episode_cost = 0.0
        last_info = {}
        steps = 0
        while max_steps <= 0 or steps < max_steps:
            norm_state = preprocess_state(state, obs_rms, update_stats=False)
            action, _, _, _, _ = agent.get_action(norm_state, deterministic=True)
            state, reward, terminated, truncated, info = step_env(env, action)
            episode_reward += reward
            episode_cost += float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0
            last_info = info if isinstance(info, dict) else {}
            steps += 1
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        costs.append(episode_cost)
        path_lengths.append(float(last_info.get("path_length", 0.0)))
        smoothness.append(float(last_info.get("smoothness", 0.0)))
        steps_list.append(steps)
        success += int(bool(last_info.get("success", False)))
        collision += int(bool(last_info.get("collision", False)))
        timeout += int(bool(last_info.get("timeout", False)))
    env.close()
    n = max(1, episodes)
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": success / n,
        "collision_rate": collision / n,
        "timeout_rate": timeout / n,
        "mean_cost": float(np.mean(costs)),
        "mean_path_length": float(np.mean(path_lengths)),
        "mean_smoothness": float(np.mean(smoothness)),
        "mean_steps": float(np.mean(steps_list)),
    }


def save_normalizer(path, obs_rms):
    if obs_rms is None:
        return
    np.savez(path, obs_mean=obs_rms.mean, obs_var=obs_rms.var, obs_count=obs_rms.count)


def main():
    args = parse_args()
    set_seed(args.seed)
    env_kwargs = json.loads(args.env_kwargs)
    env = make_env(args.env_id, args.env_module, env_kwargs)
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except AttributeError:
        pass

    state_shape = env.observation_space.shape
    if len(state_shape) != 1:
        raise TypeError("Expected flat 1D observation vector.")
    state_dim = state_shape[0]
    action_dim = env.action_space.shape[0]
    max_steps = args.max_steps
    if max_steps <= 0 and getattr(env, "spec", None) is not None:
        max_steps = env.spec.max_episode_steps or 0

    agent = PPOLagrangianAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=args.batch_size,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_cost_critic=args.lr_cost_critic,
        gamma=args.gamma,
        cost_gamma=args.cost_gamma,
        gae_lambda=args.gae_lambda,
        epochs=args.epochs,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        cost_value_coef=args.cost_value_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        log_std_init=args.log_std_init,
        cost_limit=args.cost_limit,
        lambda_lr=args.lambda_lr,
        lambda_init=args.lambda_init,
    )

    current_path = os.path.dirname(os.path.realpath(__file__))
    run_name = args.run_name or f"{args.env_id}_PPOLag_{time.strftime('%Y%m%d%H%M%S')}"
    model_dir = os.path.join(current_path, args.save_dir)
    log_dir = os.path.join(current_path, args.log_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if SummaryWriter is not None else NullWriter()
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config["parsed_env_kwargs"] = env_kwargs
        json.dump(config, f, indent=2)

    obs_rms = RunningMeanStd(shape=state_shape) if args.normalize_obs else None
    reward_buffer = np.empty(args.episodes, dtype=np.float32)
    cost_buffer = np.empty(args.episodes, dtype=np.float32)
    metric_rows = []
    best_score = -np.inf
    total_steps = 0
    last_state_for_update = None
    last_terminated_for_update = False

    for episode_i in range(args.episodes):
        state = reset_env(env, seed=args.seed + episode_i)
        episode_reward = 0.0
        episode_cost = 0.0
        steps = 0
        while max_steps <= 0 or steps < max_steps:
            norm_state = preprocess_state(state, obs_rms, update_stats=True)
            action, norm_action, value, cost_value, log_prob = agent.get_action(norm_state)
            next_state, reward, terminated, truncated, info = step_env(env, action)
            cost = float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0

            done_for_buffer = 1.0 if terminated else 0.0
            agent.buffer.add(
                norm_state,
                norm_action,
                action,
                reward * args.reward_scale,
                cost,
                value,
                cost_value,
                log_prob,
                done_for_buffer,
            )
            state = next_state
            last_state_for_update = state
            last_terminated_for_update = terminated
            episode_reward += reward
            episode_cost += cost
            steps += 1
            total_steps += 1

            if len(agent.buffer) >= args.steps_per_update:
                last_state = preprocess_state(state, obs_rms, update_stats=False)
                last_reward_value, last_cost_value = (0.0, 0.0) if terminated else agent.get_values(last_state)
                update_info = agent.update(last_reward_value, last_cost_value)
                for key, value_item in update_info.items():
                    writer.add_scalar(f"Update/{key}", value_item, total_steps)

            if terminated or truncated:
                break

        cost_violation = agent.update_lagrange(episode_cost)
        reward_buffer[episode_i] = episode_reward
        cost_buffer[episode_i] = episode_cost
        writer.add_scalar("Train/episode_reward", episode_reward, episode_i)
        writer.add_scalar("Train/episode_cost", episode_cost, episode_i)
        writer.add_scalar("Train/cost_violation", cost_violation, episode_i)
        writer.add_scalar("Train/lambda", agent.lagrange_lambda, episode_i)
        writer.add_scalar("Train/episode_steps", steps, episode_i)

        if (episode_i + 1) % args.eval_interval == 0:
            recent_reward = reward_buffer[max(0, episode_i - args.eval_interval + 1):episode_i + 1]
            recent_cost = cost_buffer[max(0, episode_i - args.eval_interval + 1):episode_i + 1]
            eval_metrics = evaluate(agent, args.env_id, args.env_module, env_kwargs, args.eval_episodes, args.seed, max_steps, obs_rms)
            for key, metric_value in eval_metrics.items():
                writer.add_scalar(f"Eval/{key}", metric_value, episode_i)
            print(
                f"Ep {episode_i + 1}/{args.episodes} "
                f"train_reward={recent_reward.mean():.2f} "
                f"train_cost={recent_cost.mean():.2f} "
                f"eval_reward={eval_metrics['mean_reward']:.2f} "
                f"success={eval_metrics['success_rate']:.2f} "
                f"collision={eval_metrics['collision_rate']:.2f} "
                f"eval_cost={eval_metrics['mean_cost']:.2f} "
                f"lambda={agent.lagrange_lambda:.3f} "
                f"steps={total_steps}"
            )
            metric_rows.append(
                [
                    episode_i + 1,
                    total_steps,
                    float(recent_reward.mean()),
                    float(recent_cost.mean()),
                    eval_metrics["mean_reward"],
                    eval_metrics["success_rate"],
                    eval_metrics["collision_rate"],
                    eval_metrics["timeout_rate"],
                    eval_metrics["mean_cost"],
                    eval_metrics["mean_path_length"],
                    eval_metrics["mean_smoothness"],
                    eval_metrics["mean_steps"],
                    agent.lagrange_lambda,
                ]
            )
            score = eval_metrics["mean_reward"] - 50.0 * eval_metrics["collision_rate"] - 2.0 * eval_metrics["mean_cost"]
            if score > best_score:
                best_score = score
                agent.save_policy(os.path.join(model_dir, f"best_ppo_lagrangian_{run_name}.pth"))
                save_normalizer(os.path.join(model_dir, f"obs_normalizer_{run_name}.npz"), obs_rms)

    if len(agent.buffer) > 0:
        last_state = preprocess_state(last_state_for_update, obs_rms, update_stats=False)
        last_reward_value, last_cost_value = (0.0, 0.0) if last_terminated_for_update else agent.get_values(last_state)
        agent.update(last_reward_value, last_cost_value)

    env.close()
    writer.close()
    np.savetxt(os.path.join(log_dir, "episode_rewards.txt"), reward_buffer)
    np.savetxt(os.path.join(log_dir, "episode_costs.txt"), cost_buffer)
    if metric_rows:
        np.savetxt(
            os.path.join(log_dir, "eval_metrics.csv"),
            np.asarray(metric_rows, dtype=np.float32),
            delimiter=",",
            header="episode,total_steps,train_reward,train_cost,eval_mean_reward,success_rate,collision_rate,timeout_rate,mean_cost,mean_path_length,mean_smoothness,mean_steps,lambda",
            comments="",
        )
    print(
        "Training done. Last 20 reward:",
        reward_buffer[-20:].mean(),
        "Last 20 cost:",
        cost_buffer[-20:].mean(),
        "lambda:",
        agent.lagrange_lambda,
    )


if __name__ == "__main__":
    main()
