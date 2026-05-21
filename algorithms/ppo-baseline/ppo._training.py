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
except ImportError:  # Keep compatibility with older Ubuntu/Gazebo stacks.
    import gym

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None

from ppo_agent import PPOAgent


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


class RunningMeanStd:
    """Online normalizer for observations or discounted returns."""

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


class ReturnNormalizer:
    def __init__(self, gamma):
        self.gamma = gamma
        self.running_return = 0.0
        self.rms = RunningMeanStd(shape=())

    def normalize(self, reward, done):
        self.running_return = self.running_return * self.gamma * (1.0 - float(done)) + reward
        self.rms.update(np.array([self.running_return], dtype=np.float32))
        return reward / np.sqrt(self.rms.var + 1e-8)


def parse_args():
    parser = argparse.ArgumentParser(description="General PPO trainer for continuous UGV/Gym-style environments.")
    parser.add_argument("--env-id", default="Pendulum-v1", help="Gym/Gymnasium environment id or Gazebo wrapper id.")
    parser.add_argument("--env-module", default="", help="Optional module to import before gym.make, e.g. a Gazebo env registration package.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses env.spec.max_episode_steps when available.")
    parser.add_argument("--steps-per-update", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--no-clip-value-loss", action="store_true", help="Disable PPO value-function clipping.")
    parser.add_argument("--anneal-lr", action="store_true", help="Linearly anneal learning rates to zero during training.")
    parser.add_argument("--reward-scale", type=float, default=0.125, help="Pendulum-compatible default. Use 1.0 for already scaled UGV rewards.")
    parser.add_argument("--normalize-obs", action="store_true", help="Recommended for Gazebo/UGV state vectors with mixed units.")
    parser.add_argument("--normalize-reward", action="store_true", help="Useful when reward magnitude changes across vehicles.")
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=3)
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


def make_env(env_id, env_module=""):
    if env_module:
        importlib.import_module(env_module)
    env = gym.make(env_id)
    if not hasattr(env.action_space, "low") or not hasattr(env.action_space, "high"):
        raise TypeError("This PPO implementation expects a continuous Box action space.")
    if not np.all(np.isfinite(env.action_space.low)) or not np.all(np.isfinite(env.action_space.high)):
        raise ValueError("Action bounds must be finite. Add an action wrapper before training PPO.")
    return env


def preprocess_state(state, obs_rms, update_stats):
    state = np.asarray(state, dtype=np.float32)
    if obs_rms is None:
        return state
    if update_stats:
        obs_rms.update(state)
    return obs_rms.normalize(state)


def evaluate(agent, env_id, env_module, episodes, seed, max_steps, obs_rms):
    env = make_env(env_id, env_module)
    rewards = []
    for ep in range(episodes):
        state = reset_env(env, seed=seed + 10000 + ep)
        episode_reward = 0.0
        steps = 0
        while max_steps <= 0 or steps < max_steps:
            norm_state = preprocess_state(state, obs_rms, update_stats=False)
            action, _, _, _ = agent.get_action(norm_state, deterministic=True)
            state, reward, terminated, truncated, _ = step_env(env, action)
            episode_reward += reward
            steps += 1
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    env.close()
    return float(np.mean(rewards))


def save_normalizer(path, obs_rms):
    if obs_rms is None:
        return
    np.savez(
        path,
        obs_mean=obs_rms.mean,
        obs_var=obs_rms.var,
        obs_count=obs_rms.count,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    env = make_env(args.env_id, args.env_module)
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except AttributeError:
        pass

    state_shape = env.observation_space.shape
    if len(state_shape) != 1:
        raise TypeError("This baseline expects a flat 1D observation vector. Add a wrapper for images/LiDAR grids.")
    state_dim = state_shape[0]
    action_dim = env.action_space.shape[0]
    max_steps = args.max_steps
    if max_steps <= 0 and getattr(env, "spec", None) is not None:
        max_steps = env.spec.max_episode_steps or 0

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=args.batch_size,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        epochs=args.epochs,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        log_std_init=args.log_std_init,
        clip_value_loss=not args.no_clip_value_loss,
    )

    current_path = os.path.dirname(os.path.realpath(__file__))
    run_name = args.run_name or f"{args.env_id}_{time.strftime('%Y%m%d%H%M%S')}"
    model_dir = os.path.join(current_path, args.save_dir)
    log_dir = os.path.join(current_path, args.log_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if SummaryWriter is not None else NullWriter()

    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    obs_rms = RunningMeanStd(shape=state_shape) if args.normalize_obs else None
    reward_norm = ReturnNormalizer(args.gamma) if args.normalize_reward else None
    reward_buffer = np.empty(args.episodes, dtype=np.float32)
    best_eval_reward = -np.inf
    total_steps = 0
    update_count = 0
    max_total_steps = args.episodes * max_steps if max_steps > 0 else None
    last_state_for_update = None
    last_terminated_for_update = False

    for episode_i in range(args.episodes):
        state = reset_env(env, seed=args.seed + episode_i)
        episode_reward = 0.0
        episode_cost = 0.0
        steps = 0

        while max_steps <= 0 or steps < max_steps:
            norm_state = preprocess_state(state, obs_rms, update_stats=True)
            action, norm_action, value, log_prob = agent.get_action(norm_state)
            next_state, reward, terminated, truncated, info = step_env(env, action)

            done_for_buffer = 1.0 if terminated else 0.0
            train_reward = reward * args.reward_scale
            if reward_norm is not None:
                train_reward = reward_norm.normalize(train_reward, terminated or truncated)

            agent.buffer.add(norm_state, norm_action, action, train_reward, value, log_prob, done_for_buffer)
            state = next_state
            last_state_for_update = state
            last_terminated_for_update = terminated
            episode_reward += reward
            episode_cost += float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0
            steps += 1
            total_steps += 1

            if len(agent.buffer) >= args.steps_per_update:
                if args.anneal_lr and max_total_steps:
                    progress = 1.0 - min(total_steps, max_total_steps) / max_total_steps
                    agent.set_lr_scale(max(progress, 0.0))
                last_state = preprocess_state(state, obs_rms, update_stats=False)
                last_value = 0.0 if terminated else agent.get_value(last_state)
                update_info = agent.update(last_value=last_value)
                update_count += 1
                for key, value_item in update_info.items():
                    writer.add_scalar(f"Update/{key}", value_item, total_steps)
                writer.add_scalar("Update/count", update_count, total_steps)

            if terminated or truncated:
                break

        reward_buffer[episode_i] = episode_reward
        writer.add_scalar("Train/episode_reward", episode_reward, episode_i)
        writer.add_scalar("Train/episode_cost", episode_cost, episode_i)
        writer.add_scalar("Train/episode_steps", steps, episode_i)

        if (episode_i + 1) % args.eval_interval == 0:
            recent = reward_buffer[max(0, episode_i - args.eval_interval + 1):episode_i + 1]
            eval_reward = evaluate(agent, args.env_id, args.env_module, args.eval_episodes, args.seed, max_steps, obs_rms)
            writer.add_scalar("Eval/mean_reward", eval_reward, episode_i)
            print(
                f"Ep {episode_i + 1}/{args.episodes} "
                f"train_mean={recent.mean():.2f} "
                f"eval_mean={eval_reward:.2f} "
                f"steps={total_steps}"
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_policy(os.path.join(model_dir, f"best_ppo_actor_{run_name}.pth"))
                save_normalizer(os.path.join(model_dir, f"obs_normalizer_{run_name}.npz"), obs_rms)

    if len(agent.buffer) > 0:
        last_state = preprocess_state(last_state_for_update, obs_rms, update_stats=False)
        last_value = 0.0 if last_terminated_for_update else agent.get_value(last_state)
        agent.update(last_value=last_value)

    env.close()
    writer.close()
    np.savetxt(os.path.join(log_dir, "episode_rewards.txt"), reward_buffer)
    print("Training done. Last 20 mean:", reward_buffer[-20:].mean(), "best eval:", best_eval_reward)


if __name__ == "__main__":
    main()
