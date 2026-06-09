import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.optim as optim

from ppo_agent import Actor, Critic, _to_tensor, device


class CostReplayMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear()

    def add(self, s, norm_a, env_a, r, cost, v, cv, logp, done):
        self.states.append(s)
        self.norm_actions.append(norm_a)
        self.env_actions.append(env_a)
        self.rewards.append(r)
        self.costs.append(cost)
        self.values.append(v)
        self.cost_values.append(cv)
        self.log_probs.append(logp)
        self.dones.append(done)

    def clear(self):
        self.states, self.norm_actions, self.env_actions = [], [], []
        self.rewards, self.costs = [], []
        self.values, self.cost_values, self.log_probs, self.dones = [], [], [], []

    def __len__(self):
        return len(self.states)

    def as_numpy(self):
        return (
            np.asarray(self.states, dtype=np.float32),
            np.asarray(self.norm_actions, dtype=np.float32),
            np.asarray(self.env_actions, dtype=np.float32),
            np.asarray(self.rewards, dtype=np.float32),
            np.asarray(self.costs, dtype=np.float32),
            np.asarray(self.values, dtype=np.float32),
            np.asarray(self.cost_values, dtype=np.float32),
            np.asarray(self.log_probs, dtype=np.float32),
            np.asarray(self.dones, dtype=np.float32),
        )

    def iter_batches(self, n):
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, self.batch_size):
            yield idx[start:start + self.batch_size]


class PPOLagrangianAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        batch_size,
        action_low=None,
        action_high=None,
        hidden_dim=64,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_cost_critic=3e-4,
        gamma=0.99,
        cost_gamma=0.99,
        gae_lambda=0.95,
        epochs=10,
        clip_epsilon=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        cost_value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        log_std_init=0.0,
        cost_limit=5.0,
        lambda_lr=0.05,
        lambda_init=0.0,
        lambda_max=100.0,
        clip_value_loss=True,
        run_device=None,
    ):
        self.device = run_device or device
        self.GAMMA = gamma
        self.COST_GAMMA = cost_gamma
        self.LAMBDA = gae_lambda
        self.EPOCH = epochs
        self.EPSILON_CLIP = clip_epsilon
        self.ENTROPY_COEF = entropy_coef
        self.VF_COEF = value_coef
        self.COST_VF_COEF = cost_value_coef
        self.MAX_GRAD_NORM = max_grad_norm
        self.TARGET_KL = target_kl
        self.CLIP_VALUE_LOSS = clip_value_loss
        self.COST_LIMIT = cost_limit
        self.LAMBDA_LR = lambda_lr
        self.LAMBDA_MAX = lambda_max
        self.lagrange_lambda = float(lambda_init)

        self.actor = Actor(state_dim, action_dim, hidden_dim, log_std_init).to(self.device)
        self.reward_critic = Critic(state_dim, hidden_dim).to(self.device)
        self.cost_critic = Critic(state_dim, hidden_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.reward_critic_optim = optim.Adam(self.reward_critic.parameters(), lr=lr_critic)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr_cost_critic)
        self.buffer = CostReplayMemory(batch_size)

        low = np.full(action_dim, -1.0, dtype=np.float32) if action_low is None else np.asarray(action_low, dtype=np.float32)
        high = np.full(action_dim, 1.0, dtype=np.float32) if action_high is None else np.asarray(action_high, dtype=np.float32)
        self.action_low = _to_tensor(low, self.device)
        self.action_high = _to_tensor(high, self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

    def _scale_action(self, norm_action):
        clipped = torch.clamp(norm_action, -1.0, 1.0)
        return clipped * self.action_scale + self.action_bias

    @staticmethod
    def _log_prob(dist, norm_action):
        return dist.log_prob(norm_action).sum(dim=-1)

    @torch.no_grad()
    def get_action(self, state, deterministic=False):
        s = _to_tensor(state, self.device).unsqueeze(0)
        dist = self.actor.get_dist(s)
        norm_action = dist.mean if deterministic else dist.sample()
        action = self._scale_action(norm_action)
        log_prob = self._log_prob(dist, norm_action)
        value = self.reward_critic(s).squeeze(-1)
        cost_value = self.cost_critic(s).squeeze(-1)
        return (
            action.cpu().numpy()[0],
            norm_action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            cost_value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
        )

    @torch.no_grad()
    def get_values(self, state):
        s = _to_tensor(state, self.device).unsqueeze(0)
        return (
            self.reward_critic(s).squeeze(-1).cpu().numpy()[0],
            self.cost_critic(s).squeeze(-1).cpu().numpy()[0],
        )

    def update_lagrange(self, episode_cost):
        violation = float(episode_cost) - self.COST_LIMIT
        self.lagrange_lambda = float(np.clip(self.lagrange_lambda + self.LAMBDA_LR * violation, 0.0, self.LAMBDA_MAX))
        return violation

    def _compute_gae(self, rewards, values, dones, last_value, gamma):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            gae = delta + gamma * self.LAMBDA * non_terminal * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def _critic_loss(self, pred, target, old_value):
        if self.CLIP_VALUE_LOSS:
            clipped = old_value + torch.clamp(pred - old_value, -self.EPSILON_CLIP, self.EPSILON_CLIP)
            loss = (pred - target).pow(2)
            clipped_loss = (clipped - target).pow(2)
            return 0.5 * torch.max(loss, clipped_loss).mean()
        return 0.5 * (pred - target).pow(2).mean()

    def update(self, last_reward_value, last_cost_value):
        (
            states,
            norm_actions,
            _,
            rewards,
            costs,
            values,
            cost_values,
            old_log_probs,
            dones,
        ) = self.buffer.as_numpy()
        reward_adv, reward_returns = self._compute_gae(rewards, values, dones, last_reward_value, self.GAMMA)
        cost_adv, cost_returns = self._compute_gae(costs, cost_values, dones, last_cost_value, self.COST_GAMMA)

        reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1e-8)
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)
        penalty_adv = (reward_adv - self.lagrange_lambda * cost_adv) / (1.0 + self.lagrange_lambda)

        states_t = _to_tensor(states, self.device)
        norm_actions_t = _to_tensor(norm_actions, self.device)
        old_logp_t = _to_tensor(old_log_probs, self.device)
        adv_t = _to_tensor(penalty_adv, self.device)
        reward_returns_t = _to_tensor(reward_returns, self.device)
        cost_returns_t = _to_tensor(cost_returns, self.device)
        values_t = _to_tensor(values, self.device)
        cost_values_t = _to_tensor(cost_values, self.device)

        T = len(states)
        update_info = {}
        for _ in range(self.EPOCH):
            approx_kl_epoch = 0.0
            for batch in self.buffer.iter_batches(T):
                dist = self.actor.get_dist(states_t[batch])
                logp = self._log_prob(dist, norm_actions_t[batch])
                entropy = dist.entropy().sum(dim=-1).mean()
                logratio = logp - old_logp_t[batch]
                ratio = torch.exp(logratio)
                s1 = ratio * adv_t[batch]
                s2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * adv_t[batch]
                actor_loss = -torch.min(s1, s2).mean() - self.ENTROPY_COEF * entropy

                reward_pred = self.reward_critic(states_t[batch]).squeeze(-1)
                reward_loss = self._critic_loss(reward_pred, reward_returns_t[batch], values_t[batch])
                cost_pred = self.cost_critic(states_t[batch]).squeeze(-1)
                cost_loss = self._critic_loss(cost_pred, cost_returns_t[batch], cost_values_t[batch])

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.MAX_GRAD_NORM)
                self.actor_optim.step()

                self.reward_critic_optim.zero_grad()
                (self.VF_COEF * reward_loss).backward()
                nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.MAX_GRAD_NORM)
                self.reward_critic_optim.step()

                self.cost_critic_optim.zero_grad()
                (self.COST_VF_COEF * cost_loss).backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.MAX_GRAD_NORM)
                self.cost_critic_optim.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    approx_kl_epoch = max(approx_kl_epoch, approx_kl)
                    clip_fraction = ((ratio - 1.0).abs() > self.EPSILON_CLIP).float().mean().item()

                update_info = {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "reward_critic_loss": float(reward_loss.detach().cpu()),
                    "cost_critic_loss": float(cost_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approx_kl": approx_kl_epoch,
                    "clip_fraction": clip_fraction,
                    "lambda": self.lagrange_lambda,
                    "mean_rollout_cost": float(np.mean(costs)),
                }
            if self.TARGET_KL and approx_kl_epoch > self.TARGET_KL:
                break

        self.buffer.clear()
        return update_info

    def save_policy(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "reward_critic": self.reward_critic.state_dict(),
                "cost_critic": self.cost_critic.state_dict(),
                "lambda": self.lagrange_lambda,
                "action_low": self.action_low.detach().cpu().numpy(),
                "action_high": self.action_high.detach().cpu().numpy(),
            },
            path,
        )
