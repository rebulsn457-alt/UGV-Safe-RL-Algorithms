import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.optim as optim
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x, target_device):
    return torch.as_tensor(x, dtype=torch.float32, device=target_device)
 
 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, log_std_init=-0.5):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # A state-independent log_std is stable for small continuous-control
        # tasks and avoids early variance collapse.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.apply(self._orthogonal_init)
        nn.init.orthogonal_(self.fc_mean.weight, 0.01)
        nn.init.constant_(self.fc_mean.bias, 0.0)

    @staticmethod
    def _orthogonal_init(module):
        if isinstance(module, nn.Linear):
            gain = np.sqrt(2.0)
            if module.out_features == 1:
                gain = 1.0
            nn.init.orthogonal_(module.weight, gain)
            nn.init.constant_(module.bias, 0.0)
 
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std
 
    def get_dist(self, s):
        mu, sigma = self.forward(s)
        return Normal(mu, sigma)
 
 
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.apply(self._orthogonal_init)
        nn.init.orthogonal_(self.fc3.weight, 1.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    @staticmethod
    def _orthogonal_init(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2.0))
            nn.init.constant_(module.bias, 0.0)
 
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
 
 
class ReplayMemory:
    """Stores one rollout worth of transitions plus the cached log-probs."""
 
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear()
 
    def add(self, s, raw_a, env_a, r, v, logp, done):
        self.states.append(s)
        self.raw_actions.append(raw_a)
        self.env_actions.append(env_a)
        self.rewards.append(r)
        self.values.append(v)
        self.log_probs.append(logp)
        self.dones.append(done)
 
    def clear(self):
        self.states, self.raw_actions, self.env_actions, self.rewards = [], [], [], []
        self.values, self.log_probs, self.dones = [], [], []

    def __len__(self):
        return len(self.states)
 
    def as_numpy(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.raw_actions, dtype=np.float32),
            np.array(self.env_actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
        )
 
    def iter_batches(self, n):
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, self.batch_size):
            yield idx[start:start + self.batch_size]
 
 
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        batch_size,
        action_low=None,
        action_high=None,
        hidden_dim=128,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        epochs=10,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        log_std_init=-0.5,
        run_device=None,
    ):
        self.device = run_device or device
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.GAMMA = gamma
        self.LAMBDA = gae_lambda
        self.EPOCH = epochs
        self.EPSILON_CLIP = clip_epsilon
        self.ENTROPY_COEF = entropy_coef
        self.VF_COEF = value_coef
        self.MAX_GRAD_NORM = max_grad_norm
        self.TARGET_KL = target_kl

        self.actor = Actor(state_dim, action_dim, hidden_dim, log_std_init).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.buffer = ReplayMemory(batch_size)
        low = np.full(action_dim, -1.0, dtype=np.float32) if action_low is None else np.asarray(action_low, dtype=np.float32)
        high = np.full(action_dim, 1.0, dtype=np.float32) if action_high is None else np.asarray(action_high, dtype=np.float32)
        self.action_low = _to_tensor(low, self.device)
        self.action_high = _to_tensor(high, self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        self._eps = 1e-6

    def _squash_action(self, raw_action):
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def _log_prob_from_raw_action(self, dist, raw_action):
        squashed = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + self._eps)
        return log_prob.sum(dim=-1)
 
    # ------------------------------------------------------------------ sample
    @torch.no_grad()
    def get_action(self, state, deterministic=False):
        s = _to_tensor(state, self.device).unsqueeze(0)
        dist = self.actor.get_dist(s)
        raw_action = dist.mean if deterministic else dist.rsample()
        action = self._squash_action(raw_action)
        log_prob = self._log_prob_from_raw_action(dist, raw_action)
        value = self.critic(s).squeeze(-1)
        return (
            action.cpu().numpy()[0],
            raw_action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
        )
 
    @torch.no_grad()
    def get_value(self, state):
        s = _to_tensor(state, self.device).unsqueeze(0)
        return self.critic(s).squeeze(-1).cpu().numpy()[0]
 
    # ------------------------------------------------------------------ update
    def _compute_gae(self, rewards, values, dones, last_value):
        """Standard GAE-lambda, reverse recursion. `dones[t]` marks whether
        step t ends an episode due to a real terminal (bootstrap cut)."""
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.GAMMA * next_value * non_terminal - values[t]
            gae = delta + self.GAMMA * self.LAMBDA * non_terminal * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + values
        return adv, returns
 
    def update(self, last_value):
        states, raw_actions, _, rewards, values, old_log_probs, dones = self.buffer.as_numpy()
        adv, returns = self._compute_gae(rewards, values, dones, last_value)
 
        # Advantage normalization is a well-known stabilizer.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
 
        states_t = _to_tensor(states, self.device)
        raw_actions_t = _to_tensor(raw_actions, self.device)
        old_logp_t = _to_tensor(old_log_probs, self.device)
        adv_t = _to_tensor(adv, self.device)
        returns_t = _to_tensor(returns, self.device)
 
        T = len(states)
        update_info = {}
        for _ in range(self.EPOCH):
            approx_kl_epoch = 0.0
            for batch in self.buffer.iter_batches(T):
                dist = self.actor.get_dist(states_t[batch])
                logp = self._log_prob_from_raw_action(dist, raw_actions_t[batch])
                entropy = dist.entropy().sum(dim=-1).mean()
 
                ratio = torch.exp(logp - old_logp_t[batch])
                s1 = ratio * adv_t[batch]
                s2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * adv_t[batch]
                actor_loss = -torch.min(s1, s2).mean() - self.ENTROPY_COEF * entropy
 
                value_pred = self.critic(states_t[batch]).squeeze(-1)
                critic_loss = nn.MSELoss()(value_pred, returns_t[batch])
 
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.MAX_GRAD_NORM)
                self.actor_optim.step()
 
                self.critic_optim.zero_grad()
                (self.VF_COEF * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.MAX_GRAD_NORM)
                self.critic_optim.step()

                with torch.no_grad():
                    approx_kl_epoch = max(
                        approx_kl_epoch,
                        (old_logp_t[batch] - logp).mean().abs().item(),
                    )

                update_info = {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approx_kl": approx_kl_epoch,
                }

            if self.TARGET_KL and approx_kl_epoch > self.TARGET_KL:
                break
 
        self.buffer.clear()
        return update_info
 
    def save_policy(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "action_low": self.action_low.detach().cpu().numpy(),
                "action_high": self.action_high.detach().cpu().numpy(),
            },
            path,
        )
