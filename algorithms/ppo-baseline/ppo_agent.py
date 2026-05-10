import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.optim as optim
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # state-independent log_std is a common and stable choice for PPO on
        # continuous-action tasks; it also prevents the std collapsing to 0.
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
 
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x)) * 2.0  # Pendulum action in [-2, 2]
        std = self.log_std.exp().expand_as(mean)
        return mean, std
 
    def get_dist(self, s):
        mu, sigma = self.forward(s)
        return Normal(mu, sigma)
 
 
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
 
 
class ReplayMemory:
    """Stores one rollout worth of transitions plus the cached log-probs."""
 
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear()
 
    def add(self, s, a, r, v, logp, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.log_probs.append(logp)
        self.dones.append(done)
 
    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
 
    def as_numpy(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
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
    def __init__(self, state_dim, action_dim, batch_size):
        self.LR_ACTOR = 3e-4
        self.LR_CRITIC = 1e-3
        self.GAMMA = 0.9
        self.LAMBDA = 0.95
        self.EPOCH = 10
        self.EPSILON_CLIP = 0.2
        self.ENTROPY_COEF = 0.0
        self.VF_COEF = 0.5
        self.MAX_GRAD_NORM = 0.5
 
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.buffer = ReplayMemory(batch_size)
 
    # ------------------------------------------------------------------ sample
    @torch.no_grad()
    def get_action(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        dist = self.actor.get_dist(s)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(s).squeeze(-1)
        action = action.clamp(-2.0, 2.0)
        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
        )
 
    @torch.no_grad()
    def get_value(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
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
        states, actions, rewards, values, old_log_probs, dones = self.buffer.as_numpy()
        adv, returns = self._compute_gae(rewards, values, dones, last_value)
 
        # Advantage normalization is a well-known stabilizer.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
 
        states_t = torch.as_tensor(states, device=device)
        actions_t = torch.as_tensor(actions, device=device)
        old_logp_t = torch.as_tensor(old_log_probs, device=device)
        adv_t = torch.as_tensor(adv, device=device)
        returns_t = torch.as_tensor(returns, device=device)
 
        T = len(states)
        for _ in range(self.EPOCH):
            for batch in self.buffer.iter_batches(T):
                dist = self.actor.get_dist(states_t[batch])
                logp = dist.log_prob(actions_t[batch]).sum(dim=-1)
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
 
        self.buffer.clear()
 
    def save_policy(self, path):
        torch.save(self.actor.state_dict(), path)
