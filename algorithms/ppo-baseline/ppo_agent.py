import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        # ⭐ 优化：连续控制务必使用 Tanh
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.mean = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh() 
        )
        self.max_action = max_action
        
        # ⭐ 关键修改：降低初始探索噪声，从 0.0 改为 -0.5
        # 这会让 AI 的“手脚”更稳，不再乱抖
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        x = self.net(x)
        mean = self.max_action * self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

class Buffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.dones, self.values = [], [], []

    def add(self, s, a, r, lp, d, v):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.log_probs.append(lp); self.dones.append(d); self.values.append(v)

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.dones, self.values = [], [], []

class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # 分开设置学习率，Critic 可以学得稍快一点
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': 3e-4},
            {'params': self.critic.parameters(), 'lr': 1e-3}
        ])

        self.buffer = Buffer()
        self.gamma, self.lam, self.clip = 0.99, 0.95, 0.2
        self.epochs, self.batch_size = 10, 64

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            mean, std = self.actor(state)
            value = self.critic(state)
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        advantages, gae = [], 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self):
        states = torch.FloatTensor(self.buffer.states).to(device)
        actions = torch.FloatTensor(self.buffer.actions).to(device)
        old_lp = torch.FloatTensor(self.buffer.log_probs).to(device)
        
        advs, rets = self.compute_gae(self.buffer.rewards, self.buffer.values, self.buffer.dones)
        advs = torch.FloatTensor(advs).to(device)
        rets = torch.FloatTensor(rets).to(device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(self.epochs):
            indices = torch.randperm(states.size(0))
            for i in range(0, states.size(0), self.batch_size):
                idx = indices[i:i+self.batch_size]
                
                m, s = self.actor(states[idx])
                dist = Normal(m, s)
                new_lp = dist.log_prob(actions[idx]).sum(dim=1)
                entropy = dist.entropy().sum(dim=1)

                ratio = torch.exp(new_lp - old_lp[idx])
                surr1 = ratio * advs[idx]
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advs[idx]

                # ⭐ 关键修改：降低熵系数到 0.001
                a_loss = -torch.min(surr1, surr2).mean()
                v_loss = (rets[idx] - self.critic(states[idx]).squeeze()).pow(2).mean()
                loss = a_loss + 0.5 * v_loss - 0.001 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：稳住更新步长
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()
        self.buffer.clear()