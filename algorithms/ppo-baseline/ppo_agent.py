import torch
import torch.nn as nn

class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer()
        
        # 你的神经网络定义
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 【重点：添加以下这两行】
        import torch.optim as optim
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def get_action(self, state):
        # 视频第 18 行逻辑
        s = torch.tensor(state, dtype=torch.float)
        return self.actor(s).detach().numpy()*2, self.critic(s).item()

    def update(self):
        # 1. 将 buffer 里的数据转为 Tensor
        states = torch.tensor(self.replay_buffer.states, dtype=torch.float)
        actions = torch.tensor(self.replay_buffer.actions, dtype=torch.float)
        rewards = torch.tensor(self.replay_buffer.rewards, dtype=torch.float)
        old_values = torch.tensor(self.replay_buffer.values, dtype=torch.float)
        dones = torch.tensor(self.replay_buffer.dones, dtype=torch.float)

        # 2. 计算优势函数 (GAE 简化版)
        advantages = []
        gae = 0
        with torch.no_grad():
            # 计算 TD 误差
            next_value = 0 # 简化处理
            for r, v, d in zip(reversed(rewards), reversed(old_values), reversed(dones)):
                td_error = r + 0.9 * next_value * (1 - d) - v
                gae = td_error + 0.9 * 0.95 * gae * (1 - d)
                advantages.insert(0, gae)
                next_value = v
        advantages = torch.tensor(advantages, dtype=torch.float)

        # 3. PPO 核心：Clipping 更新 (重复训练 10 次)
        for _ in range(10):
            # 获取当前策略的动作概率 (这里简化为均值回归)
            curr_values = self.critic(states).squeeze()
            
            # 计算 Critic Loss (均方误差)
            critic_loss = nn.MSELoss()(curr_values, rewards + 0.9 * old_values)
            
            # 更新网络
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            
        self.replay_buffer.clear()

class ReplayBuffer:
    def __init__(self):
        # 必须定义这几个名字，update 函数才能找到它们
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add_memo(self, state, action, reward, value, done):
        # 存入数据
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        # 清空数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []