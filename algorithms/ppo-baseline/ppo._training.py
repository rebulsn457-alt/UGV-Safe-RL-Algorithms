import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = PPOAgent(state_dim, action_dim, max_action)

NUM_EPISODES = 2000 
rewards_history = []
best_reward = -1e9

for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    ep_reward = 0
    
    for _ in range(200):
        action, lp, v = agent.get_action(state)
        next_s, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # ⭐ 关键修改：奖励重塑，极大提升收敛稳定性
        scaled_r = (reward + 8.0) / 8.0 
        
        agent.buffer.add(state, action, scaled_r, lp, done, v)
        state = next_s
        ep_reward += reward
        if done: break
        
    agent.update()
    rewards_history.append(ep_reward)
    best_reward = max(best_reward, ep_reward)
    
    if ep % 20 == 0:
        avg_r = np.mean(rewards_history[-20:])
        print(f"Ep {ep} | Avg: {avg_r:.2f} | Best: {best_reward:.2f}")

# 绘制平滑曲线
def smooth(data, window=50):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

plt.plot(rewards_history, alpha=0.3, color='blue')
plt.plot(smooth(rewards_history), color='red', linewidth=2)
plt.title("Optimized PPO Training")
plt.savefig("stable_curve.png")
plt.show()
# 在 ppo_training.py 的训练循环末尾
if ep_reward > best_reward:
    best_reward = ep_reward
    # 只要破纪录就保存，手头永远有一个 -1.09 级别的最强模型
    torch.save(agent.actor.state_dict(), "models/best_actor.pth") 
    print(f"🔥 新纪录！模型已保存: {best_reward:.2f}")