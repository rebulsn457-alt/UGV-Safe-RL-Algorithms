import gym
import torch
import time
import os
from ppo_agent import PPOAgent

# 1. 基础配置 (对应视频截图)
scenario = "Pendulum-v1"
env = gym.make(scenario)
current_path = os.path.dirname(os.path.realpath(__file__))
model_path = current_path + "/models/"
if not os.path.exists(model_path): os.makedirs(model_path)
timestamp = time.strftime("%Y%m%d%H%M%S")

# 2. 参数设置
NUM_EPISODE, NUM_STEP = 3000, 200
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BATCH_SIZE, UPDATE_INTERVAL = 25, 50

agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)
best_reward = -2000

# 3. 训练循环
for episode_i in range(NUM_EPISODE):
    state, info = env.reset() # Gym 0.26.2 返回两个值
    episode_reward = 0
    
    for step_i in range(NUM_STEP):
        action, value = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 存入 buffer
        agent.replay_buffer.add_memo(state, action, reward, value, done)
        
        episode_reward += reward
        state = next_state
        
        # 定期更新
        if (step_i + 1) % UPDATE_INTERVAL == 0:
            agent.update()
        if done: break
            
    # 保存最优模型
    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(agent.actor.state_dict(), model_path + f"ppo_actor_{timestamp}.pth")
        
    print(f"Episode: {episode_i}, Reward: {round(episode_reward, 2)}")