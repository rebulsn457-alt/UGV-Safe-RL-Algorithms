import os
import time
import numpy as np
import torch
import gym
from tensorboardX import SummaryWriter
 
from ppo_agent import PPOAgent
 
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
 
scenario = "Pendulum-v1"
env = gym.make(id=scenario)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
 
NUM_EPISODE = 400
NUM_STEP = 200                    # Pendulum-v1 max episode length
EPISODES_PER_UPDATE = 5           # collect 5 * 200 = 1000 steps per update
BATCH_SIZE = 64
REWARD_SCALE = 1.0 / 8.0          # maps reward from [-16.27, 0] to ~[-2, 0]
 
agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)
 
current_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_path, "models")
os.makedirs(model_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")
writer = SummaryWriter("fixed_logs")
 
REWARD_BUFFER = np.empty(NUM_EPISODE)
best_reward = -np.inf
 
episode_in_batch = 0
for episode_i in range(NUM_EPISODE):
    state, _ = env.reset(seed=SEED + episode_i)
    episode_reward = 0.0
 
    for step_i in range(NUM_STEP):
        action, value, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
 
        # Pendulum-v1 has no real terminal. `truncated` signals the time limit
        # and we should bootstrap with V(next_state), so store done=0.
        done_for_buffer = 1.0 if terminated else 0.0
        agent.buffer.add(
            state, action, reward * REWARD_SCALE, value, log_prob, done_for_buffer,
        )
        state = next_state
        if terminated or truncated:
            break
 
    episode_in_batch += 1
    REWARD_BUFFER[episode_i] = episode_reward
    writer.add_scalar("Episode reward", episode_reward, episode_i)
 
    if episode_in_batch >= EPISODES_PER_UPDATE:
        # Bootstrap with the value of the last next_state (Pendulum never
        # terminates, so this is always meaningful).
        last_value = agent.get_value(state)
        agent.update(last_value=last_value)
        episode_in_batch = 0
 
    if (episode_i + 1) % 20 == 0:
        recent = REWARD_BUFFER[max(0, episode_i - 19):episode_i + 1]
        print(
            f"[FIXED] Ep {episode_i + 1}/{NUM_EPISODE} "
            f"last20_mean={recent.mean():.2f} "
            f"last20_max={recent.max():.2f}"
        )
 
    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy(os.path.join(model_dir, f"ppo_actor_fixed_{timestamp}.pth"))
 
env.close()
np.savetxt(os.path.join(current_path, f"fixed_reward_{timestamp}.txt"), REWARD_BUFFER)
print("Fixed run done. Last 20 mean:", REWARD_BUFFER[-20:].mean(),
      "best:", best_reward)
