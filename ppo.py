import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import ale_py
import numpy as np

# Custom reward shaping
def reward_shaping(original_reward, info):
    # If a certain condition is met, add additional reward
    if 'win' in info and info['win']:
        shaped_reward = original_reward + 10 # Added bonus for winning
    else:
        shaped_reward = original_reward
    return shaped_reward

# Wrap environment to include reward shaping
class RewardShapeWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward_shaping(reward, self.env.unwrapped.ale.getRAM())
# Create environment
# A I am testing the PPO model I am using make_vec_env instead of gym.make with render as it is more useful for training purposes 
env = make_vec_env('ALE/Backgammon-v5', n_envs=8, seed=42, wrapper_class=RewardShapeWrapper)
        # n_envs tested: 2,4,8

# Define neural network policy
# Using a Convolutional Neural Network (CNN) policy as the Backgammon environment returns images
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_backgammon_tensorboard/", learning_rate=0.0003)
        # learning_rate tested: 0.0001, 0.0003, 0.0005

# Train the model
model.learn(total_timesteps=10000)
model.save("ppo_backgammon")

# Evaluation function
def evaluate_model(model, env, num_steps):
    obs = env.reset()
    total_reward = 0.0
    num_episodes = 0
    
    for step in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards

        # Reset environment if done
        if np.any(dones):
            num_episodes += 1
            print(f"Episode {num_episodes} finished with reward: {total_reward}")
            total_reward = 0.0
            obs = env.reset()

    env.close()

# Evaluate the model
evaluate_model(model, env, 10000)

#PPO11 0.0001 timestep 10000 envs 8
#PPO10 0.0001 timestep 10000 envs 4
#PPO12 0.0001 timestep 50000 envs 8
#PPO13 0.0003 timestep 10000 envs 2
#PPO15 0.0003 timestep 10000 envs 4
#PPO16 0.0003 timestep 10000 envs 8
