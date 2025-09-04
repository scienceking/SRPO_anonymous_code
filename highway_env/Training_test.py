import ray
from ray.rllib.algorithms.ppo import PPOConfig
from register_env1 import register_custom_env
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import csv

# Initialize Ray
ray.init()

# Register the custom environment
register_custom_env()

# Configure PPO
config = (
    PPOConfig()
    .environment("CustomEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=1)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        grad_clip=0.2
    )
    .training(model={
        "fcnet_hiddens": [512, 512, 512],
        "fcnet_activation": "tanh",
    })
)

# Build and restore trainer

#3000, 3005 3105, 3320
trainer = config.build()
checkpoint_path = "ppo_model_checkpoint/ppo_model_6700"
trainer.restore(checkpoint_path)

# Use native gym env for rendering
env = gym.make("CustomEnv", render_mode="rgb_array")
obs, _ = env.reset()

reward_set = []
num_episodes =100
total_rewards = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    episode_reward_set = []

    while not done:
        episode_reward_set.append(total_reward)

        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
        else:
            obs_tensor = torch.tensor(obs[0], dtype=torch.float32)

        action = trainer.compute_single_action(obs_tensor.numpy(), explore=False)
        obs, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        total_reward += reward


    total_rewards.append(total_reward)
    reward_set.append(episode_reward_set)
    print(f"Episode {episode + 1}: Total reward = {total_reward}")



# Output stats
print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
print(f"Max reward over {num_episodes} episodes: {np.max(total_rewards)}")
print(f"Min reward over {num_episodes} episodes: {np.min(total_rewards)}")

# Save to CSV
with open('PPO_reward_env_g.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for episode_index, episode in enumerate(reward_set):
        writer.writerow(['Episode ' + str(episode_index)])
        for reward in episode:
            writer.writerow([reward])

# Shutdown Ray
ray.shutdown()
