import ray
from ray.rllib.algorithms.ppo import PPOConfig
from register_env import register_custom_env
import numpy as np
import torch

# Initialize Ray
ray.init()

# Register the custom environment
register_custom_env()
import re
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages

class RewardShapingCallbacks(DefaultCallbacks):
    def __init__(self, alpha=2, lambd=3):
        super().__init__()
        self.alpha = alpha
        self.lambd = lambd

    def on_postprocess_trajectory(self, *, worker, episode, agent_id,
                                  policy_id, policies, postprocessed_batch,
                                  original_batches, **kwargs):
        rewards = postprocessed_batch[SampleBatch.REWARDS]
        episode_ids = postprocessed_batch["eps_id"]

        # === Step 1: 读取历史回报 ===
        all_past_rewards = []
        try:
            with open("reward.txt", "r") as f:
                for line in f:
                    match = re.match(r"Iteration (\d+): ([\-\d\.eE]+)", line.strip())
                    if match:
                        reward_value = float(match.group(2))
                        all_past_rewards.append(reward_value)
        except Exception as e:
            return

        if not all_past_rewards:
            return

        J_ref_mean = float(np.mean(all_past_rewards))


        if J_ref_mean == 0:
            print("[Warning] J_ref_mean is zero. Skipping shaping.")
            return

        # === Step 2: 计算 reward shaping bonus ===
        rewards = np.array(rewards, dtype=np.float32)
        episode_ids = np.array(episode_ids)
        modified_rewards = rewards.copy()

        unique_eps_ids = np.unique(episode_ids)
        # for eps_id in unique_eps_ids:
        #     mask = episode_ids == eps_id
        #     traj_rewards = rewards[mask]
        #     J_pi_t = float(np.sum(traj_rewards))
        #     delta = self.alpha * (J_pi_t - J_ref_mean) / abs(J_ref_mean)
        #     bonus = self.lambd * delta
        #     modified_rewards[mask] = 1*(traj_rewards*0 + bonus)  # 可调成 traj_rewards + bonus*ones

        for eps_id in unique_eps_ids:
            mask = episode_ids == eps_id
            traj_rewards = rewards[mask]
            J_pi_t = float(np.sum(traj_rewards))
            delta = self.alpha * (J_pi_t - J_ref_mean) / abs(J_ref_mean)
            bonus = self.lambd * delta

            N = len(traj_rewards)


            if bonus >= 0:
                ranks = np.argsort(np.argsort(-traj_rewards))
            else:
                ranks = np.argsort(np.argsort(traj_rewards))


            weights = (N - ranks).astype(np.float32)
            weights /= np.sum(weights)


            bonus_distribution = bonus * weights
            modified_rewards[mask] = traj_rewards + bonus_distribution


        postprocessed_batch[SampleBatch.REWARDS] = modified_rewards

        policy = policies[policy_id]

        compute_advantages(
            rollout=postprocessed_batch,
            last_r=postprocessed_batch[SampleBatch.VALUES_BOOTSTRAPPED][-1],
            gamma=policy.config["gamma"],
            lambda_=policy.config["lambda"],
            use_gae=policy.config["use_gae"],
            use_critic=policy.config.get("use_critic", True),
            vf_preds=postprocessed_batch[SampleBatch.VF_PREDS],
            rewards=postprocessed_batch[SampleBatch.REWARDS],
        )
# Configure the PPO algorithm, ensure it matches the training setup
config = (
    PPOConfig()
    .environment("CustomEnv")  # Name of the registered environment
    .framework("torch")        # Use PyTorch
    .rollouts(num_rollout_workers=1)  # Single worker for rollouts
    .resources(num_gpus=1)  # Use GPU if available
    .callbacks(RewardShapingCallbacks)
    .training(
        train_batch_size=4000,  # Total batch size for training
        sgd_minibatch_size=128,  # Minibatch size for SGD
        num_sgd_iter=10,  # Number of SGD iterations per batch
        lr=1e-4,  # Learning rate (1e-5 is smaller than 1e-4)
        gamma=0.99,  # Discount factor
        lambda_=0.95,  # Lambda for GAE
        clip_param=0.2,  # PPO clipping parameter
        grad_clip=0.2  # Gradient clipping value
    )
    .training(model={
        "fcnet_hiddens": [256, 256, 256],  # Three fully connected layers
        "fcnet_activation": "tanh",        # Activation function for all layers
    })
)

# Build the trainer
trainer = config.build()

# Load the trained model's weights
checkpoint_path = "ppo_model_checkpoint/ppo_model1_375"
trainer.restore(checkpoint_path)

# Create a test environment
env = trainer.env_creator({})  # Create an instance of the environment
reward_set = []
# Perform 100 test episodes
num_episodes = 50
total_rewards = []  # Record the total reward for each test episode


for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    episode_reward = 0
    episode_reward_set = []

    while not done:
        episode_reward_set.append(total_reward)
        # Extract the first element of the tuple and convert to tensor if necessary
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the array part
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)  # Convert to a tensor

        # Use the trained model to compute the action
        action = trainer.compute_single_action(
            obs.numpy() if isinstance(obs, torch.Tensor) else obs, explore=False
        )

        # Interact with the environment
        result = env.step(action)

        # Unpack the environment's return values
        obs, reward, done, truncated, info = result
        #print(obs)
        done = done or truncated  # Combine `done` and `truncated`

        # Accumulate the reward
        total_reward += reward

    # Record the total reward for this episode
    total_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total reward = {total_reward}")

    reward_set.append(episode_reward_set)
# Output statistical results
print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
print(f"Max reward over {num_episodes} episodes: {np.max(total_rewards)}")
print(f"Min reward over {num_episodes} episodes: {np.min(total_rewards)}")

# Shutdown Ray
ray.shutdown()
import csv
with open('PPO_reward_env_123.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for episode_index, episode in enumerate(reward_set):
        writer.writerow(['Episode ' + str(episode_index)])
        for reward in episode:
            writer.writerow([reward])