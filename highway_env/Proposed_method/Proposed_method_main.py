from register_env import register_custom_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray
ray.init()
register_custom_env()
from ray.rllib.algorithms.ppo import PPOConfig
import ray
import os
import numpy as np
import re
import ast
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import torch

import numpy as np
import ast
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

        # === Step 1: read historical return ===
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

        # === Step 2: calculate reward shaping bonus ===
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
        #     modified_rewards[mask] = 1*(traj_rewards*0 + bonus)

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

        # === Step 3: recalculate GAE and value target ===
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


config = (
    PPOConfig()
    .environment("CustomEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=40)
    .resources(num_gpus=1)
    .callbacks(RewardShapingCallbacks)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=20,
        lr=5e-6,
        gamma=0.9,
        lambda_=0.95,
        clip_param=0.2,
        grad_clip=1,
        model={
            "fcnet_hiddens": [512,512, 512],
            "fcnet_activation": "tanh",
        }

    )
)

log_dir = "logs/ppo_custom_env"
os.makedirs(log_dir, exist_ok=True)

# checkpoint_path = "ppo_model_checkpoint/ppo_model_1025"

trainer = config.build()
# trainer.restore(checkpoint_path)



reward_list = []

for i in range(10000):
    with open("iteration.txt", "w") as f:
        f.write(str(i))

    result = trainer.train()
    reward_mean = result["episode_reward_mean"]
    reward_list.append(reward_mean)


    print(f"Iteration {i}: reward_mean = {reward_mean}")

    if i % 60 == 0:
        should_update = True
        try:
            with open("reward.txt", "r") as f:
                line = f.readline().strip()
                match = re.match(r"Iteration \d+: ([\-\d\.eE]+)", line)
                if match:
                    previous_reward = float(match.group(1))
                    if reward_mean <= previous_reward:
                        should_update = False
        except FileNotFoundError:
            should_update = True
        except Exception as e:
            print(f"[Warning] Failed to read reward.txt: {e}")
            should_update = True

        if should_update:
            with open("reward.txt", "w") as f:  # "w"
                f.write(f"Iteration {i}: {reward_mean}\n")
                f.flush()

    with open("reward_log_shape.txt", "w") as reward_file:
        reward_file.write(f"Iteration {i}: {reward_list}\n")

    #
    if i % 5 == 0:
        checkpoint_dir = "ppo_model_checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"ppo_model_{i}")
        trainer.save(checkpoint_path)

ray.shutdown()
