"""
Train SAC model-free baseline directly on the BioSim simulator.

Usage:
    python -m biosim.train_sac_modelfree --timesteps 60000 --save data/policies/sac_modelfree
"""

import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from biosim.env import BioSimEnv


class RewardLogger(BaseCallback):
    """Log episode rewards for comparison plot."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_steps = []
        self._current_reward = 0
        self._current_steps = 0

    def _on_step(self):
        self._current_reward += self.locals.get("rewards", [0])[0]
        self._current_steps += 1
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_steps.append(self.num_timesteps)
            self._current_reward = 0
            self._current_steps = 0
        return True


def train_modelfree(
    total_timesteps: int = 60000,
    save_path: str = "data/policies/sac_modelfree",
    seed: int = 42,
):
    env = BioSimEnv()

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=1,
        seed=seed,
    )

    logger = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=logger)
    model.save(save_path)

    # Save training curve
    np.savez(
        save_path + "_curve.npz",
        rewards=np.array(logger.episode_rewards),
        steps=np.array(logger.episode_steps),
    )

    print(f"\nModel-free SAC trained for {total_timesteps} steps")
    print(f"Episodes completed: {len(logger.episode_rewards)}")
    if logger.episode_rewards:
        print(f"Final 10 episode mean reward: {np.mean(logger.episode_rewards[-10:]):.3f}")
    print(f"Saved to: {save_path}")

    return model, logger


def evaluate_policy(model_path: str, num_episodes: int = 50):
    """Evaluate a trained policy on the real simulator."""
    env = BioSimEnv()
    model = SAC.load(model_path)

    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    print(f"\nEvaluation ({num_episodes} episodes):")
    print(f"  Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f}")
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--save", default="data/policies/sac_modelfree")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_policy(args.save)
    else:
        train_modelfree(args.timesteps, args.save, args.seed)
        evaluate_policy(args.save)
