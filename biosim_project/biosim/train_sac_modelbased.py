"""
Model-based SAC with DAgger-style distribution shift correction.

The headline experiment:
1. Train SAC inside the World Model's imagination
2. Every N dream-steps, refresh the World Model with real-simulator transitions
3. Compare sample efficiency and policy transfer vs model-free baseline

Usage:
    python -m biosim.train_sac_modelbased \
        --world-model data/world_model.pt \
        --buffer data/replay_buffer.npz \
        --save data/policies/sac_modelbased
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

from biosim.env import BioSimEnv
from biosim.world_model import WorldModel
from biosim.train_world_model import train_world_model


class WorldModelEnv(gym.Env):
    """
    Wraps the trained World Model as a Gymnasium environment.
    The RL agent interacts with this instead of the real simulator.
    """

    def __init__(self, world_model: WorldModel, real_env: BioSimEnv, device="cpu"):
        super().__init__()
        self.wm = world_model
        self.real_env = real_env
        self.device = device

        self.observation_space = real_env.observation_space
        self.action_space = real_env.action_space

        self._current_z = None
        self._current_state = None
        self._block = 0
        self._max_blocks = 6

    def reset(self, seed=None, options=None):
        # Reset from real env to get a valid starting state
        state, info = self.real_env.reset(seed=seed)
        self._current_state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            self._current_z, _, _ = self.wm.encode(self._current_state.unsqueeze(0))

        self._block = 0
        return state, info

    def step(self, action):
        action_t = torch.FloatTensor(action).to(self.device).unsqueeze(0)

        with torch.no_grad():
            z_next, _, _ = self.wm.predict_next(self._current_z, action_t)
            state_pred = self.wm.decode(z_next).squeeze(0).cpu().numpy()
            reward_pred = self.wm.predict_reward(z_next).squeeze(0).item()

        self._current_z = z_next
        self._block += 1
        state_pred[11] = float(self._block)  # ensure block counter is correct

        done = self._block >= self._max_blocks
        return state_pred, reward_pred, done, False, {}


class DAggerRewardLogger(BaseCallback):
    """Log episode rewards during model-based training."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_steps = []
        self.real_env_steps = 0  # track actual simulator interactions
        self._current_reward = 0

    def _on_step(self):
        self._current_reward += self.locals.get("rewards", [0])[0]
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_steps.append(self.num_timesteps)
            self._current_reward = 0
        return True


def collect_on_policy_data(model, real_env, num_steps=500):
    """Run the current policy on the REAL simulator and collect transitions."""
    states, actions, rewards, next_states = [], [], [], []
    state, _ = real_env.reset()
    steps = 0

    while steps < num_steps:
        action, _ = model.predict(state, deterministic=False)
        next_state, reward, done, truncated, _ = real_env.step(action)

        states.append(state.copy())
        actions.append(action.copy())
        rewards.append(reward)
        next_states.append(next_state.copy())

        state = next_state
        steps += 1

        if done:
            state, _ = real_env.reset()

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
    }


def train_modelbased_with_dagger(
    world_model_path: str = "data/world_model.pt",
    buffer_path: str = "data/replay_buffer.npz",
    save_path: str = "data/policies/sac_modelbased",
    dream_steps_per_iter: int = 5000,
    dagger_iterations: int = 12,
    dagger_real_steps: int = 500,
    wm_finetune_epochs: int = 3,
    seed: int = 42,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    real_env = BioSimEnv()

    # Load World Model
    wm = WorldModel(
        obs_dim=real_env.obs_dim,
        act_dim=real_env.act_dim,
    ).to(device)
    wm.load(world_model_path, device)
    wm.eval()

    # Create dream environment
    dream_env = WorldModelEnv(wm, real_env, device)

    # Initialize SAC on dream environment
    model = SAC(
        "MlpPolicy",
        dream_env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=0,
        seed=seed,
    )

    logger = DAggerRewardLogger()
    total_real_steps = 0

    # Load existing replay buffer for WM fine-tuning
    buffer_data = np.load(buffer_path)
    all_states = list(buffer_data["states"])
    all_actions = list(buffer_data["actions"])
    all_rewards = list(buffer_data["rewards"])
    all_next_states = list(buffer_data["next_states"])

    print(f"Starting model-based training with DAgger")
    print(f"  {dagger_iterations} iterations × {dream_steps_per_iter} dream steps")
    print(f"  {dagger_real_steps} real steps per DAgger refresh")
    print(f"  Device: {device}")

    for iteration in range(dagger_iterations):
        # 1. Train in dreams
        print(f"\n--- DAgger Iteration {iteration + 1}/{dagger_iterations} ---")
        print(f"  Training SAC in World Model for {dream_steps_per_iter} steps...")
        model.learn(total_timesteps=dream_steps_per_iter, callback=logger, reset_num_timesteps=False)

        # 2. Collect real-env data with current policy
        print(f"  Collecting {dagger_real_steps} real env steps...")
        new_data = collect_on_policy_data(model, real_env, dagger_real_steps)
        total_real_steps += dagger_real_steps

        all_states.extend(new_data["states"])
        all_actions.extend(new_data["actions"])
        all_rewards.extend(new_data["rewards"])
        all_next_states.extend(new_data["next_states"])

        # 3. Fine-tune World Model on expanded buffer
        print(f"  Fine-tuning World Model ({wm_finetune_epochs} epochs on {len(all_states)} transitions)...")
        temp_buffer = "data/_dagger_buffer.npz"
        np.savez_compressed(
            temp_buffer,
            states=np.array(all_states[-100000:], dtype=np.float32),  # keep last 100k
            actions=np.array(all_actions[-100000:], dtype=np.float32),
            rewards=np.array(all_rewards[-100000:], dtype=np.float32),
            next_states=np.array(all_next_states[-100000:], dtype=np.float32),
        )

        wm_new = WorldModel(
            obs_dim=real_env.obs_dim,
            act_dim=real_env.act_dim,
        ).to(device)
        wm_new.load_state_dict(wm.state_dict())  # start from current weights

        optimizer = torch.optim.Adam(wm_new.parameters(), lr=5e-4)

        buffer_np = np.load(temp_buffer)
        n = len(buffer_np["states"])
        for epoch in range(wm_finetune_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, 256):
                batch_idx = idx[start : start + 256]
                s = torch.FloatTensor(buffer_np["states"][batch_idx]).to(device)
                a = torch.FloatTensor(buffer_np["actions"][batch_idx]).to(device)
                r = torch.FloatTensor(buffer_np["rewards"][batch_idx]).to(device)
                ns = torch.FloatTensor(buffer_np["next_states"][batch_idx]).to(device)

                outputs = wm_new(s, a, ns)
                from biosim.world_model import compute_world_model_loss
                losses = compute_world_model_loss(
                    outputs, {"state": s, "next_state": ns, "reward": r}, beta=0.01
                )
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(wm_new.parameters(), 1.0)
                optimizer.step()

        # Update dream environment with fine-tuned World Model
        wm = wm_new
        wm.eval()
        dream_env.wm = wm

        # Log progress
        recent_rewards = logger.episode_rewards[-10:] if logger.episode_rewards else [0]
        print(f"  Dream reward (last 10): {np.mean(recent_rewards):.3f}")
        print(f"  Total real env steps used: {total_real_steps}")

    # Save final policy and World Model
    model.save(save_path)
    wm.save(world_model_path.replace(".pt", "_finetuned.pt"))

    np.savez(
        save_path + "_curve.npz",
        rewards=np.array(logger.episode_rewards),
        steps=np.array(logger.episode_steps),
        real_env_steps=total_real_steps,
    )

    print(f"\n{'='*60}")
    print(f"Model-based training complete")
    print(f"  Total dream steps: {dagger_iterations * dream_steps_per_iter}")
    print(f"  Total real env steps: {total_real_steps}")
    print(f"  Policy saved to: {save_path}")

    # Evaluate on real simulator
    from biosim.train_sac_modelfree import evaluate_policy
    real_rewards = evaluate_policy(save_path, num_episodes=50)

    return model, logger, real_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model", default="data/world_model.pt")
    parser.add_argument("--buffer", default="data/replay_buffer.npz")
    parser.add_argument("--save", default="data/policies/sac_modelbased")
    parser.add_argument("--dream-steps", type=int, default=5000)
    parser.add_argument("--dagger-iters", type=int, default=12)
    parser.add_argument("--dagger-real-steps", type=int, default=500)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_modelbased_with_dagger(
        world_model_path=args.world_model,
        buffer_path=args.buffer,
        save_path=args.save,
        dream_steps_per_iter=args.dream_steps,
        dagger_iterations=args.dagger_iters,
        dagger_real_steps=args.dagger_real_steps,
        device=args.device,
    )
