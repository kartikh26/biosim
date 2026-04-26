"""
Collect training data by running a random agent on the simulator.
Saves transitions as a .npz replay buffer.

Usage:
    python -m biosim.collect_data --episodes 8500 --output data/replay_buffer.npz
"""

import argparse
import numpy as np
from tqdm import tqdm
from biosim.env import BioSimEnv


def collect_data(num_episodes: int = 8500, output_path: str = "data/replay_buffer.npz"):
    """
    Run random agent for num_episodes episodes.
    Each episode = 6 transitions (6 blocks).
    ~8500 episodes × 6 = ~51,000 transitions.
    """
    env = BioSimEnv()

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for ep in tqdm(range(num_episodes), desc="Collecting data"):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state.copy())
            actions.append(action.copy())
            rewards.append(reward)
            next_states.append(next_state.copy())
            dones.append(done)

            state = next_state

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    dones = np.array(dones, dtype=bool)

    np.savez_compressed(
        output_path,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    print(f"\nCollected {len(states)} transitions from {num_episodes} episodes")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8500)
    parser.add_argument("--output", default="data/replay_buffer.npz")
    args = parser.parse_args()
    collect_data(args.episodes, args.output)
