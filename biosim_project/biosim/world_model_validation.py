"""
World Model validation tests from Design Doc Part 4.5.
Run these BEFORE model-based RL training.

Usage:
    python -m biosim.world_model_validation --world-model data/world_model.pt --buffer data/replay_buffer.npz
"""

import argparse
import numpy as np
import torch
from biosim.world_model import WorldModel
from biosim.env import BioSimEnv


def k_step_prediction_mae(
    world_model: WorldModel,
    buffer_path: str,
    k_values=(1, 2, 4, 6),
    num_trajectories: int = 200,
    device: str = "cpu",
):
    """
    Test 1: k-step prediction MAE normalized by per-dimension std.
    Thresholds: 1-step < 0.05, 2-step < 0.10, 4-step < 0.20, 6-step < 0.35
    """
    data = np.load(buffer_path)
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]

    # Compute per-dimension std for normalization
    state_std = np.std(states, axis=0) + 1e-8

    # Group into episodes (6 steps each)
    n = len(states)
    episode_len = 6
    num_episodes = n // episode_len
    assert num_episodes >= num_trajectories, f"Need {num_trajectories} episodes, have {num_episodes}"

    results = {k: [] for k in k_values}
    world_model.eval()

    with torch.no_grad():
        for ep in range(num_trajectories):
            start = ep * episode_len
            ep_states = states[start : start + episode_len]
            ep_actions = actions[start : start + episode_len]
            ep_next = next_states[start : start + episode_len]

            # Encode initial state
            s0 = torch.FloatTensor(ep_states[0]).unsqueeze(0).to(device)
            z, _, _ = world_model.encode(s0)

            pred_states = []
            for t in range(min(max(k_values), episode_len)):
                a = torch.FloatTensor(ep_actions[t]).unsqueeze(0).to(device)
                z, _, _ = world_model.predict_next(z, a)
                s_pred = world_model.decode(z).squeeze(0).cpu().numpy()
                pred_states.append(s_pred)

            for k in k_values:
                if k <= len(pred_states):
                    true = ep_next[k - 1]
                    pred = pred_states[k - 1]
                    mae_normalized = np.mean(np.abs(true - pred) / state_std)
                    results[k].append(mae_normalized)

    print("\n=== Test 1: k-Step Prediction MAE (normalized by std) ===")
    thresholds = {1: 0.05, 2: 0.10, 4: 0.20, 6: 0.35}
    all_pass = True
    for k in k_values:
        mean_mae = np.mean(results[k])
        threshold = thresholds.get(k, 0.5)
        passed = mean_mae < threshold
        status = "PASS" if passed else "FAIL"
        print(f"  k={k}: MAE={mean_mae:.4f} (threshold={threshold:.2f}) [{status}]")
        if not passed:
            all_pass = False
    return all_pass, results


def divergence_horizon(
    world_model: WorldModel,
    buffer_path: str,
    num_trajectories: int = 200,
    threshold_std: float = 2.0,
    device: str = "cpu",
):
    """
    Test 2: At what k does the WM prediction drift > 2 SD from truth?
    Target: divergence horizon > 4 blocks (16 weeks).
    """
    data = np.load(buffer_path)
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    state_std = np.std(states, axis=0) + 1e-8

    episode_len = 6
    horizons = []

    world_model.eval()
    with torch.no_grad():
        for ep in range(num_trajectories):
            start = ep * episode_len
            if start + episode_len > len(states):
                break

            ep_states = states[start : start + episode_len]
            ep_actions = actions[start : start + episode_len]
            ep_next = next_states[start : start + episode_len]

            s0 = torch.FloatTensor(ep_states[0]).unsqueeze(0).to(device)
            z, _, _ = world_model.encode(s0)

            horizon = episode_len  # assume full horizon until proven otherwise
            for t in range(episode_len):
                a = torch.FloatTensor(ep_actions[t]).unsqueeze(0).to(device)
                z, _, _ = world_model.predict_next(z, a)
                s_pred = world_model.decode(z).squeeze(0).cpu().numpy()
                true = ep_next[t]

                max_deviation = np.max(np.abs(true - s_pred) / state_std)
                if max_deviation > threshold_std:
                    horizon = t + 1
                    break

            horizons.append(horizon)

    mean_horizon = np.mean(horizons)
    passed = mean_horizon > 4
    status = "PASS" if passed else "FAIL"

    print(f"\n=== Test 2: Divergence Horizon ===")
    print(f"  Mean horizon: {mean_horizon:.1f} blocks (threshold: > 4) [{status}]")
    print(f"  Distribution: min={np.min(horizons)}, median={np.median(horizons):.0f}, max={np.max(horizons)}")

    return passed, mean_horizon


def policy_transfer_gap(
    modelfree_path: str = "data/policies/sac_modelfree",
    modelbased_path: str = "data/policies/sac_modelbased",
    num_episodes: int = 50,
):
    """
    Test 3: Compare model-based vs model-free policy on REAL simulator.
    Target: gap < 0.30.
    Run AFTER both policies are trained.
    """
    from stable_baselines3 import SAC

    env = BioSimEnv()

    def evaluate(model_path):
        model = SAC.load(model_path)
        rewards = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_r = 0
            done = False
            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, r, done, _, _ = env.step(action)
                total_r += r
            rewards.append(total_r)
        return np.mean(rewards), np.std(rewards)

    mf_mean, mf_std = evaluate(modelfree_path)
    mb_mean, mb_std = evaluate(modelbased_path)

    gap = (mf_mean - mb_mean) / (abs(mf_mean) + 1e-8)
    passed = gap < 0.30
    status = "PASS" if passed else "FAIL"

    print(f"\n=== Test 3: Policy Transfer Gap ===")
    print(f"  Model-free  (real sim): {mf_mean:.3f} ± {mf_std:.3f}")
    print(f"  Model-based (real sim): {mb_mean:.3f} ± {mb_std:.3f}")
    print(f"  Gap: {gap:.3f} (threshold: < 0.30) [{status}]")

    return passed, gap


def run_all_validation(
    world_model_path: str,
    buffer_path: str,
    device: str = "cpu",
):
    """Run Tests 1 and 2 (pre-flight checks before model-based RL)."""
    wm = WorldModel(obs_dim=12, act_dim=8)
    wm.load(world_model_path, device)
    wm.to(device)

    pass1, _ = k_step_prediction_mae(wm, buffer_path, device=device)
    pass2, _ = divergence_horizon(wm, buffer_path, device=device)

    print(f"\n{'='*60}")
    if pass1 and pass2:
        print("PRE-FLIGHT GATE: PASS — proceed to model-based RL training")
    else:
        print("PRE-FLIGHT GATE: FAIL — iterate on World Model before proceeding")
        if not pass1:
            print("  → k-step MAE too high. Try: more data, larger hidden layers, more epochs")
        if not pass2:
            print("  → Divergence horizon too short. Try: DAgger data, reduce latent dim")

    return pass1 and pass2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model", default="data/world_model.pt")
    parser.add_argument("--buffer", default="data/replay_buffer.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test3", action="store_true", help="Also run policy transfer gap (requires trained policies)")
    args = parser.parse_args()

    passed = run_all_validation(args.world_model, args.buffer, args.device)

    if args.test3:
        policy_transfer_gap()
