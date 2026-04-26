"""
Train the VAE World Model on replay buffer data.

Usage:
    python -m biosim.train_world_model --buffer data/replay_buffer.npz --epochs 100
"""

import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from biosim.world_model import WorldModel, compute_world_model_loss


def load_replay_buffer(path):
    """Load replay buffer from .npz file."""
    data = np.load(path)
    return {
        "states": data["states"],
        "actions": data["actions"],
        "rewards": data["rewards"],
        "next_states": data["next_states"],
    }


def train_world_model(
    buffer_path: str,
    save_path: str = "data/world_model.pt",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta_start: float = 0.0,
    beta_end: float = 0.01,
    beta_anneal_steps: int = 1000,
    latent_dim: int = 8,
    hidden_dim: int = 256,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load data
    buffer = load_replay_buffer(buffer_path)
    n = len(buffer["states"])
    print(f"Replay buffer: {n} transitions")

    # Split 80/20
    split = int(0.8 * n)
    idx = np.random.permutation(n)

    def make_dataset(indices):
        return TensorDataset(
            torch.FloatTensor(buffer["states"][indices]),
            torch.FloatTensor(buffer["actions"][indices]),
            torch.FloatTensor(buffer["rewards"][indices]),
            torch.FloatTensor(buffer["next_states"][indices]),
        )

    train_ds = make_dataset(idx[:split])
    val_ds = make_dataset(idx[split:])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    obs_dim = buffer["states"].shape[1]
    act_dim = buffer["actions"].shape[1]

    model = WorldModel(obs_dim, act_dim, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in train_dl:
            states, actions, rewards, next_states = [b.to(device) for b in batch]

            # Beta annealing
            beta = beta_start + (beta_end - beta_start) * min(1.0, global_step / beta_anneal_steps)

            outputs = model(states, actions, next_states)
            targets = {"state": states, "next_state": next_states, "reward": rewards}
            losses = compute_world_model_loss(outputs, targets, beta=beta)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append({k: v for k, v in losses.items() if k != "total"})
            global_step += 1

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                states, actions, rewards, next_states = [b.to(device) for b in batch]
                outputs = model(states, actions, next_states)
                targets = {"state": states, "next_state": next_states, "reward": rewards}
                losses = compute_world_model_loss(outputs, targets, beta=beta_end)
                val_losses.append(losses["total"].item())

        mean_val = np.mean(val_losses)
        mean_train = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | β={beta:.4f} | "
                f"recon={mean_train['recon']:.4f} trans={mean_train['transition']:.4f} "
                f"rew={mean_train['reward']:.4f} kl={mean_train['kl']:.4f} | "
                f"val_total={mean_val:.4f}"
            )

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", default="data/replay_buffer.npz")
    parser.add_argument("--save", default="data/world_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_world_model(
        buffer_path=args.buffer,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        device=args.device,
    )
