"""
VAE World Model for BioSim.
Architecture: Encoder(12→8) + TransitionModel(8+8→8) + Decoder(8→12) + RewardPredictor(8→1)
Simplified Ha & Schmidhuber 2018 without recurrence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class Encoder(nn.Module):
    """State (12-dim) → latent (8-dim) as Gaussian parameters."""
    def __init__(self, obs_dim=12, latent_dim=8, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class Decoder(nn.Module):
    """Latent (8-dim) → reconstructed state (12-dim)."""
    def __init__(self, latent_dim=8, obs_dim=12, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


class TransitionModel(nn.Module):
    """(latent + action) → next_latent."""
    def __init__(self, latent_dim=8, act_dim=8, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, z, a):
        h = self.net(torch.cat([z, a], dim=-1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class RewardPredictor(nn.Module):
    """Latent (8-dim) → predicted reward (scalar)."""
    def __init__(self, latent_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class WorldModel(nn.Module):
    """Complete VAE World Model combining all four sub-networks."""

    def __init__(self, obs_dim=12, act_dim=8, latent_dim=8, hidden=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(obs_dim, latent_dim, hidden)
        self.decoder = Decoder(latent_dim, obs_dim, hidden)
        self.transition = TransitionModel(latent_dim, act_dim, hidden)
        self.reward_pred = RewardPredictor(latent_dim, hidden // 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, state):
        mu, logvar = self.encoder(state)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def predict_next(self, z, action):
        mu, logvar = self.transition(z, action)
        z_next = self.reparameterize(mu, logvar)
        return z_next, mu, logvar

    def predict_reward(self, z):
        return self.reward_pred(z)

    def forward(self, state, action, next_state):
        """Full forward pass for training. Returns all outputs needed for loss."""
        # Encode current state
        z, enc_mu, enc_logvar = self.encode(state)

        # Reconstruct current state
        state_recon = self.decode(z)

        # Predict next latent
        z_next_pred, trans_mu, trans_logvar = self.predict_next(z, action)

        # Decode predicted next state
        next_state_pred = self.decode(z_next_pred)

        # Encode actual next state (target)
        z_next_actual, _, _ = self.encode(next_state)

        # Predict reward from predicted next latent
        reward_pred = self.predict_reward(z_next_pred)

        return {
            "state_recon": state_recon,
            "next_state_pred": next_state_pred,
            "z_next_pred_mu": trans_mu,
            "z_next_actual": z_next_actual.detach(),
            "reward_pred": reward_pred,
            "enc_mu": enc_mu,
            "enc_logvar": enc_logvar,
            "trans_logvar": trans_logvar,
        }

    def imagine_rollout(self, initial_state, actions_sequence):
        """
        Imagine a full episode inside the World Model.
        initial_state: (obs_dim,) tensor
        actions_sequence: (num_blocks, act_dim) tensor
        Returns: list of predicted states, list of predicted rewards
        """
        self.eval()
        with torch.no_grad():
            z, _, _ = self.encode(initial_state.unsqueeze(0))
            states = [self.decode(z).squeeze(0)]
            rewards = []
            for t in range(actions_sequence.shape[0]):
                a = actions_sequence[t].unsqueeze(0)
                z, _, _ = self.predict_next(z, a)
                s_pred = self.decode(z).squeeze(0)
                r_pred = self.predict_reward(z).squeeze(0)
                states.append(s_pred)
                rewards.append(r_pred)
        return states, rewards

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))


def compute_world_model_loss(outputs, targets, beta=0.01):
    """
    Combined loss for the World Model.
    targets: dict with keys "state", "next_state", "reward"
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(outputs["state_recon"], targets["state"])

    # Transition loss (next-state prediction in observation space)
    trans_loss = F.mse_loss(outputs["next_state_pred"], targets["next_state"])

    # Reward prediction loss
    reward_loss = F.mse_loss(outputs["reward_pred"], targets["reward"])

    # KL divergence (encoder)
    enc_mu = outputs["enc_mu"]
    enc_logvar = outputs["enc_logvar"]
    kl_loss = -0.5 * torch.mean(1 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp())

    total = recon_loss + trans_loss + reward_loss + beta * kl_loss

    return {
        "total": total,
        "recon": recon_loss.item(),
        "transition": trans_loss.item(),
        "reward": reward_loss.item(),
        "kl": kl_loss.item(),
    }
