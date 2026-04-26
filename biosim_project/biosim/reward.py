"""
Reward function for BioSim environment.
Implements all penalty terms from Design Doc Part 3: reward function.
"""

import numpy as np


# Default weights — overridden by LLM agent per user goal
DEFAULT_WEIGHTS = {
    "w_muscle": 0.35,
    "w_fat_loss": 0.35,
    "w_health": 0.25,     # hard floor: never below 0.15
    "w_recovery": 0.05,
}

# Penalty magnitudes (tunable hyperparameters)
PENALTIES = {
    "high_estrogen": 0.3,
    "mild_crashed_e2": 0.15,
    "severe_crashed_e2": 0.4,
    "severe_liver": 0.5,
    "androgenic": 0.2,
    "failed_pct": 0.6,
    "progestogenic": 0.2,
    "glp1_overstacking": 0.3,
    "lambda_smooth": 0.02,
}


def compute_reward(
    state: np.ndarray,
    prev_state: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray | None,
    weights: dict | None = None,
) -> float:
    """
    Compute single-step reward from state transition.

    State indices (12-dim):
        0: lean_mass (kg)
        1: fat_subcut (kg)
        2: fat_visc (kg)
        3: testosterone (ng/dL)
        4: estrogen (pg/mL)
        5: igf1 (ng/mL)
        6: glp1_occupancy (0-1)
        7: hpta_suppression (0-1)
        8: recovery (0-1)
        9: hunger (0-1)
        10: androgenic_load (0-1)
        11: block_number (0-6)
    """
    w = weights or DEFAULT_WEIGHTS
    w_muscle = w["w_muscle"]
    w_fat = w["w_fat_loss"]
    w_health = max(w["w_health"], 0.15)  # HARD FLOOR
    w_recovery = w["w_recovery"]

    # Normalize weight sum to 1
    total = w_muscle + w_fat + w_health + w_recovery
    w_muscle /= total
    w_fat /= total
    w_health /= total
    w_recovery /= total

    # ── Deltas ──
    delta_lean = state[0] - prev_state[0]
    delta_fat = (prev_state[1] + prev_state[2]) - (state[1] + state[2])  # fat loss is positive

    # Normalize by reasonable per-block maxima
    delta_lean_norm = np.clip(delta_lean / 2.0, -1, 1)     # ±2 kg/block is large
    delta_fat_norm = np.clip(delta_fat / 2.0, -1, 1)

    # ── Tamoxifen mid-cycle blunt ──
    # action[7] = nolvadex dose; any androgen = action[2] (test) > 0
    nolvadex_active = action[7] > 0.05 if len(action) > 7 else False
    androgen_active = action[2] > 0.05 if len(action) > 2 else False
    if nolvadex_active and androgen_active:
        delta_lean_norm *= (1.0 - 0.30)  # 30% blunt

    # ── Base reward ──
    estrogen = state[4]
    hpta = state[7]
    liver_stress = 0.0  # simplified — no orals in MVP
    recovery = state[8]
    androgenic = state[10]

    reward = (
        w_muscle * delta_lean_norm
        + w_fat * delta_fat_norm
        + w_health * (1.0 - hpta)
        + w_recovery * recovery
    )

    # ── Estrogen penalties (two-tier) ──
    if estrogen > 60:
        reward -= PENALTIES["high_estrogen"]
    if 10 <= estrogen < 15:
        reward -= PENALTIES["mild_crashed_e2"]
    if estrogen < 10:
        reward -= PENALTIES["severe_crashed_e2"]

    # ── Safety thresholds ──
    if androgenic > 0.85:
        reward -= PENALTIES["androgenic"]

    block = int(state[11])
    if hpta > 0.95 and block >= 5:
        reward -= PENALTIES["failed_pct"]

    # ── GLP-1 over-stacking ──
    glp1_occ = state[6]
    sema_dose = action[3] if len(action) > 3 else 0
    tirz_dose = action[4] if len(action) > 4 else 0
    if glp1_occ > 0.85 and sema_dose > 0.3 and tirz_dose > 0.3:
        reward -= PENALTIES["glp1_overstacking"]

    # ── Action smoothness ──
    if prev_action is not None:
        smoothness_penalty = np.sum((action - prev_action) ** 2)
        reward -= PENALTIES["lambda_smooth"] * smoothness_penalty

    return float(reward)
