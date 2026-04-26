"""
BioSim Gymnasium Environment (MVP: 6 compounds, 12 state dims, 8 action dims).

Episode = 24 weeks = 6 blocks of 4 weeks.
Each env.step() integrates PK equations at daily resolution for 28 days,
then returns the end-of-block state.

State (12 dims):
    0: lean_mass (kg)           1: fat_subcut (kg)
    2: fat_visc (kg)            3: testosterone (ng/dL)
    4: estrogen (pg/mL)         5: igf1 (ng/mL)
    6: glp1_occupancy (0-1)     7: hpta_suppression (0-1)
    8: recovery (0-1)           9: hunger (0-1)
    10: androgenic_load (0-1)   11: block_number (0-6)

Action (8 dims, all [0,1]):
    0: training_volume          1: caloric_balance (-1 to 1, remapped)
    2: testosterone_dose        3: semaglutide_dose
    4: tirzepatide_dose         5: hgh_dose
    6: arimidex_dose            7: nolvadex_dose
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from biosim.compounds import (
    decay_factor,
    michaelis_menten_aromatization,
    igf1_dose_response,
    glp1_receptor_occupancy,
    compute_hunger_drive,
    COMPOUNDS,
)
from biosim.reward import compute_reward


class BioSimEnv(gym.Env):
    """MVP BioSim environment with 6 compounds."""

    metadata = {"render_modes": []}

    def __init__(self, reward_weights=None, age=30):
        super().__init__()

        self.obs_dim = 12
        self.act_dim = 8
        self.blocks_per_episode = 6
        self.days_per_block = 28
        self.age = age

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        # All actions normalized [0, 1] except caloric_balance remapped to [-1, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
        )

        self.reward_weights = reward_weights
        self.state = None
        self.prev_state = None
        self.prev_action = None
        self.block = 0

        # Internal drug levels (tracked at daily resolution, not exposed in obs)
        self._drug_levels = {}

    def _get_baseline(self):
        """Age-stratified baselines."""
        if self.age < 30:
            t_base = np.random.uniform(413, 1000)
            igf1_base = np.random.uniform(116, 358)
        elif self.age < 40:
            t_base = np.random.uniform(352, 1000)
            igf1_base = np.random.uniform(94, 284)
        else:
            t_base = np.random.uniform(252, 916)
            igf1_base = np.random.uniform(75, 238)
        return t_base, igf1_base

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        t_base, igf1_base = self._get_baseline()

        self.state = np.array([
            75.0,          # lean_mass (kg)
            15.0,          # fat_subcut (kg)
            5.0,           # fat_visc (kg)
            t_base,        # testosterone (ng/dL)
            30.0,          # estrogen (pg/mL)
            igf1_base,     # igf1 (ng/mL)
            0.0,           # glp1_occupancy
            0.0,           # hpta_suppression
            1.0,           # recovery
            0.5,           # hunger (baseline)
            0.1,           # androgenic_load
            0.0,           # block_number
        ], dtype=np.float32)

        self._drug_levels = {
            "testosterone_blood": 0.0,
            "semaglutide_blood": 0.0,
            "tirzepatide_blood": 0.0,
            "hgh_blood": 0.0,
            "arimidex_blood": 0.0,
            "nolvadex_blood": 0.0,
        }
        self._endogenous_t = float(self.state[3])
        self._baseline_igf1 = float(self.state[5])
        self.block = 0
        self.prev_state = None
        self.prev_action = None

        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(action, 0, 1).astype(np.float32)

        # Remap caloric_balance from [0,1] to [-1,1]
        cal_balance = action[1] * 2.0 - 1.0

        # Convert normalized doses to actual doses
        test_dose_mg_week = action[2] * COMPOUNDS["testosterone"]["max_dose_per_week"]
        sema_dose_mg_week = action[3] * COMPOUNDS["semaglutide"]["max_dose_per_week"]
        tirz_dose_mg_week = action[4] * COMPOUNDS["tirzepatide"]["max_dose_per_week"]
        hgh_dose_iu_day = action[5] * COMPOUNDS["hgh"]["max_dose_per_day"]
        arim_dose_mg_day = action[6] * COMPOUNDS["arimidex"]["max_dose_per_day"]
        nolv_dose_mg_day = action[7] * COMPOUNDS["nolvadex"]["max_dose_per_day"]

        training_vol = float(action[0])

        self.prev_state = self.state.copy()

        # ─── Daily integration loop (28 days) ───
        for day in range(self.days_per_block):
            dl = self._drug_levels

            # 1. Decay existing drug levels
            dl["testosterone_blood"] *= decay_factor(COMPOUNDS["testosterone"]["half_life_days"])
            dl["semaglutide_blood"] *= decay_factor(COMPOUNDS["semaglutide"]["half_life_days"])
            dl["tirzepatide_blood"] *= decay_factor(COMPOUNDS["tirzepatide"]["half_life_days"])
            dl["hgh_blood"] *= decay_factor(COMPOUNDS["hgh"]["half_life_days"])
            dl["arimidex_blood"] *= decay_factor(COMPOUNDS["arimidex"]["half_life_days"])
            dl["nolvadex_blood"] *= decay_factor(COMPOUNDS["nolvadex"]["half_life_days"])

            # 2. Add today's dose
            dl["testosterone_blood"] += test_dose_mg_week / 7.0 * 3.0  # mg → approximate ng/dL contribution
            dl["semaglutide_blood"] += sema_dose_mg_week / 7.0
            dl["tirzepatide_blood"] += tirz_dose_mg_week / 7.0
            dl["hgh_blood"] += hgh_dose_iu_day
            dl["arimidex_blood"] += arim_dose_mg_day
            dl["nolvadex_blood"] += nolv_dose_mg_day

            # 3. Testosterone level = endogenous (suppressed) + exogenous
            hpta_supp = float(self.state[7])
            endogenous_t = self._endogenous_t * (1.0 - hpta_supp)
            total_t = endogenous_t + dl["testosterone_blood"]
            self.state[3] = np.clip(total_t, 0, 5000)

            # 4. HPTA suppression
            if test_dose_mg_week > 10:
                # Rapid suppression from exogenous T
                supp_rate = 0.015 * (test_dose_mg_week / 500.0)
                self.state[7] = min(1.0, self.state[7] + supp_rate)
            elif nolv_dose_mg_day > 5 and test_dose_mg_week < 10:
                # PCT recovery with nolvadex
                recovery_rate = COMPOUNDS["nolvadex"]["hpta_recovery_rate"] * (nolv_dose_mg_day / 20.0)
                self.state[7] = max(0.0, self.state[7] - recovery_rate)
            else:
                # Slow natural recovery
                self.state[7] = max(0.0, self.state[7] - 0.002)

            # 5. Aromatization → estrogen
            arimidex_supp = 0.0
            if dl["arimidex_blood"] > 0.1:
                if hpta_supp > 0.5:
                    arimidex_supp = COMPOUNDS["arimidex"]["e2_suppression_hpta_suppressed"]
                else:
                    arimidex_supp = COMPOUNDS["arimidex"]["e2_suppression_hpta_active"]
                arimidex_supp *= min(1.0, dl["arimidex_blood"] / 1.0)

            e2_from_aromatization = michaelis_menten_aromatization(
                testosterone_level=self.state[3],
                arimidex_suppression=arimidex_supp,
            )
            # Baseline estrogen contribution + aromatization
            self.state[4] = max(0.0, 15.0 + e2_from_aromatization * 100.0)

            # 6. GLP-1 receptor occupancy
            self.state[6] = glp1_receptor_occupancy(
                dl["semaglutide_blood"], dl["tirzepatide_blood"]
            )

            # 7. Hunger drive
            self.state[9] = compute_hunger_drive(0.5, self.state[6])

            # 8. IGF-1 from HGH
            if hgh_dose_iu_day > 0.1:
                igf1_mult = igf1_dose_response(hgh_dose_iu_day)
                self.state[5] = self._baseline_igf1 * (1.0 + igf1_mult)
                # Suppress natural GH
                # (represented implicitly via IGF-1 override)
            else:
                # Slow return to baseline
                self.state[5] += (self._baseline_igf1 - self.state[5]) * 0.05

            # Nolvadex mid-cycle IGF-1 blunt
            if nolv_dose_mg_day > 5 and test_dose_mg_week > 50:
                self.state[5] *= (1.0 - COMPOUNDS["nolvadex"]["igf1_blunt_factor"] * 0.5)

            # 9. DHT / androgenic load
            if self.state[3] > 800:
                self.state[10] = min(1.0, self.state[10] + 0.003)
            else:
                self.state[10] = max(0.0, self.state[10] - 0.002)

            # 10. Recovery (depleted by training, restored by rest)
            recovery_drain = training_vol * 0.03
            recovery_restore = (1.0 - training_vol) * 0.04
            self.state[8] = np.clip(self.state[8] - recovery_drain + recovery_restore, 0, 1)

        # ─── End of daily loop: compute body composition changes ───

        # Fat loss from caloric deficit + GLP-1 effect
        glp1_boost = self.state[6] * 0.3  # GLP-1 amplifies deficit
        effective_deficit = -cal_balance + glp1_boost  # positive = deficit
        fat_loss_kg = effective_deficit * 0.5 * (self.days_per_block / 7.0)  # ~0.5 kg/week at full deficit

        # Tirzepatide GIP effect: better lean preservation
        lean_preservation_factor = 1.0
        if self.state[6] > 0.3 and action[4] > 0.1:
            lean_preservation_factor = 0.5  # half the muscle loss from deficit

        self.state[1] -= 0.4 * fat_loss_kg  # 40% from subcutaneous
        self.state[2] -= 0.6 * fat_loss_kg  # 60% from visceral
        self.state[1] = max(1.0, self.state[1])
        self.state[2] = max(0.5, self.state[2])

        # Muscle gain from training + testosterone + caloric surplus
        t_level = self.state[3]
        t_factor = np.clip(t_level / 600.0, 0, 3.0)  # 600 ng/dL = 1x, scaled
        calorie_factor = 1.0 if cal_balance >= -0.2 else max(0.2, 1.0 + cal_balance)
        calorie_factor *= lean_preservation_factor

        recovery_factor = np.clip(self.state[8], 0.3, 1.0)
        igf1_factor = np.clip(self.state[5] / self._baseline_igf1, 0.5, 2.5)

        muscle_gain_kg = (
            training_vol * 0.35  # base gain from training (calibrated to Bhasin)
            * t_factor            # testosterone amplification
            * calorie_factor      # caloric support
            * recovery_factor     # can't grow if overtrained
            * igf1_factor         # IGF-1 contribution
            * (self.days_per_block / 7.0)  # scale to block duration
        )
        self.state[0] += muscle_gain_kg

        # ─── Update block counter ───
        self.block += 1
        self.state[11] = float(self.block)

        # ─── Compute reward ───
        reward = compute_reward(
            self.state, self.prev_state, action, self.prev_action,
            self.reward_weights,
        )

        done = self.block >= self.blocks_per_episode
        truncated = False

        self.prev_action = action.copy()

        return self.state.copy(), reward, done, truncated, {}

    def get_state_labels(self):
        return [
            "lean_mass", "fat_subcut", "fat_visc", "testosterone",
            "estrogen", "igf1", "glp1_occupancy", "hpta_suppression",
            "recovery", "hunger", "androgenic_load", "block_number",
        ]

    def get_action_labels(self):
        return [
            "training_volume", "caloric_balance", "testosterone_dose",
            "semaglutide_dose", "tirzepatide_dose", "hgh_dose",
            "arimidex_dose", "nolvadex_dose",
        ]
