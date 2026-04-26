"""
Pharmacokinetic equations for 6 MVP compounds.
Each compound class implements daily_update(state, dose) -> state mutations.
All parameters from primary clinical literature (see docs/DESIGN.md Part 3).
"""

import numpy as np


# ─────────────────────────────────────────────
# Compound parameter dataclasses
# ─────────────────────────────────────────────

COMPOUNDS = {
    "testosterone": {
        "half_life_days": 4.5,
        "max_dose_per_week": 500.0,    # mg/week
        "aromatization_rate": 0.004,    # 0.4% per-pass (Bhasin 2001)
        "dht_conversion_rate": 0.06,    # 5-7%
        "hpta_suppression_rate": 0.015, # per day at max dose
        "muscle_efficacy": 1.0,         # reference compound
    },
    "semaglutide": {
        "half_life_days": 7.0,         # ~165 hours (Wilding 2021)
        "max_dose_per_week": 2.4,       # mg/week
        "glp1_efficacy": 1.0,
        "hunger_suppression": 0.8,
        "nausea_rate": 0.442,
        "dose_escalation_weeks": 16,    # mandatory titration
    },
    "tirzepatide": {
        "half_life_days": 5.0,         # ~116 hours (Jastreboff 2022)
        "max_dose_per_week": 15.0,      # mg/week
        "glp1_efficacy": 0.7,          # weaker GLP-1R affinity than semaglutide
        "gip_efficacy": 1.0,           # strong GIP component
        "hunger_suppression": 0.85,
        "nausea_rate": 0.333,
        "dose_escalation_weeks": 20,
    },
    "hgh": {
        "half_life_days": 0.125,       # 2-3 hours
        "max_dose_per_day": 10.0,       # IU/day
        "igf1_plateau_iu": 5.0,        # linear response up to here, then saturates
        "natural_gh_suppression": 0.9,  # near-total suppression of endogenous GH
    },
    "arimidex": {
        "half_life_days": 2.0,         # 40-50 hours
        "max_dose_per_day": 1.0,        # mg/day
        "e2_suppression_hpta_suppressed": 0.85,  # 80-95% when on exogenous T
        "e2_suppression_hpta_active": 0.35,      # ~35% when HPTA intact
    },
    "nolvadex": {
        "half_life_days": 6.0,         # 5-7 days
        "max_dose_per_day": 40.0,       # mg/day
        "hpta_recovery_rate": 0.02,     # per day, added to natural recovery
        "igf1_blunt_factor": 0.30,      # 16-50% IGF-1 reduction, use midpoint
    },
}


def decay_factor(half_life_days: float) -> float:
    """Daily exponential decay factor."""
    return 0.5 ** (1.0 / half_life_days)


def michaelis_menten_aromatization(
    testosterone_level: float,
    v_max: float = 0.08,
    k_m: float = 400.0,  # ng/dL — calibrated so 600mg/wk → E2 ~60-100 pg/mL
    arimidex_suppression: float = 0.0,
) -> float:
    """
    Testosterone → estradiol conversion via CYP19A1.
    Non-linear saturable kinetics (Michaelis-Menten).
    Arimidex acts as competitive inhibitor reducing effective V_max.
    """
    effective_vmax = v_max * (1.0 - arimidex_suppression)
    if testosterone_level <= 0:
        return 0.0
    return effective_vmax * testosterone_level / (k_m + testosterone_level)


def igf1_dose_response(hgh_dose_iu: float, plateau_iu: float = 5.0) -> float:
    """
    IGF-1 response to HGH: linear up to plateau_iu, then saturates.
    Returns a multiplier (0 to ~2.5) on baseline IGF-1.
    """
    if hgh_dose_iu <= 0:
        return 0.0
    if hgh_dose_iu <= plateau_iu:
        return hgh_dose_iu / plateau_iu * 1.5  # linear to 1.5x at plateau
    else:
        # Saturating beyond plateau
        excess = hgh_dose_iu - plateau_iu
        return 1.5 + 0.5 * (1 - np.exp(-0.2 * excess))  # asymptotes ~2.0


def glp1_receptor_occupancy(
    semaglutide_level: float,
    tirzepatide_level: float,
    sema_max: float = 1.0,
    tirz_max: float = 1.0,
) -> float:
    """
    Combined GLP-1 receptor occupancy. Saturates at 1.0.
    Semaglutide has higher GLP-1R affinity than tirzepatide.
    """
    sema_occ = min(1.0, semaglutide_level / (sema_max * 0.5 + semaglutide_level + 1e-8))
    tirz_occ = min(1.0, tirzepatide_level / (tirz_max * 1.5 + tirzepatide_level + 1e-8)) * 0.7
    return min(1.0, sema_occ + tirz_occ)


def compute_hunger_drive(
    baseline: float,
    glp1_occ: float,
) -> float:
    """
    Hunger drive modulated by GLP-1 occupancy.
    GLP-1 agonists suppress hunger proportional to receptor occupancy.
    """
    suppression = glp1_occ * 0.7  # max 70% hunger suppression
    return max(0.0, baseline * (1.0 - suppression))
