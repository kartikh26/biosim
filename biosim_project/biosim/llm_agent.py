"""
LLM Agent: Claude API with tool calling for BioSim.
Two tools: configure_and_optimize, explain_trajectory.

Usage:
    python -m biosim.llm_agent --goal "Cut fat but keep muscle, 180 lbs 22% BF, age 34"
"""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ─── Tool definitions ───

TOOLS = [
    {
        "name": "configure_and_optimize",
        "description": (
            "Configure the BioSim environment with user parameters and run the "
            "RL agent to produce an optimized 24-week protocol. Returns the "
            "6-block action sequence and predicted state trajectory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "starting_lean_mass_kg": {"type": "number", "minimum": 40, "maximum": 130},
                "starting_fat_subcut_kg": {"type": "number", "minimum": 2, "maximum": 60},
                "starting_fat_visc_kg": {"type": "number", "minimum": 1, "maximum": 20},
                "age": {"type": "integer", "minimum": 18, "maximum": 70},
                "allowed_compounds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["testosterone", "semaglutide", "tirzepatide",
                                 "hgh", "arimidex", "nolvadex"]
                    }
                },
                "reward_weights": {
                    "type": "object",
                    "properties": {
                        "w_muscle": {"type": "number", "minimum": 0, "maximum": 1},
                        "w_fat_loss": {"type": "number", "minimum": 0, "maximum": 1},
                        "w_health": {"type": "number", "minimum": 0.15, "maximum": 1},
                        "w_recovery": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["w_muscle", "w_fat_loss", "w_health", "w_recovery"],
                },
            },
            "required": ["starting_lean_mass_kg", "starting_fat_subcut_kg",
                         "starting_fat_visc_kg", "age", "allowed_compounds",
                         "reward_weights"],
        },
    },
    {
        "name": "explain_trajectory",
        "description": (
            "Take the action sequence and predicted state trajectory from "
            "configure_and_optimize and produce a human-readable block-by-block narrative."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "actions": {"type": "array", "description": "6 action vectors"},
                "states": {"type": "array", "description": "7 state vectors (initial + 6 post-block)"},
                "user_goal": {"type": "string"},
            },
            "required": ["actions", "states", "user_goal"],
        },
    },
]

# ─── User phrase → weight mapping ───

PHRASE_WEIGHTS = {
    "bulk": {"w_muscle": 0.55, "w_fat_loss": 0.05, "w_health": 0.25, "w_recovery": 0.15},
    "cut": {"w_muscle": 0.30, "w_fat_loss": 0.45, "w_health": 0.20, "w_recovery": 0.05},
    "lose weight": {"w_muscle": 0.10, "w_fat_loss": 0.60, "w_health": 0.25, "w_recovery": 0.05},
    "recomp": {"w_muscle": 0.35, "w_fat_loss": 0.35, "w_health": 0.25, "w_recovery": 0.05},
    "pct": {"w_muscle": 0.10, "w_fat_loss": 0.05, "w_health": 0.65, "w_recovery": 0.20},
    "default": {"w_muscle": 0.30, "w_fat_loss": 0.30, "w_health": 0.25, "w_recovery": 0.15},
}

ACTION_LABELS = [
    "training_volume", "caloric_balance", "testosterone_dose",
    "semaglutide_dose", "tirzepatide_dose", "hgh_dose",
    "arimidex_dose", "nolvadex_dose",
]


def execute_configure_and_optimize(params: dict) -> dict:
    """
    Actually run the RL policy to generate a protocol.
    Uses the trained model-based policy if available, else model-free.
    """
    from biosim.env import BioSimEnv
    from stable_baselines3 import SAC

    # Find best available policy
    policy_path = None
    for p in ["data/policies/sac_modelbased", "data/policies/sac_modelfree"]:
        if Path(p + ".zip").exists():
            policy_path = p
            break

    if policy_path is None:
        return {"error": "No trained policy found. Run training first."}

    env = BioSimEnv(
        reward_weights=params["reward_weights"],
        age=params["age"],
    )
    state, _ = env.reset()

    # Override starting conditions
    env.state[0] = params["starting_lean_mass_kg"]
    env.state[1] = params["starting_fat_subcut_kg"]
    env.state[2] = params["starting_fat_visc_kg"]
    state = env.state.copy()

    model = SAC.load(policy_path)

    states = [state.tolist()]
    actions = []
    rewards = []
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)

        # Zero out disallowed compounds
        compound_map = {
            "testosterone": 2, "semaglutide": 3, "tirzepatide": 4,
            "hgh": 5, "arimidex": 6, "nolvadex": 7,
        }
        allowed = params.get("allowed_compounds", list(compound_map.keys()))
        for compound, idx in compound_map.items():
            if compound not in allowed:
                action[idx] = 0.0

        state, reward, done, _, _ = env.step(action)
        actions.append(action.tolist())
        states.append(state.tolist())
        rewards.append(reward)

    return {
        "actions": actions,
        "states": states,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "action_labels": ACTION_LABELS,
        "state_labels": env.get_state_labels(),
    }


def execute_explain_trajectory(params: dict) -> str:
    """Generate a narrative explanation of the protocol."""
    actions = params["actions"]
    states = params["states"]
    goal = params.get("user_goal", "optimize fitness")

    lines = [f"## Protocol for: {goal}\n"]

    for block_idx, (action, state_after) in enumerate(zip(actions, states[1:])):
        week_start = block_idx * 4 + 1
        week_end = (block_idx + 1) * 4
        lines.append(f"### Block {block_idx + 1} (Weeks {week_start}-{week_end})")

        # Training
        training = action[0]
        cal = action[1] * 2 - 1
        lines.append(f"- Training: {'high' if training > 0.7 else 'moderate' if training > 0.4 else 'light'} ({training:.0%})")
        lines.append(f"- Calories: {'surplus' if cal > 0.1 else 'deficit' if cal < -0.1 else 'maintenance'} ({cal:+.1f})")

        # Active compounds
        compounds = [
            ("Testosterone", action[2], "mg/week", 500),
            ("Semaglutide", action[3], "mg/week", 2.4),
            ("Tirzepatide", action[4], "mg/week", 15),
            ("HGH", action[5], "IU/day", 10),
            ("Arimidex", action[6], "mg/day", 1.0),
            ("Nolvadex", action[7], "mg/day", 40),
        ]
        for name, dose_norm, unit, max_dose in compounds:
            if dose_norm > 0.05:
                actual = dose_norm * max_dose
                lines.append(f"- {name}: {actual:.1f} {unit}")

        # State snapshot
        lines.append(f"- → Lean mass: {state_after[0]:.1f} kg | Fat: {state_after[1] + state_after[2]:.1f} kg")
        lines.append(f"  E2: {state_after[4]:.0f} pg/mL | HPTA: {state_after[7]:.0%} suppressed")
        lines.append("")

    return "\n".join(lines)


def run_agent(goal: str, api_key: str = None):
    """Run the full LLM agent flow."""
    if not HAS_ANTHROPIC:
        print("anthropic package not installed. Running in offline mode.")
        # Offline fallback: parse goal manually
        weights = PHRASE_WEIGHTS["default"]
        for key, w in PHRASE_WEIGHTS.items():
            if key in goal.lower():
                weights = w
                break

        result = execute_configure_and_optimize({
            "starting_lean_mass_kg": 75.0,
            "starting_fat_subcut_kg": 15.0,
            "starting_fat_visc_kg": 5.0,
            "age": 30,
            "allowed_compounds": ["testosterone", "semaglutide", "tirzepatide",
                                  "hgh", "arimidex", "nolvadex"],
            "reward_weights": weights,
        })

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        narrative = execute_explain_trajectory({
            "actions": result["actions"],
            "states": result["states"],
            "user_goal": goal,
        })

        print(narrative)
        print(f"\nTotal reward: {result['total_reward']:.3f}")
        return result, narrative

    # Full Claude API flow
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are a fitness protocol optimizer powered by BioSim.
    You have two tools: configure_and_optimize (runs the RL agent) and explain_trajectory
    (generates a narrative). When a user describes their goal, parse it into parameters
    and call configure_and_optimize. Then call explain_trajectory on the result.

    Weight mapping guide:
    - "bulk" / "get jacked" → w_muscle=0.55, w_fat_loss=0.05, w_health=0.25, w_recovery=0.15
    - "cut" / "lose fat keep muscle" → w_muscle=0.30, w_fat_loss=0.45, w_health=0.20, w_recovery=0.05
    - "lose weight" → w_muscle=0.10, w_fat_loss=0.60, w_health=0.25, w_recovery=0.05
    - "recomp" → w_muscle=0.35, w_fat_loss=0.35, w_health=0.25, w_recovery=0.05
    - "PCT" → w_muscle=0.10, w_fat_loss=0.05, w_health=0.65, w_recovery=0.20

    CRITICAL: w_health minimum is 0.15. Never set it lower.
    If age/weight not specified, ask or use defaults (age=30, 75kg lean, 15kg subcut fat, 5kg visc fat).
    """

    messages = [{"role": "user", "content": goal}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        tools=TOOLS,
        messages=messages,
    )

    # Handle tool calls
    for block in response.content:
        if block.type == "tool_use":
            if block.name == "configure_and_optimize":
                result = execute_configure_and_optimize(block.input)
                print(json.dumps(result, indent=2)[:500])
            elif block.name == "explain_trajectory":
                narrative = execute_explain_trajectory(block.input)
                print(narrative)
        elif block.type == "text":
            print(block.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="Cut fat but keep muscle, 180 lbs 22% BF, age 34")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    run_agent(args.goal, args.api_key)
