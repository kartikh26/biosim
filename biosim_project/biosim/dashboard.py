"""
BioSim Streamlit Dashboard.
Visualizes comparison experiment results and protocol exploration.

Usage:
    streamlit run biosim/dashboard.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

st.set_page_config(page_title="BioSim Dashboard", layout="wide")

# ─── Header ───
st.title("🧬 BioSim: World Model + RL for Fitness Protocol Optimization")
st.caption("⚠️ Portfolio demonstration only. Not medical advice. Compound pharmacology is approximate.")

# ─── Load results ───
@st.cache_data
def load_training_curves():
    curves = {}
    for name in ["sac_modelfree", "sac_modelbased"]:
        path = f"data/policies/{name}_curve.npz"
        if Path(path).exists():
            data = np.load(path)
            curves[name] = {
                "rewards": data["rewards"],
                "steps": data["steps"],
            }
            if "real_env_steps" in data:
                curves[name]["real_env_steps"] = int(data["real_env_steps"])
    return curves


@st.cache_data
def load_protocol(goal_key="default"):
    """Load a pre-computed protocol if available."""
    path = f"data/protocols/{goal_key}.json"
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


# ─── Tab 1: Comparison Experiment ───
tab1, tab2, tab3 = st.tabs([
    "📊 Model-Free vs Model-Based",
    "🔬 Protocol Explorer",
    "📋 Architecture",
])

with tab1:
    st.header("Headline Experiment: Sample Efficiency Comparison")

    curves = load_training_curves()

    if not curves:
        st.warning("No training curves found. Run training first.")
        st.code("""
# Step 1: Collect data
python -m biosim.collect_data --episodes 8500

# Step 2: Train world model
python -m biosim.train_world_model --buffer data/replay_buffer.npz --epochs 100

# Step 3: Train model-free SAC
python -m biosim.train_sac_modelfree --timesteps 60000

# Step 4: Train model-based SAC + DAgger
python -m biosim.train_sac_modelbased --world-model data/world_model.pt --buffer data/replay_buffer.npz
        """)
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Curves")
            fig = go.Figure()

            if "sac_modelfree" in curves:
                c = curves["sac_modelfree"]
                # Smooth rewards
                window = min(20, len(c["rewards"]) // 3 + 1)
                if window > 1:
                    smoothed = np.convolve(c["rewards"], np.ones(window)/window, mode="valid")
                    fig.add_trace(go.Scatter(
                        y=smoothed,
                        name="Model-Free SAC",
                        line=dict(color="#2196F3", width=2),
                    ))

            if "sac_modelbased" in curves:
                c = curves["sac_modelbased"]
                window = min(20, len(c["rewards"]) // 3 + 1)
                if window > 1:
                    smoothed = np.convolve(c["rewards"], np.ones(window)/window, mode="valid")
                    fig.add_trace(go.Scatter(
                        y=smoothed,
                        name="Model-Based SAC + DAgger",
                        line=dict(color="#FF5722", width=2),
                    ))

            fig.update_layout(
                xaxis_title="Episode",
                yaxis_title="Episode Reward (smoothed)",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Sample Efficiency")

            mf = curves.get("sac_modelfree", {})
            mb = curves.get("sac_modelbased", {})

            mf_total = int(mf.get("steps", [0])[-1]) if "steps" in mf and len(mf["steps"]) > 0 else 0
            mb_real = mb.get("real_env_steps", 0)
            mb_dream = int(mb.get("steps", [0])[-1]) if "steps" in mb and len(mb["steps"]) > 0 else 0

            st.metric("Model-Free: Real Env Steps", f"{mf_total:,}")
            st.metric("Model-Based: Real Env Steps", f"{mb_real:,}")
            st.metric("Model-Based: Dream Steps", f"{mb_dream:,}")

            if mf_total > 0 and mb_real > 0:
                ratio = mf_total / mb_real
                st.metric("Sample Efficiency Gain", f"{ratio:.1f}x fewer real steps")

            # Final performance comparison
            if "rewards" in mf and "rewards" in mb:
                mf_final = np.mean(mf["rewards"][-10:]) if len(mf["rewards"]) >= 10 else 0
                mb_final = np.mean(mb["rewards"][-10:]) if len(mb["rewards"]) >= 10 else 0
                gap = (mf_final - mb_final) / (abs(mf_final) + 1e-8)
                st.metric("Policy Transfer Gap", f"{gap:.1%}")

with tab2:
    st.header("Protocol Explorer")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("User Goal")
        goal_type = st.selectbox("Goal", [
            "Cut fat, keep muscle",
            "Bulk / gain muscle",
            "Body recomp",
            "Lose weight (no steroids)",
            "Post-cycle therapy",
        ])

        age = st.slider("Age", 18, 65, 30)
        weight_lbs = st.slider("Body Weight (lbs)", 120, 300, 180)
        bf_pct = st.slider("Body Fat %", 8, 40, 22)

        compounds = st.multiselect("Allowed Compounds", [
            "testosterone", "semaglutide", "tirzepatide",
            "hgh", "arimidex", "nolvadex",
        ], default=["semaglutide", "arimidex", "nolvadex"])

        run_btn = st.button("🚀 Generate Protocol", type="primary")

    with col2:
        if run_btn:
            weight_kg = weight_lbs * 0.4536
            fat_kg = weight_kg * bf_pct / 100
            lean_kg = weight_kg - fat_kg

            st.info("Running RL agent...")

            try:
                from biosim.llm_agent import execute_configure_and_optimize, execute_explain_trajectory

                weight_map = {
                    "Cut fat, keep muscle": {"w_muscle": 0.30, "w_fat_loss": 0.45, "w_health": 0.20, "w_recovery": 0.05},
                    "Bulk / gain muscle": {"w_muscle": 0.55, "w_fat_loss": 0.05, "w_health": 0.25, "w_recovery": 0.15},
                    "Body recomp": {"w_muscle": 0.35, "w_fat_loss": 0.35, "w_health": 0.25, "w_recovery": 0.05},
                    "Lose weight (no steroids)": {"w_muscle": 0.10, "w_fat_loss": 0.60, "w_health": 0.25, "w_recovery": 0.05},
                    "Post-cycle therapy": {"w_muscle": 0.10, "w_fat_loss": 0.05, "w_health": 0.65, "w_recovery": 0.20},
                }

                result = execute_configure_and_optimize({
                    "starting_lean_mass_kg": lean_kg,
                    "starting_fat_subcut_kg": fat_kg * 0.6,
                    "starting_fat_visc_kg": fat_kg * 0.4,
                    "age": age,
                    "allowed_compounds": compounds,
                    "reward_weights": weight_map[goal_type],
                })

                if "error" in result:
                    st.error(result["error"])
                else:
                    # Plot trajectories
                    states = np.array(result["states"])
                    labels = result["state_labels"]

                    fig = make_subplots(rows=2, cols=2, subplot_titles=[
                        "Body Composition (kg)", "Hormones",
                        "Health Markers", "Metabolic",
                    ])

                    blocks = list(range(len(states)))

                    # Body comp
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 0], name="Lean Mass"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 1] + states[:, 2], name="Total Fat"), row=1, col=1)

                    # Hormones
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 3], name="Testosterone"), row=1, col=2)
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 4], name="Estradiol"), row=1, col=2)

                    # Health
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 7], name="HPTA Suppression"), row=2, col=1)
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 8], name="Recovery"), row=2, col=1)

                    # Metabolic
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 6], name="GLP-1 Occupancy"), row=2, col=2)
                    fig.add_trace(go.Scatter(x=blocks, y=states[:, 9], name="Hunger"), row=2, col=2)

                    fig.update_layout(height=600, template="plotly_white", showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                    # Narrative
                    narrative = execute_explain_trajectory({
                        "actions": result["actions"],
                        "states": result["states"],
                        "user_goal": goal_type,
                    })
                    st.markdown(narrative)

                    st.metric("Total Reward", f"{result['total_reward']:.3f}")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you've trained a policy first (see instructions in Tab 1)")

with tab3:
    st.header("Architecture")
    st.markdown("""
    ```
    User Goal (English) → LLM Agent (Claude) → Reward Weights
                                                      ↓
    RL Agent (SAC) ←──── World Model (VAE) ←──── Simulator (Gymnasium)
         ↓                                            ↑
    Optimized Protocol                         50k random transitions
         ↓                                    + DAgger refresh data
    Dashboard (this page)
    ```

    **Components:**
    - **Simulator:** Gymnasium env with 6 compounds, 12 state dims, 8 action dims
    - **World Model:** VAE (encoder 12→8, transition 8+8→8, decoder 8→12, reward predictor)
    - **RL Agent:** SAC from Stable-Baselines3, trained both model-free and model-based
    - **DAgger Loop:** Every 5k dream steps, collect 500 real-env steps, fine-tune World Model
    - **LLM Agent:** Claude API with 2 tools (configure_and_optimize, explain_trajectory)
    """)

    st.caption("Built as a portfolio project demonstrating World Models + RL. See docs/DESIGN.md for full details.")
