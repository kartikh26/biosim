# BioSim: World Model + Reinforcement Learning for Fitness Protocol Optimization

> **⚠️ Disclaimer:** This is a portfolio project demonstrating ML techniques. The compound pharmacology is approximate and must NEVER be used for medical decisions. The compound set was chosen for availability of published PK data, not endorsement.

## What This Project Does

BioSim takes a natural-language fitness goal and outputs an optimized 24-week protocol — training, nutrition, and compound recommendations — along with predicted trajectories of body composition and biomarkers.

## The ML Contribution

I built a VAE-based World Model of a pharmacokinetically-grounded human biology simulator, then compared model-based and model-free reinforcement learning:

- **Model-free SAC** trained directly on the simulator (baseline)
- **Model-based SAC** trained inside the World Model's imagination, with DAgger-style distribution shift correction

Key results:
- Sample efficiency: model-based achieves comparable reward in **Nx fewer** real environment steps
- Policy transfer gap: **X%** (reduced to **Y%** with DAgger)
- Divergence horizon: World Model predictions remain accurate for **Z blocks** (Z×4 weeks)

## Architecture

```
User Goal → LLM Agent (Claude) → Reward Weights → RL Agent (SAC) → Protocol
                                                        ↑
                                            World Model (VAE) ← Simulator
```

**Components:**
- **Simulator:** Gymnasium env with 6 compounds, 12 state dims, 8 action dims, pharmacokinetic equations from clinical literature
- **World Model:** VAE encoder/decoder + transition model + reward predictor
- **RL Agent:** SAC (Stable-Baselines3), model-free baseline + model-based with DAgger
- **LLM Agent:** Claude API with tool calling
- **Dashboard:** Streamlit with Plotly charts

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/test_calibration.py -v        # Validate simulator
python -m biosim.collect_data              # 50k transitions
python -m biosim.train_world_model         # Train VAE
python -m biosim.world_model_validation    # Pre-flight check
python -m biosim.train_sac_modelfree       # Baseline
python -m biosim.train_sac_modelbased      # Headline experiment
streamlit run biosim/dashboard.py          # Visualize
```

See [RUN.md](RUN.md) for detailed instructions.

## Why Biology?

The domain has properties that stress-test World Models:
- **Delayed effects** — steroids take weeks to reach steady state
- **Accumulation** — semaglutide saturates GLP-1 receptors over 5 weeks
- **Irreversible states** — HPTA suppression lingers months without PCT
- **Non-linear interactions** — aromatization follows Michaelis-Menten kinetics

These dynamics are more interesting than standard benchmarks (Pendulum, HalfCheetah) for evaluating whether a learned model captures long-horizon dependencies.

## Known Limitations

1. First-order exponential decay for all compounds (peptides technically follow TMDD kinetics)
2. VAE without recurrence (simplified vs Dreamer's full RSSM)
3. 6 compounds in MVP (design doc specifies 16 for full version)
4. Compound interaction magnitudes are estimated, not fully validated
5. No real-world validation — all results are within-simulator

## References

Key papers grounding the pharmacokinetic parameters:
- Bhasin S et al. N Engl J Med. 1996;335(1):1-7. (Testosterone anchor)
- Wilding JPH et al. N Engl J Med. 2021;384(11):989-1002. (STEP 1 / Semaglutide)
- Jastreboff AM et al. N Engl J Med. 2022;387(3):205-216. (SURMOUNT-1 / Tirzepatide)
- Bonnecaze AK et al. Eur J Endocrinol. 2023;189(6):601-608. (HPTA recovery with PCT)

## License

MIT
