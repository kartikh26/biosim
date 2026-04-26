# BioSim: Run Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step-by-Step Execution (in order)

### Phase 1: Validate Simulator (~30 min)

```bash
# Run calibration tests — ALL must pass before proceeding
pytest tests/test_calibration.py -v

# If tests fail, tune parameters in biosim/compounds.py and biosim/env.py
# Common issues:
#   - Bhasin anchor too low → increase muscle_efficacy or t_factor scaling in env.py
#   - Semaglutide weight loss too low → increase glp1 hunger suppression
#   - HPTA not suppressing → increase hpta_suppression_rate
```

### Phase 2: Collect Training Data (~5 min)

```bash
python -m biosim.collect_data --episodes 8500 --output data/replay_buffer.npz
# Creates ~51,000 transitions from random agent
```

### Phase 3: Train World Model (~15 min on GPU)

```bash
python -m biosim.train_world_model \
    --buffer data/replay_buffer.npz \
    --save data/world_model.pt \
    --epochs 100 \
    --device auto
```

### Phase 4: Validate World Model (~2 min)

```bash
python -m biosim.world_model_validation \
    --world-model data/world_model.pt \
    --buffer data/replay_buffer.npz \
    --device auto

# Must see "PRE-FLIGHT GATE: PASS" before proceeding
# If FAIL: try more epochs, more data, or larger hidden dim
```

### Phase 5: Train Model-Free SAC Baseline (~30-60 min)

```bash
python -m biosim.train_sac_modelfree \
    --timesteps 60000 \
    --save data/policies/sac_modelfree \
    --seed 42
```

### Phase 6: Train Model-Based SAC + DAgger (~20-40 min) ⭐ HEADLINE

```bash
python -m biosim.train_sac_modelbased \
    --world-model data/world_model.pt \
    --buffer data/replay_buffer.npz \
    --save data/policies/sac_modelbased \
    --dream-steps 5000 \
    --dagger-iters 12 \
    --dagger-real-steps 500 \
    --device auto
```

### Phase 7: Policy Transfer Gap Test (~2 min)

```bash
python -m biosim.world_model_validation \
    --world-model data/world_model.pt \
    --buffer data/replay_buffer.npz \
    --test3
```

### Phase 8: LLM Agent Test (~1 min)

```bash
# Without API key (offline mode — uses trained policy directly):
python -m biosim.llm_agent --goal "Cut fat but keep muscle, 180 lbs at 22% BF, age 34"

# With API key:
ANTHROPIC_API_KEY=sk-... python -m biosim.llm_agent \
    --goal "Cut fat but keep muscle" \
    --api-key $ANTHROPIC_API_KEY
```

### Phase 9: Launch Dashboard

```bash
streamlit run biosim/dashboard.py
# Opens in browser at http://localhost:8501
```

## Google Colab Notes

If running in Colab:

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Symlink data directory
!ln -sf /content/drive/MyDrive/biosim_data data

# Install deps
!pip install gymnasium numpy torch stable-baselines3 anthropic streamlit plotly

# Clone repo
!git clone https://github.com/yourusername/biosim.git
%cd biosim
```

For Streamlit in Colab:
```python
!pip install pyngrok
from pyngrok import ngrok
!streamlit run biosim/dashboard.py &>/dev/null &
public_url = ngrok.connect(8501)
print(public_url)
```

## Checkpoint Saves (for Colab disconnects)

After each phase, copy outputs to Google Drive:
```bash
cp data/replay_buffer.npz /content/drive/MyDrive/biosim_data/
cp data/world_model.pt /content/drive/MyDrive/biosim_data/
cp -r data/policies/ /content/drive/MyDrive/biosim_data/
```

## What to Cut If Time Runs Short

Priority order (cut from bottom):
1. ❌ Dashboard polish (skip, show terminal output)
2. ❌ LLM agent (skip, demonstrate RL directly)
3. ⚠️ DAgger loop (reduce iterations from 12 to 4)
4. ✅ Model-free vs model-based comparison (NEVER CUT — this is the project)
5. ✅ World Model training + validation (NEVER CUT — prerequisite)
6. ✅ Simulator + calibration tests (NEVER CUT — foundation)
