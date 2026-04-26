# BioSim: Complete Design Document

## Part 0: The Intuitive Explanation (Read This First)

Imagine you're designing a fitness app that recommends a 24-week plan. A user says "help me lose fat." You could hire a thousand human coaches, have each one design a different plan, run each plan on a volunteer, measure the outcomes, and pick the best. That's impossibly expensive in the real world.

BioSim replaces that process with AI. The clearest analogy: **think of it like training a chess-playing AI, but for biology.** In chess, AlphaZero plays millions of games against itself to learn which moves lead to winning. It needs a "game engine" that knows the rules of chess to play those games. BioSim is the same idea — the RL agent is the "player," the simulator is the "game engine" (the rules of human biology), and "winning" means achieving the user's fitness goals without destroying the body.

### The Components in Plain English

**The simulator is the rulebook.** It's not AI. It's a Python file where you hand-code the pharmacology. "If someone takes 500 mg of testosterone and trains hard over this 4-week block, their muscle mass will increase by this much, their estrogen will increase by this much, their liver stress will increase by this much." You get these numbers from clinical research papers. The simulator is the *ground truth* of the project — everything else is trying to either learn from it or play games within it.

**The World Model is a learned approximation of the rulebook — and it's here to demonstrate a technique, not to save compute.** This is the most important thing to internalize: your simulator is cheap Python equations. If all you wanted was a working optimizer, you'd skip the World Model and just train an RL agent directly on the simulator. You're building a World Model because (a) it's the ML technique you're learning and showcasing, (b) it demonstrates the Dreamer-style "imagine futures and learn from dreams" paradigm, and (c) comparing model-based vs model-free RL is *the* headline experiment of this project. In domains where each real interaction is genuinely expensive (robotics, drug discovery, wet labs), this paradigm is the whole point; your project shows you can execute it in a controlled setting where you can also measure its accuracy precisely.

**The RL agent is the student.** It tries random plans first. Each plan it tries, it gets a score (reward). Over thousands of attempts, it learns patterns: "running testosterone without arimidex crashes my reward because estrogen blows up. Stacking MK-677 with semaglutide fights itself because one raises hunger and the other suppresses it. Planning a proper PCT in the last block matters for the health score."

**The LLM agent is the translator.** A user types "I want to cut fat but keep muscle." The LLM (Claude) converts that into numerical reward weights — e.g., `fat_loss = 0.45, muscle = 0.30, health = 0.20, recovery = 0.05`. Then it calls the RL agent and gets back the recommended plan. Then it translates the raw numbers back into a human explanation: "Block 1 (weeks 1-4): start semaglutide at low dose, maintain moderate training..."

**The dashboard is how the user sees everything.** Charts, timelines, a narrative. Nothing technical.

### What You Actually Execute and Train

You'll train **three separate things** in sequence:

1. **The VAE World Model** — You run a script that shows it ~50,000 examples from the simulator. It learns to predict the simulator. This is supervised learning. Takes minutes on a Colab GPU.
2. **The Model-Free RL Agent** — You run Stable-Baselines3 SAC directly on the simulator. It learns by trial and error on the real simulator. Takes hours.
3. **The Model-Based RL Agent** — You run the same SAC algorithm, but the "environment" is now your World Model, not the real simulator. Runs much faster, and also periodically refreshes itself with real-simulator experience (the DAgger loop, see Part 5).

You do NOT train the simulator. You do NOT train the LLM agent. Those are both *configured* rather than *learned* — the simulator is hand-coded equations, and the LLM agent uses prompt engineering + tool calling.

### The MVP Fallback (If You Get Overwhelmed)

If you feel stuck at any point, build the simplest possible version first:

- Skip the VAE entirely.
- Use a simulator with only 2 compounds (testosterone + semaglutide) and a handful of action dimensions.
- Train SAC model-free on this tiny simulator.
- Verify you can generate a 24-week protocol.

This takes about one day and tells you whether your core pipeline works. Then scale up. A working small project is infinitely better than an ambitious project that isn't finished by the deadline.

### Ethical Framing (Read Before Publishing)

This project predicts outcomes from anabolic steroids at supraphysiological doses. That is sensitive content. Be deliberate in how you frame it publicly:

- The compound choice is driven by **availability of published pharmacokinetic data**, not endorsement. The same architecture works just as well for FDA-approved medications, training interventions, or nutritional regimes — the pharmacology set here was picked because clinical trials have published PK parameters you can plug in.
- The system is a **portfolio demonstration of ML technique**, not a clinical tool. Do not use it for medical decisions.
- The **health reward weight has a hard floor of 0.15** that cannot be lowered by the LLM agent regardless of user request (see Part 6). This prevents the optimizer from producing protocols that trade away liver function, HPTA axis integrity, or cardiovascular health in pursuit of faster results.
- Both the README and the dashboard display a prominent disclaimer on every screen.

---

## Part 1: Formal Project Summary

BioSim is an ML system that takes a natural-language fitness goal and outputs an optimized 24-week protocol — week-by-week training, nutrition, and compound recommendations — along with predicted trajectories of body composition and key biomarkers.

The technical architecture has four learned or configured components layered on a pharmacokinetic simulator: (1) a hand-coded biology simulator implemented as a Gymnasium environment, (2) a VAE-based World Model trained on simulator trajectories, (3) a SAC-based reinforcement learning agent trained both model-free (baseline) and model-based (main experiment, via Dreamer-style imagination rollouts), and (4) a Claude-powered LLM agent that translates user intent into reward configurations and narrates results.

The headline experiment compares sample efficiency of model-free vs model-based RL on a controlled problem, with World Model accuracy measured explicitly via k-step prediction error and policy-transfer gap (Part 4.5). A DAgger-style iteration loop (Part 5) addresses distribution shift between the random-agent-generated training data and the on-policy trajectories the RL agent cares about.

The end deliverable is a portfolio project: a GitHub repo with runnable code, a React dashboard for interactive exploration of protocols, and a writeup that honestly discusses the project's scope, limitations, and what was learned. See the architecture diagram in Part 2 for the end-to-end flow.

---

## Part 2: The Architecture

```
END-TO-END FLOW
===============

USER TYPES: "I'm 200 lbs at 25% body fat. Cut to 15%. 
             Use semaglutide and testosterone. No orals."
        │
        ▼
┌─────────────────────────────────────────────────┐
│  COMPONENT 4: LLM AGENT (Claude API)            │
│                                                 │
│  - Parses English into numerical parameters.    │
│  - Calls the RL Agent via function calling.     │
│  - Narrates the results back to the user.       │
└────────────────────┬────────────────────────────┘
                     │ configure_and_optimize(params)
                     ▼
┌─────────────────────────────────────────────────┐
│  COMPONENT 3: RL AGENT (SAC algorithm)          │
│                                                 │
│  Goal: find the best 24-week protocol that      │
│  maximizes the user's reward function.          │
│                                                 │
│  Method: "dreams" thousands of protocols        │
│  inside the World Model's imagination.          │
│  Periodically refreshes with real-sim data      │
│  (DAgger loop).                                 │
└────────────────────┬────────────────────────────┘
                     │ imagined rollouts
                     ▼
┌─────────────────────────────────────────────────┐
│  COMPONENT 2: VAE WORLD MODEL                   │
│                                                 │
│  Four neural networks working together:         │
│                                                 │
│   ┌─────────┐         ┌──────────────┐          │
│   │ Encoder │ ──────► │  Transition  │          │
│   │ 18 → 8  │         │    Model     │          │
│   └─────────┘         │ predicts z'  │          │
│                       └──────┬───────┘          │
│                              │                  │
│   ┌─────────┐         ┌──────▼───────┐          │
│   │ Decoder │ ◄────── │   Reward     │          │
│   │ 8 → 18  │         │  Predictor   │          │
│   └─────────┘         └──────────────┘          │
│                                                 │
│  Trained on data from the simulator.            │
│  Periodically refreshed by DAgger loop.         │
└────────────────────┬────────────────────────────┘
                     │ learned dynamics
                     ▲
                     │
┌─────────────────────────────────────────────────┐
│  COMPONENT 1: THE SIMULATOR                     │
│                                                 │
│  Gymnasium environment. env.step() applies      │
│  pharmacokinetic equations across a 4-week      │
│  decision block to update body state.           │
│                                                 │
│  18 state dims + 18 action dims per block.      │
│  6 decision blocks per 24-week episode.         │
│  Runs deterministically. No AI — just math.     │
└─────────────────────────────────────────────────┘
                     ▲
                     │ random agent generates
                     │ ~50,000 transitions
                     │ + DAgger refresh every 5k RL steps
                     │
                [REPLAY BUFFER]
                     ▲
                     │
            ┌────────┴─────────┐
            │  Used to train   │
            │  World Model AND │
            │  for model-free  │
            │  RL baseline     │
            └──────────────────┘
```

---

## Part 3: The Simulator (Component 1)

### What It Is

The simulator is a Python file that contains the "physics" of the human body. You write it as a Gymnasium environment (the standard API for RL). It is NOT an AI — it's a collection of equations. Every call to `env.step(action)` does this:

1. Takes the current body state (18 numbers) and this block's actions (18 numbers, held constant for 4 weeks).
2. Integrates the pharmacokinetic equations forward for 4 weeks at daily resolution internally.
3. Returns the new body state (18 updated numbers), reward (1 number), and whether the episode is done.

**Episode structure:** one episode = 24 weeks = 6 decision blocks of 4 weeks each. The agent makes 6 decisions per episode, not 24.

**Why block-structured actions (4-week blocks, not weekly):** Real protocols don't change all 18 knobs every week. Compounds run for cycles, PCTs last several weeks, caloric targets are held. If the agent were allowed to flip doses weekly, it would discover noisy, unrealistic policies ("tren week, off week, tren week"). Block structure matches how real protocols are designed, keeps the sequential RL structure intact (6 sequential decisions), and reduces exploration difficulty. See Part 9 for why we didn't pick other variants.

### The Observation Space (18 Dimensions)

"Observation" means the 18 numbers the agent sees when deciding what to do. At any point during an episode, the state vector looks roughly like this (numbers are illustrative):

```
[lean_mass=72.3, fat_subcut=11.2, fat_visc=4.1,
 testosterone=580, estrogen=32, progesterone=0.4,
 igf1=220, natural_gh=0.9, glp1_occupancy=0.0,
 liver_stress=0.1, recovery=0.8, hpta_suppression=0.2,
 water=0.1, hunger=0.5, thermogenesis=0.3,
 e2_blockade=0.0, androgenic_load=0.2, block=1]
```

Full variable list:

| # | Variable | What It Represents |
|---|---|---|
| 1 | lean_mass | Muscle tissue (kg) |
| 2 | fat_mass_subcutaneous | Pinchable fat under skin (kg) |
| 3 | fat_mass_visceral | Fat around organs (kg) — Tesamorelin target |
| 4 | testosterone_level | Circulating testosterone (ng/dL) |
| 5 | estrogen_level | Circulating estradiol (pg/mL) |
| 6 | progesterone_level | Elevated by Nandrolone specifically |
| 7 | igf1_level | Insulin-like Growth Factor 1 (ng/mL) |
| 8 | natural_gh_production | 0-1, suppressed by exogenous HGH |
| 9 | glp1_receptor_occupancy | 0-1, driven by GLP-1 agonists |
| 10 | liver_stress | 0-1, accumulates with orals and RAD-140 |
| 11 | recovery_status | 0-1, depleted by training, restored by rest/BPC-157 |
| 12 | hpta_suppression | 0-1, all androgens suppress, Nolvadex recovers (see naming note below) |
| 13 | water_retention | 0-1, from Dianabol and high estrogen |
| 14 | hunger_drive | 0-1, MK-677 increases, GLP-1 agonists suppress |
| 15 | thermogenesis_rate | 0-1, elevated by Retatrutide glucagon, Trenbolone |
| 16 | estrogen_receptor_blockade | 0-1, Nolvadex mechanism |
| 17 | androgenic_load | 0-1, DHT-driven side effects (hair loss, etc.) |
| 18 | block_number | Episode clock, 0-6 |

> **Naming note (HPA vs HPTA):** Bodybuilding-community shorthand often refers to "HPA suppression" when discussing androgen cycles, but the technically correct term is **HPTA** — Hypothalamic-Pituitary-Testicular Axis. The HPA axis (Hypothalamic-Pituitary-Adrenal) regulates cortisol/stress, not testosterone. This simulator models HPTA suppression specifically. The variable is named `hpta_suppression` throughout the codebase.

### The Action Space (18 Dimensions, 6 Decisions Per Episode)

At each of the 6 decision points (blocks), the agent outputs 18 continuous values in [0, 1]. Each value is a normalized dose held constant for the next 4 weeks.

| # | Variable | Range | Maps to |
|---|---|---|---|
| 1 | training_volume | 0-1 | Weekly training load |
| 2 | caloric_balance | -1 to 1 | Deficit or surplus |
| 3 | testosterone_dose | 0-1 | Up to 500 mg/week |
| 4 | trenbolone_dose | 0-1 | Up to 400 mg/week |
| 5 | nandrolone_dose | 0-1 | Up to 400 mg/week |
| 6 | dianabol_dose | 0-1 | Up to 50 mg/day oral |
| 7 | anavar_dose | 0-1 | Up to 50 mg/day oral |
| 8 | rad140_dose | 0-1 | Up to 20 mg/day |
| 9 | hgh_dose | 0-1 | Up to 10 IU/day |
| 10 | cjc_ipamorelin_dose | 0-1 | Up to 300 mcg/day |
| 11 | tesamorelin_dose | 0-1 | Up to 2 mg/day |
| 12 | bpc157_dose | 0-1 | Up to 500 mcg/day |
| 13 | mk677_dose | 0-1 | Up to 25 mg/day |
| 14 | semaglutide_dose | 0-1 | Up to 2.4 mg/week |
| 15 | tirzepatide_dose | 0-1 | Up to 15 mg/week |
| 16 | retatrutide_dose | 0-1 | Up to 12 mg/week |
| 17 | arimidex_dose | 0-1 | Up to 1 mg/day |
| 18 | nolvadex_dose | 0-1 | Up to 40 mg/day |

### Pharmacokinetic (PK) Equations

**What is a PK equation?** Pharmacokinetics is the branch of pharmacology that describes how a drug's concentration changes in the body over time: absorption, distribution, metabolism, elimination. A PK equation is a mathematical model of those dynamics. For most compounds in this project, the dominant dynamic is **first-order exponential decay** — the drug clears the body at a rate proportional to how much is present, which produces the familiar half-life curve.

#### Core Exponential Decay Model (Applied Daily Within a Block)

```python
level(t+1_day) = level(t) * 0.5 ** (1 / half_life_days) + daily_dose * absorption_factor
```

Steady state is reached after 4-5 half-lives.

#### Michaelis-Menten Aromatization (Non-Linear)

Testosterone → estradiol conversion saturates at high doses, so use Michaelis-Menten kinetics rather than linear scaling:

```python
V_max = V_max_base * (1.4 if age >= 40 else 1.0)  # older males aromatize more
K_m = 1.83  # nM, from clinical literature
estrogen_from_aromatization = V_max * testosterone_level / (K_m + testosterone_level)
```

#### Age-Stratified Baselines (applied at episode reset)

```python
if age < 30:
    endogenous_T_baseline_ng_dL = uniform(413, 1000)
    igf1_baseline_ng_mL = uniform(116, 358)
elif age < 40:
    endogenous_T_baseline_ng_dL = uniform(352, 1000)
    igf1_baseline_ng_mL = uniform(94, 284)
elif age < 50:
    endogenous_T_baseline_ng_dL = uniform(252, 916)
    igf1_baseline_ng_mL = uniform(75, 238)
```

#### Per-Compound Parameters

All parameters below are sourced from primary clinical literature. Citations are inline. Where a value depends on context (e.g., aromatase inhibitor potency in suppressed vs intact HPTA states), the contextual rule is given. The simulator's PK module must implement these as parameterized constants in `biosim/envs/compounds/`.

**Testosterone Enanthate (Bhasin 1996, PMID 8637535)**
- Half-life: 4.5 ± 1.5 days
- Time to steady state: 21-35 days
- Max dose: 500 mg/week (performance range 300-600 mg/week)
- Anchor: 600 mg/week + resistance training → +6.1 kg fat-free mass over 10 weeks (Bhasin et al., N Engl J Med, 1996)
- Aromatization: ~0.4% per-pass conversion (Bhasin 2001, PMID 11701429); cumulative daily conversion produces 60-100 pg/mL estradiol at 600 mg/week. Calibrate Michaelis-Menten V_max so steady-state E2 lands in this range.
- 5α-reductase: ~5-7% conversion to DHT
- HPTA suppression: rapid onset (LH/FSH near-zero within 14 days at >200 mg/week)
- HPTA recovery: median 13 weeks with PCT (IQR 8-19), median 26 weeks without (Bonnecaze 2023, n=641)
- Acne incidence: 42% at supraphysiological dose; erythrocytosis (Hct >50%): 20% (Coviello 2008, PMID 18160461)

**Trenbolone Acetate (Yarrow 2010, PMID 20133493)**
- Half-life: 1-3 days (rodent data; human PK not directly characterized)
- Time to steady state: 5-15 days
- Max dose: 400 mg/week
- Aromatization: 0% (no estrogen conversion)
- 5α-reductase: 0% (binds AR directly)
- Progesterone receptor binding: significant — drives prolactin elevation; **Arimidex does not address this pathway**
- Stronger anabolic effect per mg than testosterone via direct AR binding (~3x affinity)
- Confidence: LOW — primary data is rodent. Human side-effect incidence (insomnia, night sweats, dyslipidemia) is not RCT-quantified.

**Nandrolone Decanoate (Sattler 2002, PMID 12213858)**
- Half-life: 6-12 days
- Time to steady state: 30-60 days
- Max dose: 400 mg/week (performance range 200-600 mg/week)
- Anchor: 200 mg/week → +3.0 kg lean body mass over 12 weeks in HIV-wasting men, CI [2.2, 3.8]
- Aromatization: ~20% the rate of testosterone (Bergink 1985, PMID 4021486)
- 5α-reductase: converts to 5α-dihydronandrolone (DHN), which has *lower* AR affinity than parent — explains tissue selectivity (high muscle anabolism, lower prostate effect)
- Progesterone receptor: ~20% affinity vs endogenous progesterone — significant prolactin elevation
- Joint/connective tissue benefit via collagen synthesis

**Methandrostenolone / Dianabol (Hervey 1981, PMID 7018798)**
- Half-life: 3-6 hours
- Time to steady state: 15-30 hours
- Max dose: 50 mg/day (performance range 20-50 mg/day)
- Anchor: 100 mg/day for 6 weeks → +3.3 ± 1.2 kg body weight (large fluid component)
- Aromatization: yields 17α-methylestradiol — a metabolism-resistant, exceptionally potent estrogen analog. Weight estrogen receptor occupancy higher than molar conversion would suggest.
- Liver stress: 100% of users at >50 mg/day show transient ALT/AST elevation; hypertension via Na+ retention >50%
- Confidence: MEDIUM — human data exists but largely from older trials predating modern ethical restrictions on supraphysiological dosing.

**Oxandrolone / Anavar (Schroeder 2004, PMID 14594863)**
- Half-life: 9.4-10.4 hours
- Time to steady state: 2-3 days
- Max dose: 50 mg/day (performance range 20-50 mg/day)
- Anchor: 20 mg/day for 12 weeks → +2.0 kg total lean body mass (CI [1.3, 2.7]) in older men
- Aromatization: 0% (DHT-derived)
- 5α-reductase: 0%
- Clearance: ~30% renal (unique among 17α-alkylated compounds — much lower hepatic burden than Dianabol)
- HDL reduction: 40-50% from baseline; ALT/AST elevation: 15%
- Preferentially oxidizes visceral fat

**RAD-140 / Testolone (LoRusso 2022, PMID 34815233)**
- Half-life: ~60 hours (significant accumulation over 12-15 days at once-daily dosing)
- Time to steady state: 12-15 days
- Max dose: 20 mg/day (performance); therapeutic oncology dose 50-150 mg/day
- Aromatization: 0%
- 5α-reductase: 0% (non-steroidal)
- Significant HPTA suppression despite tissue-selective design (still exerts central negative feedback)
- Hepatotoxicity: ~20% ALT/AST elevation at oncology doses; documented DILI cases at performance doses
- Confidence: MEDIUM — first-in-human Phase 1 data only, in postmenopausal breast cancer cohort. Performance use is extrapolation.

**Exogenous HGH / Somatropin (Blackman 2002, PMID 12425705)**
- Half-life: 2-3 hours
- Time to IGF-1 steady state: 3-5 days
- Max dose: 10 IU/day (replacement 1-2 IU/day, performance 4-10 IU/day)
- Anchor: ~2-3 IU/day equivalent for 26 weeks → +2.1 kg lean body mass in older men, CI [1.3, 2.9]
- **IGF-1 dose-response is non-linear**: linear up to 4-6 IU/day, then plateaus. Above the plateau, you get insulin resistance and side effects without further IGF-1 benefit. The simulator must model this as a saturable Emax curve, not a linear scaling.
- **Female sensitivity factor: 1.7x.** Women require 1.7x the dose for equivalent IGF-1 response (males ED50 ≈ 9.5 µg/kg/day, females 16.5 µg/kg/day). Oral estrogen attenuates hepatic GH action via STAT5 downregulation.
- Suppresses endogenous GH pulses via somatostatin negative feedback
- Side effects: peripheral edema 39%, carpal tunnel/arthralgia 24%

**CJC-1295 (with DAC) + Ipamorelin (Teichman 2006, PMID 16352683)**
- CJC-1295 with DAC half-life: 5.8-8.1 days (binds endogenous albumin via DAC moiety)
- Ipamorelin half-life: ~2 hours
- Single CJC dose at 30-60 µg/kg → mean GH AUC +2x to +10x baseline; IGF-1 +1.5x to +3x; trough GH +7.5x
- IGF-1 elevation persists 9-11 days after a single dose; 28+ days with chronic dosing
- Preserves endogenous GH pulsatility (does NOT suppress natural GH like exogenous HGH does)
- Ipamorelin amplifies pulse magnitude via ghrelin receptor without cortisol/prolactin spike
- Confidence: MEDIUM — solid Phase 1/2 data for CJC-1295; the specific stack with Ipamorelin is mechanistic extrapolation.

**Tesamorelin (Falutz 2010, PMID 20554713)**
- Half-life: ~38 minutes
- Time to steady state: 1-2 days
- Max dose: 2 mg/day
- Anchor: 2 mg/day for 26 weeks → -15.4% visceral adipose tissue (-24 ± 41 cm²) in HIV lipodystrophy, n=806
- Subcutaneous fat: non-significant change (the simulator must enforce VAT specificity — this is a hallmark)
- Lean mass: maintained or improved
- Effect reverses on cessation (rapid VAT regain within months)
- Side effects: arthralgia 13.3%, injection site erythema 8.5%

**BPC-157 (Sikiric 2010, PMID 20166993)**
- Half-life: NOT AVAILABLE — minutes to hours estimated from rodent data
- Max dose: 500 mcg/day (extrapolated from animal equivalent dosing)
- **No human clinical trials exist.** All efficacy data is rodent.
- Modeled effect: multiplier on recovery_status restoration; localized angiogenesis via VEGFR2 upregulation
- Stable in extreme gastric pH (oral bioavailability higher than typical peptides)
- Confidence: LOW — purely extrapolated. Flag in code comments as "qualitative direction only, magnitude speculative."

**MK-677 / Ibutamoren (Murphy 1999, PMID 9920063 / Chapman 1996, PMID 8954023)**
- Half-life: ~24 hours (orally active)
- Time to steady state: 3-5 days
- Max dose: 25 mg/day
- Anchor: 25 mg/day for 4 weeks → 24-hour GH AUC +97% ± 23%; IGF-1 from 141 → 264 µg/L (+87.9%)
- Strongly increases hunger drive (>50% of users) via ghrelin receptor — directly conflicts with GLP-1 agonist appetite suppression
- Tachyphylaxis: appetite-stimulating effect attenuates within months
- Insulin sensitivity: declines (fasting glucose +1.4 mmol/L over 4 weeks)
- Long-term: +1.1 kg fat-free mass over 12 months; transient +2.7 kg total weight (mostly fluid + intake)
- Mild cortisol elevation (+47 nmol/L)

**Semaglutide (Wilding 2021, STEP 1, PMID 33567185)**
- Half-life: 165 hours (~7 days)
- Time to steady state: 4-5 weeks
- Max dose: 2.4 mg/week
- **Dose escalation required:** start 0.25 mg/week, step up every 4 weeks over 16 weeks. The simulator must enforce this titration; jumping straight to 2.4 mg produces unrealistic GI side effect burden.
- Anchor: 2.4 mg/week for 68 weeks → -14.9% body weight ± 9.3% (n=1961, ≥5% loss in 86.4%, ≥15% loss in 50.5%)
- **Cessation rebound:** patients regain ~2/3 of lost weight within 52 weeks of stopping (STEP 1 extension). The simulator must implement a hyperphagic rebound curve when GLP-1 receptor occupancy decays.
- **Metabolic adaptation plateau:** weight loss curve flattens around weeks 60-68 as resting energy expenditure drops to match drug-induced caloric deficit
- Side effects: nausea 44.2%, diarrhea 31.5%

**Tirzepatide (Jastreboff 2022, SURMOUNT-1, PMID 35658024)**
- Half-life: 116 hours (~5 days)
- Time to steady state: 4 weeks
- Max dose: 15 mg/week (with mandatory 20-week dose escalation: start 2.5 mg, +2.5 mg every 4 weeks)
- Anchor: 15 mg/week for 72 weeks → **-20.9% body weight (CI [-21.8, -19.9]), n=2539** — *Note: this is the corrected value. The previous 22.5% figure in the doc was incorrect.*
- Dose-response: 5 mg → -15.0%, 10 mg → -19.5%, 15 mg → -20.9%
- **Time to weight plateau (TTWP) varies by BMI:** overweight 24w, class I obesity 26w, class II/III obesity 36w. The simulator should reflect that higher-BMI users plateau later.
- Dual GLP-1/GIP mechanism: GIP component reduces nausea relative to pure GLP-1 agonists at equivalent weight loss
- Side effects: nausea 33.3% (lower than Semaglutide), constipation 17.1%

**Retatrutide (Jastreboff 2023, TRIUMPH Phase 2, PMID 37366315)**
- Half-life: ~144 hours (~6 days)
- Time to steady state: 4-5 weeks
- Max dose: 12 mg/week
- Anchor: 12 mg/week for 48 weeks → -24.2% body weight; 8 mg → -22.8%; placebo -2.1%; n=338
- **Does NOT plateau within 48 weeks** — weight continues to decline through trial end. Unique among incretin-class drugs.
- **Triple GLP-1/GIP/GCG mechanism:** glucagon receptor agonism actively raises basal metabolic rate (E_out term in energy balance), unlike pure GLP-1/GIP which only suppress E_in
- **Liver fat normalization:** -82.4% liver fat at 12 mg by week 24 (effectively normalizes liver fat in 86% of patients)
- **"Super-responder" rate elevated:** ~40% of patients achieve ≥30% body weight loss vs 10-15% for tirzepatide
- Transient dose-dependent heart rate elevation (glucagon chronotropy)
- Side effects: nausea ~45%; HR elevation peaks at week 24 then declines
- Confidence: MEDIUM — Phase 2 only; Phase 3 not yet complete.

**Anastrozole / Arimidex (Mauras 2000, PMID 10902781 / Leder 2004, PMID 15001605)**
- Half-life: 40-50 hours
- Time to steady state: 7-10 days
- Max dose: 1 mg/day (typical performance: 0.25-0.5 mg, 2x/week)
- **E2 suppression is context-dependent — this is the most important parameter to model correctly:**
  - In men with **HPTA suppressed by exogenous androgens**: 80-95% E2 suppression at 1 mg/day (testicular aromatization is the only remaining contributor and gets blocked)
  - In men with **active HPTA** (TRT-only, post-cycle, or healthy baseline): only ~35% E2 suppression — testicular aromatase is more resistant to inhibition AND endogenous testosterone surges (+58-67%) as estrogen-mediated negative feedback is removed, partly compensating
  - The simulator's anastrozole module must check the user's HPTA state and apply the appropriate suppression magnitude.
- Restores LH/FSH by removing estrogen negative feedback
- Side effects in men: arthralgia ~15%, decreased BMD on long-term use
- Crashed E2 (<10 pg/mL) penalty: see reward function

**Tamoxifen / Nolvadex (Tenover 1992, PMID 1400877 / Bonnecaze 2023)**
- Half-life: 5-7 days
- Time to steady state: 3-4 weeks
- Max dose: 40 mg/day (typical PCT: 20-25 mg/day)
- **Tissue-selective action — different effect mid-cycle vs PCT:**
  - **Mid-cycle:** Tamoxifen acts as estrogen agonist in the liver, downregulating IGF-1 synthesis by 16-50% and elevating IGFBP-1. This **blunts mid-cycle anabolic gains** by reducing free IGF-1 available to muscle. The simulator should apply this penalty to muscle gain when tamoxifen is co-administered with active androgens.
  - **Post-cycle (PCT):** Tamoxifen acts as estrogen antagonist at hypothalamic/pituitary receptors, restarting LH/FSH and accelerating endogenous T recovery. Median recovery 13 weeks (IQR 8-19) with PCT vs 26 weeks (IQR 10.5-52) without (Bonnecaze 2023, n=641).
- Liver: estrogen agonist effect actually *improves* lipid profile (unlike AIs which crush HDL)
- Side effects: hot flashes ~80%, VTE risk <1% (statistically significant)

### Reward Function

```python
# Tamoxifen mid-cycle muscle gain blunt:
# Tamoxifen + active androgens → IGF-1 synthesis reduced 16-50%, blunting muscle gains
if tamoxifen_dose > 0 and any_active_androgen():
    delta_lean_mass_normalized *= (1 - 0.30 * tamoxifen_blunt_factor)

reward = (
    w_muscle    * delta_lean_mass_normalized +
    w_fat_loss  * delta_fat_loss_normalized +
    w_health    * (1.0 - liver_stress) +
    w_health    * (1.0 - hpta_suppression) +
    w_recovery  * recovery_status
)

# Two-tier estrogen penalty (replaces previous single threshold)
if estrogen_level > 60:                  reward -= high_estrogen_penalty
if 10 <= estrogen_level < 15:            reward -= mild_crashed_e2_penalty   # symptoms emerge
if estrogen_level < 10:                  reward -= severe_crashed_e2_penalty # severe symptoms

# Safety thresholds
if liver_stress > 0.8:                                      reward -= severe_liver_penalty
if androgenic_load > 0.85:                                  reward -= androgenic_penalty
if hpta_suppression > 0.95 and block_number >= 5:           reward -= failed_pct_penalty

# Progestogenic load penalty (independent of E2 — Arimidex doesn't help)
# Driven by trenbolone + nandrolone PR binding, elevates prolactin
if progesterone_level > 0.7:                                reward -= progestogenic_penalty

# GLP-1 receptor saturation cap (semaglutide + tirzepatide stacking)
# Stacking does NOT increase efficacy but DOES increase adverse events
if glp1_receptor_occupancy > 0.85 and (semaglutide_dose > 0.3 and tirzepatide_dose > 0.3):
    reward -= glp1_overstacking_penalty  # supratherapeutic GI side-effect risk

# Action smoothness (prevents oscillating protocols)
reward -= lambda_smooth * ||action_t - action_{t-1}||^2
```

Weights `w_muscle`, `w_fat_loss`, `w_health`, `w_recovery` are configured by the LLM agent per user request, subject to the hard floor `w_health >= 0.15` (Part 6). `lambda_smooth` is a small constant (start 0.02).

### Caloric Economics (Calibration Anchors)

- 7,700 kcal deficit = 1 kg fat loss
- 1,500-2,000 kJ/day surplus required per 1 kg lean muscle
- Protein threshold: 1.6 g/kg/day
- Natural trainee baseline: ~1.5 kg FFM over 8-12 weeks without drugs
- Bhasin: 600 mg/week testosterone + training → +6.1 kg over 10 weeks

---

## Part 3.5: Simulator Calibration Tests

Before you train anything else, you must verify that your hand-coded simulator produces results that match published clinical data. Without this, everything downstream is building on sand. These are standard `pytest` tests that live at `tests/test_simulator_calibration.py`.

### Structure

Each calibration test: (a) initializes the simulator to a known starting state, (b) runs a specific protocol for a specific duration, (c) asserts the output matches a published clinical result within a tolerance.

### Required Tests

Each test below cites the primary trial. Tolerances are derived from reported confidence intervals or 1.5× standard deviation. Confidence ratings indicate source quality (HIGH = primary RCT directly verified; MEDIUM = primary trial with secondary uncertainty; LOW = extrapolated, conference abstract, or observational data).

#### PK Shape Tests (verifying drug kinetics, no biological outcome)

1. `test_testosterone_halflife`: inject 500 mg testosterone enanthate, simulate 4 weeks. Verify blood level curve has half-life 4-5 days (±10%). [Bhasin 1996, PMID 8637535] **Confidence: HIGH**
2. `test_semaglutide_steady_state`: dose 2.4 mg/week for 8 weeks, verify steady state reached by week 5 (±1 week). [Wilding 2021, PMID 33567185] **Confidence: HIGH**
3. `test_trenbolone_no_aromatization`: 400 mg/week trenbolone × 8 weeks in isolation, verify estradiol stays near baseline. [Yarrow 2010, PMID 20133493] **Confidence: HIGH**
4. `test_arimidex_suppression_suppressed_hpta`: 1 mg/day arimidex + 600 mg/week test × 8 weeks. Verify estradiol suppressed 80-95%. [Mauras 2000, PMID 10902781] **Confidence: HIGH**
5. `test_arimidex_suppression_active_hpta`: 1 mg/day arimidex × 12 weeks in mildly hypogonadal man with intact HPTA, no exogenous T. Verify only ~25-45% E2 suppression AND endogenous testosterone surges +50-85%. [Leder 2004, PMID 15001605] **Confidence: HIGH**
6. `test_rad140_accumulation`: 20 mg/day RAD-140 × 15 days. Verify steady state reached at 12-15 days due to ~60-hour half-life. [LoRusso 2022, PMID 34815233] **Confidence: MEDIUM**

#### Biological Outcome Tests

7. `test_bhasin_testosterone_anchor`: 600 mg/week testosterone enanthate + resistance training × 10 weeks on healthy 19-40 y/o male. Target: 4.5 ≤ delta_FFM_kg ≤ 8.0 (Bhasin 1996, PMID 8637535, n=21 in relevant arms; +6.1 kg mean). **Confidence: MEDIUM** — original paper confirms +6.1 kg but primary source confidence interval [4.5, 7.7] is widened here to account for biological variability and uncertainty in reproducing the trial population.
8. `test_step1_semaglutide_anchor`: 2.4 mg/week semaglutide × 68 weeks on obese baseline (BMI ≥ 30), with mandatory 16-week dose escalation. Target: 13.0% ≤ |body_weight_pct_change| ≤ 16.5%. Verify ≥5% loss in 80-92% of simulated trials. [Wilding 2021, PMID 33567185, n=1961, -14.9% ± 9.3% mean] **Confidence: HIGH**
9. `test_step1_semaglutide_cessation_rebound`: continue test 8, then cease semaglutide at week 68 and simulate 52 more weeks. Target: net body weight regain of ~2/3 of lost weight. The simulator's GLP-1 receptor decay must trigger a hyperphagic rebound. **Confidence: HIGH**
10. `test_surmount_tirzepatide_anchor`: 15 mg/week tirzepatide × 72 weeks with mandatory 20-week dose escalation, on BMI ≥ 30 baseline. Target: 18.8% ≤ |body_weight_pct_change| ≤ 23.0% [Jastreboff 2022, PMID 35658024, n=2539, -20.9% mean, CI [-21.8, -19.9]]. **Confidence: HIGH** — *note: this corrects a previous erroneous 22.5% target.*
11. `test_tirzepatide_dose_response`: run 5/10/15 mg/week tirzepatide × 72 weeks. Verify monotonic dose-response: 5 mg → ~-15.0%, 10 mg → ~-19.5%, 15 mg → ~-20.9%. **Confidence: HIGH**
12. `test_triumph_retatrutide_anchor`: 12 mg/week retatrutide × 48 weeks. Target: 21.0% ≤ |body_weight_pct_change| ≤ 27.0% [Jastreboff 2023, PMID 37366315, n=338 in 12 mg arm, -24.2% mean]. Also verify weight loss does NOT plateau by week 48. **Confidence: MEDIUM** — Phase 2 only.
13. `test_retatrutide_super_responder_rate`: simulate 100 trials at 12 mg/week × 48 weeks. Target: 30-50% of simulated patients achieve ≥30% body weight loss (vs only ~10-15% on tirzepatide). **Confidence: MEDIUM**
14. `test_tesamorelin_visceral_specificity`: 2 mg/day tesamorelin × 26 weeks on HIV lipodystrophy baseline. Target: -10% ≤ delta_VAT_pct ≤ -20% (centered on -15.4%); SAT change must be statistically non-significant (within ±3%); lean mass maintained or increased [Falutz 2010, PMID 20554713, n=806]. **Confidence: HIGH**
15. `test_tesamorelin_responder_rate`: simulate 100 trials. Target: ≥30% of simulated patients achieve ≥8% VAT reduction. **Confidence: HIGH**
16. `test_hgh_igf1_dose_response`: vary HGH dose from 0.5 to 10 IU/day for 12 weeks. Verify IGF-1 increase is linear up to 4-6 IU/day, then plateaus. The simulator's IGF-1 generator MUST saturate above this dose. [Page 2008 meta-analysis; secondary source — flagged as needing better citation in future Deep Research pass]. **Confidence: MEDIUM** — well-established physiology but specific dose-response source is a conference abstract; primary trials exist but were not cleanly surfaced. Re-run Deep Research with target query: "primary RCTs measuring IGF-1 response curve to recombinant HGH dose escalation in adult GH-deficient or healthy populations."
17. `test_hgh_female_sensitivity_factor`: simulate identical HGH dose (3 IU/day × 8 weeks) on male and female avatars. Verify female IGF-1 response is ~60% of male (1.7x dose-equivalence factor). **Confidence: MEDIUM**
18. `test_mk677_igf1_anchor`: 25 mg/day MK-677 × 4 weeks on healthy elderly avatar. Target: IGF-1 increase 75-105% (Murphy 1999, PMID 9920063, +87.9% mean from 141 → 264 µg/L); 24-hour GH AUC increase 70-125% (+97% ± 23% mean). **Confidence: HIGH**
19. `test_hpta_recovery_with_pct`: 500 mg/week testosterone × 16 weeks → 20 mg/day nolvadex PCT × 13 weeks. Target: full HPTA recovery (hpta_suppression < 0.2) within 8-19 weeks of PCT initiation [Bonnecaze 2023, n=641, median 13 weeks IQR 8-19]. **Confidence: MEDIUM** — observational, not RCT.
20. `test_hpta_recovery_no_pct`: same cycle, no PCT. Target: full recovery within 21-32 weeks (median 26 weeks IQR 10.5-52). **Confidence: MEDIUM**
21. `test_tamoxifen_mid_cycle_blunt`: run 600 mg/week test × 10 weeks (matching Bhasin) WITH co-administered 20 mg/day tamoxifen throughout. Target: lean mass gain reduced by 16-50% vs no-tamoxifen baseline (i.e., target shifts to 2.5-5.5 kg). [Tamoxifen reduces hepatic IGF-1 by 16-50%]. **Confidence: HIGH**

#### Safety-Threshold Tests

22. `test_crashed_estrogen_mild_threshold`: 500 mg/week test + 1.5 mg/day arimidex × 6 weeks. Target: estradiol falls into 10-15 pg/mL range, mild_crashed_e2_penalty active. **Confidence: HIGH**
23. `test_crashed_estrogen_severe_threshold`: 500 mg/week test + 2 mg/day arimidex × 8 weeks. Target: estradiol < 10 pg/mL, severe_crashed_e2_penalty active, joint pain & libido symptoms simulated. **Confidence: MEDIUM** — symptom thresholds are well-documented clinical pattern but specific incidence numbers come from observational/clinical guideline sources rather than RCTs. Re-run Deep Research with target query: "peer-reviewed studies quantifying symptom prevalence in men with iatrogenic estradiol < 10 pg/mL on aromatase inhibitor + testosterone protocols."
24. `test_liver_stress_dianabol`: 30 mg/day dianabol × 8 weeks. Target: ALT elevated 3-10x ULN by week 4-6 (Solimini 2017 + DILI registry data). Verify peak occurs 2-10 weeks after initiation, R-value transitions to cholestatic pattern. **Confidence: LOW** — DILI registry data with weak primary citation. Re-run Deep Research with target query: "peer-reviewed prospective cohort studies of liver enzyme dynamics in healthy men using 17-alpha-alkylated AAS at supraphysiological doses."
25. `test_liver_stress_oral_stack`: 30 mg/day dianabol + 30 mg/day anavar × 6 weeks. Target: liver_stress reaches 0.8 threshold faster (~week 4-5) than either alone, demonstrating synergistic accumulation per Part 3.6 interaction 7. **Confidence: HIGH** (mechanism well-established; magnitude approximate)
26. `test_failed_pct`: 500 mg/week test × 16 weeks, no PCT in blocks 5-6. Verify failed_pct_penalty active in final reward; hpta_suppression remains > 0.95 in final block. **Confidence: HIGH**
27. `test_progestogenic_arimidex_impotence`: 400 mg/week trenbolone + 1 mg/day arimidex × 8 weeks. Verify progesterone_level still rises to penalty threshold (Arimidex must NOT reduce progestogenic load — this validates the Part 3.6 interaction 8 modeling). **Confidence: HIGH**

### Caveats and Trial Population Notes

The calibration tests use trial populations that may not match user input distributions. Document these caveats in code comments:

- **Bhasin 1996** used healthy 19-40 y/o trained men. The simulator's androgen receptor sensitivity should inversely scale with age; older avatars get lower FFM accretion at the same dose. Initial weeks of an enanthate cycle produce disproportionate fluid retention; lean mass gain in tests should be measured at week 8+ to filter glycogen/water.
- **STEP 1** required gradual dose escalation. Skipping titration produces unrealistic GI side-effect burden. The simulator must enforce a 16-week dose ramp for semaglutide.
- **SURMOUNT-1** TTWP varies by BMI — overweight users plateau at ~24 weeks, class III obesity at ~36 weeks. Long-running tirzepatide simulations should reproduce this.
- **REDUCE trials** on tesamorelin enrolled HIV lipodystrophy patients with elevated baseline VAT. Healthy avatar with low VAT may show smaller absolute reductions — calibrate the relative percentage.
- **HGH IGF-1 dose-response** reflects average physiological saturation; obese or insulin-resistant avatars show blunted IGF-1 elevation due to hepatic GH receptor downregulation. Premenopausal women on oral estrogen show further blunting.

### How to Generate the Exact Targets (For Future Refinement)

When updating these targets:

1. Use Deep Research with a focused prompt: "For [specific endpoint], find the primary trial. Report: trial name, sample size, mean outcome, 95% confidence interval, duration, population characteristics."
2. Cross-check the citation by opening the actual paper on PubMed.
3. Encode the numbers as test tolerances using reported CIs or `mean ± 1.5 × std_dev`.
4. Update the confidence rating in this doc.

---

## Part 3.6: Compound Interaction Table

Most compounds have explicit interaction effects that must be modeled. Interactions are either additive (two compounds affect the same state variable, sum the effects), modulatory (one compound modifies another's effect), competitive (compounds compete for a shared receptor/pathway), or synergistic (combined effect exceeds the sum of individual effects, often via cross-talk between distinct molecular pathways).

The structural mechanisms below are derived from primary pharmacology literature. Magnitudes are cited or flagged as estimated where literature is sparse.

| # | Interaction | Type | Effect | Magnitude / Notes | Where in Code | Confidence |
|---|---|---|---|---|---|---|
| 1 | Any aromatizable androgen + Arimidex | Modulatory (competitive inhibition) | Anastrozole occupies CYP19A1 active site, blocking conversion of ALL aromatizable substrates (testosterone, dianabol, nandrolone) proportionally to their plasma concentrations. Anastrozole K_i = 14.6 nM. | Suppression magnitude depends on HPTA state: 80-95% if HPTA suppressed by exogenous androgens; ~35% if HPTA active (testicular aromatase resistance). | `update_estrogen()` scales V_max by suppression factor; check HPTA state and apply context-dependent magnitude | HIGH |
| 2 | Testosterone + Nandrolone | Competitive (substrate-substrate) | Both bind aromatase, but nandrolone aromatizes at only ~20% of testosterone's rate. Nandrolone occupies aromatase enzymes inefficiently, displacing testosterone — acts functionally as a weak competitive inhibitor. Net E2 from the stack is **sub-additive**. | E2 yield from stack < sum of individual E2 yields | Multi-substrate Michaelis-Menten: shared denominator weighted by relative K_m / V_max for each compound | HIGH |
| 3 | MK-677 + GLP-1 agonists | Antagonistic (functional, central) | Ghrelin agonism (MK-677) raises hunger via NPY/AgRP hypothalamic neurons; GLP-1 agonism suppresses hunger via POMC/CART. **GLP-1 dominates.** Additionally, MK-677's appetite effect undergoes tachyphylaxis within months, while GLP-1 satiety persists. | Net hunger trajectory: blunted weight loss in first 4-8 weeks, then GLP-1 effect dominates as MK-677 tolerance develops | `hunger_drive` updated as: GLP-1 effect minus time-decaying MK-677 effect (decay constant ~0.05/week) | MEDIUM |
| 4 | Exogenous HGH + CJC-1295 | Antagonistic (somatostatin-mediated suppression) | Exogenous HGH triggers somatostatin surge, which fully blocks pituitary GHRH receptor signaling. CJC-1295 (a GHRH analog) becomes ~90-95% inert when exogenous HGH is present. | CJC-1295 efficacy reduced 90-95% under exogenous HGH | `cjc_efficacy *= (1 - 0.95 * normalized_exogenous_hgh)` | HIGH |
| 5 | Exogenous HGH + Ipamorelin | Antagonistic (partial breakthrough) | Ipamorelin uses ghrelin receptor (IP3/PKC pathway), partially bypassing somatostatin inhibition. Exogenous HGH still suppresses Ipamorelin's GH pulse magnitude by 60-70% — but unlike CJC-1295, some breakthrough remains. | Ipamorelin efficacy reduced 60-70% under exogenous HGH | `ipamorelin_efficacy *= (1 - 0.65 * normalized_exogenous_hgh)` | HIGH |
| 6 | Tesamorelin + MK-677 | **Synergistic** (cross-pathway amplification) | Tesamorelin (GHRH analog → cAMP/PKA) and MK-677 (ghrelin agonist → IP3/PKC) activate distinct pituitary signaling cascades. Combined activation produces multiplicative GH release — total GH AUC increases by 200-400% vs either alone. | Use cross-talk multiplier in GH release | `gh_release = base + tesa_effect + mk677_effect + sigma * tesa_effect * mk677_effect`, sigma ≈ 0.5 | HIGH |
| 7 | Tesamorelin + Exogenous HGH | Antagonistic | Same somatostatin mechanism as interaction #4. Exogenous HGH renders Tesamorelin almost fully inert. | Tesamorelin efficacy reduced ~95% under exogenous HGH | `tesamorelin_efficacy *= (1 - 0.95 * normalized_exogenous_hgh)` | HIGH |
| 8 | Semaglutide + Tirzepatide | Saturating (receptor competition + redundancy) | Both occupy GLP-1 receptor; standard doses already drive receptor toward saturation. Stacking yields **no additional efficacy** but **stacks adverse events additively** (severe nausea, gastroparesis risk). Semaglutide has higher GLP-1R affinity (K_i <1 nM vs Tirzepatide ~7.2 nM) and competitively displaces tirzepatide at GLP-1R. Tirzepatide retains independent GIP receptor binding. | Efficacy = max(individual effects) capped at receptor saturation; side effects = sum + nonlinear penalty | `glp1_occupancy = min(1.0, sema_occ + tirz_occ)`; apply `glp1_overstacking_penalty` to reward when both > 0.3 | HIGH |
| 9 | Tirzepatide + Retatrutide | Saturating (receptor competition + redundancy) | Same logic as #8 — overlapping GLP-1 and GIP pathways. No clinical or pharmacokinetic rationale for combining. | Efficacy capped; AE risk multiplied | Same pattern as #8 with extra receptor overlap | HIGH |
| 10 | Multiple androgens | Saturating (HPTA suppression) **+ additive (suppression duration)** | HPTA suppression saturates at ~100% rapidly; one moderate androgen achieves the same depth as a heavy stack. **However, recovery duration accumulates additively** based on the longest-acting depot ester in the stack — a stack with nandrolone decanoate (12-day half-life) extends recovery significantly beyond a stack with only testosterone enanthate (5-day half-life). | Depth of suppression: saturates at 1.0; recovery time: gated by longest-clearing compound | `hpta_suppression = min(1.0, sum(suppression_per_compound))`; `recovery_clock` blocks until longest-half-life compound clears below threshold | HIGH |
| 11 | Multiple oral 17α-alkylated steroids | **Synergistic** (greater than additive) | Hepatic clearance pathways (CYP, glucuronidation) saturate; concurrent orals compete for finite enzyme capacity. Antioxidant buffers (glutathione) deplete, then ROS damage cascades nonlinearly. | Hepatotoxicity scales as sum + cross-product term | `liver_stress += k * (S1 + S2 + alpha * S1 * S2)^beta`, alpha≈0.3, beta≈1.4 | HIGH |
| 12 | Trenbolone + Nandrolone (19-nor stack) | **Additive** progestogenic load — Arimidex IMPOTENT against this | Both bind progesterone receptor (~20% affinity vs progesterone for nandrolone, similar for trenbolone). Combined PR occupancy elevates prolactin and amplifies estrogen-receptor sensitivity in mammary tissue. **Anastrozole has zero effect** on this pathway since it's not an aromatization issue. Cabergoline (dopamine agonist) is the appropriate countermeasure — but it's not in the simulator's compound set. | Progestogenic load = sum of PR occupancies | `progesterone_level += k_tren * tren_dose + k_nan * nandrolone_dose`; Arimidex must NOT modify this term | HIGH |
| 13 | Tamoxifen mid-cycle | Antagonistic to anabolism (hepatic IGF-1 suppression) | Tamoxifen is an estrogen agonist in the liver; downregulates IGF-1 synthesis (16-50% reduction), upregulates IGFBP-1 (sequesters free IGF-1). Result: **mid-cycle tamoxifen blunts muscle gains** by reducing free IGF-1 available for satellite cell recruitment. | IGF-1 reduction: 16-50%; muscle gain reduction: ~30% (midpoint) | `delta_lean_mass *= (1 - 0.30)` when tamoxifen + active androgen co-occur | HIGH |
| 14 | Tamoxifen post-cycle | **Synergistic** with HPTA recovery | Tamoxifen acts as estrogen antagonist in hypothalamus/pituitary, blocking estrogenic negative feedback. LH/FSH surge → testicular reactivation. Recovery time roughly halves: 13 weeks vs 26 weeks median. | LH/FSH stimulation strong | Different code path: when no active androgen present, tamoxifen accelerates `hpta_recovery_rate` by ~2x | HIGH |
| 15 | BPC-157 + anything | Modulatory (recovery only) | Localized angiogenesis (VEGFR2 upregulation) and tendon/connective tissue repair. No central endocrine effects. | Magnitude purely speculative | `recovery_status_rate *= (1 + bpc157_factor)` — flag confidence LOW in code | LOW |

### How to Verify Interaction Magnitudes

The directions and mechanisms above are well-supported. The exact numerical magnitudes (e.g., the 0.30 muscle gain reduction for tamoxifen mid-cycle, the sigma=0.5 synergy multiplier for tesamorelin+MK-677) are ranges synthesized from clinical literature where exact values are not always available. Validate during simulator calibration: run each interaction in isolation, confirm the simulator produces clinically plausible outcomes, then tune the magnitude constants until calibration tests pass.

For interactions where no good clinical data exists (e.g., BPC-157 + anything, polypharmacy stacks beyond 3 compounds), document this clearly in code comments as "qualitative only, direction correct but magnitude speculative."

---

## Part 4: The VAE World Model (Component 2)

### What It Is

A neural network that learns to predict what the simulator does by observing many (state, action, next_state, reward) examples.

Four parts:

**Encoder** — compresses the 18-dim body state into an 8-dim latent representation. Think of it as a zip file. It learns which dimensions of the body state are correlated (e.g., testosterone_level and hpta_suppression are inversely related) and packs them efficiently.

**Transition Model** — takes the current 8-dim latent + 18-dim action = 26 dims, predicts the next 8-dim latent. Chained forward 6 times, it produces an imagined 24-week (6-block) trajectory.

**Decoder** — expands the 8-dim latent back to the 18-dim body state so you can display it.

**Reward Predictor** — takes the 8-dim latent and predicts reward (so the RL agent can evaluate dreams without decoding).

### Why a VAE Specifically

A plain VAE is a building block of World Models. Real production World Models (Dreamer) add a recurrent memory (GRU) on top to handle temporal dependencies. You're building a simpler version because:

- Your state variables already encode duration effects (hpta_suppression *is* accumulated history), so explicit recurrent memory is less critical here.
- 2.5-week timeline — adding recurrent components doubles implementation time.
- VAE + transition model is still a legitimate World Model (simplified Ha & Schmidhuber 2018).

State this as a scoping decision in the writeup.

### Training

1. Random agent interacts with the simulator for ~50,000 steps. Records (state, action, reward, next_state).
2. Train VAE + transition model + reward predictor jointly on the replay buffer.
3. Loss = reconstruction_loss + β · KL_divergence + transition_loss + reward_loss.
4. β annealed from 0 to 0.01 over first 1000 training steps.
5. Train 3 copies with different random seeds for the ensemble.

### KL Balancing

- Too high β → posterior collapse (latent ignores inputs).
- Too low β → no structure (latent is a lookup table).
- Default: start β = 0.001, anneal to 0.01 over first 1000 steps.

### Hyperparameter Decisions

Every hyperparameter below has a default, a range to try, and a condition under which to tune it. This is the honest way to present these — they are not magic numbers.

| Hyperparameter | Default | Range | Tune If |
|---|---|---|---|
| Latent dim | 8 | 4-16 | Reconstruction loss plateaus high (try larger); latent space shows redundant dims in PCA (try smaller) |
| Ensemble size | 3 | 1-5 | Model-based RL policy is unstable across seeds (add more); compute-limited (drop to 1, lose uncertainty estimate) |
| KL β target | 0.01 | 0.001-0.1 | Posterior collapse (lower); no structure in latent (raise) |
| KL β anneal steps | 1000 | 500-5000 | Training unstable at start (lengthen); convergence too slow (shorten) |
| Replay buffer size | 50k transitions | 20k-200k | World Model underfits on coverage (grow); disk/memory constrained (shrink) |
| Hidden layer width | 256 | 128-512 | Capacity bottleneck (widen); overfitting (narrow) |
| Training batch size | 256 | 64-1024 | GPU memory constrained (shrink); training noisy (grow) |

A principled default for latent_dim: run PCA on your raw simulator state data from the replay buffer, find the number of components needed for 90-95% variance, set latent_dim to roughly that. For biology with 18 correlated variables, 8 typically captures the right dimensionality. Validate this empirically on your actual data before committing.

---

## Part 4.5: World Model Validation

Before trusting the World Model to train your RL agent, run three validation tests. These determine whether model-based RL is viable at all — if they fail, fall back to model-free and report that honestly.

### Test 1: k-Step Prediction MAE

Take held-out trajectories (20% test split of your replay buffer). For each trajectory, compute the World Model's prediction error at k=1, 2, 4, and 6 steps (block-step horizons). Report mean absolute error per state dimension, normalized by that dimension's standard deviation over the full buffer.

**Thresholds:**

- 1-step MAE: < 0.05 × std (prediction is much smaller than variability — easy target)
- 2-step MAE: < 0.10 × std
- 4-step MAE: < 0.20 × std
- 6-step MAE: < 0.35 × std

If 1-step MAE fails, the World Model hasn't learned the dynamics — retrain with more data, larger hidden layers, or more epochs. If 6-step fails but 1-step passes, your model is accurate locally but error compounds over long horizons — this is usually a distribution shift or overfitting issue, addressed by the DAgger loop (Part 5).

### Test 2: Divergence Horizon

For each held-out trajectory, find the smallest k at which the World Model's prediction drifts beyond 2 standard deviations from the true trajectory on *any* state dimension. Average this across trajectories to get the "divergence horizon."

**Threshold:** divergence horizon > 4 blocks (so at least 16 weeks of accurate imagination before the model loses the plot).

If the divergence horizon is less than 4 blocks, your model can't support full-episode imagination rollouts. Options: (a) train the model-based agent on shorter imagined rollouts (e.g., only 3-block lookahead, with bootstrapped critic value for the rest), or (b) accept that this experiment will show model-free winning and write that up honestly.

### Test 3: Policy Transfer Gap

This is the test that actually matters for the RL experiment. Train SAC in the World Model (model-based). Evaluate the trained policy by running it on the *real simulator*. Compare that real-sim reward to the reward of an SAC agent trained directly on the real simulator (model-free baseline) for the same number of real-simulator steps.

```
policy_transfer_gap = (reward_model_free_realsim - reward_model_based_realsim) / reward_model_free_realsim
```

**Threshold:** policy_transfer_gap < 0.30 (model-based within 30% of model-free on real simulator).

If the gap exceeds 0.30, the World Model's errors are large enough that the policy learned inside it doesn't transfer well. The DAgger loop (Part 5) is specifically designed to close this gap — if you haven't already added it, add it.

### Pre-flight Gate

Run all three tests before Day 11 (model-based RL training day). If Test 1 or Test 2 fails, you need to iterate on the World Model before doing the main experiment. If Test 3 fails after DAgger has been added, report that model-based RL did not transfer well in this setting and discuss why — that's still a valid portfolio result, and the ability to identify and honestly report failure modes is itself a valuable ML engineering skill.

---

## Part 5: The RL Agent (Component 3)

### What It Is

An AI "coach" that learns to find the best 24-week protocol for any starting condition and reward function. Uses the SAC (Soft Actor-Critic) algorithm from Stable-Baselines3.

**SAC in one paragraph:** SAC is a reinforcement learning algorithm designed for continuous action spaces. It has two neural networks: an *actor* (picks actions) and a *critic* (rates how good an action is in a given state). The critic is trained via the Bellman equation — its prediction for "value of this action here" should equal the immediate reward plus the discounted value of the next state. The actor is trained to pick actions the critic rates highly, plus a small entropy bonus that keeps it from collapsing onto one strategy too early. SAC uses twin critics to reduce overestimation bias and is generally stable and sample-efficient. In Stable-Baselines3 it's five lines of code: `SAC("MlpPolicy", env).learn(total_timesteps=N)`.

### Two Training Modes

**Model-Free:** Agent interacts directly with the simulator. Every learning step needs a real `env.step()`. Honest baseline, slower wall-clock.

**Model-Based:** Agent "dreams" inside the World Model. Much faster per step, risky if the World Model is inaccurate. This is the headline experiment.

### The DAgger Loop (Distribution Shift Fix)

**Problem this solves:** The World Model was trained on trajectories from a *random agent*. But the RL agent you eventually train will NOT act randomly — it concentrates in high-reward regions of state space. The World Model may be accurate on random-like trajectories and inaccurate on the trajectories the RL agent actually visits. This is the classic failure mode of model-based RL.

**Solution — DAgger-style iteration:** every 5,000 RL training steps, the partially-trained RL agent is given 500 steps on the *real* simulator. Those transitions are added to the replay buffer. The World Model is then retrained (or fine-tuned) on the updated buffer for 2-3 epochs. This makes the World Model increasingly accurate on trajectories that look like the RL agent's current policy, closing the distribution shift gap.

```
PSEUDOCODE:

replay_buffer = random_agent_rollout(real_sim, 50_000 steps)
world_model = train_world_model(replay_buffer)

for iteration in range(12):   # 12 outer iterations × 5k steps = 60k total
    rl_agent = train_sac(env=world_model, steps=5000, init_from=rl_agent)
    
    # DAgger refresh
    new_transitions = rl_agent_rollout(real_sim, 500 steps)
    replay_buffer.extend(new_transitions)
    world_model = finetune_world_model(world_model, replay_buffer, epochs=3)

final_policy = rl_agent
```

Report: sample efficiency gain vs model-free (total real-sim steps used), final policy transfer gap (Part 4.5, Test 3), and how much DAgger closed the gap (run ablation: with DAgger vs without).

### What You Measure (Final Experiment Report)

1. **Sample efficiency:** total real `env.step()` calls to reach reward threshold. Model-free vs model-based + DAgger.
2. **Policy transfer:** does the dream-trained agent perform well on the real simulator?
3. **Imagination accuracy:** at what block does the World Model's prediction diverge from the real simulator? (Report divergence horizon from Part 4.5, Test 2.)
4. **DAgger ablation:** model-based with vs without DAgger, on policy transfer gap.

---

## Part 6: The LLM Agent (Component 4)

### What It Is

A Python script using the Claude API with tool calling. It is the conversational layer between the user and the RL machinery. It does not learn — it uses prompt engineering and tool calls.

### Tool Schemas (Exact JSON)

**Tool 1: `configure_and_optimize`**

```json
{
  "name": "configure_and_optimize",
  "description": "Configure the BioSim environment with user parameters and run the RL agent to produce an optimized 24-week protocol.",
  "input_schema": {
    "type": "object",
    "properties": {
      "starting_lean_mass_kg":        {"type": "number", "minimum": 40, "maximum": 130},
      "starting_fat_subcut_kg":       {"type": "number", "minimum": 2,  "maximum": 60},
      "starting_fat_visc_kg":         {"type": "number", "minimum": 1,  "maximum": 20},
      "age":                          {"type": "integer", "minimum": 18, "maximum": 70},
      "allowed_compounds":            {"type": "array", "items": {"type": "string", "enum": ["testosterone", "trenbolone", "nandrolone", "dianabol", "anavar", "rad140", "hgh", "cjc_ipamorelin", "tesamorelin", "bpc157", "mk677", "semaglutide", "tirzepatide", "retatrutide", "arimidex", "nolvadex"]}},
      "reward_weights": {
        "type": "object",
        "properties": {
          "w_muscle":   {"type": "number", "minimum": 0, "maximum": 1},
          "w_fat_loss": {"type": "number", "minimum": 0, "maximum": 1},
          "w_health":   {"type": "number", "minimum": 0.15, "maximum": 1},
          "w_recovery": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["w_muscle", "w_fat_loss", "w_health", "w_recovery"]
      }
    },
    "required": ["starting_lean_mass_kg", "starting_fat_subcut_kg", "starting_fat_visc_kg", "age", "allowed_compounds", "reward_weights"]
  }
}
```

**Note the hard floor `w_health >= 0.15`** — the JSON schema enforces it. The LLM cannot set it lower. This is the ethical guardrail discussed in Part 0.

**Tool 2: `explain_trajectory`**

```json
{
  "name": "explain_trajectory",
  "description": "Take the action sequence and predicted state trajectory from configure_and_optimize and produce a human-readable week-by-week narrative.",
  "input_schema": {
    "type": "object",
    "properties": {
      "actions":    {"type": "array", "description": "6-block × 18-dim action sequence"},
      "states":     {"type": "array", "description": "7 × 18-dim state trajectory (initial + 6 post-block)"},
      "user_goal":  {"type": "string", "description": "The original natural-language goal for framing"}
    },
    "required": ["actions", "states", "user_goal"]
  }
}
```

**Tool 3: `request_clarification`** (new — handles underspecified queries)

```json
{
  "name": "request_clarification",
  "description": "Call this when the user's goal is missing critical parameters (age, starting body composition, or compound preferences). Do NOT call this for minor/optional parameters.",
  "input_schema": {
    "type": "object",
    "properties": {
      "missing_parameters": {"type": "array", "items": {"type": "string"}},
      "clarifying_question": {"type": "string"}
    },
    "required": ["missing_parameters", "clarifying_question"]
  }
}
```

### LLM Decision Rule (Embedded in System Prompt)

```
CRITICAL parameters (if missing, call request_clarification):
  - age
  - starting body composition (lean mass, fat mass)
  - compound restrictions (must ask if user doesn't specify)

NON-CRITICAL parameters (use sensible defaults, note in narrative):
  - training experience level → default "intermediate"
  - recovery capacity → default 0.7
  - specific goal deadline → default 24 weeks

NEVER:
  - Set w_health below 0.15 (hardcoded schema floor)
  - Include compounds the user explicitly disallowed
  - Produce a protocol without running configure_and_optimize
```

### User Phrase → Reward Weight Mapping

The LLM maps natural-language goals to reward weights using this table as few-shot guidance in its system prompt. These are starting points; the LLM may adjust ±0.05 based on user context.

| User phrase | w_muscle | w_fat_loss | w_health | w_recovery | Justification |
|---|---|---|---|---|---|
| "Help me get jacked" / "bulk" | 0.55 | 0.05 | 0.25 | 0.15 | Muscle dominant; health floor prevents liver blowup |
| "Cut fat but keep muscle" | 0.30 | 0.45 | 0.20 | 0.05 | Balanced; muscle preservation via retention term |
| "Lose weight, no steroids" | 0.10 | 0.60 | 0.25 | 0.05 | Fat dominant; higher health due to GLP-1 GI side effects |
| "Body recomp" | 0.35 | 0.35 | 0.25 | 0.05 | Equal muscle/fat; moderate health weight |
| "Prep for photoshoot in 16 weeks" | 0.25 | 0.45 | 0.15 | 0.15 | Fat lead; recovery higher for peak-week performance |
| "Recover from a cycle" / "PCT focus" | 0.10 | 0.05 | 0.65 | 0.20 | Health dominant; HPA recovery is the whole goal |
| "Strongest version of myself" | 0.45 | 0.15 | 0.25 | 0.15 | Muscle lead; balance health/recovery for sustainability |
| "Anti-aging / longevity" | 0.10 | 0.20 | 0.50 | 0.20 | Health dominant; minimal body composition pressure |

### Example Flow

```
User: "I'm 180 lbs at 22% BF, 34 years old. Cut to 15% without losing muscle.
       I'll use semaglutide, no testosterone."

Claude parses:
  starting_lean_mass_kg = 63.5, starting_fat_subcut_kg = 14, 
  starting_fat_visc_kg = 5.9, age = 34
  allowed_compounds = ["semaglutide", "arimidex", "nolvadex"]
  weights = {w_muscle: 0.30, w_fat_loss: 0.45, w_health: 0.20, w_recovery: 0.05}

Claude calls configure_and_optimize(...)
  → returns 6-block action sequence and state trajectory

Claude calls explain_trajectory(actions, states, "Cut to 15% without losing muscle")
  → returns narrative: "Block 1 (weeks 1-4): moderate training, semaglutide 
     titrates to full dose. Fat loss begins slowly as GLP-1 receptors 
     saturate. Block 2..."
```

---

## Part 7: The Dashboard (Component 5)

React + Recharts + Tailwind. Shows:

1. **Prominent disclaimer banner** (every screen): "Portfolio demonstration only. Not medical advice. Compound pharmacology is approximate."
2. **Hero metrics:** final lean mass, final fat %, peak liver stress, sample efficiency ratio (model-based vs model-free).
3. **Trajectory charts:** one per state variable, showing World Model prediction (dashed) vs simulator ground truth (solid) with ensemble uncertainty bands.
4. **Protocol timeline:** 6-block action recommendations as color-coded bars, one row per compound.
5. **LLM narrative:** Claude-generated block-by-block explanation.
6. **Latent space explorer:** 2D PCA projection of the learned 8D latent space, colored by outcome.

---

## Part 8: Getting Started Today (The Practical Guide)

### Hardware Plan: Google Colab / Cloud GPU

You're using Colab. That shapes the setup significantly:

- **Colab Pro or Pro+** is worth the $10-50/mo for this project — free Colab disconnects aggressively and bumps you to slower GPUs. For 3 weeks, Pro is ~$10, well worth it.
- **Keep the simulator and training code in a GitHub repo.** Each Colab session clones the repo fresh, runs training, saves checkpoints to Google Drive.
- **Store replay buffer and model checkpoints in Google Drive**, not Colab's ephemeral storage. You will lose work otherwise.
- **Use `%%capture` and `tqdm`** for clean training logs in notebooks. Log to Weights & Biases for persistent metrics across sessions.

Training time estimates are in Part 10.

### What Order to Build Things

**Day 1: Set up the repo and the simulator skeleton.**

```bash
git clone https://github.com/yourusername/biosim.git
cd biosim
python -m venv venv
source venv/bin/activate
pip install gymnasium numpy torch stable-baselines3 anthropic matplotlib wandb
mkdir -p biosim/envs biosim/world_model biosim/rl biosim/agent docs tests
```

Create minimal Gymnasium env at `biosim/envs/biology_env.py`. Return random states first; fill in pharmacology next. Verify with a random agent for 10 steps.

**Days 2-3: Build pharmacokinetic equations.** Implement the 18 obs vars, 18 action vars, reward function, and PK equations for all 16 compounds. Sanity-plot testosterone clearance, semaglutide accumulation, random-agent 24-week stability.

**Day 3 (afternoon, parallel task): Run Deep Research for calibration anchors.** Produce the exact numbers for Parts 3.5 and 3.6. Pass to Claude Code to fill them into the simulator tests.

**Day 4: Calibration tests.** Implement the 12 tests from Part 3.5. All must pass before proceeding. If any fail, revisit the equations — don't move on with a broken simulator.

**Day 5: Collect training data and build the VAE.** Random agent for 50k steps, save replay buffer. VAE in PyTorch (encoder, transition model, decoder, reward predictor).

**Days 6-7: Train the World Model and run Part 4.5 validation.** Train on replay buffer, monitor all four losses. Run Tests 1 and 2 (Test 3 comes after Day 11). If Test 1 fails, iterate. Train 2 more copies for ensemble if time.

**Day 8: Polish and document Week 1 progress.** Clean code, commits, README update.

**Days 9-11: Model-free RL baseline.** SB3 SAC on the simulator directly. Track real env steps to reward threshold. Baseline number.

**Day 12: Model-based RL + DAgger loop.** Train SAC on World Model with DAgger iteration. Run Part 4.5 Test 3 (policy transfer gap).

**Day 13: Ablation.** Model-based without DAgger, compare to with DAgger. Results go in the writeup.

**Days 14-15: LLM Agent.** Build the script with tool calling. Test with 8 example scenarios (one per row of the phrase→weight table).

**Days 16-17: Dashboard.** React + Recharts. Pre-compute 3-5 scenarios as JSON, load statically. Live backend as stretch.

**Day 18: Portfolio polish.** README, architecture diagram, demo video, GitHub push. Writeup honest about scope, limitations, and what model-based vs model-free actually showed.

### How to Use Claude Code

Claude Code is a command-line tool that runs Claude in your terminal. It reads your codebase, writes code, runs tests, fixes bugs.

```bash
npm install -g @anthropic-ai/claude-code
cd biosim
claude
```

**Workflow:**

1. Put this design doc in your repo at `docs/DESIGN.md`.
2. Start each session: "Read docs/DESIGN.md. We're building BioSim. Today we're working on [component X]."
3. Give scoped tasks: "Implement the Testosterone Enanthate PK equations in biosim/envs/compounds/testosterone.py. Use the half-life and dose-response from Part 3."
4. Iterate: Claude Code writes, you test, you feed back errors.
5. Unblock stuck moments: "The VAE's reconstruction loss isn't decreasing. Here's my training loop."

**Three rules:**

- Always reference the design doc by part number, not by description.
- Scope tasks tightly. One concrete deliverable per prompt.
- Verify before moving on. Don't accumulate tech debt.

---

## Part 9: Design Decisions (Why We Didn't Pick the Alternatives)

**Why 4-week blocks (not weekly, not full-protocol-at-once)?** Weekly produces unrealistic noisy protocols. Full-protocol-at-once collapses to contextual bandits, weakening the RL narrative. 4-week blocks match real protocol structure, give the agent 6 sequential decisions per episode (keeping meaningful temporal RL), and reduce exploration difficulty vs 24 weekly decisions.

**Why VAE without recurrence (not Dreamer)?** State variables already encode accumulated history (hpta_suppression, liver_stress are integrators). 2.5-week timeline. Simplified Ha & Schmidhuber architecture is a legitimate stopping point for a portfolio project.

**Why SAC specifically (not PPO, not DDPG)?** Continuous action space, sample efficiency, stable training (PPO oscillates more), off-policy (uses replay buffer which we already have). Stable-Baselines3 implementation is mature.

**Why 8-dim latent?** Empirical — PCA on simulator state usually shows ~8 components capturing 90%+ variance for 18 correlated biological variables. This is a hyperparameter with a principled default, not a magic number. See Part 4 hyperparameter table.

**Why DAgger over MBPO or other model-based RL variants?** Simplicity. DAgger is one concept (re-gather data from current policy, retrain model) and is sufficient to demonstrate the fix for distribution shift. MBPO, DreamerV3, etc. are stronger methods but add implementation complexity that pushes you past your 18-day timeline.

---

## Part 10: Training Time and Performance Optimization

### Expected Training Times on Colab Pro (T4 / A100)

**VAE World Model training** (50k transitions, 100 epochs, batch size 256):

- T4: 5-15 minutes
- A100: 1-3 minutes
- Ensemble of 3: triple the above, run sequentially

**Model-free SAC** (60k real env steps, block-level — meaning 60k `env.step()` calls, each representing 4 weeks):

- T4: 30-60 minutes
- A100: 20-40 minutes
- Wall-clock dominated by neural network updates, not simulator steps (simulator is Python math, very fast)

**Model-based SAC with DAgger** (60k imagined steps + 12 refresh cycles × 500 real steps + World Model fine-tuning):

- T4: 20-40 minutes total (imagined rollouts are faster than real sim, World Model fine-tunes are fast)
- A100: 10-25 minutes

**Expected speedup ratio:** model-based ~1.5-3x faster wall-clock. On real simulators where `env.step()` is expensive, this would be 10-100x. State honestly in the writeup that because your simulator is cheap, the model-based speedup here is modest, and the value of the exercise is in demonstrating the paradigm.

### Key Performance Optimizations

**1. Vectorize the simulator.** Don't loop one step at a time if you can help it. Gymnasium has `VectorEnv`; run 16-32 parallel episodes on CPU. SAC training on a vectorized env is 5-10x faster without changing algorithm.

**2. Move VAE training to GPU.** Simulator stays on CPU (it's Python math, GPU transfer overhead isn't worth it). VAE is the heavy compute — move tensors to GPU, run batches of 256+, use AMP (mixed precision) for ~2x speedup on A100.

**3. Replay buffer on disk, not in GPU memory.** 50k transitions × 18 floats × 4 bytes ≈ 3.6 MB, tiny. But if you grow the buffer (DAgger), keep it in CPU RAM or a memory-mapped file, stream batches to GPU on demand.

**4. Weights & Biases for logging, not prints.** Print statements slow down Colab notebooks significantly at scale. Log to W&B every N steps, view in browser. Also persistent across Colab disconnects.

**5. Save checkpoints every 5k steps to Google Drive.** Colab disconnects. You will lose training runs if you don't checkpoint. Load on resume.

**6. Precompile the PK equations.** If simulator step latency matters, use `numba` `@jit` on the hot path (the daily PK integration loop). Typical speedup 10-50x on numerical Python. Skip this if step latency isn't the bottleneck — don't optimize prematurely.

**7. Use `stable_baselines3.SAC` defaults first.** Don't tune SAC hyperparameters until your baseline works. The defaults (lr=3e-4, gamma=0.99, tau=0.005, batch_size=256) are good for most continuous control problems.

**8. Train ensemble of 3 World Models in parallel, not sequentially.** If on A100, each World Model uses ~1 GB VRAM. Run all three training jobs as separate Python processes with different GPUs or different CUDA streams.

### When to Stop Optimizing

If you find yourself spending more than one day on performance tuning, you're over-optimizing for a portfolio project. The scientific result (model-free vs model-based comparison) is more important than training in 10 minutes vs 30 minutes. Ship the slower version and move on.

---

## Part 11: Known Limitations (State Upfront in Writeup)

1. The simulator uses first-order exponential decay for all compounds. Peptides and biologics technically follow TMDD kinetics. At block-level timesteps, the approximation is fine but worth noting.
2. The VAE is a simplified World Model — no recurrent memory (GRU) as in Dreamer. Scoping decision.
3. Compound interaction effects are estimated and partially validated against clinical literature. Magnitude of some interactions (especially polypharmacy stacks) is speculative.
4. RAD-140 and BPC-157 efficacy values are extrapolated from limited (RAD-140) or only animal (BPC-157) data. Confidence flagged LOW in code.
5. Block-structured actions (4 weeks) reduce the granularity of the RL problem. Real protocols sometimes require weekly adjustments (e.g., AI dose based on bloodwork). This project does not model that.
6. The simulator should NEVER be used for medical decisions. Portfolio project demonstrating ML techniques, not a clinical tool.
7. The project trains on simulated data only — there is no real-world validation. All reported numbers are within-simulator.

---

## Quick Reference: The Three Questions People Will Ask You

**"What does this project do?"**
*"It's a World Model + reinforcement learning system that learns to predict human biological response to fitness and pharmacological interventions, and optimizes 24-week protocols based on user-specified goals."*

**"What's the ML contribution?"**
*"I built a VAE-based World Model of a pharmacokinetically-grounded human biology simulator, then compared model-based and model-free RL with and without DAgger-style distribution shift correction. I measured sample efficiency, policy transfer gap, and divergence horizon — producing a controlled study of when model-based RL helps and when it doesn't."*

**"Why biology specifically?"**
*"The domain has delayed effects (steroids take weeks), accumulation (Semaglutide reaches steady state over 5 weeks), and irreversible states (liver damage, HPA suppression). It stress-tests the World Model's ability to capture long-horizon dependencies — more so than standard benchmarks like Pendulum or HalfCheetah. Compound choice was driven by availability of published PK data, not endorsement of any regimen."*
