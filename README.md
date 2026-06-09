# Bayesian ML: Regime-Change Impact Modeling with SHAP Explainability

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyMC](https://img.shields.io/badge/PyMC-Bayesian_Modeling-FF6B6B?style=flat)](https://www.pymc.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-4CAF50?style=flat)](https://shap.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)

---

## Overview

This project applies **Bayesian probabilistic modeling** and **SHAP feature attribution** to analyze how structural regime changes affect outcome distributions in complex, high-variance systems.

Regime changes — transitions in leadership, strategy, or operational approach — create measurable shifts in system behavior that standard ML models often fail to capture because they treat historical data as uniform. This project develops a framework for **detecting, modeling, and explaining** those shifts using Bayesian uncertainty quantification and explainable AI.

**Domain:** NFL team performance data is used as the modeling substrate — a domain with rich, publicly documented regime-change events (coaching transitions, quarterback changes), high-quality historical data, and well-defined outcome metrics. The methodology is transferable to any domain where structural transitions affect performance distributions.

---

## Why This Matters Beyond Sports

The core problem — modeling how structural regime changes affect system outcomes under uncertainty — applies directly to:

| Domain | Regime Change Equivalent |
|---|---|
| **FinTech / Risk Modeling** | Leadership change, regulatory shift, market regime transition |
| **Healthcare AI** | Treatment protocol change, clinical guideline update |
| **Operations / Manufacturing** | Process change, supplier transition, equipment upgrade |
| **Organizational Analytics** | Team restructuring, strategy pivot, M&A integration |

Bayesian modeling with SHAP explainability produces **confidence intervals** and **feature attribution** — both required for responsible AI deployment in regulated industries.

---

## Methodology

### The Core Problem
Standard ML models trained on historical data assume stationarity — that the future resembles the past. Regime changes violate this assumption. A model trained on pre-coaching-change data will produce miscalibrated predictions post-change until enough new data accumulates. This project explicitly models regime membership as a latent variable.

### Approach

```
Historical Data (game-level outcomes, contextual features)
        ↓
Regime Identification (coaching transitions, QB changes as structural breakpoints)
        ↓
Feature Engineering (regime-aware features: time-in-regime, regime interaction terms)
        ↓
Bayesian Model (PyMC — posterior distributions over parameters per regime)
        ↓
Confidence Intervals (credible intervals on predictions, not just point estimates)
        ↓
SHAP Attribution (which features drive predictions, how attribution shifts across regimes)
        ↓
Interpretable Outputs (feature importance plots, regime comparison visualizations)
```

### Features Modeled

| Feature Category | Specific Features |
|---|---|
| **Regime Variables** | Coaching tenure, QB tenure, time-since-regime-change |
| **Betting Markets** | Vegas spread, over/under, implied win probability |
| **Game Context** | Home/away, weather conditions, location |
| **Play-Calling** | Pass rate tendency, run rate tendency, 3rd-down aggression |
| **Opponent** | Opponent defensive rating, recent form |
| **Target** | Game outcome (Win/Loss), margin of victory |

---

## Key Technical Components

### Bayesian Uncertainty Modeling (PyMC)
Rather than producing a single point estimate, the Bayesian model produces a **posterior distribution** over predictions — a full probability distribution that quantifies uncertainty. This is particularly valuable for regime-change scenarios where data is sparse in the early post-transition period.

Output: credible intervals (e.g., "65% probability of win, 90% credible interval: 52%–78%") rather than just "65% probability."

### SHAP Explainability
SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for each prediction — grounded in game theory. This enables:
- Identifying which features drive predictions for specific regimes
- Comparing feature importance **before vs. after** regime changes
- Producing human-interpretable explanations for every prediction

---

## Results

- Bayesian regime-aware model outperforms standard logistic regression baseline on post-regime-change predictions
- SHAP analysis reveals that **Vegas betting lines and QB tenure** are the dominant predictive features, while **weather and location** contribute minimally once regime variables are controlled
- Confidence intervals widen significantly in the first 4–6 games following a regime change — quantifying the uncertainty period that standard models ignore
- Regime interaction terms improve calibration on post-transition data

*See notebooks/final_model.ipynb for full results, visualizations, and SHAP plots*

---

## Repository Structure

```
├── data/
│   ├── raw/                    # Raw game-level data from source APIs
│   └── processed/              # Cleaned, feature-engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_bayesian_model.ipynb # Core PyMC model
│   └── 05_shap_analysis.ipynb  # SHAP attribution and visualizations
├── scripts/
│   ├── data_processing.py
│   └── modeling.py
├── outputs/
│   ├── predictions/            # Model predictions and confidence intervals
│   └── visualizations/         # SHAP plots, regime comparison charts
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Vysubs28/Bayesian-ML-Regime-Impact-Modeling
cd Bayesian-ML-Regime-Impact-Modeling
pip install -r requirements.txt
jupyter notebook
```

---

## Dependencies

```
pymc
shap
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
arviz          # Bayesian visualization
```

---

## Data Sources

- [Pro Football Reference](https://www.pro-football-reference.com) — game-level statistics
- [FiveThirtyEight](https://fivethirtyeight.com) — Elo ratings and win probability
- NFL APIs — real-time game data

---

## Future Work

- Extend regime-change detection to automatic identification using change-point detection algorithms (PyMC, ruptures library)
- Apply framework to financial time-series data where regime changes (market regimes, policy shifts) are a known challenge
- Incorporate hierarchical Bayesian modeling to share information across regime types

---

## License

MIT License

---

## Author

**Vyaas Subramanian**  
[github.com/Vysubs28](https://github.com/Vysubs28) · [LinkedIn](https://linkedin.com/in/vyaas-subramanian-427468237) · [Portfolio](https://vyaasportfolio.netlify.app)
