# Stochastic-Switching-Bandits

This repository reproduces and extends the behavioural modelling analysis from Beron et al. (2022) for stochastic switching in a probabilistic two-armed bandit task.

The project has three main parts:

- `reproduction/`: baseline reproduction of the released 2ABT behaviour-model code and figures
- `Extensions/`: project-specific extensions beyond the original study
- `Paper/`: the manuscript and compiled paper output

## Repository Layout

### `reproduction/`

- `reproduction/2ABT_behaviour_models/`: the original behaviour-model codebase tracked as a submodule
- `reproduction/generate_figures.py`: script for regenerating the baseline reproduction figures
- `reproduction/figures/`: reproduced figures used by the paper
- `reproduction/2ABT_behaviour_models_updated/`: local updated copy of the baseline code retained in the repo

### `Extensions/`

- `Extensions/p2_generative_fits/`: Phase 2 generative-fit analysis, including `Phase2_analysis.ipynb`, `cv_results_fitted.csv`, and generated figures
- `Extensions/p3_nonlinear_stickiness/`: Phase 3 nonlinear stickiness model implementation, centred on `nonlinear_stickiness_refined_v2.py`
- `Extensions/p4_model_comparison/`: Phase 4 predictive and generative model comparison outputs, including `Phase4_analysis.ipynb`, `phase4_final_model_comparison.csv`, and figures

### `Paper/`

- `Paper/paper.pdf`: compiled paper.

## Typical Workflow

### Reproduction figures

Run from the repository root:

```powershell
python reproduction/generate_figures.py
```

This writes baseline figures into `reproduction/figures/`.

### Extension analyses

- Phase 2 is organised around `Extensions/p2_generative_fits/Phase2_analysis.ipynb`
- Phase 3 model code lives in `Extensions/p3_nonlinear_stickiness/nonlinear_stickiness_refined_v2.py`
- Phase 4 is organised around `Extensions/p4_model_comparison/Phase4_analysis.ipynb`

These analyses produce the CSVs and figures used in the paper.

