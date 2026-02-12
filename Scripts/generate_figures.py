"""
Reproduction script for 2ABT behaviour models.

Runs the full pipeline from the submodule's demo_models.ipynb notebook
and saves all generated figures to reproduction/figures/.

DO NOT MODIFY

"""

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMODULE_DIR = os.path.join(SCRIPT_DIR, "2ABT_behaviour_models")
sys.path.insert(0, SUBMODULE_DIR)

# Change working directory to the submodule so relative data paths work
os.chdir(SUBMODULE_DIR)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

import plot_models_v_mouse as bp
import model_policies as models
import conditional_probs as cprobs
import resample_and_model_reps as reps
import model_fitting as fit
from sklearn.model_selection import train_test_split


# Output directory for figures
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_current_figures(prefix: str, close: bool = True):
    """Save all open matplotlib figures with a numbered prefix, then close them."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        path = os.path.join(FIGURES_DIR, f"{prefix}_{i + 1}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    if close:
        plt.close("all")



# 1. Load data and train-test split
print("=" * 60)
print("Loading data and splitting into train/test sets...")
print("=" * 60)

data_path = "bandit_data.csv"
data = pd.read_csv(data_path)

probs = "80-20"           # P(high)-P(low)
seq_nback = 3             # history length for conditional probabilities
train_prop = 0.7          # train/test proportion
seed = np.random.randint(1000)

data = data.loc[data.Condition == probs]
data = cprobs.add_history_cols(data, seq_nback)

train_session_ids, test_session_ids = train_test_split(
    data.Session.unique(), train_size=train_prop, random_state=seed
)

data["block_pos_rev"] = data["blockTrial"] - data["blockLength"]
data["model"] = "mouse"
data["highPort"] = data["Decision"] == data["Target"]

train_features, _, _ = reps.pull_sample_dataset(train_session_ids, data)
test_features, _, block_pos_core = reps.pull_sample_dataset(test_session_ids, data)

bpos_mouse = bp.get_block_position_summaries(block_pos_core)
bpos_mouse["condition"] = "mouse"

# 2. Mouse-only conditional switch probabilities
print("\n" + "=" * 60)
print("Plotting mouse-only conditional switch probabilities...")
print("=" * 60)

df_mouse_symm_reference = (
    cprobs.calc_conditional_probs(data, symm=True, action=["Switch"])
    .sort_values("pswitch")
)

df_mouse_symm = cprobs.calc_conditional_probs(
    block_pos_core, symm=True, action=["Switch", "Decision"]
)
df_mouse_symm = cprobs.sort_cprobs(
    df_mouse_symm, df_mouse_symm_reference.history.values
)

bp.plot_sequences(df_mouse_symm, alpha=0.5)
save_current_figures("01_mouse_cprobs")



# Helper: given model_probs, apply policy and produce all comparison figures
def plot_model_results(model_probs, model_name: str, fig_num: int):
    """Apply stochastic policy, build comparison dataframes, and save figures."""
    print(f"\n  Applying stochastic policy and generating plots for {model_name}...")

    model_choices, model_switches = models.model_to_policy(
        model_probs, test_features, policy="stochastic"
    )

    block_pos_model = reps.reconstruct_block_pos(
        block_pos_core, model_choices, model_switches
    )

    # --- Block-position comparison (P(high port) and P(switch)) ---
    bpos_model = bp.get_block_position_summaries(block_pos_model)
    bpos_model["condition"] = "model"
    bpos_model_v_mouse = pd.concat((bpos_mouse, bpos_model))

    color_dict = {"mouse": "gray", "model": sns.color_palette()[0]}
    bp.plot_by_block_position(
        bpos_model_v_mouse, subset="condition", color_dict=color_dict
    )
    save_current_figures(f"{fig_num:02d}_{model_name}_block_position")

    # --- Conditional probabilities overlay: mouse vs model ---
    symm_cprobs_model = cprobs.calc_conditional_probs(
        block_pos_model, symm=True, action=["Switch"]
    )
    symm_cprobs_model = cprobs.sort_cprobs(
        symm_cprobs_model, df_mouse_symm.history.values
    )
    bp.plot_sequences(
        df_mouse_symm,
        overlay=symm_cprobs_model,
        main_label="mouse",
        overlay_label="model",
    )
    save_current_figures(f"{fig_num:02d}_{model_name}_cprobs_overlay")

    # --- Scatter: mouse P(switch) vs model P(switch) ---
    bp.plot_scatter(df_mouse_symm, symm_cprobs_model)
    save_current_figures(f"{fig_num:02d}_{model_name}_scatter")



# 3. Logistic Regression
print("\n" + "=" * 60)
print("Fitting Logistic Regression model...")
print("=" * 60)

L1 = 1   # choice history
L2 = 5   # choice * reward history
L3 = 0
memories = [L1, L3, L2, 1]

lr = models.fit_logreg_policy(train_features, memories)
model_probs_lr = models.compute_logreg_probs(test_features, lr_args=[lr, memories])

plot_model_results(model_probs_lr, "logistic_regression", fig_num=2)



# 4. Recursively Formulated Logistic Regression (RFLR)
print("\n" + "=" * 60)
print("Fitting RFLR model (this may take a minute)...")
print("=" * 60)

params, nll = fit.fit_with_sgd(fit.log_probability_rflr, train_features)
alpha, beta, tau = params
print(f"  alpha = {alpha[0]:.2f}")
print(f"  beta  = {beta[0]:.2f}")
print(f"  tau   = {tau[0]:.2f}")

model_probs_rflr = models.RFLR(test_features, params)

plot_model_results(model_probs_rflr, "rflr", fig_num=3)



# 5. Hidden Markov Model (HMM)
print("\n" + "=" * 60)
print("Computing HMM model predictions...")
print("=" * 60)

q = 0.98  # 1 - p(block transition)
p = 0.8   # p(reward | high port)

model_probs_hmm = models.compute_hmm_probs(
    test_features, parameters={"q": q, "p": p}
)

plot_model_results(model_probs_hmm, "hmm", fig_num=4)



# 6. Forgetting Q-Learning (F-Q model)
print("Computing F-Q learning model predictions...")
print("=" * 60)

T_fq = (1 - np.exp(-1 / tau)) / beta
k_fq = 1 - np.exp(-1 / tau)
a_fq = alpha

model_probs_fq = models.fq_learning_model(
    test_features, parameters=[a_fq, k_fq, T_fq]
)

plot_model_results(model_probs_fq, "fq_learning", fig_num=5)


# ============================================================================
print("\n" + "=" * 60)
print(f"All figures saved to: {FIGURES_DIR}")
print("=" * 60)
