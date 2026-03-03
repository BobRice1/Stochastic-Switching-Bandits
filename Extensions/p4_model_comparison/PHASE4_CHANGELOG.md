# Phase 4 Change Log

This file tracks code changes made to `Phase4_analysis.ipynb` so the paper can be updated against the actual implementation.

## Code fixes

1. FQ held-out evaluation now uses the fitted `gamma` parameter rather than hard-coding `gamma = 0.0`.
   Impact: the FQ rows in `phase4_final_model_comparison.csv` and Table I must be regenerated before they are cited in the paper.

2. The final Phase 4 results table is now built from `final_results_v2` rather than a manually typed array.
   Impact: the CSV becomes a direct output of the cross-validation pipeline after rerunning the notebook.

3. Predictive Sticky HMM plotting cells now use the empirical session-derived hazard estimate for each condition instead of a hard-coded `q = 0.95`.
   Impact: the predictive HMM history and reversal figures should be regenerated, and any paper claims based on those figures should be checked against the new outputs.

4. Dynamic RFLR generative prediction error now uses the pre-update evidence state, matching `nonlinear_stickiness_refined_v2.py`.
   Impact: generative RFLR reward-switch comparisons and reversal dynamics should be regenerated.

5. Value-gated FQ generative simulations now match the model definition used in `nonlinear_stickiness_refined_v2.py`: frozen Q-values within each window, decision policy based on `(Q_right - Q_left) / temp`, and boundary updates using mean reward per action within the window.
   Impact: any generative FQ comparisons in Phase 4 need to be rerun and reinterpreted from the new outputs.

6. The static RFLR baseline used in Phase 4 generative comparisons now samples reward on trial 0, matching the other simulated agents.
   Impact: reward comparisons against static RFLR in Phase 4 should be regenerated.

## Paper sections to revisit after rerun

- Methods: adaptive model optimisation, predictive comparison, and generative simulation details.
- Results: Table I, the predictive HMM paragraphs, the dynamic-model efficiency comparison, and any direct statements about FQ ranking.
- Discussion/Conclusion: keep claims about adaptive stickiness narrow until regenerated figures and updated likelihood values are confirmed.

## Minimum rerun checklist

1. Run the 5-fold Phase 4 fitting block and regenerate `phase4_final_model_comparison.csv`.
2. Regenerate the predictive RFLR/HMM/FQ figures.
3. Regenerate the generative comparison figures for dynamic models.
4. Update `paper.tex` using the regenerated numbers and plots, not the previous table values.
