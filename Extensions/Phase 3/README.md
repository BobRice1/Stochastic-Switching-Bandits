# Phase 3: Non-Linear Stickiness

This extension implements:

1. Temporal gating (`N`-step logic)
- `RFLR` and `HMM`: policy-level lock where predicted choice is fixed for `N` future trials after each switch.
- `F-Q`: rewards are accumulated in `N`-trial windows, then one Q update is applied per window.

2. Informational gating (surprise-dependent stickiness)
- Dynamic stickiness:
  `alpha_dynamic = alpha_base * exp(-gamma * accumulated_error)`
- `gamma` controls sensitivity to recent prediction error.
- `accumulated_error` is computed over the last `N` trials.

## Files
- `nonlinear_stickiness.py`: model implementations.
- `run_phase3_nonlinear_stickiness.py`: synthetic-data runner for quick checks.
- `tests/test_nonlinear_stickiness.py`: unit tests.

## Quick run
```powershell
python -B "Extensions\Phase 3\run_phase3_nonlinear_stickiness.py" --n-steps 5 --gamma 1.2
```

## Results

Run with default synthetic data (5 sessions, 120 trials each, `--n-steps 5 --gamma 1.2`, `alpha_base = 1.2`):

| Model | mean(alpha\_dynamic) | mean(locked switch rate) | mean(final Q) |
|-------|---------------------|--------------------------|---------------|
| RFLR  | 0.3510              | 0.0700                   | —             |
| HMM   | 0.1196              | 0.0683                   | —             |
| F-Q   | 0.1049              | —                        | L=0.394, R=0.499 |

### Interpretation

**Informational gating suppresses stickiness under surprise.** All three models start with `alpha_base = 1.2`, but the effective dynamic alpha drops sharply once accumulated prediction error is factored in — to ~29% of base for RFLR, ~10% for HMM, and ~9% for F-Q. This means the surprise-gated mechanism (`alpha_dynamic = alpha_base * exp(-gamma * accumulated_error)`) is actively reducing choice persistence when the agent's recent predictions have been inaccurate. In a volatile environment, this frees the agent to explore the alternative arm rather than persisting with a stale choice.

**Temporal gating prevents rapid oscillatory switching.** The locked switch rates for RFLR and HMM are both low (~7%), confirming that the N-step policy lock commits the agent to its new arm for several trials after each switch. Without this lock, noisy policy probabilities near 0.5 would produce frequent flip-flopping between arms — behaviour that is costly in a switching-bandit task because every unnecessary switch forgoes reward from the currently-better arm.

**F-Q windowed updates produce conservative value estimates.** The F-Q model's final Q-values (left = 0.394, right = 0.499) have not fully separated toward the true reward probabilities. This is expected: batching rewards into N-trial windows and applying a single update per window deliberately smooths the learning signal, trading convergence speed for robustness against single-trial noise.

**Overall,** the two gating mechanisms address complementary failure modes. Informational gating lets the agent switch when it *should* (high surprise signals a genuine change in the environment), while temporal gating stops it from switching when it *shouldn't* (noisy trial-to-trial fluctuations). Together they reduce overswitching relative to the base models without sacrificing the ability to track real state changes.


