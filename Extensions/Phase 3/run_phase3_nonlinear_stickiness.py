from __future__ import annotations
import argparse
from typing import List, Tuple
import numpy as np

from nonlinear_stickiness import (
    run_fq_nonlinear_stickiness,
    run_hmm_nonlinear_stickiness,
    run_rflr_nonlinear_stickiness,
)


Session = Tuple[np.ndarray, np.ndarray]


def generate_synthetic_sessions(
    n_sessions: int,
    n_trials: int,
    p_switch: float,
    p_reward: float,
    seed: int,
) -> List[Session]:
    rng = np.random.default_rng(seed)
    sessions: List[Session] = []

    for _ in range(n_sessions):
        states = np.zeros(n_trials, dtype=int)
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)

        states[0] = int(rng.integers(0, 2))
        choices[0] = int(rng.integers(0, 2))
        rewards[0] = int(rng.random() < (p_reward if choices[0] == states[0] else (1.0 - p_reward)))

        for t in range(1, n_trials):
            states[t] = states[t - 1] if rng.random() > p_switch else (1 - states[t - 1])

            if rewards[t - 1] == 1:
                choices[t] = choices[t - 1]
            else:
                choices[t] = (1 - choices[t - 1]) if rng.random() < 0.7 else choices[t - 1]

            prob_reward = p_reward if choices[t] == states[t] else (1.0 - p_reward)
            rewards[t] = int(rng.random() < prob_reward)

        sessions.append((choices, rewards))

    return sessions


def summarize(name: str, outputs: dict) -> None:
    alpha_means = [float(np.mean(a)) for a in outputs["alpha_traces"]]
    print(f"[{name}] mean(alpha_dynamic): {np.mean(alpha_means):.4f}")

    if "locked_switches" in outputs:
        switch_rates = [float(np.mean(s)) for s in outputs["locked_switches"]]
        print(f"[{name}] mean(locked switch rate): {np.mean(switch_rates):.4f}")

    if "q_traces" in outputs:
        q_last = np.mean([q[-1, :] for q in outputs["q_traces"]], axis=0)
        print(f"[{name}] mean(final Q): left={q_last[0]:.4f}, right={q_last[1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 non-linear stickiness runner")
    parser.add_argument("--n-sessions", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=120)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=20260220)
    args = parser.parse_args()

    sessions = generate_synthetic_sessions(
        n_sessions=args.n_sessions,
        n_trials=args.n_trials,
        p_switch=0.1,
        p_reward=0.8,
        seed=args.seed,
    )

    rflr_out = run_rflr_nonlinear_stickiness(
        behavior_features=sessions,
        parameters=(1.2, 2.0, 6.0),
        n_steps=args.n_steps,
        gamma=args.gamma,
        policy_mode="stochastic",
        seed=args.seed,
    )

    hmm_out = run_hmm_nonlinear_stickiness(
        behavior_features=sessions,
        parameters={"q": 0.9, "p": 0.8, "alpha": 1.2, "beta": 2.0, "tau": 6.0},
        n_steps=args.n_steps,
        gamma=args.gamma,
        policy_mode="stochastic",
        seed=args.seed,
    )

    fq_out = run_fq_nonlinear_stickiness(
        behavior_features=sessions,
        parameters=(1.2, 0.35, 1.0),
        n_steps=args.n_steps,
        gamma=args.gamma,
    )

    summarize("RFLR", rflr_out)
    summarize("HMM", hmm_out)
    summarize("FQ", fq_out)


if __name__ == "__main__":
    main()

