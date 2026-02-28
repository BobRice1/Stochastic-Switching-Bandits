"""Phase 3 extension: non-linear stickiness with temporal and informational gating.

Implements:
1. Temporal gating
   - RFLR/HMM: policy-level lock for N trials after each predicted switch.
   - F-Q: reward accumulation over N trials with one Q update per window.
2. Informational gating
   - Dynamic alpha controlled by surprise:
     alpha_dynamic = alpha_base * exp(-gamma * accumulated_error)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ArrayLikeInt = Sequence[int]
Session = Tuple[ArrayLikeInt, ArrayLikeInt]

_EPS = 1e-8


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _pm1(x: np.ndarray) -> np.ndarray:
    return 2 * x - 1


def dynamic_alpha(alpha_base: float, gamma: float, accumulated_error: float) -> float:
    """Compute surprise-gated alpha."""
    return float(alpha_base * np.exp(-gamma * accumulated_error))


def apply_temporal_policy_lock(
    policy_probs: np.ndarray,
    n_lock_trials: int,
    mode: str = "stochastic",
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply policy-level lock for n_lock_trials after each predicted switch.

    Args:
        policy_probs: (T, 2) policy probabilities.
        n_lock_trials: number of *future* trials to lock after each switch.
        mode: "stochastic" or "greedy" action sampling from policy_probs.
    """
    if n_lock_trials < 0:
        raise ValueError("n_lock_trials must be >= 0.")
    if mode not in {"stochastic", "greedy"}:
        raise ValueError("mode must be 'stochastic' or 'greedy'.")

    if rng is None:
        rng = np.random.default_rng()

    t_max = policy_probs.shape[0]
    choices = np.zeros(t_max, dtype=int)
    switches = np.zeros(t_max, dtype=int)

    lock_remaining = 0
    locked_choice = 0

    for t in range(t_max):
        if t == 0:
            p_right = policy_probs[t, 1]
            choices[t] = int(rng.random() < p_right) if mode == "stochastic" else int(round(p_right))
            continue

        if lock_remaining > 0:
            choices[t] = locked_choice
            lock_remaining -= 1
        else:
            p_right = policy_probs[t, 1]
            choices[t] = int(rng.random() < p_right) if mode == "stochastic" else int(round(p_right))

        if choices[t] != choices[t - 1]:
            switches[t] = 1
            locked_choice = choices[t]
            lock_remaining = n_lock_trials

    return choices, switches


def _ensure_array_session(session: Session) -> Tuple[np.ndarray, np.ndarray]:
    choices, rewards = session
    return np.asarray(choices, dtype=int), np.asarray(rewards, dtype=float)


def _windowed_q_update(q_prev: np.ndarray, choices: np.ndarray, rewards: np.ndarray, k: float) -> np.ndarray:
    """Single Q update from a reward window."""
    q_next = q_prev.copy()
    for arm in (0, 1):
        mask = choices == arm
        if np.any(mask):
            reward_mean = float(np.mean(rewards[mask]))
            q_next[arm] = q_prev[arm] + k * (reward_mean - q_prev[arm])
        else:
            q_next[arm] = (1.0 - k) * q_prev[arm]
    return np.clip(q_next, 0.0, 1.0)


def run_rflr_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
    policy_mode: str = "stochastic",
    seed: int | None = None,
) -> Dict[str, List[np.ndarray]]:
    """RFLR with surprise-gated alpha + policy-level temporal lock."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")

    alpha_base, beta, tau = parameters
    decay = float(np.exp(-1.0 / tau))

    rng = np.random.default_rng(seed)

    policies: List[np.ndarray] = []
    alpha_traces: List[np.ndarray] = []
    locked_choices: List[np.ndarray] = []
    locked_switches: List[np.ndarray] = []

    for session in behavior_features:
        choices, rewards = _ensure_array_session(session)
        t_max = len(choices)
        if t_max == 0:
            policies.append(np.zeros((0, 2), dtype=float))
            alpha_traces.append(np.zeros(0, dtype=float))
            locked_choices.append(np.zeros(0, dtype=int))
            locked_switches.append(np.zeros(0, dtype=int))
            continue

        policy = np.zeros((t_max, 2), dtype=float)
        policy[0, :] = [0.5, 0.5]
        alpha_trace = np.full(t_max, alpha_base, dtype=float)

        error_buffer: deque[float] = deque(maxlen=n_steps)
        error_buffer.append(float(abs(rewards[0] - 0.5)))

        cbar = _pm1(choices.astype(int))
        phi = float(beta * rewards[0] * cbar[0])

        for t in range(1, t_max):
            accumulated_error = float(sum(error_buffer))
            alpha_t = dynamic_alpha(alpha_base, gamma, accumulated_error)
            alpha_trace[t] = alpha_t

            drive = phi + alpha_t * cbar[t - 1]
            p_right = float(_sigmoid(drive))
            policy[t, 1] = p_right
            policy[t, 0] = 1.0 - p_right

            expected_reward_prev = float(_sigmoid(phi * cbar[t - 1]))
            error_buffer.append(float(abs(rewards[t - 1] - expected_reward_prev)))

            phi = decay * phi + float(beta * rewards[t] * cbar[t])

        pred_choices, pred_switches = apply_temporal_policy_lock(
            policy_probs=policy,
            n_lock_trials=n_steps,
            mode=policy_mode,
            rng=rng,
        )

        policies.append(policy)
        alpha_traces.append(alpha_trace)
        locked_choices.append(pred_choices)
        locked_switches.append(pred_switches)

    return {
        "policies": policies,
        "alpha_traces": alpha_traces,
        "locked_choices": locked_choices,
        "locked_switches": locked_switches,
    }


def run_hmm_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    policy_mode: str = "stochastic",
    seed: int | None = None,
) -> Dict[str, List[np.ndarray]]:
    """2-state HMM with surprise-gated stickiness + policy-level lock.

    Required parameter keys:
      q: state self-transition probability
      p: reward probability when choosing the correct arm
      alpha: base stickiness scale
      beta: stickiness offset term used in the original sticky-HMM
      tau: stickiness decay timescale
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")

    q = float(parameters["q"])
    p = float(parameters["p"])
    alpha_base = float(parameters["alpha"])
    beta = float(parameters["beta"])
    tau = float(parameters["tau"])

    decay = float(np.exp(-1.0 / tau))
    transition = np.array([[q, 1.0 - q], [1.0 - q, q]], dtype=float)

    rng = np.random.default_rng(seed)

    beliefs: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    alpha_traces: List[np.ndarray] = []
    stickiness_traces: List[np.ndarray] = []
    locked_choices: List[np.ndarray] = []
    locked_switches: List[np.ndarray] = []

    for session in behavior_features:
        choices, rewards = _ensure_array_session(session)
        t_max = len(choices)
        if t_max == 0:
            beliefs.append(np.zeros((0, 2), dtype=float))
            policies.append(np.zeros((0, 2), dtype=float))
            alpha_traces.append(np.zeros(0, dtype=float))
            stickiness_traces.append(np.zeros(0, dtype=float))
            locked_choices.append(np.zeros(0, dtype=int))
            locked_switches.append(np.zeros(0, dtype=int))
            continue

        belief = np.zeros((t_max, 2), dtype=float)
        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.full(t_max, alpha_base, dtype=float)
        stickiness = np.zeros(t_max, dtype=float)

        error_buffer: deque[float] = deque(maxlen=n_steps)
        posterior_prev = np.array([0.5, 0.5], dtype=float)
        cbar = _pm1(choices.astype(int))

        for t in range(t_max):
            prior = transition.T @ posterior_prev
            choice_t = int(choices[t])
            reward_t = float(rewards[t])

            # P(reward=1 | state=k, choice=choice_t)
            p_reward_state = np.array(
                [
                    p if choice_t == 0 else 1.0 - p,
                    p if choice_t == 1 else 1.0 - p,
                ],
                dtype=float,
            )

            expected_reward = float(np.dot(prior, p_reward_state))
            likelihood = p_reward_state if reward_t > 0.5 else (1.0 - p_reward_state)
            posterior = prior * likelihood
            posterior /= np.maximum(posterior.sum(), _EPS)

            belief[t, :] = posterior

            if t == 0:
                policy[t, 1] = posterior[1]
                policy[t, 0] = 1.0 - policy[t, 1]
            else:
                accumulated_error = float(sum(error_buffer))
                alpha_t = dynamic_alpha(alpha_base, gamma, accumulated_error)
                alpha_trace[t] = alpha_t

                s1 = alpha_t + beta / 2.0
                s2 = -alpha_t * decay
                if t == 1:
                    stickiness[t] = s1 * cbar[t - 1]
                else:
                    stickiness[t] = decay * stickiness[t - 1]
                    stickiness[t] += s1 * cbar[t - 1]
                    stickiness[t] += s2 * cbar[t - 2]

                p_right = float(_sigmoid(_logit(posterior[1]) + stickiness[t]))
                policy[t, 1] = p_right
                policy[t, 0] = 1.0 - p_right

            error_buffer.append(float(abs(reward_t - expected_reward)))
            posterior_prev = posterior

        pred_choices, pred_switches = apply_temporal_policy_lock(
            policy_probs=policy,
            n_lock_trials=n_steps,
            mode=policy_mode,
            rng=rng,
        )

        beliefs.append(belief)
        policies.append(policy)
        alpha_traces.append(alpha_trace)
        stickiness_traces.append(stickiness)
        locked_choices.append(pred_choices)
        locked_switches.append(pred_switches)

    return {
        "beliefs": beliefs,
        "policies": policies,
        "alpha_traces": alpha_traces,
        "stickiness_traces": stickiness_traces,
        "locked_choices": locked_choices,
        "locked_switches": locked_switches,
    }


def run_fq_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
) -> Dict[str, List[np.ndarray]]:
    """F-Q model with surprise-gated alpha + N-step reward-window updates."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")

    alpha_base, k, temp = parameters

    policies: List[np.ndarray] = []
    alpha_traces: List[np.ndarray] = []
    q_traces: List[np.ndarray] = []

    for session in behavior_features:
        choices, rewards = _ensure_array_session(session)
        choices = choices.astype(int)
        t_max = len(choices)

        if t_max == 0:
            policies.append(np.zeros((0, 2), dtype=float))
            alpha_traces.append(np.zeros(0, dtype=float))
            q_traces.append(np.zeros((0, 2), dtype=float))
            continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.full(t_max, alpha_base, dtype=float)
        q_trace = np.zeros((t_max, 2), dtype=float)

        policy[0, :] = [0.5, 0.5]
        q_curr = np.array([0.0, 0.0], dtype=float)
        q_trace[0, :] = q_curr

        error_buffer: deque[float] = deque(maxlen=n_steps)
        error_buffer.append(float(abs(rewards[0] - q_curr[choices[0]])))

        window_choices: List[int] = [int(choices[0])]
        window_rewards: List[float] = [float(rewards[0])]

        if len(window_rewards) == n_steps:
            q_curr = _windowed_q_update(
                q_prev=q_curr,
                choices=np.asarray(window_choices, dtype=int),
                rewards=np.asarray(window_rewards, dtype=float),
                k=float(k),
            )
            window_choices.clear()
            window_rewards.clear()
            q_trace[0, :] = q_curr

        for t in range(1, t_max):
            accumulated_error = float(sum(error_buffer))
            alpha_t = dynamic_alpha(alpha_base, gamma, accumulated_error)
            alpha_trace[t] = alpha_t

            drive = ((q_curr[1] - q_curr[0]) / temp) + alpha_t * (2 * choices[t - 1] - 1)
            p_right = float(_sigmoid(drive))
            policy[t, 1] = p_right
            policy[t, 0] = 1.0 - p_right

            expected_reward_t = float(q_curr[choices[t]])
            error_buffer.append(float(abs(rewards[t] - expected_reward_t)))

            window_choices.append(int(choices[t]))
            window_rewards.append(float(rewards[t]))

            if len(window_rewards) == n_steps or t == t_max - 1:
                q_curr = _windowed_q_update(
                    q_prev=q_curr,
                    choices=np.asarray(window_choices, dtype=int),
                    rewards=np.asarray(window_rewards, dtype=float),
                    k=float(k),
                )
                window_choices.clear()
                window_rewards.clear()

            q_trace[t, :] = q_curr

        policies.append(policy)
        alpha_traces.append(alpha_trace)
        q_traces.append(q_trace)

    return {
        "policies": policies,
        "alpha_traces": alpha_traces,
        "q_traces": q_traces,
    }

