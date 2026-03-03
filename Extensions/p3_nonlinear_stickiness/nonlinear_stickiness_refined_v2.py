from collections import deque
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

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


def dynamic_alpha(alpha_base: float, gamma: float, accumulated_pe: float) -> float:
    """
    Asymmetric informational gating based on signed prediction error.
    """
    return float(alpha_base * np.exp(gamma * accumulated_pe))


def _ensure_array_session(session: Session) -> Tuple[np.ndarray, np.ndarray]:
    choices, rewards = session
    return np.asarray(choices, dtype=int), np.asarray(rewards, dtype=float)


def _windowed_q_update(q_prev: np.ndarray, choices: np.ndarray, rewards: np.ndarray, k: float) -> np.ndarray:
    """
    Temporal gating: update values once per N-trial window using mean reward per action.
    """
    q_next = q_prev.copy()
    for arm in (0, 1):
        mask = choices == arm
        if np.any(mask):
            reward_mean = float(np.mean(rewards[mask]))
            q_next[arm] = q_prev[arm] + k * (reward_mean - q_prev[arm])
        else:
            q_next[arm] = (1.0 - k) * q_prev[arm]
    return np.clip(q_next, 0.0, 1.0)


def _buffer_mean(buffer: deque[float]) -> float:
    return float(sum(buffer) / len(buffer))


def _select_choice(
    rng: np.random.Generator,
    mouse_choices: np.ndarray,
    t: int,
    p_right: float,
    fit_mode: bool,
) -> int:
    if fit_mode:
        return int(mouse_choices[t])
    return int(rng.random() < p_right)


def _finalize_session_results(
    policy: np.ndarray,
    alpha_trace: np.ndarray,
    mouse_choices: np.ndarray,
    total_ll: float,
    return_ll_only: bool,
    policies: List[np.ndarray],
    alpha_traces: List[np.ndarray],
) -> float:
    if return_ll_only:
        chosen_probs = policy[np.arange(len(mouse_choices)), mouse_choices]
        return total_ll + float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))

    policies.append(policy)
    alpha_traces.append(alpha_trace)
    return total_ll


def _phase2_stickiness_update(
    stickiness: np.ndarray,
    agent_choices_pm1: np.ndarray,
    t: int,
    alpha_t: float,
    beta: float,
    decay: float,
) -> None:
    if t == 1:
        stickiness[t] = (alpha_t + beta / 2.0) * agent_choices_pm1[0]
    elif t > 1:
        s1 = alpha_t + beta / 2.0
        s2 = -alpha_t * decay
        stickiness[t] = (
            decay * stickiness[t - 1]
            + s1 * agent_choices_pm1[t - 1]
            + s2 * agent_choices_pm1[t - 2]
        )


def _hmm_reward_probabilities(current_choice: int, p_reward: float) -> np.ndarray:
    if current_choice == 0:
        return np.array([p_reward, 1.0 - p_reward], dtype=float)
    return np.array([1.0 - p_reward, p_reward], dtype=float)


def _asymmetric_reset_signal(avg_pe: float, gamma: float) -> float:
    negative_pe = min(0.0, avg_pe)
    return float(1.0 - np.exp(gamma * negative_pe))


def _symmetric_reset_signal(avg_pe: float, gamma: float) -> float:
    return float(1.0 - np.exp(-gamma * abs(avg_pe)))


def _run_hmm_variant(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    seed: int | None,
    fit_mode: bool,
    return_ll_only: bool,
    reset_signal_fn: Callable[[float, float], float] | None,
) -> Dict[str, List[np.ndarray]] | float:
    q_val = parameters["q"]
    p_reward = parameters["p"]
    alpha_base = parameters["alpha"]
    beta = parameters["beta"]
    tau = parameters["tau"]

    decay = float(np.exp(-1.0 / tau))
    transition = np.array([[q_val, 1.0 - q_val], [1.0 - q_val, q_val]], dtype=float)
    rng = np.random.default_rng(seed)

    total_ll = 0.0
    policies: List[np.ndarray] = []
    alpha_traces: List[np.ndarray] = []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0:
            continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.zeros(t_max, dtype=float)
        stickiness = np.zeros(t_max, dtype=float)
        pe_buffer: deque[float] = deque(maxlen=n_steps)
        posterior_prev = np.array([0.5, 0.5], dtype=float)
        agent_choices_pm1 = np.zeros(t_max, dtype=float)

        for t in range(t_max):
            prior = transition.T @ posterior_prev

            if t == 0:
                alpha_t = alpha_base
                reset_signal = 0.0
            else:
                avg_pe = _buffer_mean(pe_buffer)
                alpha_t = dynamic_alpha(alpha_base, gamma, avg_pe)
                reset_signal = 0.0 if reset_signal_fn is None else reset_signal_fn(avg_pe, gamma)

            _phase2_stickiness_update(stickiness, agent_choices_pm1, t, alpha_t, beta, decay)

            p_right = float(_sigmoid(_logit(prior[1]) + stickiness[t]))
            policy[t, :] = [1.0 - p_right, p_right]
            alpha_trace[t] = alpha_t

            current_choice = _select_choice(rng, mouse_choices, t, p_right, fit_mode)
            agent_choices_pm1[t] = _pm1(np.array([current_choice]))[0]

            p_reward_state = _hmm_reward_probabilities(current_choice, p_reward)
            expected_reward_t = float(np.dot(prior, p_reward_state))
            pe_buffer.append(float(rewards[t] - expected_reward_t))

            likelihood = p_reward_state if rewards[t] > 0.5 else (1.0 - p_reward_state)
            posterior_raw = (prior * likelihood) / np.maximum((prior * likelihood).sum(), _EPS)

            if reset_signal_fn is None:
                posterior_prev = posterior_raw
            else:
                posterior_prev = (
                    (1.0 - reset_signal) * posterior_raw
                    + reset_signal * np.array([0.5, 0.5], dtype=float)
                )

        total_ll = _finalize_session_results(
            policy,
            alpha_trace,
            mouse_choices,
            total_ll,
            return_ll_only,
            policies,
            alpha_traces,
        )

    if return_ll_only:
        return total_ll
    return {"policies": policies, "alpha_traces": alpha_traces}


def _run_fq_variant(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
    seed: int | None,
    fit_mode: bool,
    return_ll_only: bool,
    error_fn: Callable[[float, np.ndarray, int], float] | None,
) -> Dict[str, List[np.ndarray]] | float:
    alpha_base, k_val, temp = parameters
    rng = np.random.default_rng(seed)

    total_ll = 0.0
    policies: List[np.ndarray] = []
    alpha_traces: List[np.ndarray] = []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0:
            continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.zeros(t_max, dtype=float)
        q_curr = np.array([0.5, 0.5], dtype=float)
        signal_buffer: deque[float] | None = deque(maxlen=n_steps) if error_fn is not None else None
        window_choices: List[int] = []
        window_rewards: List[float] = []

        policy[0, :] = [0.5, 0.5]
        alpha_trace[0] = alpha_base
        current_choice = _select_choice(rng, mouse_choices, 0, 0.5, fit_mode)
        agent_pm1_prev = _pm1(np.array([current_choice]))[0]

        if signal_buffer is not None:
            signal_buffer.append(error_fn(rewards[0], q_curr, current_choice))

        window_choices.append(current_choice)
        window_rewards.append(rewards[0])

        for t in range(1, t_max):
            if signal_buffer is None:
                alpha_t = alpha_base
            else:
                alpha_t = dynamic_alpha(alpha_base, gamma, _buffer_mean(signal_buffer))

            alpha_trace[t] = alpha_t
            drive = ((q_curr[1] - q_curr[0]) / max(temp, _EPS)) + alpha_t * agent_pm1_prev
            p_right = float(_sigmoid(drive))
            policy[t, :] = [1.0 - p_right, p_right]

            current_choice = _select_choice(rng, mouse_choices, t, p_right, fit_mode)
            if signal_buffer is not None:
                signal_buffer.append(error_fn(rewards[t], q_curr, current_choice))

            window_choices.append(current_choice)
            window_rewards.append(rewards[t])

            if len(window_rewards) == n_steps or t == t_max - 1:
                q_curr = _windowed_q_update(
                    q_curr,
                    np.asarray(window_choices, dtype=int),
                    np.asarray(window_rewards, dtype=float),
                    k_val,
                )
                window_choices = []
                window_rewards = []

            agent_pm1_prev = _pm1(np.array([current_choice]))[0]

        total_ll = _finalize_session_results(
            policy,
            alpha_trace,
            mouse_choices,
            total_ll,
            return_ll_only,
            policies,
            alpha_traces,
        )

    if return_ll_only:
        return total_ll
    return {"policies": policies, "alpha_traces": alpha_traces}


def run_rflr_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    alpha_base, beta, tau = parameters
    decay = float(np.exp(-1.0 / tau))
    rng = np.random.default_rng(seed)

    total_ll = 0.0
    policies: List[np.ndarray] = []
    alpha_traces: List[np.ndarray] = []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0:
            continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.zeros(t_max, dtype=float)
        pe_buffer: deque[float] = deque(maxlen=n_steps)

        policy[0, :] = [0.5, 0.5]
        alpha_trace[0] = alpha_base

        current_choice = _select_choice(rng, mouse_choices, 0, 0.5, fit_mode)
        agent_cbar_prev = _pm1(np.array([current_choice]))[0]

        pe_buffer.append(float(rewards[0] - 0.5))
        phi = float(beta * rewards[0] * agent_cbar_prev)

        for t in range(1, t_max):
            alpha_t = dynamic_alpha(alpha_base, gamma, _buffer_mean(pe_buffer))
            alpha_trace[t] = alpha_t

            drive = phi + alpha_t * agent_cbar_prev
            p_right = float(_sigmoid(drive))
            policy[t, :] = [1.0 - p_right, p_right]

            current_choice = _select_choice(rng, mouse_choices, t, p_right, fit_mode)
            agent_cbar_curr = _pm1(np.array([current_choice]))[0]

            expected_reward_t = float(_sigmoid(phi * agent_cbar_curr))
            pe_buffer.append(float(rewards[t] - expected_reward_t))

            phi = decay * phi + float(beta * rewards[t] * agent_cbar_curr)
            agent_cbar_prev = agent_cbar_curr

        total_ll = _finalize_session_results(
            policy,
            alpha_trace,
            mouse_choices,
            total_ll,
            return_ll_only,
            policies,
            alpha_traces,
        )

    if return_ll_only:
        return total_ll
    return {"policies": policies, "alpha_traces": alpha_traces}


def run_hmm_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    return _run_hmm_variant(
        behavior_features,
        parameters,
        n_steps,
        gamma,
        seed,
        fit_mode,
        return_ll_only,
        reset_signal_fn=None,
    )


def run_hmm_reset_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    fit_mode: bool = False,
    return_ll_only: bool = False,
    seed: int | None = None,
) -> Dict[str, List[np.ndarray]] | float:
    return _run_hmm_variant(
        behavior_features,
        parameters,
        n_steps,
        gamma,
        seed,
        fit_mode,
        return_ll_only,
        reset_signal_fn=_asymmetric_reset_signal,
    )


def run_hmm_resetv2_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    fit_mode: bool = False,
    return_ll_only: bool = False,
    seed: int | None = None,
) -> Dict[str, List[np.ndarray]] | float:
    return _run_hmm_variant(
        behavior_features,
        parameters,
        n_steps,
        gamma,
        seed,
        fit_mode,
        return_ll_only,
        reset_signal_fn=_symmetric_reset_signal,
    )


def run_fq_value_gated_only(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    del gamma
    return _run_fq_variant(
        behavior_features,
        parameters,
        n_steps,
        gamma=0.0,
        seed=seed,
        fit_mode=fit_mode,
        return_ll_only=return_ll_only,
        error_fn=None,
    )


def run_fq_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    def fq_error(reward: float, q_curr: np.ndarray, current_choice: int) -> float:
        return float(abs(reward - q_curr[current_choice]))

    return _run_fq_variant(
        behavior_features,
        parameters,
        n_steps,
        gamma,
        seed,
        fit_mode,
        return_ll_only,
        error_fn=fq_error,
    )
