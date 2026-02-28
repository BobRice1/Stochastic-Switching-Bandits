from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
from collections import deque

ArrayLikeInt = Sequence[int]
Session = Tuple[ArrayLikeInt, ArrayLikeInt]

_EPS = 1e-8

# ==========================================
# MATH HELPERS
# ==========================================

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))

def _pm1(x: np.ndarray) -> np.ndarray:
    return 2 * x - 1

def dynamic_alpha(alpha_base: float, gamma: float, accumulated_error: float) -> float:
    """
    INFORMATIONAL GATING: Compute surprise-gated base alpha.
    Allows base stickiness to decrease exponentially in proportion to the 
    accumulated prediction error over the last N trials.
    """
    return float(alpha_base * np.exp(-gamma * accumulated_error))

def _ensure_array_session(session: Session) -> Tuple[np.ndarray, np.ndarray]:
    choices, rewards = session
    return np.asarray(choices, dtype=int), np.asarray(rewards, dtype=float)

def _windowed_q_update(q_prev: np.ndarray, choices: np.ndarray, rewards: np.ndarray, k: float) -> np.ndarray:
    """
    TEMPORAL GATING (VALUE-BASED): Performs a smoothed value update 
    using the mean reward aggregated over an N-step macro-action window.
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


# ==========================================
# PHASE 4: RFLR DYNAMIC AGENT
# ==========================================

def run_rflr_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
    policy_mode: str = "stochastic",
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    
    alpha_base, beta, tau = parameters
    decay = float(np.exp(-1.0 / tau))
    rng = np.random.default_rng(seed)

    total_ll = 0.0
    policies, alpha_traces = [], []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0: continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.zeros(t_max, dtype=float)
        error_buffer = deque(maxlen=n_steps)
        
        # Init trial 0
        policy[0, :] = [0.5, 0.5]
        alpha_trace[0] = alpha_base
        
        current_choice = int(mouse_choices[0]) if fit_mode else (int(rng.random() < 0.5))
        agent_cbar_prev = _pm1(np.array([current_choice]))[0]
        
        error_buffer.append(float(abs(rewards[0] - 0.5)))
        phi = float(beta * rewards[0] * agent_cbar_prev)

        for t in range(1, t_max):
            # INFORMATIONAL GATING
            accumulated_error = sum(error_buffer) / len(error_buffer)
            alpha_t = dynamic_alpha(alpha_base, gamma, accumulated_error)
            alpha_trace[t] = alpha_t

            drive = phi + alpha_t * agent_cbar_prev
            p_right = float(_sigmoid(drive))
            policy[t, :] = [1.0 - p_right, p_right]

            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            agent_cbar_curr = _pm1(np.array([current_choice]))[0]

            expected_reward_t = float(_sigmoid(phi * agent_cbar_curr))
            error_buffer.append(float(abs(rewards[t] - expected_reward_t)))
            phi = decay * phi + float(beta * rewards[t] * agent_cbar_curr)
            agent_cbar_prev = agent_cbar_curr

        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}


# ==========================================
# PHASE 4: HMM DYNAMIC AGENT
# ==========================================

def run_hmm_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    policy_mode: str = "stochastic",
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    
    q, p, alpha_base, beta, tau = parameters["q"], parameters["p"], parameters["alpha"], parameters["beta"], parameters["tau"]
    decay = float(np.exp(-1.0 / tau))
    transition = np.array([[q, 1.0 - q], [1.0 - q, q]], dtype=float)
    rng = np.random.default_rng(seed)

    total_ll = 0.0
    policies, alpha_traces = [], []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0: continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.zeros(t_max, dtype=float)
        stickiness = np.zeros(t_max, dtype=float)
        error_buffer = deque(maxlen=n_steps)
        posterior_prev = np.array([0.5, 0.5])
        agent_choices_pm1 = np.zeros(t_max)

        for t in range(t_max):
            prior = transition.T @ posterior_prev
            
            # 1. Calculate alpha_t
            if t == 0:
                alpha_t = alpha_base
            else:
                accumulated_error = sum(error_buffer) / len(error_buffer)
                alpha_t = dynamic_alpha(alpha_base, gamma, accumulated_error)

            # 2. Update Stickiness (Top-of-loop, perfectly matching RFLR)
            if t > 0:
                s1, s2 = alpha_t + beta/2.0, -alpha_t * decay
                stickiness[t] = decay * stickiness[t-1] + s1 * agent_choices_pm1[t-1]
                if t > 1: stickiness[t] += s2 * agent_choices_pm1[t-2]

            # 3. Policy Calculation
            p_right = float(_sigmoid(_logit(prior[1]) + stickiness[t]))
            alpha_trace[t] = alpha_t
            policy[t, :] = [1.0 - p_right, p_right]

            # 4. Make and Record the Choice
            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            agent_choices_pm1[t] = _pm1(np.array([current_choice]))[0]

            # 5. Record Error and Update Posterior
            p_reward_state = np.array([p if current_choice == 0 else 1.0-p, p if current_choice == 1 else 1.0-p])
            error_buffer.append(float(abs(rewards[t] - float(np.dot(prior, p_reward_state)))))
            
            likelihood = p_reward_state if rewards[t] > 0.5 else (1.0 - p_reward_state)
            posterior_prev = (prior * likelihood) / np.maximum((prior * likelihood).sum(), _EPS)
                        
        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}

def run_hmm_reset_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    policy_mode: str = "stochastic",  
    seed: int | None = None,          
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    
    # Extract baseline parameters
    q, p, alpha_base = parameters["q"], parameters["p"], parameters["alpha"]
    beta, tau = parameters["beta"], parameters["tau"]
    decay = float(np.exp(-1.0 / tau))
    transition = np.array([[q, 1.0 - q], [1.0 - q, q]], dtype=float)
    
    rng = np.random.default_rng(seed)

    total_ll = 0.0
    policies, alpha_traces = [], []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0: continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.zeros(t_max, dtype=float)
        stickiness = np.zeros(t_max, dtype=float)
        error_buffer = deque(maxlen=n_steps)
        posterior_prev = np.array([0.5, 0.5])
        agent_choices_pm1 = np.zeros(t_max)

        for t in range(t_max):
            prior = transition.T @ posterior_prev
            
            # 1. NORMALIZED SURPRISE CALCULATION
            if t == 0:
                alpha_t, surprise_signal = alpha_base, 0.0
            else:
                avg_error = sum(error_buffer) / len(error_buffer)
                alpha_t = dynamic_alpha(alpha_base, gamma, avg_error)
                surprise_signal = 1.0 - np.exp(-gamma * avg_error) # 0 to 1 scale

            # 2. STICKINESS UPDATE (Motor Gating)
            if t > 0:
                s1, s2 = alpha_t + beta/2.0, -alpha_t * decay
                stickiness[t] = decay * stickiness[t-1] + s1 * agent_choices_pm1[t-1]
                if t > 1: stickiness[t] += s2 * agent_choices_pm1[t-2]
            
            # 3. POLICY CALCULATION
            p_right = float(_sigmoid(_logit(prior[1]) + stickiness[t]))
            policy[t, :] = [1.0 - p_right, p_right]
            alpha_trace[t] = alpha_t

            # 4. CHOICE & REWARD OBSERVATION
            # RESTORED: Generative vs Fit mode logic
            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            agent_choices_pm1[t] = _pm1(np.array([current_choice]))[0]

            # 5. BAYESIAN UPDATE + RESET (Cognitive Gating)
            p_reward_state = np.array([p if current_choice == 0 else 1.0-p, 
                                       p if current_choice == 1 else 1.0-p])
            likelihood = p_reward_state if rewards[t] > 0.5 else (1.0 - p_reward_state)
            
            # Raw Bayesian Posterior
            posterior_raw = (prior * likelihood) / np.maximum((prior * likelihood).sum(), _EPS)
            
            # ABLATION: Uncertainty Reset (Mix toward 0.5 based on surprise)
            posterior_prev = (1.0 - surprise_signal) * posterior_raw + surprise_signal * np.array([0.5, 0.5])
            
            # Store error for next trial
            error_buffer.append(float(abs(rewards[t] - float(np.dot(prior, p_reward_state)))))

        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}


# ==========================================
# PHASE 4: F-Q DYNAMIC AGENT
# ==========================================

def run_fq_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
    policy_mode: str = "stochastic",
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    
    alpha_base, k, temp = parameters
    rng = np.random.default_rng(seed)
    total_ll = 0.0
    policies, alpha_traces = [], []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0: continue

        policy, alpha_trace = np.zeros((t_max, 2)), np.zeros(t_max)
        q_curr = np.array([0.5, 0.5])
        error_buffer = deque(maxlen=n_steps)
        window_choices, window_rewards = [], []
        
        # Init trial 0
        policy[0, :] = [0.5, 0.5]
        alpha_trace[0] = alpha_base
        current_choice = int(mouse_choices[0]) if fit_mode else (int(rng.random() < 0.5))
        agent_pm1_prev = _pm1(np.array([current_choice]))[0]
        error_buffer.append(float(abs(rewards[0] - q_curr[current_choice])))
        window_choices.append(current_choice); window_rewards.append(rewards[0])

        for t in range(1, t_max):
            accumulated_error = sum(error_buffer) / len(error_buffer)
            alpha_t = dynamic_alpha(alpha_base, gamma, accumulated_error)
            alpha_trace[t] = alpha_t
            
            p_right = float(_sigmoid(((q_curr[1]-q_curr[0])/max(temp, _EPS)) + alpha_t * agent_pm1_prev))
            policy[t, :] = [1.0 - p_right, p_right]

            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            error_buffer.append(float(abs(rewards[t] - q_curr[current_choice])))
            
            window_choices.append(current_choice); window_rewards.append(rewards[t])
            if len(window_rewards) == n_steps or t == t_max - 1:
                q_curr = _windowed_q_update(q_curr, np.array(window_choices), np.array(window_rewards), k)
                window_choices, window_rewards = [], []
            
            agent_pm1_prev = _pm1(np.array([current_choice]))[0]

        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}

def run_fq_value_gated_only(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,  # Gamma remains in signature for compatibility but is unused
    policy_mode: str = "stochastic",
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    
    alpha_base, k, temp = parameters
    rng = np.random.default_rng(seed)
    total_ll = 0.0
    policies, alpha_traces = [], []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0: continue

        policy, alpha_trace = np.zeros((t_max, 2)), np.zeros(t_max)
        q_curr = np.array([0.5, 0.5])
        window_choices, window_rewards = [], []
        
        # Init trial 0
        policy[0, :] = [0.5, 0.5]
        alpha_trace[0] = alpha_base
        current_choice = int(mouse_choices[0]) if fit_mode else (int(rng.random() < 0.5))
        agent_pm1_prev = _pm1(np.array([current_choice]))[0]
        
        # Collect data for the first window
        window_choices.append(current_choice)
        window_rewards.append(rewards[0])

        for t in range(1, t_max):
            # VALUE-BASED GATING: alpha remains constant
            alpha_t = alpha_base 
            alpha_trace[t] = alpha_t
            
            # Choice probability calculation using current (potentially frozen) Q-values
            p_right = float(_sigmoid(((q_curr[1]-q_curr[0])/max(temp, _EPS)) + alpha_t * agent_pm1_prev))
            policy[t, :] = [1.0 - p_right, p_right]

            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            
            # Accumulate rewards and choices in the window
            window_choices.append(current_choice)
            window_rewards.append(rewards[t])
            
            # MACRO-ACTION: Update Q-values only when the window N is full
            if len(window_rewards) == n_steps or t == t_max - 1:
                q_curr = _windowed_q_update(q_curr, np.array(window_choices), np.array(window_rewards), k)
                window_choices, window_rewards = [], []
            
            agent_pm1_prev = _pm1(np.array([current_choice]))[0]

        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}