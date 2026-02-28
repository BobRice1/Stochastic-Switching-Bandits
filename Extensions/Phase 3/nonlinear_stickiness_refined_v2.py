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

def dynamic_alpha(alpha_base: float, gamma: float, accumulated_pe: float) -> float:
    """
    ASYMMETRIC INFORMATIONAL GATING:
    Uses Signed Prediction Error (PE). 
    Negative PE (omissions) decreases alpha (habit breaks).
    Positive PE (unexpected rewards) increases alpha (habit stamps in).
    """
    return float(alpha_base * np.exp(gamma * accumulated_pe))

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
# PHASE 4: RFLR DYNAMIC AGENT (ASYMMETRIC)
# ==========================================

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
    policies, alpha_traces = [], []

    for session in behavior_features:
        mouse_choices, rewards = _ensure_array_session(session)
        t_max = len(rewards)
        if t_max == 0: continue

        policy = np.zeros((t_max, 2), dtype=float)
        alpha_trace = np.zeros(t_max, dtype=float)
        pe_buffer = deque(maxlen=n_steps) # Signed PE Buffer
        
        # Init trial 0
        policy[0, :] = [0.5, 0.5]
        alpha_trace[0] = alpha_base
        
        current_choice = int(mouse_choices[0]) if fit_mode else (int(rng.random() < 0.5))
        agent_cbar_prev = _pm1(np.array([current_choice]))[0]
        
        # Trial 0 PE: Assumes 0.5 starting expectation
        pe_buffer.append(float(rewards[0] - 0.5))
        phi = float(beta * rewards[0] * agent_cbar_prev)

        for t in range(1, t_max):
            # 1. Calculate alpha_t based on Signed PE
            accumulated_pe = sum(pe_buffer) / len(pe_buffer)
            alpha_t = dynamic_alpha(alpha_base, gamma, accumulated_pe)
            alpha_trace[t] = alpha_t

            # 2. Policy Calculation
            drive = phi + alpha_t * agent_cbar_prev
            p_right = float(_sigmoid(drive))
            policy[t, :] = [1.0 - p_right, p_right]

            # 3. Choice Observation
            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            agent_cbar_curr = _pm1(np.array([current_choice]))[0]

            # 4. Update PE and Value Trace
            expected_reward_t = float(_sigmoid(phi * agent_cbar_curr))
            pe_buffer.append(float(rewards[t] - expected_reward_t))
            
            phi = decay * phi + float(beta * rewards[t] * agent_cbar_curr)
            agent_cbar_prev = agent_cbar_curr

        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}


# ==========================================
# PHASE 4: HMM DYNAMIC AGENTS (ASYMMETRIC)
# ==========================================

def run_hmm_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    seed: int | None = None,
    fit_mode: bool = False,
    return_ll_only: bool = False,
) -> Dict[str, List[np.ndarray]] | float:
    
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
        pe_buffer = deque(maxlen=n_steps) # Signed PE Buffer
        posterior_prev = np.array([0.5, 0.5])
        agent_choices_pm1 = np.zeros(t_max)

        for t in range(t_max):
            prior = transition.T @ posterior_prev
            
            # 1. Update alpha_t
            if t == 0:
                alpha_t = alpha_base
            else:
                avg_pe = sum(pe_buffer) / len(pe_buffer)
                alpha_t = dynamic_alpha(alpha_base, gamma, avg_pe)

            # 2. MATCHING PHASE 2 RECURSION EXACTLY
            if t == 1:
                # Only c[0] available
                s1 = alpha_t + beta/2.0
                stickiness[t] = s1 * agent_choices_pm1[0]
            elif t > 1:
                # c[t-1] and c[t-2] available
                s1, s2 = alpha_t + beta/2.0, -alpha_t * decay
                stickiness[t] = decay * stickiness[t-1] + s1 * agent_choices_pm1[t-1] + s2 * agent_choices_pm1[t-2]

            # 3. Policy Calculation
            p_right = float(_sigmoid(_logit(prior[1]) + stickiness[t]))
            alpha_trace[t] = alpha_t
            policy[t, :] = [1.0 - p_right, p_right]

            # 4. Choice & Reward
            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            agent_choices_pm1[t] = _pm1(np.array([current_choice]))[0]

            # 5. Record PE and Update Posterior
            p_reward_state = np.array([p if current_choice == 0 else 1.0-p, 
                                       p if current_choice == 1 else 1.0-p])
            expected_reward_t = float(np.dot(prior, p_reward_state))
            pe_buffer.append(float(rewards[t] - expected_reward_t))
            
            likelihood = p_reward_state if rewards[t] > 0.5 else (1.0 - p_reward_state)
            posterior_prev = (prior * likelihood) / np.maximum((prior * likelihood).sum(), _EPS)
                        
        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}


# ==========================================
# HMM RESET: ASYMMETRIC TRIGGER + SOFT RESET TO 0.5
# ==========================================

def run_hmm_reset_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    fit_mode: bool = False,
    return_ll_only: bool = False,
    seed: int | None = None,
) -> Dict[str, List[np.ndarray]] | float:
    
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
        pe_buffer = deque(maxlen=n_steps) 
        posterior_prev = np.array([0.5, 0.5])
        agent_choices_pm1 = np.zeros(t_max)

        for t in range(t_max):
            prior = transition.T @ posterior_prev
            
            # 1. SURPRISE & RESET SIGNAL
            if t == 0:
                alpha_t, surprise_signal = alpha_base, 0.0
            else:
                avg_pe = sum(pe_buffer) / len(pe_buffer)
                alpha_t = dynamic_alpha(alpha_base, gamma, avg_pe)
                
                # ASYMMETRIC TRIGGER: Only Negative PE (omissions) causes confusion
                npe_only = min(0.0, avg_pe)
                surprise_signal = 1.0 - np.exp(gamma * npe_only)

            # 2. STICKINESS RECURSION (Phase 2 Match)
            if t == 1:
                stickiness[t] = (alpha_t + beta/2.0) * agent_choices_pm1[0]
            elif t > 1:
                s1, s2 = alpha_t + beta/2.0, -alpha_t * decay
                stickiness[t] = decay * stickiness[t-1] + s1 * agent_choices_pm1[t-1] + s2 * agent_choices_pm1[t-2]
            
            # 3. POLICY
            p_right = float(_sigmoid(_logit(prior[1]) + stickiness[t]))
            policy[t, :] = [1.0 - p_right, p_right]
            alpha_trace[t] = alpha_t

            # 4. CHOICE
            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            agent_choices_pm1[t] = _pm1(np.array([current_choice]))[0]

            # 5. BAYESIAN UPDATE + SOFT RESET TO 0.5/0.5
            p_reward_state = np.array([p if current_choice == 0 else 1.0-p, 
                                       1.0-p if current_choice == 0 else p])
            expected_reward_t = float(np.dot(prior, p_reward_state))
            
            likelihood = p_reward_state if rewards[t] > 0.5 else (1.0 - p_reward_state)
            posterior_raw = (prior * likelihood) / np.maximum((prior * likelihood).sum(), _EPS)
            
            # SOFT RESET: Mix raw belief with maximum uncertainty (0.5/0.5)
            posterior_prev = (1.0 - surprise_signal) * posterior_raw + surprise_signal * np.array([0.5, 0.5])

            # Record Signed Prediction Error
            pe_buffer.append(float(rewards[t] - expected_reward_t))

        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}


def run_hmm_resetv2_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Dict[str, float],
    n_steps: int,
    gamma: float,
    fit_mode: bool = False,
    return_ll_only: bool = False,
    seed: int | None = None,
) -> Dict[str, List[np.ndarray]] | float:
    
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
        pe_buffer = deque(maxlen=n_steps) 
        posterior_prev = np.array([0.5, 0.5])
        agent_choices_pm1 = np.zeros(t_max)

        for t in range(t_max):
            prior = transition.T @ posterior_prev
            
            # 1. INFORMATIONAL GATING
            if t == 0:
                alpha_t, surprise_signal = alpha_base, 0.0
            else:
                avg_pe = sum(pe_buffer) / len(pe_buffer)
                
                # HABIT ASYMMETRY: alpha_t reacts differently to + vs - errors
                alpha_t = dynamic_alpha(alpha_base, gamma, avg_pe)
                
                # COGNITIVE SYMMETRY: Reset signal reacts to the MAGNITUDE of surprise
                # surprise_signal = 1.0 - exp(-gamma * |avg_pe|)
                surprise_signal = 1.0 - np.exp(-gamma * abs(avg_pe))

            # 2. STICKINESS RECURSION
            if t == 1:
                stickiness[t] = (alpha_t + beta/2.0) * agent_choices_pm1[0]
            elif t > 1:
                s1, s2 = alpha_t + beta/2.0, -alpha_t * decay
                stickiness[t] = decay * stickiness[t-1] + s1 * agent_choices_pm1[t-1] + s2 * agent_choices_pm1[t-2]
            
            # 3. POLICY & CHOICE
            p_right = float(_sigmoid(_logit(prior[1]) + stickiness[t]))
            policy[t, :] = [1.0 - p_right, p_right]
            alpha_trace[t] = alpha_t

            current_choice = int(mouse_choices[t]) if fit_mode else (int(rng.random() < p_right))
            agent_choices_pm1[t] = _pm1(np.array([current_choice]))[0]

            # 4. BAYESIAN UPDATE
            p_reward_state = np.array([p if current_choice == 0 else 1.0-p, 
                                       1.0-p if current_choice == 0 else p])
            expected_reward_t = float(np.dot(prior, p_reward_state))
            
            likelihood = p_reward_state if rewards[t] > 0.5 else (1.0 - p_reward_state)
            posterior_raw = (prior * likelihood) / np.maximum((prior * likelihood).sum(), _EPS)
            
            # SOFT RESET: Mix raw belief with 0.5/0.5 based on absolute surprise
            posterior_prev = (1.0 - surprise_signal) * posterior_raw + surprise_signal * np.array([0.5, 0.5])

            # Record Signed Prediction Error for the buffer
            pe_buffer.append(float(rewards[t] - expected_reward_t))

        if return_ll_only:
            chosen_probs = policy[np.arange(t_max), mouse_choices]
            total_ll += float(np.sum(np.log(np.clip(chosen_probs, _EPS, 1.0))))
        else:
            policies.append(policy); alpha_traces.append(alpha_trace)

    return total_ll if return_ll_only else {"policies": policies, "alpha_traces": alpha_traces}


# ==========================================
# PHASE 4: F-Q DYNAMIC AGENT
# ==========================================

def run_fq_value_gated_only(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,  # Gamma remains in signature for compatibility but is unused
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


def run_fq_nonlinear_stickiness(
    behavior_features: Iterable[Session],
    parameters: Tuple[float, float, float],
    n_steps: int,
    gamma: float,
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
