"""
Overestimation in Q-Learning vs. Double Q-Learning

This script compares standard Q-learning with Double Q-learning in a simple MDP setup involving
two non-terminal states (A and B) and one terminal state. The agent starts in state A and decides
between two actions. One action leads directly to the terminal state, while the other leads to
state B, which has 10 possible actions, each yielding stochastic rewards.

The focus is on illustrating overestimation bias in standard Q-learning and how Double Q-learning
mitigates it. The figure shows the frequency with which the suboptimal "left" action is taken
from state A over 300 episodes, averaged across 1000 runs.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# --- MDP Setup ---

# States
STATE_A = 0
STATE_B = 1
STATE_TERMINAL = 2
STATE_START = STATE_A

# Actions for state A
ACTION_A_RIGHT = 0
ACTION_A_LEFT = 1

# Action set for each state
ACTIONS_B = range(10)
STATE_ACTIONS = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTIONS_B]

# Transition table: TRANSITION[state][action] = next_state
TRANSITION = [
    [STATE_TERMINAL, STATE_B],  # From state A
    [STATE_TERMINAL] * len(ACTIONS_B),  # From state B
]

# Q-value initialization for all states (zero-initialized)
INITIAL_Q = [np.zeros(2), np.zeros(len(ACTIONS_B)), np.zeros(1)]

# Hyperparameters
EPSILON = 0.1  # ε-greedy exploration
ALPHA = 0.1  # learning rate
GAMMA = 1.0  # discount factor


def choose_action(state, q_value):
    """
    Epsilon-greedy action selection.

    Parameters:
        state (int): Current state.
        q_value (list of np.array): Q-value tables for each state.

    Returns:
        int: Selected action.
    """
    if np.random.rand() < EPSILON:
        return np.random.choice(STATE_ACTIONS[state])
    values_ = q_value[state]
    max_value = np.max(values_)
    return np.random.choice([a for a, v in enumerate(values_) if v == max_value])


def take_action(state, action):
    """
    Return reward from taking action in given state.

    Parameters:
        state (int): Current state.
        action (int): Selected action.

    Returns:
        float: Reward received.
    """
    if state == STATE_A:
        return 0  # deterministic transition, no reward
    return np.random.normal(-0.1, 1)  # stochastic reward from state B


def q_learning(q1, q2=None):
    """
    Perform one episode of (Double) Q-learning.

    Parameters:
        q1 (list of np.array): Q-value tables for Q-learning or first Q-table in Double Q-learning.
        q2 (list of np.array, optional): Second Q-table if using Double Q-learning.

    Returns:
        int: 1 if the agent chose the left action from state A during the episode, 0 otherwise.
    """
    state = STATE_START
    left_count = 0

    while state != STATE_TERMINAL:
        if q2 is None:
            action = choose_action(state, q1)
        else:
            # For Double Q-learning: use combined estimates to select action
            action = choose_action(state, [q1_s + q2_s for q1_s, q2_s in zip(q1, q2)])

        if state == STATE_A and action == ACTION_A_LEFT:
            left_count += 1

        reward = take_action(state, action)
        next_state = TRANSITION[state][action]

        if q2 is None:
            # Standard Q-learning update
            best_next_action_value = np.max(q1[next_state])
            q1[state][action] += ALPHA * (
                reward + GAMMA * best_next_action_value - q1[state][action]
            )
        else:
            # Double Q-learning update
            if np.random.rand() < 0.5:
                best_action = np.argmax(q1[next_state])
                q1[state][action] += ALPHA * (
                    reward + GAMMA * q2[next_state][best_action] - q1[state][action]
                )
            else:
                best_action = np.argmax(q2[next_state])
                q2[state][action] += ALPHA * (
                    reward + GAMMA * q1[next_state][best_action] - q2[state][action]
                )

        state = next_state

    return left_count


def figure_6_7():
    """
    Generate and save Figure 6.7 comparing standard Q-learning with Double Q-learning.
    """
    episodes = 300
    runs = 1000

    left_counts_q = np.zeros((runs, episodes))
    left_counts_double_q = np.zeros((runs, episodes))

    for run in tqdm(range(runs)):
        q = copy.deepcopy(INITIAL_Q)
        q1 = copy.deepcopy(INITIAL_Q)
        q2 = copy.deepcopy(INITIAL_Q)
        for ep in range(episodes):
            left_counts_q[run, ep] = q_learning(q)
            left_counts_double_q[run, ep] = q_learning(q1, q2)

    # Average over runs
    avg_left_q = left_counts_q.mean(axis=0)
    avg_left_double_q = left_counts_double_q.mean(axis=0)

    plt.plot(avg_left_q, label="Q-Learning")
    plt.plot(avg_left_double_q, label="Double Q-Learning")
    plt.axhline(y=0.05, linestyle="--", color="black", label="Optimal")

    plt.xlabel("Episodes")
    plt.ylabel("% Left Actions from A")
    plt.title("Figure 6.7: Overestimation in Q-Learning vs. Double Q-Learning")
    plt.legend()

    plt.savefig("../images/figure_6_7.png")
    plt.close()


if __name__ == "__main__":
    figure_6_7()
