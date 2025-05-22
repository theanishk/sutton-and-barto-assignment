"""
Cliff Walking Environment and Evaluation of TD Control Methods

This module implements the gridworld from Sutton and Barto's Reinforcement Learning book (Figure 6.4, 6.6).
It includes Q-Learning, Sarsa, and Expected Sarsa agents interacting with the environment,
and compares their performance through learning curves and optimal policy extraction.

The cliff is located between (3,1) to (3,10). Stepping into it gives a large penalty and resets the agent to the start.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from tqdm import tqdm

# Grid dimensions
WORLD_HEIGHT = 4
WORLD_WIDTH = 12

# Start and goal positions
START = [3, 0]
GOAL = [3, 11]

# Hyperparameters
EPSILON = 0.1  # Exploration rate
ALPHA = 0.5  # Step size
GAMMA = 1.0  # Discount factor

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


def step(state, action):
    """
    Take a step in the environment.

    Args:
        state (list): Current position as [row, col].
        action (int): Action to take (0-3).

    Returns:
        next_state (list): Resulting state after action.
        reward (float): Reward received for the transition.
    """
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    else:
        raise ValueError("Invalid action")

    reward = -1
    # Cliff condition
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
        action == ACTION_RIGHT and state == START
    ):
        reward = -100
        next_state = START

    return next_state, reward


def choose_action(state, q_value):
    """
    Choose an action using epsilon-greedy strategy.

    Args:
        state (list): Current state.
        q_value (ndarray): Q-table.

    Returns:
        int: Selected action.
    """
    if np.random.rand() < EPSILON:
        return np.random.choice(ACTIONS)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice(np.flatnonzero(values == np.max(values)))


def sarsa(q_value, expected=False, step_size=ALPHA):
    """
    Perform one episode using Sarsa or Expected Sarsa.

    Args:
        q_value (ndarray): Q-table to be updated.
        expected (bool): Use Expected Sarsa if True.
        step_size (float): Learning rate.

    Returns:
        float: Cumulative reward of the episode.
    """
    state = START
    action = choose_action(state, q_value)
    total_reward = 0.0

    while state != GOAL:
        next_state, reward = step(state, action)
        total_reward += reward
        next_action = choose_action(next_state, q_value)

        if expected:
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.flatnonzero(q_next == np.max(q_next))
            expected_value = 0.0
            for a in ACTIONS:
                prob = (
                    (1 - EPSILON) / len(best_actions) if a in best_actions else 0
                ) + EPSILON / len(ACTIONS)
                expected_value += prob * q_next[a]
            target = reward + GAMMA * expected_value
        else:
            target = reward + GAMMA * q_value[next_state[0], next_state[1], next_action]

        q_value[state[0], state[1], action] += step_size * (
            target - q_value[state[0], state[1], action]
        )
        state, action = next_state, next_action

    return total_reward


def q_learning(q_value, step_size=ALPHA):
    """
    Perform one episode using Q-Learning.

    Args:
        q_value (ndarray): Q-table to be updated.
        step_size (float): Learning rate.

    Returns:
        float: Cumulative reward of the episode.
    """
    state = START
    total_reward = 0.0

    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        total_reward += reward
        best_q = np.max(q_value[next_state[0], next_state[1], :])
        q_value[state[0], state[1], action] += step_size * (
            reward + GAMMA * best_q - q_value[state[0], state[1], action]
        )
        state = next_state

    return total_reward


def print_optimal_policy(q_value):
    """
    Print optimal policy using Q-table.
    """
    symbols = {ACTION_UP: "U", ACTION_DOWN: "D", ACTION_LEFT: "L", ACTION_RIGHT: "R"}
    for i in range(WORLD_HEIGHT):
        row = []
        for j in range(WORLD_WIDTH):
            if [i, j] == GOAL:
                row.append("G")
            else:
                best_action = np.argmax(q_value[i, j, :])
                row.append(symbols[best_action])
        print(row)


def figure_6_4():
    """
    Generate and save learning curves comparing Q-Learning and Sarsa (Figure 6.4).
    """
    episodes = 500
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q = np.zeros(episodes)

    for _ in tqdm(range(runs)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q = np.copy(q_sarsa)
        for ep in range(episodes):
            rewards_sarsa[ep] += sarsa(q_sarsa)
            rewards_q[ep] += q_learning(q_q)

    rewards_sarsa /= runs
    rewards_q /= runs

    plt.plot(rewards_sarsa, label="Sarsa")
    plt.plot(rewards_q, label="Q-Learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.ylim([-100, 0])
    plt.legend()
    plt.savefig("../images/figure_6_4.png")
    plt.close()

    print("Sarsa Optimal Policy:")
    print_optimal_policy(q_sarsa)
    print("Q-Learning Optimal Policy:")
    print_optimal_policy(q_q)


def figure_6_6():
    """
    Evaluate performance of Sarsa, Expected Sarsa, and Q-Learning with various step sizes (Figure 6.6).
    """
    step_sizes = np.arange(0.1, 1.1, 0.1)
    episodes = 1000
    runs = 10

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_Q = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_Q = 5

    performance = np.zeros((6, len(step_sizes)))

    for run in range(runs):
        for idx, alpha in tqdm(list(enumerate(step_sizes))):
            q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
            q_exp_sarsa = np.copy(q_sarsa)
            q_q = np.copy(q_sarsa)
            for ep in range(episodes):
                r_sarsa = sarsa(q_sarsa, expected=False, step_size=alpha)
                r_exp_sarsa = sarsa(q_exp_sarsa, expected=True, step_size=alpha)
                r_q = q_learning(q_q, step_size=alpha)

                performance[ASY_SARSA, idx] += r_sarsa
                performance[ASY_EXPECTED_SARSA, idx] += r_exp_sarsa
                performance[ASY_Q, idx] += r_q

                if ep < 100:
                    performance[INT_SARSA, idx] += r_sarsa
                    performance[INT_EXPECTED_SARSA, idx] += r_exp_sarsa
                    performance[INT_Q, idx] += r_q

    performance[:3, :] /= episodes * runs
    performance[3:, :] /= 100 * runs

    labels = [
        "Asymptotic Sarsa",
        "Asymptotic Expected Sarsa",
        "Asymptotic Q-Learning",
        "Interim Sarsa",
        "Interim Expected Sarsa",
        "Interim Q-Learning",
    ]

    for method, label in zip(range(6), labels):
        plt.plot(step_sizes, performance[method], label=label)

    plt.xlabel("Alpha (Step Size)")
    plt.ylabel("Average Reward per Episode")
    plt.legend()
    plt.savefig("../images/figure_6_6.png")
    plt.close()


if __name__ == "__main__":
    figure_6_4()
    figure_6_6()
