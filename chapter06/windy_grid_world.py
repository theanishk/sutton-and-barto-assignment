"""
Windy Gridworld Example using SARSA

This script implements the Windy Gridworld example using the SARSA algorithm,
as described in Sutton & Barto's *Reinforcement Learning: An Introduction* (Figure 6.3).

The environment consists of a 7x10 grid where each column has an associated wind strength
that pushes the agent upward. The agent's objective is to reach a fixed goal position
from a fixed start position while minimizing the number of steps.

Key Features:
- Epsilon-greedy policy for exploration
- SARSA (on-policy TD control) for learning
- Visualization of cumulative time steps vs episodes
- Display of the learned optimal policy
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Environment dimensions
WORLD_HEIGHT = 7
WORLD_WIDTH = 10

# Wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# Action definitions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# Sarsa hyperparameters
EPSILON = 0.1  # Exploration rate
ALPHA = 0.5  # Learning rate
REWARD = -1.0  # Reward for each step

# Start and goal positions
START = [3, 0]
GOAL = [3, 7]


def step(state, action):
    """
    Takes a step in the environment given a state and action,
    and returns the resulting next state accounting for wind.

    Args:
        state (list): Current position [row, column].
        action (int): Action to take (0: up, 1: down, 2: left, 3: right).

    Returns:
        list: Next state after applying the action and wind effect.
    """
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        raise ValueError("Invalid action.")


def episode(q_value):
    """
    Runs a single episode using the Sarsa algorithm.

    Args:
        q_value (np.ndarray): Q-table of shape (WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)).

    Returns:
        int: Number of time steps taken to reach the goal.
    """
    time = 0
    state = START

    # Choose initial action using epsilon-greedy policy
    if np.random.rand() < EPSILON:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice(
            [a for a, v in enumerate(values_) if v == np.max(values_)]
        )

    while state != GOAL:
        next_state = step(state, action)

        if np.random.rand() < EPSILON:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [a for a, v in enumerate(values_) if v == np.max(values_)]
            )

        # Sarsa update rule
        q_value[state[0], state[1], action] += ALPHA * (
            REWARD
            + q_value[next_state[0], next_state[1], next_action]
            - q_value[state[0], state[1], action]
        )

        state = next_state
        action = next_action
        time += 1

    return time


def figure_6_3():
    """
    Reproduces Figure 6.3 from Sutton & Barto’s book.
    Trains the agent using Sarsa in the windy gridworld environment,
    and plots cumulative time steps per episode.

    Also prints the optimal policy and wind strengths.
    """
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        ep += 1

    steps = np.add.accumulate(steps)

    # Plotting cumulative time steps per episode
    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.title("Windy Gridworld: SARSA Performance")
    plt.savefig("../images/figure_6_3.png")
    plt.close()

    # Print the optimal policy
    optimal_policy = []
    for i in range(WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append("G")
                continue
            best_action = np.argmax(q_value[i, j, :])
            if best_action == ACTION_UP:
                optimal_policy[-1].append("U")
            elif best_action == ACTION_DOWN:
                optimal_policy[-1].append("D")
            elif best_action == ACTION_LEFT:
                optimal_policy[-1].append("L")
            elif best_action == ACTION_RIGHT:
                optimal_policy[-1].append("R")

    print("Optimal policy is:")
    for row in optimal_policy:
        print(row)
    print("Wind strength for each column:\n{}".format(WIND))


if __name__ == "__main__":
    figure_6_3()
