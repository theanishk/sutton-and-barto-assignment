"""
Value Iteration for Gambler's Problem

This script solves the Gambler’s Problem using value iteration.
In this RL example, a gambler bets money on a coin toss and aims to reach a goal of 100 units of capital.
Each bet wins with a fixed probability (e.g., 0.4). The gambler can stake any amount from 1 up to the minimum
of their current capital and the difference between the current capital and the goal.

Features:
- Performs value iteration until value estimates converge.
- Tracks value function changes across sweeps.
- Derives the optimal policy (betting strategy) based on final value function.
- Plots the value function evolution and final policy.
- Saves the result as `../images/figure_4_3.png`.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Final goal for capital
GOAL = 100

# Set of all states: capital from 0 to GOAL
STATES = np.arange(GOAL + 1)

# Probability of getting heads
HEAD_PROB = 0.4


def figure_4_3():
    """
    Perform value iteration to solve the Gambler’s Problem and generate a plot.

    Computes value estimates over multiple sweeps and derives an optimal policy.
    Then, saves the visualizations for value updates and final betting policy.
    """
    # Initialize state values: goal state has reward of 1
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    # History of value function per sweep
    sweeps_history = []

    # ---------- Value Iteration Loop ----------
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:  # Skip terminal states
            # Possible actions: all possible stakes
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                # Expected value from this action
                reward = 0
                win_state = state + action
                lose_state = state - action
                expected_return = (
                    HEAD_PROB * state_value[win_state]
                    + (1 - HEAD_PROB) * state_value[lose_state]
                )
                action_returns.append(expected_return)

            # Optimal value function update
            state_value[state] = np.max(action_returns)

        # Check for convergence
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value.copy())
            break

    # ---------- Derive Optimal Policy ----------
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            expected_return = (
                HEAD_PROB * state_value[state + action]
                + (1 - HEAD_PROB) * state_value[state - action]
            )
            action_returns.append(expected_return)

        # Round values to reduce numerical instability (matches book)
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    # ---------- Plotting ----------
    plt.figure(figsize=(10, 20))

    # Value function over sweeps
    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label=f"sweep {sweep}")
    plt.xlabel("Capital")
    plt.ylabel("Value Estimates")
    plt.title("Value Function Over Iterations")
    plt.legend(loc="best")

    # Final policy
    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel("Capital")
    plt.ylabel("Final Policy (Stake)")
    plt.title("Derived Optimal Policy")

    plt.savefig("../images/figure_4_3.png")
    plt.close()


if __name__ == "__main__":
    figure_4_3()
