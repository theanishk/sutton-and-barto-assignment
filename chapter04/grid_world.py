"""
Iterative Policy Evaluation in a Gridworld (Figure 4.1 - Sutton & Barto)

This script evaluates the state values of a 4x4 gridworld using iterative policy evaluation.
The environment is a deterministic grid where each move yields a -1 reward until a terminal state is reached.
Two approaches are compared:
- In-place (asynchronous) update
- Out-of-place (synchronous) update

Features:
- Implements Bellman expectation backup for uniform random policy
- Uses both in-place and synchronous value updates
- Visualizes the final state value function
- Saves the figure as '../images/figure_4_1.png'
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use("Agg")

# Gridworld size (4x4)
WORLD_SIZE = 4

# Action space: [left, up, right, down]
ACTIONS = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]

# Equal probability for each action under random policy
ACTION_PROB = 0.25


def is_terminal(state):
    """
    Check if the state is terminal.

    Terminal states are at the top-left (0,0) and bottom-right (3,3) corners.
    """
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state, action):
    """
    Take an action in the environment.

    Parameters:
        state (list): Current state as [row, col]
        action (np.array): Action to take

    Returns:
        next_state (list): Resulting state after action
        reward (int): Reward received (-1 per move)
    """
    if is_terminal(state):
        return state, 0  # No reward or transition from terminal

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    # If move takes agent off the grid, stay in the same state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    return next_state, -1  # Reward is -1 for all actions


def draw_image(image):
    """
    Render the state value function using a matplotlib table.

    Parameters:
        image (np.array): 2D array of state values
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells to the table
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")

    # Add row and column indices
    for i in range(len(image)):
        tb.add_cell(
            i,
            -1,
            width,
            height,
            text=i + 1,
            loc="right",
            edgecolor="none",
            facecolor="none",
        )
        tb.add_cell(
            -1,
            i,
            width,
            height / 2,
            text=i + 1,
            loc="center",
            edgecolor="none",
            facecolor="none",
        )

    ax.add_table(tb)


def compute_state_value(in_place=True, discount=1.0):
    """
    Perform iterative policy evaluation.

    Parameters:
        in_place (bool): If True, updates values in-place (asynchronous); otherwise, uses a copy (synchronous)
        discount (float): Discount factor for future rewards

    Returns:
        new_state_values (np.array): Evaluated state values
        iteration (int): Number of iterations to convergence
    """
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0

    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()

        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (
                        reward + discount * state_values[next_i, next_j]
                    )
                new_state_values[i, j] = value

        # Check for convergence
        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


def figure_4_1():
    """
    Generate and save Figure 4.1:
    Compares value iteration results between in-place and synchronous updates.
    """
    # In-place and synchronous evaluations
    _, async_iterations = compute_state_value(in_place=True)
    values, sync_iterations = compute_state_value(in_place=False)

    draw_image(np.round(values, decimals=2))

    print(f"In-place: {async_iterations} iterations")
    print(f"Synchronous: {sync_iterations} iterations")

    plt.savefig("../images/figure_4_1.png")
    plt.close()


if __name__ == "__main__":
    figure_4_1()
