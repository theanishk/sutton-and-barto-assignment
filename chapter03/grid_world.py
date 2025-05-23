"""
Gridworld Value Iteration and Policy Visualization

This script implements the classical Gridworld example from Reinforcement Learning.
It includes:
- Value iteration using Bellman expectation equations (Figure 3.2)
- Solving the system as a linear system (exact solution)
- Value iteration for optimal values and policies (Figure 3.5)

Features:
- Special teleporting states A and B with associated rewards.
- Uniform random policy or greedy policy over the optimal value function.
- Visualization using matplotlib tables with cell annotations.

Saves:
- Value functions and policies as images in the `../images/` directory.
"""

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# Environment constants
WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# Action space: left, up, right, down
ACTIONS = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]
ACTIONS_FIGS = ["←", "↑", "→", "↓"]
ACTION_PROB = 0.25


def step(state, action):
    """
    Take a step from the current state using the given action.

    Special transitions:
    - From A → A' with reward 10
    - From B → B' with reward 5

    Normal transitions incur a reward of 0 or -1 (if hitting the wall).

    Args:
        state (list): [row, col] coordinates
        action (np.array): direction vector

    Returns:
        (next_state, reward)
    """
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        return state, -1.0
    return next_state, 0


def draw_image(image):
    """
    Render a matrix as a table image with Gridworld state annotations.

    Args:
        image (np.array): 2D array of values to render
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i, j), val in np.ndenumerate(image):
        label = str(val)
        if [i, j] == A_POS:
            label += " (A)"
        if [i, j] == A_PRIME_POS:
            label += " (A')"
        if [i, j] == B_POS:
            label += " (B)"
        if [i, j] == B_PRIME_POS:
            label += " (B')"
        tb.add_cell(i, j, width, height, text=label, loc="center", facecolor="white")

    for i in range(nrows):
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


def draw_policy(optimal_values):
    """
    Draw the greedy policy over a value function.

    Args:
        optimal_values (np.array): 2D array of state values
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i, j), _ in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])
        best_actions = np.where(next_vals == np.max(next_vals))[0]
        arrows = "".join([ACTIONS_FIGS[ba] for ba in best_actions])

        if [i, j] == A_POS:
            arrows += " (A)"
        if [i, j] == A_PRIME_POS:
            arrows += " (A')"
        if [i, j] == B_POS:
            arrows += " (B)"
        if [i, j] == B_PRIME_POS:
            arrows += " (B')"

        tb.add_cell(i, j, width, height, text=arrows, loc="center", facecolor="white")

    for i in range(nrows):
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


def figure_3_2():
    """
    Iteratively evaluate the value function under a uniform random policy (Figure 3.2).
    Saves: ../images/figure_3_2.png
    """
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    new_value[i, j] += ACTION_PROB * (
                        reward + DISCOUNT * value[next_i, next_j]
                    )
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig("../images/figure_3_2.png")
            plt.close()
            break
        value = new_value


def figure_3_2_linear_system():
    """
    Solve the value function exactly by setting up the system of linear Bellman equations.
    Saves: ../images/figure_3_2_linear_system.png
    """
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))
                A[index_s, index_s_] += ACTION_PROB * DISCOUNT
                b[index_s] -= ACTION_PROB * r

    x = np.linalg.solve(A, b)
    draw_image(np.round(x.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    plt.savefig("../images/figure_3_2_linear_system.png")
    plt.close()


def figure_3_5():
    """
    Perform value iteration to compute the optimal value function and greedy policy (Figure 3.5).
    Saves:
    - ../images/figure_3_5.png (value function)
    - ../images/figure_3_5_policy.png (policy)
    """
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig("../images/figure_3_5.png")
            plt.close()

            draw_policy(new_value)
            plt.savefig("../images/figure_3_5_policy.png")
            plt.close()
            break
        value = new_value


if __name__ == "__main__":
    figure_3_2_linear_system()
    figure_3_2()
    figure_3_5()
