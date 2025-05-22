"""
Random Walk Simulation

This script compares the Temporal Difference (TD(0)) and Monte Carlo (MC) methods
for estimating the value of states in a random walk problem. The environment consists
of 7 states, where the two terminal states (0 and 6) have values of 0 and 1 respectively,
and states A–E (1–5) have initial estimated values of 0.5.

The code generates:
- Value estimates over different episodes using TD(0)
- Root Mean Squared (RMS) errors across episodes for both TD(0) and MC methods
- Batch updating performance for both TD(0) and MC methods
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Disable GUI backend for plotting
import matplotlib.pyplot as plt
from tqdm import tqdm

# State values: 0 is the left terminal, 6 is the right terminal
# Initial non-terminal state values set to 0.5
VALUES = np.zeros(7)
VALUES[1:6] = 0.5
VALUES[6] = 1

# True state values for comparison
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1


def temporal_difference(values, alpha=0.1, batch=False):
    """
    Perform a TD(0) update on the value function from a single episode.

    Args:
        values (np.ndarray): Current value function estimates.
        alpha (float): Step size (learning rate).
        batch (bool): If True, do not update values online (used for batch learning).

    Returns:
        tuple: trajectory (list of visited states), rewards (list of rewards, all 0s).
    """
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        old_state = state
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        reward = 0
        trajectory.append(state)
        if not batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state in (0, 6):
            break
        rewards.append(reward)
    return trajectory, rewards


def monte_carlo(values, alpha=0.1, batch=False):
    """
    Perform a Monte Carlo update on the value function from a single episode.

    Args:
        values (np.ndarray): Current value function estimates.
        alpha (float): Step size (learning rate).
        batch (bool): If True, do not update values online (used for batch learning).

    Returns:
        tuple: trajectory (list of visited states), rewards (constant reward per step).
    """
    state = 3
    trajectory = [state]
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break

    if not batch:
        for state_ in trajectory[:-1]:
            values[state_] += alpha * (returns - values[state_])
    return trajectory, [returns] * (len(trajectory) - 1)


def compute_state_value():
    """
    Plot state value estimates over a fixed number of episodes using TD(0),
    compared with the true values.
    """
    episodes = [0, 1, 10, 100]
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(
                ("A", "B", "C", "D", "E"), current_values[1:6], label=f"{i} episodes"
            )
        temporal_difference(current_values)
    plt.plot(("A", "B", "C", "D", "E"), TRUE_VALUE[1:6], label="true values")
    plt.xlabel("State")
    plt.ylabel("Estimated Value")
    plt.legend()


def rms_error():
    """
    Compute and plot the RMS error between estimated and true values over episodes
    for both TD(0) and MC methods at various step sizes.
    """
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 101
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        method = "TD" if i < len(td_alphas) else "MC"
        linestyle = "solid" if method == "TD" else "dashdot"
        for _ in tqdm(range(runs)):
            current_values = np.copy(VALUES)
            errors = []
            for ep in range(episodes):
                errors.append(
                    np.sqrt(np.sum(np.square(TRUE_VALUE - current_values)) / 5.0)
                )
                if method == "TD":
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(
            total_errors,
            linestyle=linestyle,
            label=f"{method}, $\\alpha$ = {alpha:.2f}",
        )
    plt.xlabel("Walks/Episodes")
    plt.ylabel("Empirical RMS error, averaged over states")
    plt.legend()


def batch_updating(method, episodes, alpha=0.001):
    """
    Apply batch updating for a given method and return the RMS error per episode.

    Args:
        method (str): 'TD' or 'MC'
        episodes (int): Number of episodes
        alpha (float): Learning rate

    Returns:
        np.ndarray: RMS errors for each episode averaged over runs.
    """
    runs = 100
    total_errors = np.zeros(episodes)
    for _ in tqdm(range(runs)):
        current_values = np.copy(VALUES)
        current_values[1:6] = -1
        trajectories, rewards, errors = [], [], []
        for ep in range(episodes):
            if method == "TD":
                traj, rews = temporal_difference(current_values, batch=True)
            else:
                traj, rews = monte_carlo(current_values, batch=True)
            trajectories.append(traj)
            rewards.append(rews)

            while True:
                updates = np.zeros(7)
                for traj, rews in zip(trajectories, rewards):
                    for i in range(len(traj) - 1):
                        s = traj[i]
                        if method == "TD":
                            updates[s] += (
                                rews[i]
                                + current_values[traj[i + 1]]
                                - current_values[s]
                            )
                        else:
                            updates[s] += rews[i] - current_values[s]
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                current_values += updates
            errors.append(np.sqrt(np.sum(np.square(current_values - TRUE_VALUE)) / 5.0))
        total_errors += np.asarray(errors)
    return total_errors / runs


def example_6_2():
    """
    Reproduce both subplots of Example 6.2 showing:
    - TD value estimates over time
    - RMS error curves for TD and MC
    """
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value()

    plt.subplot(2, 1, 2)
    rms_error()
    plt.tight_layout()
    plt.savefig("../images/example_6_2.png")
    plt.close()


def figure_6_2():
    """
    Generate Figure 6.2 showing RMS error over episodes using batch updates.
    """
    episodes = 101
    td_errors = batch_updating("TD", episodes)
    mc_errors = batch_updating("MC", episodes)

    plt.plot(td_errors, label="TD")
    plt.plot(mc_errors, label="MC")
    plt.title("Batch Training")
    plt.xlabel("Walks/Episodes")
    plt.ylabel("RMS error, averaged over states")
    plt.xlim(0, 100)
    plt.ylim(0, 0.25)
    plt.legend()
    plt.savefig("../images/figure_6_2.png")
    plt.close()


if __name__ == "__main__":
    example_6_2()
    figure_6_2()
