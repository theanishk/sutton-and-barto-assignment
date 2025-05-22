"""
TD(n)-step Temporal Difference Learning Simulation for Random Walk

This script simulates the n-step temporal difference learning algorithm
for a random walk problem. The random walk is represented as a
one-dimensional state space with non-terminal states in between
two terminal states. The script implements the TD(n) algorithm to
update the value estimates of the states based on the rewards received
during the walk. The algorithm is evaluated for different values of
n (number of steps) and α (step-size parameter). The performance
is measured by the root mean square (RMS) error between the
estimated values and the true values of the states. The true values
are computed using the Bellman equation. The script runs multiple
episodes of the random walk and averages the RMS errors over
multiple runs. The results are plotted to visualize the effect
of different (n, α) combinations on the learning performance.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for saving plots to file
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
N_STATES = 19  # Number of non-terminal states
GAMMA = 1.0  # Discount factor
START_STATE = 10  # Starting state index
END_STATES = [0, N_STATES + 1]  # Terminal states

# Bellman-computed true state values
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0


def temporal_difference(value: np.ndarray, n: int, alpha: float):
    """
    Run one episode of n-step TD learning to update state values.

    Args:
        value (np.ndarray): Array of current value estimates.
        n (int): Number of steps for TD(n).
        alpha (float): Step-size parameter.
    """
    state = START_STATE
    states = [state]
    rewards = [0]

    T = float("inf")  # Episode end time
    time = 0

    while True:
        time += 1

        if time < T:
            # Take a random action: left (-1) or right (+1)
            next_state = state + 1 if np.random.rand() < 0.5 else state - 1

            # Assign reward
            if next_state == END_STATES[0]:
                reward = -1
            elif next_state == END_STATES[1]:
                reward = 1
            else:
                reward = 0

            states.append(next_state)
            rewards.append(reward)

            if next_state in END_STATES:
                T = time

        # Time of state to update
        update_time = time - n
        if update_time >= 0:
            # Compute n-step return
            G = sum(
                [
                    GAMMA ** (t - update_time - 1) * rewards[t]
                    for t in range(update_time + 1, min(T, update_time + n) + 1)
                ]
            )
            if update_time + n <= T:
                G += GAMMA**n * value[states[update_time + n]]

            state_to_update = states[update_time]
            if state_to_update not in END_STATES:
                value[state_to_update] += alpha * (G - value[state_to_update])

        if update_time == T - 1:
            break

        state = next_state


def figure_7_2():
    """
    Generate Figure 7.2 by evaluating the performance of n-step TD
    across various step sizes and α values.

    Saves the figure as '../images/figure_7_2.png'.
    """
    steps = np.power(2, np.arange(10))  # n = 1, 2, 4, ..., 512
    alphas = np.arange(0, 1.1, 0.1)  # α from 0.0 to 1.0
    episodes = 10
    runs = 100

    errors = np.zeros((len(steps), len(alphas)))  # RMS error accumulator

    for run in tqdm(range(runs), desc="Running experiments"):
        for step_idx, n in enumerate(steps):
            for alpha_idx, alpha in enumerate(alphas):
                value = np.zeros(N_STATES + 2)
                for _ in range(episodes):
                    temporal_difference(value, n, alpha)
                    rms_error = np.sqrt(np.sum((value - TRUE_VALUE) ** 2) / N_STATES)
                    errors[step_idx, alpha_idx] += rms_error

    errors /= runs * episodes  # Average over all runs and episodes

    # Plotting
    for step_idx, n in enumerate(steps):
        plt.plot(alphas, errors[step_idx, :], label=f"n = {n}")
    plt.xlabel("Alpha (α)")
    plt.ylabel("RMS Error")
    plt.title("Figure 7.2: n-step TD Prediction Performance")
    plt.ylim([0.25, 0.55])
    plt.legend()

    plt.savefig("../images/figure_7_2.png")
    plt.close()


if __name__ == "__main__":
    figure_7_2()
