"""
This script simulates the sampling process to illustrate how sample averages
converge to the true value with different branching factors `b`, as shown in
Figure 8.7 of Sutton & Barto's *Reinforcement Learning: An Introduction*.

The experiment:
- Creates `b` hypothetical next-state values drawn from a normal distribution.
- The true value of the current state is the mean of these values.
- At each time step, one of these values is randomly sampled.
- The sample average is updated and its RMS error from the true value is recorded.
- This process is repeated for 2b steps over multiple runs.

The result is a plot of RMS error versus the number of computations (normalized by `b`),
demonstrating that a larger `b` leads to slower convergence initially, but ultimately
more accurate estimates.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from tqdm import tqdm


def b_steps(b):
    """
    Simulate 2b samples to estimate the value of a state with branching factor `b`.

    Args:
        b (int): The branching factor (i.e., number of possible next states).

    Returns:
        list: RMS error at each time step as the sample average converges.
    """
    # Simulate the values of the b possible next states
    distribution = np.random.randn(b)

    # True value of the current state is the mean of next-state values
    true_v = np.mean(distribution)

    samples = []
    errors = []

    # Take 2b samples, update average, and compute error at each step
    for t in range(2 * b):
        v = np.random.choice(distribution)
        samples.append(v)
        estimated_v = np.mean(samples)
        errors.append(np.abs(estimated_v - true_v))

    return errors


def figure_8_7():
    """
    Generate Figure 8.7: RMS error in estimating state value vs. number of computations.

    This function compares the convergence rate of sample averages for various branching
    factors `b` (e.g., 2, 10, 100, 1000). The RMS error is averaged over multiple runs.
    """
    runs = 100
    branch_factors = [2, 10, 100, 1000]

    for b in branch_factors:
        errors = np.zeros((runs, 2 * b))
        for r in tqdm(range(runs), desc=f"Branch factor b = {b}"):
            errors[r] = b_steps(b)

        # Average error over all runs
        avg_errors = errors.mean(axis=0)

        # Normalize x-axis by branching factor
        x_axis = (np.arange(1, 2 * b + 1)) / float(b)

        plt.plot(x_axis, avg_errors, label=f"b = {b}")

    plt.xlabel("Number of computations (steps normalized by b)")
    plt.xticks([0, 1.0, 2.0], ["0", "b", "2b"])
    plt.ylabel("RMS error")
    plt.title("Figure 8.7: Error vs Computation for Different b Values")
    plt.legend()

    # Save figure
    plt.savefig("../images/figure_8_7.png")
    plt.close()


if __name__ == "__main__":
    figure_8_7()
