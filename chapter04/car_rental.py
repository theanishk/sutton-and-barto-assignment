"""
Policy Iteration for Jack's Car Rental Problem

This script implements policy iteration to solve Jack's Car Rental Problem,
a classical reinforcement learning example from the book "Reinforcement Learning: An Introduction"
by Sutton and Barto. It models two car rental locations where cars can be moved overnight to maximize
expected rental income while considering probabilistic rental and return demands.

Main Features:
- Policy evaluation with expected returns based on Poisson-distributed rental and return requests.
- Policy improvement using greedy selection based on current value estimates.
- Optional simplification of returns to constant values for computational efficiency.
- Heatmap visualization of policies and value functions over policy iteration steps.
- Saves the final plot of optimal policy and value to `../images/figure_4_2.png`.

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use("Agg")

# Maximum number of cars in each location
MAX_CARS = 20

# Maximum number of cars that can be moved overnight
MAX_MOVE_OF_CARS = 5

# Poisson means for rental requests and returns at both locations
RENTAL_REQUEST_FIRST_LOC = 3
RENTAL_REQUEST_SECOND_LOC = 4
RETURNS_FIRST_LOC = 3
RETURNS_SECOND_LOC = 2

# Discount factor and cost/reward settings
DISCOUNT = 0.9
RENTAL_CREDIT = 10
MOVE_CAR_COST = 2

# Action space
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# Truncation threshold for Poisson probability calculation
POISSON_UPPER_BOUND = 11

# Poisson probability cache
poisson_cache = dict()


def poisson_probability(n, lam):
    """
    Memoized computation of the Poisson probability mass function.

    Args:
        n (int): Number of occurrences.
        lam (int): Lambda parameter of Poisson distribution.

    Returns:
        float: Probability of observing `n` given Poisson(λ).
    """
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


def expected_return(state, action, state_value, constant_returned_cars):
    """
    Compute expected return for a given state and action using the Bellman expectation equation.

    Args:
        state (list): Current state as [cars at location 1, cars at location 2].
        action (int): Number of cars moved overnight (positive: 1 → 2, negative: 2 → 1).
        state_value (np.ndarray): Current value function matrix.
        constant_returned_cars (bool): If True, use fixed returns instead of sampling from Poisson.

    Returns:
        float: Expected return value.
    """
    returns = 0.0
    returns -= MOVE_CAR_COST * abs(action)

    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            prob = poisson_probability(
                rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC
            ) * poisson_probability(
                rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC
            )

            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC

            valid_rental_first_loc = min(
                num_of_cars_first_loc, rental_request_first_loc
            )
            valid_rental_second_loc = min(
                num_of_cars_second_loc, rental_request_second_loc
            )

            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc

            if constant_returned_cars:
                num_of_cars_first_loc = min(
                    num_of_cars_first_loc + RETURNS_FIRST_LOC, MAX_CARS
                )
                num_of_cars_second_loc = min(
                    num_of_cars_second_loc + RETURNS_SECOND_LOC, MAX_CARS
                )
                returns += prob * (
                    reward
                    + DISCOUNT
                    * state_value[num_of_cars_first_loc, num_of_cars_second_loc]
                )
            else:
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(
                            returned_cars_first_loc, RETURNS_FIRST_LOC
                        ) * poisson_probability(
                            returned_cars_second_loc, RETURNS_SECOND_LOC
                        )
                        prob_ = prob * prob_return

                        num_of_cars_first_loc_ = min(
                            num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS
                        )
                        num_of_cars_second_loc_ = min(
                            num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS
                        )
                        returns += prob_ * (
                            reward
                            + DISCOUNT
                            * state_value[
                                num_of_cars_first_loc_, num_of_cars_second_loc_
                            ]
                        )
    return returns


def figure_4_2(constant_returned_cars=True):
    """
    Perform policy iteration to solve the Jack’s Car Rental problem and save policy/value visualizations.

    Args:
        constant_returned_cars (bool): If True, use constant car returns for faster evaluation.
    """
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int32)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel("# cars at first location", fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel("# cars at second location", fontsize=30)
        fig.set_title(f"Policy Iteration {iterations}", fontsize=30)

        # Policy evaluation
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    value[i, j] = expected_return(
                        [i, j], policy[i, j], value, constant_returned_cars
                    )
            max_value_change = abs(old_value - value).max()
            print(f"Max value change: {max_value_change}")
            if max_value_change < 1e-4:
                break

        # Policy improvement
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(
                            expected_return(
                                [i, j], action, value, constant_returned_cars
                            )
                        )
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if old_action != new_action:
                    policy_stable = False
        print(f"Policy stable: {policy_stable}")

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel("# cars at first location", fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel("# cars at second location", fontsize=30)
            fig.set_title("Optimal Value Function", fontsize=30)
            break

        iterations += 1

    plt.savefig("../images/figure_4_2.png")
    plt.close()


if __name__ == "__main__":
    figure_4_2()
