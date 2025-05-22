"""
1. **Uniform Sampling**: Each state-action pair is updated uniformly.
2. **On-policy Sampling**: Updates are made using a soft epsilon-greedy behavior policy.

Each method is evaluated in randomly generated Markov Decision Processes (MDPs) with varying
state space sizes and branching factors. The performance is measured by the value of the
start state under the greedy policy derived from the current action-value function `q`.

Output:
    Saves `figure_8_8.png` comparing performance across different settings.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use("Agg")  # For saving plots without displaying

# Constants
ACTIONS = [0, 1]  # Two available actions
TERMINATION_PROB = 0.1  # Probability of transitioning to terminal state
MAX_STEPS = 20000  # Maximum updates
EPSILON = 0.1  # ε-greedy parameter for on-policy sampling


def argmax(values):
    """
    Randomized argmax: returns the index of the maximum value, breaking ties randomly.

    Args:
        values (np.ndarray): Array of values to choose from.

    Returns:
        int: Index of the chosen maximum value.
    """
    max_val = np.max(values)
    return np.random.choice([i for i, v in enumerate(values) if v == max_val])


class Task:
    """
    A randomly generated episodic MDP for evaluating sampling strategies.

    Attributes:
        n_states (int): Number of non-terminal states (terminal state is n_states).
        b (int): Branching factor — number of possible transitions per (s, a) pair.
        transition (ndarray): Transition matrix of shape [state, action, branch].
        reward (ndarray): Reward matrix of shape [state, action, branch].
    """

    def __init__(self, n_states, b):
        self.n_states = n_states
        self.b = b
        self.transition = np.random.randint(n_states, size=(n_states, len(ACTIONS), b))
        self.reward = np.random.randn(n_states, len(ACTIONS), b)

    def step(self, state, action):
        """
        Sample a next state and reward from the MDP with termination probability.

        Args:
            state (int): Current state.
            action (int): Action taken.

        Returns:
            (int, float): Next state and corresponding reward.
        """
        if np.random.rand() < TERMINATION_PROB:
            return self.n_states, 0  # Terminal state
        index = np.random.randint(self.b)
        return self.transition[state, action, index], self.reward[state, action, index]


def evaluate_pi(q, task):
    """
    Evaluate the value of the start state under the greedy policy derived from q.

    Args:
        q (np.ndarray): Action-value function array.
        task (Task): The MDP environment.

    Returns:
        float: Monte Carlo estimate of the value of the start state.
    """
    returns = []
    for _ in range(1000):
        total_reward = 0
        state = 0
        while state < task.n_states:
            action = argmax(q[state])
            state, reward = task.step(state, action)
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns)


def uniform(task, eval_interval):
    """
    Expected updates using uniform sampling of all state-action pairs.

    Args:
        task (Task): MDP task.
        eval_interval (int): Frequency of evaluation steps.

    Returns:
        tuple: Arrays of (steps, value estimates).
    """
    q = np.zeros((task.n_states, len(ACTIONS)))
    performance = []

    for step in tqdm(range(MAX_STEPS), desc="Uniform Sampling"):
        state = (step // len(ACTIONS)) % task.n_states
        action = step % len(ACTIONS)

        next_states = task.transition[state, action]
        rewards = task.reward[state, action]
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(
            rewards + np.max(q[next_states], axis=1)
        )

        if step % eval_interval == 0:
            performance.append([step, evaluate_pi(q, task)])

    return zip(*performance)


def on_policy(task, eval_interval):
    """
    Expected updates using on-policy sampling via ε-greedy behavior.

    Args:
        task (Task): MDP task.
        eval_interval (int): Frequency of evaluation steps.

    Returns:
        tuple: Arrays of (steps, value estimates).
    """
    q = np.zeros((task.n_states, len(ACTIONS)))
    performance = []
    state = 0

    for step in tqdm(range(MAX_STEPS), desc="On-Policy Sampling"):
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = argmax(q[state])

        next_state, _ = task.step(state, action)

        next_states = task.transition[state, action]
        rewards = task.reward[state, action]
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(
            rewards + np.max(q[next_states], axis=1)
        )

        state = 0 if next_state == task.n_states else next_state

        if step % eval_interval == 0:
            performance.append([step, evaluate_pi(q, task)])

    return zip(*performance)


def figure_8_8():
    """
    Generate and save Figure 8.8 comparing uniform vs. on-policy updates
    across different state sizes and branching factors.
    """
    num_states = [1000, 10000]
    branch_factors = [1, 3, 10]
    methods = [on_policy, uniform]
    n_tasks = 30
    x_ticks = 100

    plt.figure(figsize=(10, 20))

    for i, n in enumerate(num_states):
        plt.subplot(2, 1, i + 1)
        for b in branch_factors:
            tasks = [Task(n, b) for _ in range(n_tasks)]
            for method in methods:
                values = []
                for task in tasks:
                    steps, v = method(task, MAX_STEPS // x_ticks)
                    values.append(v)
                avg_value = np.mean(values, axis=0)
                plt.plot(steps, avg_value, label=f"b = {b}, {method.__name__}")
        plt.title(f"{n} States")
        plt.ylabel("Value of Start State")
        plt.legend()

    plt.xlabel("Computation Time (Expected Updates)")
    plt.savefig("../images/figure_8_8.png")
    plt.close()


if __name__ == "__main__":
    figure_8_8()
