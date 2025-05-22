"""
Ten-Armed Testbed for Bandit Algorithms

This script implements a testbed for evaluating various multi-armed bandit algorithms,
including epsilon-greedy, UCB, optimistic initialization, and gradient bandits.
It reproduces the experiments and figures from Sutton & Barto's "Reinforcement Learning: An Introduction" (Chapter 2).

Classes:
    Bandit: Simulates a k-armed bandit problem with configurable algorithms.

Functions:
    simulate: Runs multiple bandit instances and collects statistics.
    figure_2_1: Plots reward distributions for each action.
    figure_2_2: Compares epsilon-greedy algorithms.
    figure_2_3: Shows effect of optimistic initial values.
    figure_2_4: Compares UCB and epsilon-greedy.
    figure_2_5: Compares gradient bandit algorithms.
    figure_2_6: Parameter study for all algorithms.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for image saving

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class Bandit:
    """
    Multi-armed bandit class implementing epsilon-greedy, UCB, and gradient bandit strategies.
    """

    def __init__(
        self,
        k_arm=10,
        epsilon=0.0,
        initial=0.0,
        step_size=0.1,
        sample_averages=False,
        UCB_param=None,
        gradient=False,
        gradient_baseline=False,
        true_reward=0.0,
    ):
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.true_reward = true_reward
        self.indices = np.arange(self.k)
        self.reset()

    def reset(self):
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_estimation = np.full(self.k, self.initial)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0
        self.average_reward = 0
        self.action_prob = np.full(self.k, 1.0 / self.k)

    def act(self):
        if self.gradient:
            exp_est = np.exp(
                self.q_estimation - np.max(self.q_estimation)
            )  # numerical stability
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            ucb = self.q_estimation + self.UCB_param * np.sqrt(
                np.log(self.time + 1) / (self.action_count + 1e-5)
            )
            return np.random.choice(np.flatnonzero(ucb == np.max(ucb)))

        return np.random.choice(
            np.flatnonzero(self.q_estimation == np.max(self.q_estimation))
        )

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            self.q_estimation[action] += (
                reward - self.q_estimation[action]
            ) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            baseline = self.average_reward if self.gradient_baseline else 0
            self.q_estimation += (
                self.step_size * (reward - baseline) * (one_hot - self.action_prob)
            )
        else:
            self.q_estimation[action] += self.step_size * (
                reward - self.q_estimation[action]
            )

        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros_like(rewards)

    for i, bandit in enumerate(bandits):
        for r in trange(runs, desc=f"Bandit {i + 1}/{len(bandits)}"):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                best_action_counts[i, r, t] = int(action == bandit.best_action)

    return best_action_counts.mean(axis=1), rewards.mean(axis=1)


def save_plot(xlabel, ylabel, filename, *plot_args):
    plt.figure()
    for args in plot_args:
        plt.plot(*args[:-1], label=args[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"../images/{filename}.png")
    plt.close()


def figure_2_1():
    data = np.random.randn(200, 10) + np.random.randn(10)
    plt.violinplot(dataset=data)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig("../images/figure_2_1.png")
    plt.close()


def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    save_plot(
        "Steps",
        "Average reward",
        "figure_2_2_top",
        *((rewards[i], f"$\epsilon = {eps:.2f}$") for i, eps in enumerate(epsilons)),
    )

    save_plot(
        "Steps",
        "% Optimal action",
        "figure_2_2_bottom",
        *(
            (best_action_counts[i], f"$\epsilon = {eps:.2f}$")
            for i, eps in enumerate(epsilons)
        ),
    )


def figure_2_3(runs=2000, time=1000):
    bandits = [
        Bandit(epsilon=0, initial=5, step_size=0.1),
        Bandit(epsilon=0.1, initial=0, step_size=0.1),
    ]
    best_action_counts, _ = simulate(runs, time, bandits)
    save_plot(
        "Steps",
        "% Optimal action",
        "figure_2_3",
        (best_action_counts[0], "$\epsilon = 0, q = 5$"),
        (best_action_counts[1], "$\epsilon = 0.1, q = 0$"),
    )


def figure_2_4(runs=2000, time=1000):
    bandits = [
        Bandit(epsilon=0, UCB_param=2, sample_averages=True),
        Bandit(epsilon=0.1, sample_averages=True),
    ]
    _, rewards = simulate(runs, time, bandits)
    save_plot(
        "Steps",
        "Average reward",
        "figure_2_4",
        (rewards[0], "UCB $c = 2$"),
        (rewards[1], "epsilon-greedy $\epsilon = 0.1$"),
    )


def figure_2_5(runs=2000, time=1000):
    configs = [(0.1, True), (0.1, False), (0.4, True), (0.4, False)]
    bandits = [
        Bandit(
            gradient=True, step_size=alpha, gradient_baseline=baseline, true_reward=4
        )
        for alpha, baseline in configs
    ]
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = [
        rf"$\alpha = {alpha}$, {'with' if baseline else 'without'} baseline"
        for alpha, baseline in configs
    ]
    save_plot(
        "Steps",
        "% Optimal action",
        "figure_2_5",
        *((best_action_counts[i], labels[i]) for i in range(len(bandits))),
    )


def figure_2_6(runs=2000, time=1000):
    labels = ["epsilon-greedy", "gradient bandit", "UCB", "optimistic initialization"]
    generators = [
        lambda e: Bandit(epsilon=e, sample_averages=True),
        lambda a: Bandit(gradient=True, step_size=a, gradient_baseline=True),
        lambda c: Bandit(epsilon=0, UCB_param=c, sample_averages=True),
        lambda i: Bandit(epsilon=0, initial=i, step_size=0.1),
    ]
    param_ranges = [
        np.arange(-7, -1),
        np.arange(-5, 2),
        np.arange(-4, 3),
        np.arange(-2, 3),
    ]

    bandits, x_vals = [], []
    for gen, exponents in zip(generators, param_ranges):
        powers = [np.float_power(2, x) for x in exponents]
        x_vals.append(powers)
        bandits.extend([gen(p) for p in powers])

    _, rewards = simulate(runs, time, bandits)
    mean_rewards = np.mean(rewards, axis=1)

    plt.figure()
    idx = 0
    for label, powers in zip(labels, x_vals):
        end = idx + len(powers)
        plt.plot(np.log2(powers), mean_rewards[idx:end], label=label)
        idx = end
    plt.xlabel("log2(Parameter)")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("../images/figure_2_6.png")
    plt.close()


if __name__ == "__main__":
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()
