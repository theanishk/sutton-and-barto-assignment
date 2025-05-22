"""
Monte Carlo Methods for Blackjack

1. Monte Carlo Prediction (On-policy)
2. Monte Carlo Control (Exploring Starts)
3. Monte Carlo Prediction (Off-policy Importance Sampling)

The environment simulates a simplified Blackjack game where:
- The player aims for a sum as close to 21 as possible without going over.
- A usable ace counts as 11 unless it causes a bust.
- The player follows a fixed policy, and the dealer acts according to predefined rules.
- Rewards are +1 for winning, -1 for losing, and 0 for a draw.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Ensure output directory exists
os.makedirs("../images", exist_ok=True)

# Constants
ACTION_HIT = 0
ACTION_STAND = 1
ACTIONS = [ACTION_HIT, ACTION_STAND]

# Player policy: hit on 12–19, stand on 20–21
POLICY_PLAYER = np.zeros(22, dtype=int)
POLICY_PLAYER[12:20] = ACTION_HIT
POLICY_PLAYER[20:] = ACTION_STAND

# Dealer policy: hit on 12–16, stand on 17–21
POLICY_DEALER = np.zeros(22, dtype=int)
POLICY_DEALER[12:17] = ACTION_HIT
POLICY_DEALER[17:22] = ACTION_STAND


# Target policy function (used for on-policy & off-policy)
def target_policy_player(usable_ace, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]


# Behavior policy (used in off-policy learning)
def behavior_policy_player(usable_ace, player_sum, dealer_card):
    return np.random.choice(ACTIONS)


# Card mechanics
def get_card():
    """Draws a card uniformly between 1 and 10 (J, Q, K = 10)."""
    return min(np.random.randint(1, 14), 10)


def card_value(card):
    """Returns the value of a card (11 for Ace)."""
    return 11 if card == 1 else card


def play(policy_player, initial_state=None, initial_action=None):
    """
    Simulates a full game of Blackjack.

    Parameters:
    - policy_player: function determining player's action
    - initial_state: optional (usable_ace, player_sum, dealer_card)
    - initial_action: optional starting action

    Returns:
    - initial_state: the state the episode started from
    - reward: +1 win, 0 draw, -1 loss
    - player_trajectory: list of (state, action) tuples
    """
    player_sum = 0
    usable_ace_player = False
    player_trajectory = []

    if initial_state is None:
        while player_sum < 12:
            card = get_card()
            player_sum += card_value(card)
            if player_sum > 21:
                player_sum -= 10  # Convert ace from 11 to 1
            else:
                usable_ace_player |= card == 1
        dealer_card1, dealer_card2 = get_card(), get_card()
    else:
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    state = [usable_ace_player, player_sum, dealer_card1]
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
    if dealer_sum > 21:
        dealer_sum -= 10

    # Player's turn
    while True:
        action = (
            initial_action
            if initial_action is not None
            else policy_player(usable_ace_player, player_sum, dealer_card1)
        )
        initial_action = None
        player_trajectory.append(
            [(usable_ace_player, player_sum, dealer_card1), action]
        )
        if action == ACTION_STAND:
            break
        card = get_card()
        ace_count = int(usable_ace_player) + (card == 1)
        player_sum += card_value(card)
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1
        if player_sum > 21:
            return state, -1, player_trajectory
        usable_ace_player = ace_count >= 1

    # Dealer's turn
    while POLICY_DEALER[dealer_sum] == ACTION_HIT:
        card = get_card()
        ace_count = int(usable_ace_dealer) + (card == 1)
        dealer_sum += card_value(card)
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        if dealer_sum > 21:
            return state, 1, player_trajectory
        usable_ace_dealer = ace_count >= 1

    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory


def monte_carlo_on_policy(episodes):
    """Monte Carlo prediction using a fixed on-policy strategy."""
    usable_ace = np.zeros((10, 10))
    usable_ace_count = np.ones((10, 10))
    no_usable_ace = np.zeros((10, 10))
    no_usable_ace_count = np.ones((10, 10))

    for _ in tqdm(range(episodes)):
        _, reward, trajectory = play(target_policy_player)
        for (ua, ps, dc), _ in trajectory:
            ps -= 12
            dc -= 1
            if ua:
                usable_ace[ps, dc] += reward
                usable_ace_count[ps, dc] += 1
            else:
                no_usable_ace[ps, dc] += reward
                no_usable_ace_count[ps, dc] += 1

    return usable_ace / usable_ace_count, no_usable_ace / no_usable_ace_count


def monte_carlo_es(episodes):
    """Monte Carlo control with Exploring Starts (ES)."""
    values = np.zeros((10, 10, 2, 2))  # state-action values
    counts = np.ones_like(values)  # to avoid divide-by-zero

    def greedy_policy(ua, ps, dc):
        ps -= 12
        dc -= 1
        ua = int(ua)
        vals = values[ps, dc, ua] / counts[ps, dc, ua]
        return np.random.choice([a for a, v in enumerate(vals) if v == np.max(vals)])

    for i in tqdm(range(episodes)):
        init_state = [
            bool(np.random.randint(0, 2)),
            np.random.randint(12, 22),
            np.random.randint(1, 11),
        ]
        init_action = np.random.choice(ACTIONS)
        policy = target_policy_player if i == 0 else greedy_policy
        _, reward, trajectory = play(policy, init_state, init_action)
        visited = set()
        for (ua, ps, dc), action in trajectory:
            ps -= 12
            dc -= 1
            ua = int(ua)
            if (ua, ps, dc, action) not in visited:
                visited.add((ua, ps, dc, action))
                values[ps, dc, ua, action] += reward
                counts[ps, dc, ua, action] += 1

    return values / counts


def monte_carlo_off_policy(episodes):
    """Off-policy prediction using importance sampling."""
    init_state = [True, 13, 2]
    true_value_estimates = []
    weights = []

    for _ in range(episodes):
        _, reward, trajectory = play(behavior_policy_player, initial_state=init_state)
        rho = 1.0
        for (ua, ps, dc), action in trajectory:
            if action != target_policy_player(ua, ps, dc):
                rho = 0.0
                break
            rho *= 2  # target = deterministic, behavior = uniform
        weights.append(rho)
        true_value_estimates.append(rho * reward)

    weights = np.array(weights)
    rewards = np.array(true_value_estimates)
    cum_weights = np.cumsum(weights)
    cum_rewards = np.cumsum(rewards)

    ordinary = cum_rewards / np.arange(1, episodes + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = np.where(cum_weights != 0, cum_rewards / cum_weights, 0)

    return ordinary, weighted


def figure_5_1():
    """Plots state-value estimates for on-policy Monte Carlo."""
    ace_10k, no_ace_10k = monte_carlo_on_policy(10000)
    ace_500k, no_ace_500k = monte_carlo_on_policy(500000)

    states = [ace_10k, ace_500k, no_ace_10k, no_ace_500k]
    titles = [
        "Usable Ace, 10k",
        "Usable Ace, 500k",
        "No Usable Ace, 10k",
        "No Usable Ace, 500k",
    ]

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for data, title, ax in zip(states, titles, axes):
        sns.heatmap(
            np.flipud(data),
            cmap="YlGnBu",
            ax=ax,
            xticklabels=range(1, 11),
            yticklabels=list(reversed(range(12, 22))),
        )
        ax.set_title(title, fontsize=30)
        ax.set_xlabel("Dealer Showing", fontsize=20)
        ax.set_ylabel("Player Sum", fontsize=20)

    plt.savefig("../images/figure_5_1.png")
    plt.close()


def figure_5_2():
    """Plots optimal policy and value functions learned via Exploring Starts."""
    q = monte_carlo_es(500000)
    v_ace = np.max(q[:, :, 1, :], axis=-1)
    v_no_ace = np.max(q[:, :, 0, :], axis=-1)
    pi_ace = np.argmax(q[:, :, 1, :], axis=-1)
    pi_no_ace = np.argmax(q[:, :, 0, :], axis=-1)

    images = [pi_ace, v_ace, pi_no_ace, v_no_ace]
    titles = [
        "Policy with Ace",
        "Value with Ace",
        "Policy without Ace",
        "Value without Ace",
    ]

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for img, title, ax in zip(images, titles, axes):
        sns.heatmap(
            np.flipud(img),
            cmap="YlGnBu",
            ax=ax,
            xticklabels=range(1, 11),
            yticklabels=list(reversed(range(12, 22))),
        )
        ax.set_title(title, fontsize=30)
        ax.set_xlabel("Dealer Showing", fontsize=20)
        ax.set_ylabel("Player Sum", fontsize=20)

    plt.savefig("../images/figure_5_2.png")
    plt.close()


def figure_5_3():
    """Plots MSE of ordinary vs weighted importance sampling."""
    true_value = -0.27726
    episodes = 10000
    runs = 100
    mse_ord = np.zeros(episodes)
    mse_wtd = np.zeros(episodes)

    for _ in tqdm(range(runs)):
        ord_, wtd_ = monte_carlo_off_policy(episodes)
        mse_ord += (ord_ - true_value) ** 2
        mse_wtd += (wtd_ - true_value) ** 2

    mse_ord /= runs
    mse_wtd /= runs

    plt.plot(np.arange(1, episodes + 1), mse_ord, label="Ordinary", color="green")
    plt.plot(np.arange(1, episodes + 1), mse_wtd, label="Weighted", color="red")
    plt.xlabel("Episodes (log scale)")
    plt.ylabel(f"MSE (avg over {runs} runs)")
    plt.xscale("log")
    plt.legend()
    plt.savefig("../images/figure_5_3.png")
    plt.close()


if __name__ == "__main__":
    figure_5_1()
    figure_5_2()
    figure_5_3()
