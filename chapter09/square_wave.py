import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend suitable for file saving
import matplotlib.pyplot as plt
from tqdm import tqdm


class Interval:
    """
    Represents a half-open interval [left, right).
    """

    def __init__(self, left: float, right: float):
        """
        Initialize the interval.

        Args:
            left (float): Left endpoint of the interval.
            right (float): Right endpoint (exclusive).
        """
        self.left = left
        self.right = right

    def contain(self, x: float) -> bool:
        """
        Check if a point is within the interval.

        Args:
            x (float): The point to check.

        Returns:
            bool: True if x is in [left, right), else False.
        """
        return self.left <= x < self.right

    def size(self) -> float:
        """
        Returns the size (length) of the interval.

        Returns:
            float: Length of the interval.
        """
        return self.right - self.left


# Domain of the square wave function: [0, 2)
DOMAIN = Interval(0.0, 2.0)


def square_wave(x: float) -> int:
    """
    Square wave function.

    Args:
        x (float): Input value.

    Returns:
        int: 1 if 0.5 < x < 1.5, else 0.
    """
    return 1 if 0.5 < x < 1.5 else 0


def sample(n: int) -> list:
    """
    Generate random samples from the square wave function.

    Args:
        n (int): Number of samples to generate.

    Returns:
        list: List of [x, y] pairs.
    """
    samples = []
    for _ in range(n):
        x = np.random.uniform(DOMAIN.left, DOMAIN.right)
        y = square_wave(x)
        samples.append([x, y])
    return samples


class ValueFunction:
    """
    Approximates a function using overlapping binary features (like coarse coding).
    """

    def __init__(
        self,
        feature_width: float,
        domain: Interval = DOMAIN,
        alpha: float = 0.2,
        num_of_features: int = 50,
    ):
        """
        Initialize the value function with given parameters.

        Args:
            feature_width (float): Width of each binary feature.
            domain (Interval): Domain over which features are defined.
            alpha (float): Step size for weight updates.
            num_of_features (int): Number of features.
        """
        self.feature_width = feature_width
        self.num_of_features = num_of_features
        self.alpha = alpha
        self.domain = domain
        self.features = []

        step = (domain.size() - feature_width) / (num_of_features - 1)
        left = domain.left
        for _ in range(num_of_features - 1):
            self.features.append(Interval(left, left + feature_width))
            left += step
        self.features.append(Interval(left, domain.right))  # Last feature

        self.weights = np.zeros(num_of_features)

    def get_active_features(self, x: float) -> list:
        """
        Get indices of active (containing x) features.

        Args:
            x (float): The input point.

        Returns:
            list: Indices of features that contain x.
        """
        return [i for i, f in enumerate(self.features) if f.contain(x)]

    def value(self, x: float) -> float:
        """
        Estimate the value at a point x.

        Args:
            x (float): Input point.

        Returns:
            float: Estimated value.
        """
        active = self.get_active_features(x)
        return np.sum(self.weights[active])

    def update(self, delta: float, x: float):
        """
        Update weights based on a sample delta and input x.

        Args:
            delta (float): The error (target - estimate).
            x (float): Input point.
        """
        active = self.get_active_features(x)
        update = self.alpha * delta / len(active)
        for idx in active:
            self.weights[idx] += update


def approximate(samples: list, value_function: ValueFunction):
    """
    Train a value function on a set of samples.

    Args:
        samples (list): List of [x, y] pairs.
        value_function (ValueFunction): Function approximator to update.
    """
    for x, y in samples:
        delta = y - value_function.value(x)
        value_function.update(delta, x)


def figure_9_8():
    """
    Reproduce Figure 9.8 from Sutton & Barto’s RL book.
    Shows how the approximation improves with sample size and different feature widths.
    Saves the figure to ../images/figure_9_8.png.
    """
    num_of_samples = [10, 40, 160, 640, 2560, 10240]
    feature_widths = [0.2, 0.4, 1.0]
    axis_x = np.arange(DOMAIN.left, DOMAIN.right, 0.02)

    plt.figure(figsize=(30, 20))

    for index, num_samples in enumerate(num_of_samples):
        print(f"{num_samples} samples")
        samples = sample(num_samples)
        value_functions = [ValueFunction(width) for width in feature_widths]

        plt.subplot(2, 3, index + 1)
        plt.title(f"{num_samples} samples")

        for vf in value_functions:
            approximate(samples, vf)
            predictions = [vf.value(x) for x in axis_x]
            plt.plot(axis_x, predictions, label=f"feature width {vf.feature_width:.1f}")

        plt.legend()

    plt.savefig("../images/figure_9_8.png")
    plt.close()


if __name__ == "__main__":
    figure_9_8()
