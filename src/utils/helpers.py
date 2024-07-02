"""
Utility functions used across the project.
"""

# src/utils/helpers.py

import matplotlib.pyplot as plt
import numpy as np


def plot_rate_paths(
    simulated_rates, title="Simulated Interest Rate Paths", num_paths=None
):
    """
    Plot simulated interest rate paths.

    Parameters:
    simulated_rates (np.ndarray): Array of simulated rates with shape (n_paths, n_steps)
    title (str): Title for the plot
    num_paths (int): Number of paths to plot. If None, all paths are plotted.
    """
    plt.figure(figsize=(10, 6))

    if num_paths is None:
        num_paths = simulated_rates.shape[0]
    else:
        num_paths = min(num_paths, simulated_rates.shape[0])

    for i in range(num_paths):
        plt.plot(simulated_rates[i, :])

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Interest Rate")
    plt.grid(True)
    plt.show()
