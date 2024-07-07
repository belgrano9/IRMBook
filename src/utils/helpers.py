"""
Utility functions used across the project.
"""

# src/utils/helpers.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


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


def plot_rate_trajectories_with_mean(
    simulated_rates, title="Interest Rate Trajectories", num_trajectories=None
):
    """
    Plot simulated interest rate trajectories with mean trajectory highlighted.

    Parameters:
    simulated_rates (np.ndarray): Array of simulated rates with shape (n_paths, n_steps)
    title (str): Title for the plot
    num_trajectories (int): Number of trajectories to plot. If None, all trajectories are plotted.
    """
    if num_trajectories is None:
        num_trajectories = simulated_rates.shape[0]
    else:
        num_trajectories = min(num_trajectories, simulated_rates.shape[0])

    # Calculate mean trajectory
    mean_trajectory = np.mean(simulated_rates, axis=0)

    # Calculate distance of each trajectory from the mean
    distances = np.mean(np.abs(simulated_rates - mean_trajectory), axis=1)

    # Sort trajectories by distance from mean
    sorted_indices = np.argsort(distances)

    # Create color map
    colors = [(0, 0, 1, 1), (0, 0, 1, 0.1)]  # Blue with decreasing alpha
    n_bins = num_trajectories
    cm = LinearSegmentedColormap.from_list("custom_blue", colors, N=n_bins)

    plt.figure(figsize=(12, 8))

    # Plot trajectories
    for i in range(num_trajectories):
        idx = sorted_indices[i]
        color = cm(i / num_trajectories)
        plt.plot(simulated_rates[idx], color=color, linewidth=0.5, alpha=0.5)

    # Plot mean trajectory
    plt.plot(mean_trajectory, color="red", linewidth=2, label="Mean Trajectory")

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Interest Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
