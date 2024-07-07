# examples/interest_rate_models_example.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.factories.model_factory import ModelFactory
from src.utils.helpers import plot_rate_trajectories_with_mean


def main():
    # Parameters
    T = 1  # Time horizon
    N = 252  # Number of time steps
    n_paths = 500  # Number of paths to simulate
    r0 = 0.05  # Initial short rate

    # Create and simulate models
    models = {
        "Vasicek": ModelFactory.create_model(
            "Vasicek", r0=r0, k=0.3, theta=0.05, sigma=0.02
        ),
        "Extended Vasicek": ModelFactory.create_model(
            "Extended Vasicek", r0=r0, k=0.3, theta=0.05, sigma=0.02, eta=1, a=1
        ),
    }

    for model_name, model in models.items():
        if model_name == "HullWhite":
            rates = model.simulate(T, N, n_paths, r0)
        else:
            rates = model.simulate(T, N, n_paths)

        plot_rate_trajectories_with_mean(
            rates,
            title=f"{model_name} Model: Interest Rate Trajectories",
            num_trajectories=500,
        )


if __name__ == "__main__":
    main()
