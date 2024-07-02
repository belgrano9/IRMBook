# examples/interest_rate_models_example.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.factories.model_factory import ModelFactory
from src.utils.helpers import plot_rate_paths


def main():
    # Parameters
    T = 1  # Time horizon
    N = 252  # Number of time steps
    n_paths = 500  # Number of paths to simulate
    r0 = 0.05  # Initial short rate

    # Create and simulate Vasicek model
    vasicek = ModelFactory.create_model("Vasicek", r0=r0, k=0.3, theta=0.05, sigma=0.02)
    vasicek_rates = vasicek.simulate(T, N, n_paths)
    plot_rate_paths(
        vasicek_rates, title="Vasicek Model: Simulated Interest Rate Paths", num_paths=5
    )

    # Create and simulate CIR model
    cir = ModelFactory.create_model("CIR", r0=r0, k=0.3, theta=0.05, sigma=0.1)
    cir_rates = cir.simulate(T, N, n_paths)
    plot_rate_paths(
        cir_rates, title="CIR Model: Simulated Interest Rate Paths", num_paths=5
    )

    # Create and simulate Hull-White model
    t_array = np.linspace(0, T, N + 1)
    theta_array = 0.05 + 0.01 * np.sin(
        2 * np.pi * t_array
    )  # Example time-varying theta
    hull_white = ModelFactory.create_model(
        "HullWhite", a=0.1, sigma=0.02, t_array=t_array, theta_array=theta_array
    )
    hw_rates = hull_white.simulate(T, N, n_paths, r0)
    plot_rate_paths(
        hw_rates, title="Hull-White Model: Simulated Interest Rate Paths", num_paths=5
    )

    # Compare zero-coupon bond prices
    T_bond = 2  # 2-year bond
    vasicek_price = vasicek.zero_coupon_bond_price(T_bond, r0)
    cir_price = cir.zero_coupon_bond_price(T_bond, r0)
    hw_price = hull_white.zero_coupon_bond_price(T_bond, r0)

    print(f"Price of a {T_bond}-year zero-coupon bond:")
    print(f"  Vasicek model: {vasicek_price:.4f}")
    print(f"  CIR model: {cir_price:.4f}")
    print(f"  Hull-White model: {hw_price:.4f}")


if __name__ == "__main__":
    main()
