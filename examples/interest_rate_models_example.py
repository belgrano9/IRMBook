# examples/interest_rate_models_example.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factories.model_factory import ModelFactory
from src.utils.helpers import plot_rate_paths


def main():
    # Create a Vasicek model
    vasicek = ModelFactory.create_model(
        "Vasicek", r0=0.05, k=0.3, theta=0.05, sigma=0.02
    )

    # Simulate Vasicek paths
    vasicek_rates = vasicek.simulate(T=1, N=252, n_paths=500)

    # Plot Vasicek paths
    plot_rate_paths(
        vasicek_rates,
        title="Vasicek Model: Simulated Interest Rate Paths",
        num_paths=15,
    )

    # Create a CIR model
    cir = ModelFactory.create_model("CIR", r0=0.05, k=0.3, theta=0.05, sigma=0.1)

    # Simulate CIR paths
    cir_rates = cir.simulate(T=1, N=252, n_paths=500)

    # Plot CIR paths
    plot_rate_paths(
        cir_rates, title="CIR Model: Simulated Interest Rate Paths", num_paths=15
    )

    # Compare zero-coupon bond prices
    T = 2  # 2-year bond
    r = 0.05  # current short rate
    vasicek_price = vasicek.zero_coupon_bond_price(T, r)
    cir_price = cir.zero_coupon_bond_price(T, r)

    print(f"Price of a {T}-year zero-coupon bond:")
    print(f"  Vasicek model: {vasicek_price:.4f}")
    print(f"  CIR model: {cir_price:.4f}")


if __name__ == "__main__":
    main()
