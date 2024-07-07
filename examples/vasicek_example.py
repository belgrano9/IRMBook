# examples/vasicek_example.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factories.model_factory import ModelFactory
import matplotlib.pyplot as plt


def main():
    # Create a Vasicek model
    vasicek = ModelFactory.create_model(
        "Vasicek",
        r0=0.05,
        k=0.3,
        theta=0.05,
        sigma=0.02,
        T=1,
        N=252,
        n_paths=5,
    )

    # Simulate interest rate paths

    # simulated_rates = vasicek.simulate()

    """ # Plot the simulated paths
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(simulated_rates[i, :])
    plt.title("Vasicek Model: Simulated Interest Rate Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Interest Rate")
    plt.show() """

    # Price a zero-coupon bond
    # bond_price = vasicek.zero_coupon_bond_price(T=2, r=0.05)
    # print(f"Price of a 2-year zero-coupon bond: {bond_price:.4f}")

    # MC Path dependent
    vasicek.mc_path_dependent(error=False)


if __name__ == "__main__":
    main()
