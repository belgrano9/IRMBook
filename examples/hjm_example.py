# examples/hjm_example.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factories.model_factory import ModelFactory


def constant_volatility(t, forward_rates, maturities):
    """
    A simple constant volatility function that returns the same volatility
    for all maturities and all paths.
    
    Parameters:
    t (float): Time
    forward_rates (np.array): Forward rates with shape (n_paths, n_maturities)
    maturities (np.array): Array of maturities
    
    Returns:
    np.array: Volatility values with same shape as forward_rates
    """
    return 0.01 * np.ones_like(forward_rates)


def exponential_volatility(t, forward_rates, maturities):
    """
    An exponential volatility function that decreases with maturity.
    
    Parameters:
    t (float): Time
    forward_rates (np.array): Forward rates with shape (n_paths, n_maturities)
    maturities (np.array): Array of maturities
    
    Returns:
    np.array: Volatility values with same shape as forward_rates
    """
    n_paths, n_maturities = forward_rates.shape
    volatilities = np.zeros((n_paths, n_maturities))
    
    for i in range(n_maturities):
        volatilities[:, i] = 0.02 * np.exp(-0.2 * maturities[i])
    
    return volatilities


def plot_forward_curves(times, maturities, forward_rates, n_curves=5):
    """
    Plot forward rate curves at different times.
    
    Parameters:
    times (np.array): Simulation times
    maturities (np.array): Maturities for the forward rates
    forward_rates (np.array): Simulated forward rates with shape (n_paths, len(times), len(maturities))
    n_curves (int): Number of curves to plot
    """
    plt.figure(figsize=(12, 8))
    
    # Select time points to plot
    time_indices = np.linspace(0, len(times)-1, n_curves, dtype=int)
    
    # Plot a single path at different times
    path_idx = 0
    for i, t_idx in enumerate(time_indices):
        t = times[t_idx]
        plt.plot(maturities, forward_rates[path_idx, t_idx, :], label=f"t = {t:.2f}")
    
    plt.title("Evolution of Forward Rate Curve")
    plt.xlabel("Maturity")
    plt.ylabel("Forward Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_zero_bond_prices(model, times, max_maturity=30):
    """
    Plot zero-coupon bond prices for different maturities at t=0.
    
    Parameters:
    model (HJMModel): Calibrated HJM model
    times (np.array): Array of times for which to calculate bond prices
    max_maturity (float): Maximum maturity to consider
    """
    plt.figure(figsize=(10, 6))
    
    maturities = np.linspace(0.1, max_maturity, 50)
    prices = np.array([model.zero_coupon_bond_price(T=m) for m in maturities])
    
    plt.plot(maturities, prices)
    plt.title("Zero-Coupon Bond Prices at t=0")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def main():
    # Define parameters
    T = 10.0  # Time horizon
    N = 120  # Number of time steps
    n_paths = 100  # Number of simulation paths
    
    # Define initial forward curve (flat at 3%)
    maturities = np.linspace(0, 30, 60)  # Maturities up to 30 years
    initial_forward_curve = 0.03 * np.ones_like(maturities)
    
    # Define volatility functions (2-factor model)
    vol_functions = [constant_volatility, exponential_volatility]
    
    # Create HJM model
    hjm = ModelFactory.create_model(
        "HJM",
        initial_forward_curve=initial_forward_curve,
        volatility_functions=vol_functions,
        maturities=maturities,
        T=T,
        N=N,
        n_paths=n_paths
    )
    
    # Simulate forward rate curves
    times, maturities, forward_rates = hjm.simulate()
    
    # Store simulated data in the model instance for later use
    hjm.simulated_times = times
    hjm.simulated_forwards = forward_rates
    
    # Plot forward rate curves
    plot_forward_curves(times, maturities, forward_rates)
    
    # Plot zero-coupon bond prices
    plot_zero_bond_prices(hjm, times)
    
    # Test zero-coupon bond pricing
    price_5y = hjm.zero_coupon_bond_price(5.0)
    print(f"Price of 5-year zero-coupon bond: {price_5y:.4f}")



if __name__ == "__main__":
    main()