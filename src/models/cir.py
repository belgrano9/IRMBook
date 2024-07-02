# src/models/cir.py

import numpy as np
from .base import InterestRateModel


class CIRModel(InterestRateModel):
    def __init__(self, r0, k, theta, sigma):
        """
        Initialize the CIR (Cox-Ingersoll-Ross) model.

        Parameters:
        r0 (float): Initial short rate
        k (float): Speed of mean reversion
        theta (float): Long-term mean level
        sigma (float): Volatility
        """
        self.r0 = r0
        self.k = k
        self.theta = theta
        self.sigma = sigma

    def simulate(self, T, N, n_paths):
        """
        Simulate interest rate paths using the CIR model.

        Parameters:
        T (float): Time horizon
        N (int): Number of time steps
        n_paths (int): Number of paths to simulate

        Returns:
        numpy.ndarray: Simulated interest rate paths
        """
        dt = T / N
        r = np.zeros((n_paths, N + 1))
        r[:, 0] = self.r0

        for i in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            dr = (
                self.k * (self.theta - r[:, i - 1]) * dt
                + self.sigma * np.sqrt(np.maximum(r[:, i - 1], 0)) * dW
            )
            r[:, i] = np.maximum(r[:, i - 1] + dr, 0)  # Ensure rates are non-negative

        return r

    def zero_coupon_bond_price(self, T, r):
        """
        Calculate the price of a zero-coupon bond using the CIR model.

        Parameters:
        T (float): Time to maturity
        r (float): Current short rate

        Returns:
        float: Price of the zero-coupon bond
        """
        gamma = np.sqrt(self.k**2 + 2 * self.sigma**2)
        B = (2 * (np.exp(gamma * T) - 1)) / (
            (gamma + self.k) * (np.exp(gamma * T) - 1) + 2 * gamma
        )
        A = (
            (2 * gamma * np.exp((self.k + gamma) * T / 2))
            / ((gamma + self.k) * (np.exp(gamma * T) - 1) + 2 * gamma)
        ) ** (2 * self.k * self.theta / self.sigma**2)
        return A * np.exp(-B * r)

    def calibrate(self, market_data):
        """
        Calibrate the CIR model to market data.

        Parameters:
        market_data (dict): Market observed zero-coupon bond prices

        Returns:
        dict: Calibrated model parameters
        """
        # This is a placeholder. Actual calibration would involve
        # an optimization routine to fit the model to market data.
        print("Calibration method not implemented yet.")
        return {"r0": self.r0, "k": self.k, "theta": self.theta, "sigma": self.sigma}
