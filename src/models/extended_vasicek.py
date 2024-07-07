# src/models/vasicek.py

import numpy as np
from .base import InterestRateModel


class ExtendedVasicekModel(InterestRateModel):
    def __init__(self, r0, k, theta, sigma, a, eta):
        """
        Initialize the Vasicek model.

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
        self.a = a
        self.eta = eta

    def simulate(self, T, N, n_paths):
        """
        Simulate interest rate paths.

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
                r[:, i - 1] * (self.eta - self.a * np.log(r[:, i - 1])) * dt
                + self.sigma * r[:, i - 1] * dW
            )
            r[:, i] = r[:, i - 1] + dr

        return r

    def zero_coupon_bond_price(self, T, r):
        """
        Calculate the price of a zero-coupon bond.

        Parameters:
        T (float): Time to maturity
        r (float): Current short rate

        Returns:
        float: Price of the zero-coupon bond
        """
        B = (1 - np.exp(-self.k * T)) / self.k
        A = np.exp(
            (self.theta - self.sigma**2 / (2 * self.k**2)) * (B - T)
            - (self.sigma**2 / (4 * self.k)) * B**2
        )
        return A * np.exp(-B * r)

    def calibrate(self, market_data):
        """
        Calibrate the model to market data.

        Parameters:
        market_data (dict): Market observed zero-coupon bond prices

        Returns:
        dict: Calibrated model parameters
        """
        # This is a placeholder. Actual calibration would involve
        # an optimization routine to fit the model to market data.
        print("Calibration method not implemented yet.")
        return {"r0": self.r0, "k": self.k, "theta": self.theta, "sigma": self.sigma}
