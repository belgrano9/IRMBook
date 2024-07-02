# src/models/dothan.py

import numpy as np
from scipy.stats import norm
from .base import InterestRateModel


class DothanModel(InterestRateModel):
    def __init__(self, r0, sigma):
        """
        Initialize the Dothan model.

        Parameters:
        r0 (float): Initial short rate
        sigma (float): Volatility
        """
        self.r0 = r0
        self.sigma = sigma

    def simulate(self, T, N, n_paths):
        """
        Simulate interest rate paths using the Dothan model.

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
            dr = self.sigma * r[:, i - 1] * dW
            r[:, i] = r[:, i - 1] * np.exp(dr - 0.5 * self.sigma**2 * dt)

        return r

    def zero_coupon_bond_price(self, T, r):
        """
        Calculate the price of a zero-coupon bond using the Dothan model.

        Parameters:
        T (float): Time to maturity
        r (float): Current short rate

        Returns:
        float: Price of the zero-coupon bond
        """
        d1 = (np.log(r / self.r0) + (self.sigma**2 * T / 2)) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)

        return (self.r0 / r) * norm.cdf(-d1) + norm.cdf(d2)

    def calibrate(self, market_data):
        """
        Calibrate the Dothan model to market data.

        Parameters:
        market_data (dict): Market observed zero-coupon bond prices

        Returns:
        dict: Calibrated model parameters
        """
        # This is a placeholder. Actual calibration would involve
        # an optimization routine to fit the model to market data.
        print("Calibration method not implemented yet.")
        return {"r0": self.r0, "sigma": self.sigma}
