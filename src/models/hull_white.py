# src/models/hull_white.py

import numpy as np
from .base import InterestRateModel


class HullWhiteModel(InterestRateModel):
    def __init__(self, a, sigma, t_array=None, theta_array=None):
        """
        Initialize the Hull-White model.

        Parameters:
        a (float): Mean reversion speed
        sigma (float): Volatility
        t_array (np.array): Array of times for theta values (optional)
        theta_array (np.array): Array of theta values (optional)
        """
        self.a = a
        self.sigma = sigma
        self.t_array = t_array
        self.theta_array = theta_array

    def _theta(self, t):
        """
        Calculate theta at time t.
        If theta_array is not provided, return 0 (for simplicity).
        """
        if self.t_array is None or self.theta_array is None:
            return 0
        return np.interp(t, self.t_array, self.theta_array)

    def simulate(self, T, N, n_paths, r0):
        """
        Simulate interest rate paths using the Hull-White model.

        Parameters:
        T (float): Time horizon
        N (int): Number of time steps
        n_paths (int): Number of paths to simulate
        r0 (float): Initial short rate

        Returns:
        numpy.ndarray: Simulated interest rate paths
        """
        dt = T / N
        r = np.zeros((n_paths, N + 1))
        r[:, 0] = r0

        for i in range(1, N + 1):
            t = i * dt
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            theta = self._theta(t)
            dr = (theta - self.a * r[:, i - 1]) * dt + self.sigma * dW
            r[:, i] = r[:, i - 1] + dr

        return r

    def zero_coupon_bond_price(self, T, r, t=0):
        """
        Calculate the price of a zero-coupon bond using the Hull-White model.

        Parameters:
        T (float): Time to maturity
        r (float): Current short rate
        t (float): Current time

        Returns:
        float: Price of the zero-coupon bond
        """
        B = (1 - np.exp(-self.a * (T - t))) / self.a
        A = np.exp(
            (B - T + t) * (self._theta(t) - self.sigma**2 / (2 * self.a**2))
            - (self.sigma**2 / (4 * self.a)) * B**2
        )
        return A * np.exp(-B * r)

    def calibrate(self, market_data):
        """
        Calibrate the Hull-White model to market data.

        Parameters:
        market_data (dict): Market observed zero-coupon bond prices and yield curve

        Returns:
        dict: Calibrated model parameters
        """
        # This is a placeholder. Actual calibration would involve
        # an optimization routine to fit the model to market data.
        print("Calibration method not implemented yet.")
        return {"a": self.a, "sigma": self.sigma}
