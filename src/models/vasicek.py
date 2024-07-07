# src/models/vasicek.py

import numpy as np
from .base import InterestRateModel
import math


class VasicekModel(InterestRateModel):
    def __init__(self, r0, k, theta, sigma, T, N, n_paths):
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
        self.T = T
        self.N = N
        self.n_paths = n_paths

    def simulate(self):
        """
        Simulate interest rate paths.

        Parameters:
        T (float): Time horizon
        N (int): Number of time steps
        n_paths (int): Number of paths to simulate

        Returns:
        numpy.ndarray: Simulated interest rate paths
        """
        dt = self.T / self.N
        r = np.zeros((self.n_paths, self.N + 1))
        r[:, 0] = self.r0

        for i in range(1, self.N + 1):
            dW = np.random.normal(0, np.sqrt(dt), self.n_paths)
            dr = self.k * (self.theta - r[:, i - 1]) * dt + self.sigma * dW
            r[:, i] = r[:, i - 1] + dr

        return r

    def zero_coupon_bond_price(self, r):
        """
        Calculate the price of a zero-coupon bond.

        Parameters:
        T (float): Time to maturity
        r (float): Current short rate

        Returns:
        float: Price of the zero-coupon bond
        """
        B = (1 - np.exp(-self.k * self.T)) / self.k
        A = np.exp(
            (self.theta - self.sigma**2 / (2 * self.k**2)) * (B - self.T)
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

    @property
    def times(self):
        return np.linspace(0, self.T, self.N + 1)

    def mc_path_dependent(self, error: bool):
        r = self.simulate()
        h = np.max(r, axis=1) - r[:, -1]  # Changed to match the product definition
        P = self.compute_discount_factor(r)
        d_h = P * h
        av_d_h = np.mean(d_h)
        if error:
            std = np.std(d_h, ddof=1)
            print(f"{av_d_h:.4f} +- {1.96*std/np.sqrt(len(d_h)):.4f}")
            return av_d_h, 1.96 * std / np.sqrt(len(d_h))

        else:
            return av_d_h

    def compute_discount_factor(self, rates):
        if rates.ndim == 1:
            return self._df_loop(rates)
        else:
            return np.apply_along_axis(self._df_loop, 1, rates)

    def _df_loop(self, rates):
        dt = np.diff(self.times)
        integral_sum = np.sum(0.5 * (rates[:-1] + rates[1:]) * dt)
        return math.exp(-integral_sum)
