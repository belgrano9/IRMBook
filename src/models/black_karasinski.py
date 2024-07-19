# src/models/black_karasinski.py

import numpy as np
from scipy.optimize import newton
from .base import InterestRateModel


class BlackKarasinskiModel(InterestRateModel):
    def __init__(self, a, sigma, t_array=None, theta_array=None):
        super().__init__(a, sigma, t_array, theta_array)

    def _theta(self, t):
        if self.t_array is None or self.theta_array is None:
            return 0.05  # Default constant value
        return np.interp(t, self.t_array, self.theta_array)

    def simulate(self, T, N, n_paths, r0):
        """
        Simulate interest rate paths using the Black-Karasinski model.

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
            log_r = np.log(r[:, i - 1])
            d_log_r = (theta - self.a * log_r) * dt + self.sigma * dW
            r[:, i] = np.exp(log_r + d_log_r)

        return r

    def build_tree(self, T, N, r0, P):
        """
        Build a trinomial tree for the Black-Karasinski model.

        Parameters:
        T (float): Time horizon
        N (int): Number of time steps
        r0 (float): Initial short rate
        P (function): Function that returns the initial discount factor for a given maturity

        Returns:
        tuple: Tree nodes, probabilities, and displacement values
        """
        dt = T / N
        dx = self.sigma * np.sqrt(3 * dt)

        # Initialize tree structures
        x = np.zeros((N + 1, 2 * N + 1))
        r = np.zeros((N + 1, 2 * N + 1))
        Q = np.zeros((N + 1, 2 * N + 1))
        alpha = np.zeros(N + 1)

        # Set up initial node
        x[0, N] = 0
        Q[0, N] = 1
        alpha[0] = np.log(-np.log(P(dt)) / dt)
        r[0, N] = np.exp(alpha[0])

        # Build the tree
        for i in range(1, N + 1):
            j_min, j_max = N - i, N + i
            x[i, j_min : j_max + 1] = np.arange(j_min - N, j_max - N + 1) * dx

            # Calculate Q values
            for j in range(j_min, j_max + 1):
                Q[i, j] = (
                    Q[i - 1, j - 1] * self._pu(i - 1, j - 1, dt, dx)
                    + Q[i - 1, j] * self._pm(i - 1, j, dt, dx)
                    + Q[i - 1, j + 1] * self._pd(i - 1, j + 1, dt, dx)
                ) * np.exp(-r[i - 1, j] * dt)

            # Calculate alpha
            alpha[i] = self._calculate_alpha(i, x[i], Q[i], P(i * dt), dt)

            # Calculate r values
            r[i] = np.exp(x[i] + alpha[i])

        return x, r, Q, alpha

    def _pu(self, i, j, dt, dx):
        return (
            1 / 6
            + (self._theta(i * dt) - self.a * j * dx) * np.sqrt(dt) / (2 * dx)
            + (self._theta(i * dt) - self.a * j * dx) ** 2 * dt / (2 * dx**2)
        )

    def _pm(self, i, j, dt, dx):
        return 2 / 3 - (self._theta(i * dt) - self.a * j * dx) ** 2 * dt / dx**2

    def _pd(self, i, j, dt, dx):
        return (
            1 / 6
            - (self._theta(i * dt) - self.a * j * dx) * np.sqrt(dt) / (2 * dx)
            + (self._theta(i * dt) - self.a * j * dx) ** 2 * dt / (2 * dx**2)
        )

    def _calculate_alpha(self, i, x, Q, P_target, dt):
        def objective(alpha):
            r = np.exp(x + alpha)
            return P_target - np.sum(Q * np.exp(-r * dt))

        def derivative(alpha):
            r = np.exp(x + alpha)
            return -np.sum(Q * np.exp(-r * dt) * r * dt)

        return newton(objective, x0=0, fprime=derivative)

    def zero_coupon_bond_price(self, T, r, t=0):
        """
        Calculate the price of a zero-coupon bond using the trinomial tree.

        Parameters:
        T (float): Time to maturity
        r (float): Current short rate
        t (float): Current time

        Returns:
        float: Price of the zero-coupon bond
        """
        N = 100  # Number of time steps in the tree
        dt = (T - t) / N

        def P(t):
            # This should return the initial discount factor for maturity t
            # In a real implementation, this would come from market data
            return np.exp(-r * t)

        x, r_tree, Q, alpha = self.build_tree(T - t, N, r, P)

        # Calculate bond price
        payoff = np.ones(2 * N + 1)  # Payoff of 1 at maturity
        for i in range(N - 1, -1, -1):
            payoff = (
                payoff[:-2] * self._pu(i, N - i, dt, x[1, N + 1] - x[1, N])
                + payoff[1:-1] * self._pm(i, N - i, dt, x[1, N + 1] - x[1, N])
                + payoff[2:] * self._pd(i, N - i, dt, x[1, N + 1] - x[1, N])
            ) * np.exp(-r_tree[i, N - i : N + i + 1] * dt)

        return payoff[N]  # Price at the root node

    def calibrate(self, market_data):
        """
        Calibrate the Black-Karasinski model to market data.

        Parameters:
        market_data (dict): Market observed zero-coupon bond prices and yield curve

        Returns:
        dict: Calibrated model parameters
        """
        # Placeholder for calibration method
        print("Calibration method not implemented yet.")
        return {"a": self.a, "sigma": self.sigma}
