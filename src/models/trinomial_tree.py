# src/models/trinomial_tree.py

import numpy as np
from .numerical_methods import NumericalMethod


class TrinomialTree(NumericalMethod):
    def build_tree(self, model, T, N):
        dt = T / N
        r_tree = np.zeros((N + 1, 2 * N + 1))
        r_tree[0, N] = model.r0

        # Implement trinomial tree construction logic here
        # This will depend on the specific model you're using

        return r_tree

    def price_zero_coupon_bond(self, model, T):
        N = 100  # Number of time steps, can be adjusted
        tree = self.build_tree(model, T, N)

        # Implement backward induction to price the bond
        # Return the bond price

    def price_european_option(self, model, K, T, option_type="call"):
        N = 100  # Number of time steps, can be adjusted
        tree = self.build_tree(model, T, N)

        # Implement backward induction to price the option
        # Return the option price
