# In models/monte_carlo_pricer.py

import numpy as np
from .base import InterestRateModel


# src/models/monte_carlo_pricer.py

from .base import InterestRateModel


class MCPathDependentPricer:
    def __init__(self, rate_model: InterestRateModel):
        self.rate_model = rate_model

    def price_derivative(self, payoff_func, error=False):
        return self.rate_model.mc_path_dependent(payoff_func, error)
