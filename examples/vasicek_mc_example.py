""" examples/vasicek_mc_example.py

This example shows how to price an spot rate dependent option given the payoff using MonteCarlo method.

"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factories.model_factory import ModelFactory
import matplotlib.pyplot as plt
from src.models.montecarlo_pricer import MCPathDependentPricer
from src.utils.payoffs_mc import european_call_on_rate


vasicek = ModelFactory.create_model(
    "Vasicek", r0=0.05, k=0.1, theta=0.04, sigma=0.05, T=1, N=252, n_paths=100000
)

mc_pricer = MCPathDependentPricer(vasicek)


def payoff(rates, times):
    return european_call_on_rate(rates, times, strike=0.04, maturity=1.0)


price, error = mc_pricer.price_derivative(payoff, error=True)

print(f"Estimated price: {price:.4f} +- {error:.6f}")
