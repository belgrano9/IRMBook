# tests/test_vasicek.py

import pytest
import numpy as np
from src.factories.model_factory import ModelFactory


def test_vasicek_model_creation():
    vasicek = ModelFactory.create_model(
        "Vasicek", r0=0.05, k=0.3, theta=0.05, sigma=0.02
    )
    assert isinstance(
        vasicek,
        ModelFactory.create_model(
            "Vasicek", r0=0.05, k=0.3, theta=0.05, sigma=0.02
        ).__class__,
    )


def test_vasicek_simulation():
    vasicek = ModelFactory.create_model(
        "Vasicek", r0=0.05, k=0.3, theta=0.05, sigma=0.02
    )
    simulated_rates = vasicek.simulate(T=1, N=252, n_paths=1000)
    assert simulated_rates.shape == (1000, 253)
    assert np.isclose(simulated_rates[:, 0], 0.05).all()


def test_vasicek_zero_coupon_bond_pricing():
    vasicek = ModelFactory.create_model(
        "Vasicek", r0=0.05, k=0.3, theta=0.05, sigma=0.02
    )
    bond_price = vasicek.zero_coupon_bond_price(T=2, r=0.05)
    assert 0 < bond_price < 1  # Bond price should be between 0 and 1


@pytest.mark.parametrize(
    "r0,k,theta,sigma",
    [
        (0.05, 0.3, 0.05, 0.02),
        (0.03, 0.5, 0.04, 0.01),
        (0.07, 0.2, 0.06, 0.03),
    ],
)
def test_vasicek_parameter_variations(r0, k, theta, sigma):
    vasicek = ModelFactory.create_model("Vasicek", r0=r0, k=k, theta=theta, sigma=sigma)
    simulated_rates = vasicek.simulate(T=1, N=252, n_paths=100)
    assert simulated_rates.shape == (100, 253)
    assert np.isclose(simulated_rates[:, 0], r0).all()
