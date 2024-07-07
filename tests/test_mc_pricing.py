import pytest
import numpy as np
from src.factories.model_factory import ModelFactory


@pytest.fixture
def pricer():
    # Set up parameters for the test
    T = 1.0  # Maturity
    r0 = 0.05  # Initial interest rate
    kappa = 0.1  # Mean reversion speed
    theta = 0.05  # Long-term mean
    sigma = 0.02  # Volatility
    N = 252  # Number of time steps
    n_path = 1
    return ModelFactory.create_model(
        "Vasicek", T=T, r0=r0, k=kappa, theta=theta, sigma=sigma, N=N, n_paths=n_path
    )


def test_convergence(pricer):
    # Test if the price converges as the number of simulations increases
    prices = []
    sim_counts = [1000, 5000, 10000, 50000]

    for sim_count in sim_counts:
        pricer.n_path = sim_count  # Set number of simulations
        price = pricer.mc_path_dependent(error=False)
        prices.append(price)

    # Check if the prices are converging
    diffs = np.diff(prices)
    assert np.all(np.abs(diffs) < 0.01), "Prices are not converging as expected"


def test_price_range(pricer):
    # Test if the price is within a reasonable range
    pricer.n_path = 10000  # Number of simulations

    price = pricer.mc_path_dependent(error=False)

    # The price should be positive and less than some reasonable upper bound
    assert 0 < price < 0.1, f"Price {price} is outside the expected range"


def test_benchmark_comparison(pricer):
    # Compare with a benchmark price (if available)
    # Note: This is a placeholder. You need to replace the benchmark_price
    # with a value from a trusted source or another pricing method
    benchmark_price = 0.0234  # Replace with actual benchmark
    tolerance = 0.002  # 0.2% tolerance

    pricer.n_path = 100000  # High number of simulations for accuracy

    price = pricer.mc_path_dependent(error=False)

    assert (
        abs(price - benchmark_price) < tolerance
    ), f"Price {price} is not close enough to benchmark {benchmark_price}"


def test_error_calculation(pricer):
    # Test if the error calculation is working correctly
    pricer.n_path = 10000  # Number of simulations

    result = pricer.mc_path_dependent(error=True)

    # Assuming mc_path_dependent returns a tuple (price, error) when error=True
    assert isinstance(result, tuple), "Result should be a tuple when error=True"
    assert len(result) == 2, "Result should contain price and error"
    price, error = result
    assert isinstance(price, float), "Price should be a float"
    assert isinstance(error, float), "Error should be a float"
    assert error > 0, "Error should be positive"


def test_n_path_effect(pricer):
    # Test if increasing n_path reduces the error
    pricer.n_path = 1000
    _, error1 = pricer.mc_path_dependent(error=True)

    pricer.n_path = 10000
    _, error2 = pricer.mc_path_dependent(error=True)

    assert error2 < error1, "Error should decrease with more paths"
