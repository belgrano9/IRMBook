import numpy as np


def zero_coupon_bond(rates, times, maturity):
    """
    Payoff function for a zero-coupon bond.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    maturity (float): Maturity time of the bond

    Returns:
    np.array: Payoff for each path
    """
    maturity_index = np.searchsorted(times, maturity)
    return np.ones(rates.shape[0])  # Pays 1 at maturity


def european_call_on_rate(rates, times, strike, maturity):
    """
    Payoff function for a European call option on the interest rate.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    strike (float): Strike rate of the option
    maturity (float): Maturity time of the option

    Returns:
    np.array: Payoff for each path
    """
    maturity_index = np.searchsorted(times, maturity)
    return np.maximum(rates[:, maturity_index] - strike, 0)


def european_put_on_rate(rates, times, strike, maturity):
    """
    Payoff function for a European put option on the interest rate.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    strike (float): Strike rate of the option
    maturity (float): Maturity time of the option

    Returns:
    np.array: Payoff for each path
    """
    maturity_index = np.searchsorted(times, maturity)
    return np.maximum(strike - rates[:, maturity_index], 0)


def asian_call_on_rate(rates, times, strike, start_time, end_time):
    """
    Payoff function for an Asian call option on the interest rate.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    strike (float): Strike rate of the option
    start_time (float): Start time for averaging
    end_time (float): End time for averaging (maturity)

    Returns:
    np.array: Payoff for each path
    """
    start_index = np.searchsorted(times, start_time)
    end_index = np.searchsorted(times, end_time)
    average_rates = np.mean(rates[:, start_index : end_index + 1], axis=1)
    return np.maximum(average_rates - strike, 0)


def cap(rates, times, strike, reset_times):
    """
    Payoff function for an interest rate cap.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    strike (float): Cap rate
    reset_times (list): List of reset times for the cap

    Returns:
    np.array: Payoff for each path
    """
    payoffs = np.zeros(rates.shape[0])
    for reset_time in reset_times:
        reset_index = np.searchsorted(times, reset_time)
        payoffs += np.maximum(rates[:, reset_index] - strike, 0)
    return payoffs


def floor(rates, times, strike, reset_times):
    """
    Payoff function for an interest rate floor.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    strike (float): Floor rate
    reset_times (list): List of reset times for the floor

    Returns:
    np.array: Payoff for each path
    """
    payoffs = np.zeros(rates.shape[0])
    for reset_time in reset_times:
        reset_index = np.searchsorted(times, reset_time)
        payoffs += np.maximum(strike - rates[:, reset_index], 0)
    return payoffs


def binary_call_on_rate(rates, times, strike, maturity):
    """
    Payoff function for a binary call option on the interest rate.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    strike (float): Strike rate of the option
    maturity (float): Maturity time of the option

    Returns:
    np.array: Payoff for each path (1 if in-the-money, 0 otherwise)
    """
    maturity_index = np.searchsorted(times, maturity)
    return (rates[:, maturity_index] > strike).astype(int)


def lookback_call_on_rate(rates, times, maturity):
    """
    Payoff function for a lookback call option on the interest rate.

    Args:
    rates (np.array): Simulated interest rate paths
    times (np.array): Time points for the simulation
    maturity (float): Maturity time of the option

    Returns:
    np.array: Payoff for each path
    """
    maturity_index = np.searchsorted(times, maturity)
    return rates[:, maturity_index] - np.min(rates[:, : maturity_index + 1], axis=1)
