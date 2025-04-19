# src/models/hjm.py

import numpy as np
from .base import InterestRateModel


class HJMModel(InterestRateModel):
    def __init__(self, initial_forward_curve, volatility_functions, maturities, T=None, N=None, n_paths=None):
        """
        Initialize the Heath-Jarrow-Morton model.

        Parameters:
        initial_forward_curve (np.array): Initial forward rate curve values at specified maturities
        volatility_functions (list): List of volatility functions for each factor
        maturities (np.array): Array of maturities for which the initial forward rates are defined
        T (float): Time horizon for simulation, optional
        N (int): Number of time steps, optional
        n_paths (int): Number of simulation paths, optional
        """
        self.initial_forward_curve = initial_forward_curve
        self.volatility_functions = volatility_functions
        self.maturities = maturities
        self.n_factors = len(volatility_functions)
        self.T = T
        self.N = N
        self.n_paths = n_paths
        
    def simulate(self, T=None, N=None, n_paths=None):
        """
        Simulate forward rate curves using the HJM model.

        Parameters:
        T (float): Time horizon (optional, uses instance value if not provided)
        N (int): Number of time steps (optional, uses instance value if not provided)
        n_paths (int): Number of paths to simulate (optional, uses instance value if not provided)

        Returns:
        tuple: (times, maturities, forward_rates) where forward_rates has shape (n_paths, len(times), len(maturities))
        """
        # Use provided parameters or fall back to instance variables
        T = T if T is not None else self.T
        N = N if N is not None else self.N
        n_paths = n_paths if n_paths is not None else self.n_paths
        
        if T is None or N is None or n_paths is None:
            raise ValueError("Simulation parameters T, N, and n_paths must be provided")
        
        dt = T / N
        times = np.linspace(0, T, N+1)
        
        # Initialize forward rates array (paths, time steps, maturities)
        forward_rates = np.zeros((n_paths, N+1, len(self.maturities)))
        
        # Set initial forward curve for all paths
        forward_rates[:, 0, :] = self.initial_forward_curve
        
        # Simulate forward rate paths
        for i in range(1, N+1):
            t = times[i-1]
            
            # Generate random normal variables for each factor
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, self.n_factors))
            
            # Calculate drift adjustment for no-arbitrage (HJM drift condition)
            drift_adjustment = self._calculate_drift_adjustment(t, forward_rates[:, i-1, :], dt)
            
            # Calculate volatility effect for each factor
            vol_effect = np.zeros((n_paths, len(self.maturities)))
            for j in range(self.n_factors):
                vol_values = self._apply_volatility_function(j, t, forward_rates[:, i-1, :])
                vol_effect += vol_values * dW[:, j:j+1]
            
            # Update forward rates
            forward_rates[:, i, :] = forward_rates[:, i-1, :] + drift_adjustment + vol_effect
        
        return times, self.maturities, forward_rates
    
    def _calculate_drift_adjustment(self, t, current_forward_rates, dt):
        """
        Calculate the drift adjustment based on the HJM no-arbitrage condition.
        
        Parameters:
        t (float): Current time
        current_forward_rates (np.array): Current forward rates with shape (n_paths, n_maturities)
        dt (float): Time step size
        
        Returns:
        np.array: Drift adjustment with shape (n_paths, n_maturities)
        """
        n_paths, n_maturities = current_forward_rates.shape
        drift = np.zeros((n_paths, n_maturities))
        
        # Calculate drift based on the HJM drift condition
        for j in range(self.n_factors):
            vol_values = self._apply_volatility_function(j, t, current_forward_rates)
            vol_integrals = self._calculate_volatility_integrals(j, t, self.maturities)
            drift += vol_values * vol_integrals
            
        return drift * dt
    
    def _apply_volatility_function(self, factor_idx, t, forward_rates):
        """
        Apply the volatility function for a specific factor.
        
        Parameters:
        factor_idx (int): Index of the factor
        t (float): Current time
        forward_rates (np.array): Current forward rates
        
        Returns:
        np.array: Volatility values for each path and maturity
        """
        vol_func = self.volatility_functions[factor_idx]
        return vol_func(t, forward_rates, self.maturities)
    
    def _calculate_volatility_integrals(self, factor_idx, t, maturities):
        """
        Calculate the integrals of volatility functions needed for drift calculation.
        
        Parameters:
        factor_idx (int): Index of the factor
        t (float): Current time
        maturities (np.array): Array of maturities
        
        Returns:
        np.array: Integrated volatility values
        """
        # This is a simplified implementation
        # In practice, this would involve numerical integration
        vol_func = self.volatility_functions[factor_idx]
        # Placeholder for actual integration
        return np.ones_like(maturities) * 0.5  # Example placeholder
    
    def zero_coupon_bond_price(self, T, r=None, t=0):
        """
        Calculate the price of a zero-coupon bond using the HJM model.
        
        Parameters:
        T (float): Time to maturity
        r (float): Not used in HJM, included for API compatibility
        t (float): Current time
        
        Returns:
        float or np.array: Price(s) of the zero-coupon bond
        """
        if t >= T:
            return np.ones(self.n_paths) if self.n_paths else 1.0
        
        # Find the closest simulated maturity
        if hasattr(self, 'simulated_times') and hasattr(self, 'simulated_forwards'):
            t_idx = np.searchsorted(self.simulated_times, t)
            T_idx = np.searchsorted(self.maturities, T - t)
            
            # If we have simulated data, use it to calculate bond prices
            if t_idx < len(self.simulated_times) and T_idx < len(self.maturities):
                forward_rates = self.simulated_forwards[:, t_idx, :T_idx]
                dt = self.maturities[1] - self.maturities[0]  # Assuming uniform spacing
                
                # Approximate the integral of forward rates for each path
                integral = np.sum(forward_rates * dt, axis=1)
                return np.exp(-integral)
        
        # If we don't have simulated data or the requested time/maturity is out of range
        # We use the initial forward curve
        idx = np.searchsorted(self.maturities, T - t)
        if idx >= len(self.maturities):
            idx = len(self.maturities) - 1
            
        # Approximate the integral using the initial forward curve
        forward_rates = self.initial_forward_curve[:idx]
        dt = self.maturities[1] - self.maturities[0]  # Assuming uniform spacing
        integral = np.sum(forward_rates * dt)
        
        return np.exp(-integral)
    
    def calibrate(self, market_data):
        """
        Calibrate the HJM model to market data.
        
        Parameters:
        market_data (dict): Market observed zero-coupon bond prices and yield curve
        
        Returns:
        dict: Calibrated model parameters
        """
        # Placeholder for actual calibration method
        print("HJM calibration method not implemented yet.")
        return {"status": "not_implemented"}
    
    def mc_path_dependent(self, payoff_func, error=False):
        """
        Price a derivative using Monte Carlo simulation.
        
        Parameters:
        payoff_func (callable): Function that calculates the payoff given simulated rates
        error (bool): Whether to return the standard error
        
        Returns:
        float or tuple: The price or (price, error) if error=True
        """
        if not hasattr(self, 'simulated_times') or not hasattr(self, 'simulated_forwards'):
            # Run simulation if not already done
            self.simulated_times, _, self.simulated_forwards = self.simulate()
        
        # Extract short rates from forward curves (approximation)
        short_rates = self.simulated_forwards[:, :, 0]
        
        # Calculate payoffs
        payoffs = payoff_func(short_rates, self.simulated_times)
        
        # Calculate discount factors
        discount_factors = self.compute_discount_factor(short_rates)
        
        # Discount payoffs
        discounted_payoffs = discount_factors * payoffs
        
        # Calculate price and error
        price = np.mean(discounted_payoffs)
        
        if error:
            std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
            return price, 1.96 * std_error
        else:
            return price
    
    def compute_discount_factor(self, rates):
        """
        Compute discount factors from simulated rates.
        
        Parameters:
        rates (np.array): Simulated interest rate paths
        
        Returns:
        np.array: Discount factors for each path
        """
        if rates.ndim == 1:
            return self._df_loop(rates)
        else:
            return np.apply_along_axis(self._df_loop, 1, rates)
    
    def _df_loop(self, rates):
        """
        Helper function to compute discount factor for a single path.
        
        Parameters:
        rates (np.array): Interest rates for a single path
        
        Returns:
        float: Discount factor
        """
        if hasattr(self, 'simulated_times'):
            dt = np.diff(self.simulated_times)
            integral_sum = np.sum(0.5 * (rates[:-1] + rates[1:]) * dt)
            return np.exp(-integral_sum)
        else:
            # Fallback if simulation hasn't been run
            dt = self.T / self.N
            integral_sum = np.sum(0.5 * (rates[:-1] + rates[1:]) * dt)
            return np.exp(-integral_sum)