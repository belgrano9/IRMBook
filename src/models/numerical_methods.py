# src/models/numerical_methods.py

from abc import ABC, abstractmethod


class NumericalMethod(ABC):
    @abstractmethod
    def build_tree(self, model, T, N):
        pass

    @abstractmethod
    def price_zero_coupon_bond(self, model, T):
        pass

    @abstractmethod
    def price_european_option(self, model, K, T, option_type="call"):
        pass
