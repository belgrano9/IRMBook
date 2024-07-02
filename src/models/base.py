"""
Contains abstract base classes for all models.
"""

from abc import ABC, abstractmethod


class InterestRateModel(ABC):
    @abstractmethod
    def simulate(self, *args, **kwargs):
        pass

    @abstractmethod
    def zero_coupon_bond_price(self, *args, **kwargs):
        pass

    @abstractmethod
    def calibrate(self, *args, **kwargs):
        pass
