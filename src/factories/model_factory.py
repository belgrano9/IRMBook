"""
Contains the factory class for creating model instances.
"""

# src/factories/model_factory.py

from ..models.vasicek import VasicekModel
from ..models.cir import CIRModel
from ..models.hull_white import HullWhiteModel
from ..models.dothan import DothanModel


class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == "Vasicek":
            return VasicekModel(**kwargs)
        elif model_type == "CIR":
            return CIRModel(**kwargs)
        elif model_type == "HullWhite":
            return HullWhiteModel(**kwargs)
        elif model_type == "Dothan":
            return DothanModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
