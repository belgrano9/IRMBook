"""
Contains the factory class for creating model instances.
"""
# src/factories/model_factory.py

from ..models.vasicek import VasicekModel
from ..models.cir import CIRModel
from ..models.hull_white import HullWhiteModel
from ..models.dothan import DothanModel
from ..models.extended_vasicek import ExtendedVasicekModel
from ..models.hjm import HJMModel 


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
        elif model_type == "Extended Vasicek":
            return ExtendedVasicekModel(**kwargs)
        elif model_type == "HJM":
            return HJMModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
