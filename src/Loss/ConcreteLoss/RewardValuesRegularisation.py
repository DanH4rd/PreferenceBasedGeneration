import abc
import torch
from src.Loss.AbsLoss import AbsLoss
from DataStructures.ConcreteDataStructures.PairPreferenceData import PairPreferenceData

class RewardValuesRegularisation(AbsLoss):
    

    def __str__(self) -> str:
        """
            Returns string describing the object
        """
        return f"Reward Values Regularisator"