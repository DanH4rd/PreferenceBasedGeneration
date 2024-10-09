import abc
import torch
from src.DataStructures.AbsData import AbsData

class AbsLoss(object, metaclass=abc.ABCMeta):
    """
        Base class incupsulating the required logic for calculating loss
    """

    @abc.abstractmethod
    def CalculateLoss(self, data:AbsData) -> torch.tensor:
        """
            
            Calculate loss for the given data

            Params:
                X - [N, D] where N - number of actions, D - dim of actions 
                y - tensor of real labels

            Returns:
                tensor containing float value with grad tree
        """
        raise NotImplementedError('users must define CalculateLoss to use this base class')

    @abc.abstractmethod
    def __str__(self) -> str:
        """
            Returns string describing the object
        """
        raise NotImplementedError('users must define __str__ to use this base class')