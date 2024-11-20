import abc

import torch

from src.DataStructures.AbsData import AbsData


class AbsLoss(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for calculating the loss value
    """

    @abc.abstractmethod
    def calculate_loss(self, data: AbsData) -> torch.tensor:
        """Calculate loss for the given data

        Args:
            data (AbsData): data to calculate loss for

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            torch.tensor: tensor tensor containing loss float value with grad tree
        """        

        raise NotImplementedError(
            "users must define calculate_loss to use this base class"
        )

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str: _description_
        """        
        raise NotImplementedError("users must define __str__ to use this base class")
