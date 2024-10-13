import abc

from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData


class AbsActionDistribution(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for action sample space
    """

    @abc.abstractmethod
    def Sample(self, N: int) -> ActionData:
        """
        Sample N actions from the distribution

        Params:
            N - number of actions to sample
        """
        raise NotImplementedError("users must define Sample to use this base class")

    @abc.abstractmethod
    def Update(self, data: AbsData) -> None:
        """
        Updates the distribution based on input data

        Params:
            data - data for update
        """
        raise NotImplementedError("users must define Update to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
