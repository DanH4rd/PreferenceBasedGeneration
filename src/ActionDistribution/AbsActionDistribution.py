import abc

from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData


class AbsActionDistribution(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for action sample space
    """

    @abc.abstractmethod
    def sample(self, N: int) -> ActionData:
        """Sample N actions from the distribution

        Args:
            N (int): number of actions to sample

        Raises:
            NotImplementedError

        Returns:
            ActionData: a data object
        """
        raise NotImplementedError("users must define sample to use this base class")

    @abc.abstractmethod
    def update(self, data: AbsData) -> None:
        """Updates the distribution based on input data

        Args:
            data (AbsData): data for update

        Raises:
            NotImplementedError
        """

        raise NotImplementedError("users must define update to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError:

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
