import abc

from src.DataStructures.AbsData import AbsData


class AbsDiscModel(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for discriminative models"""

    @abc.abstractmethod
    def discriminate(self, data: AbsData) -> None:
        """Generates a discriminate score for the given data

        Args:
            data (AbsData): data serving as input for disc model

        Raises:
            NotImplementedError: this method is abstract
        """
        raise NotImplementedError("users must define discriminate to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
