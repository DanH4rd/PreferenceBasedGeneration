import abc

from src.DataStructures.AbsData import AbsData


class AbsFeedbackSource(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for generating feedback
    """

    @abc.abstractmethod
    def generate_feedback(self, data: AbsData) -> AbsData:
        """
        Returns feedback for provided data
        """
        raise NotImplementedError("users must define Filter to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
