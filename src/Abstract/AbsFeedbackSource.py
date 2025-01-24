import abc

from src.Abstract.AbsData import AbsData


class AbsFeedbackSource(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for generating feedback
    """

    @abc.abstractmethod
    def generate_feedback(self, data: AbsData) -> AbsData:
        """Returns feedback for provided data

        Args:
            data (AbsData): data for which we want to get feedback values

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            AbsData: feedback data
        """
        raise NotImplementedError(
            "users must define generate_feedback to use this base class"
        )

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
