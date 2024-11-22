import abc

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData


class AbsActionFilter(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for filtering actions
    """

    @abc.abstractmethod
    def filter(self, data: ActionData) -> ActionData:
        """Filter actions based on implemented logic and, if set up, return
        limit elements of the result list

        Args:
            data (ActionData): list of actions to filter

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            ActionData: filtered list of actions
        """

        raise NotImplementedError("users must define filter to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
