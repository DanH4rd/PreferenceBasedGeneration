import abc
import torch
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData

class AbsActionFilter(object, metaclass=abc.ABCMeta):
    """
        Base class incupsulating the required logic for filtering actions
    """

    @abc.abstractmethod
    def Filter(self, data:ActionData) -> ActionData:
        """
            Filter actions based on implemented logic and, if set up, return
            limit elements of the result list

            Params:
                data - action data to filter
        """
        raise NotImplementedError('users must define Filter to use this base class')

    @abc.abstractmethod
    def __str__(self) -> str:
        """
            Returns string describing the object
        """
        raise NotImplementedError('users must define __str__ to use this base class')