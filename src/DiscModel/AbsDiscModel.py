import abc
from src.DataStructures.AbsData import AbsData
from src.Logger.AbsLogger import AbsLogger

class AbsDiscModel(object, metaclass=abc.ABCMeta):
    """
        Base class incupsulating the required logic for discriminative model
    """
    

    @abc.abstractmethod
    def Discriminate(self, data: AbsData) -> None:
        """
            Generates a discriminate score for the given data
        """
        raise NotImplementedError('users must define Generate to use this base class')

    @abc.abstractmethod
    def __str__(self) -> str:
        """
            Returns string describing the object
        """
        raise NotImplementedError('users must define __str__ to use this base class')