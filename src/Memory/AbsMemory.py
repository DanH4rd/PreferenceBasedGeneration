import abc


from src.DataStructures.AbsData import AbsData


class AbsLoss(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for memory used in training
    """

    @abc.abstractmethod
    def AddData(self, data: AbsData) -> None:
        """

        Add new data to the memory
        Params:
            data - data to add
        """
        raise NotImplementedError("users must define AddData to use this base class")

    @abc.abstractmethod
    def GetMemoryData(self) -> AbsData:
        """
        Returns data, contained in memory

        Returns:
            AbsData object
        """
        raise NotImplementedError("users must define AddData to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
