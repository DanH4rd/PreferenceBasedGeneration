import abc

from src.DataStructures.AbsData import AbsData


class AbsLoss(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for memory used during training
    """

    @abc.abstractmethod
    def add_data(self, data: AbsData) -> None:
        """Add new data to the memory

        Args:
            data (AbsData): data to add

        Raises:
            NotImplementedError: this method is abstract
        """        

        raise NotImplementedError("users must define add_data to use this base class")

    @abc.abstractmethod
    def get_data_from_memory(self) -> AbsData:
        """Returns data kept in memory

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            AbsData: data from memory
        """        

        raise NotImplementedError("users must define get_data_from_memory to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """        
        raise NotImplementedError("users must define __str__ to use this base class")
