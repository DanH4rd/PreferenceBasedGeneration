import abc


class AbsData(object, metaclass=abc.ABCMeta):
    """
    Base class for data containers
    """

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
