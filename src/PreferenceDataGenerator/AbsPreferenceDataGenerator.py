import abc


from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.PairPreferenceData import (
    PairPreferenceData,
)


class AbsPreferenceDataGenerator(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for constructing
    preference data basing on feedback
    """

    @abc.abstractmethod
    def GeneratePreferenceData(
        self, data: ActionData, limit: int
    ) -> PairPreferenceData:
        """
        Generates preference data for the provided actions
        """
        raise NotImplementedError("users must define Filter to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
