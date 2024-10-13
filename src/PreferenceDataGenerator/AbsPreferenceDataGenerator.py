import abc

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)


class AbsPreferenceDataGenerator(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for constructing
    preference data basing on feedback
    """

    @abc.abstractmethod
    def generate_preference_data(
        self, data: ActionData, limit: int
    ) -> PreferencePairsData:
        """
        Generates preference data for the provided action data
        """
        raise NotImplementedError("users must define Filter to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
