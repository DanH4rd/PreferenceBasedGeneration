import abc

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.DataStructures.ActionData import ActionData
from src.DataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.PreferencePairsData import PreferencePairsData


class AbsPreferenceDataGenerator(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for generating
    preference data using feedback from a scecified source
    """

    feedbackSource: AbsFeedbackSource

    @abc.abstractmethod
    def generate_preference_data(
        self, data: ActionData, limit: int
    ) -> tuple[ActionPairsData, PreferencePairsData]:
        """Generates preference data for the provided data

        Args:
            data (ActionData): data to generate preferences for
            limit (int): maximum number of preferences the generator can
                ask the feedback source for preferences

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            tuple[ActionPairsData, PreferencePairsData]: action pairs and
                preferences generated for the given data
        """

        raise NotImplementedError("users must define Filter to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
