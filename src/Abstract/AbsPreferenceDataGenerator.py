import abc

import torch

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.DataStructures import ActionData, ActionPairsData, PreferencePairsData


class AbsPreferenceDataGenerator(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for generating
    preference data using feedback from a scecified source
    """

    feedbackSource: AbsFeedbackSource

    def generate_preference_data(
        self, data: ActionData, limit: int
    ) -> tuple[ActionPairsData, PreferencePairsData]:
        """Helper functions that creates action pairs from the given data and calls generate_preference_data_idx

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
        pair_idx_tensor, preference_data = self.generate_preference_data_idx(
            data, limit
        )

        if (pair_idx_tensor < 0).any():
            raise Exception(
                "Pair index tensor contains values less than 0, which is unexpected. Either check the implementation or write an overload with custom behaviour"
            )
        if (pair_idx_tensor >= len(data.actions)).any():
            raise Exception(
                f"Invalid pair_idx_tensor values: {pair_idx_tensor} for data of length {len(data.actions)}"
            )

        actions_tensor = data.actions
        action_pair_tensor = torch.stack(
            [
                actions_tensor[pair_idx_tensor[:, 0]],
                actions_tensor[pair_idx_tensor[:, 1]],
            ],
            dim=1,
        )
        action_pairs_data = ActionPairsData(action_pairs=action_pair_tensor)

        return action_pairs_data, preference_data

    @abc.abstractmethod
    def generate_preference_data_idx(
        self, data: ActionData, limit: int
    ) -> tuple[torch.Tensor, PreferencePairsData]:
        """Generates preference data for the provided data

        Args:
            data (ActionData): data to generate preferences for
            limit (int): maximum number of preferences the generator can
                ask the feedback source for preferences

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            tuple[Tensor, PreferencePairsData]: pairs of action data idx and
                preferences generated for the given data
        """

        raise NotImplementedError(
            "users must define generate_preference_data_idx to use this base class"
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
