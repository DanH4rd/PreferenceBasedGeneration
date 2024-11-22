from itertools import combinations

import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)
from src.FeedbackSource.AbsFeedbackSource import AbsFeedbackSource
from src.PreferenceDataGenerator.AbsPreferenceDataGenerator import (
    AbsPreferenceDataGenerator,
)


class RandomPreferenceDataGenerator(AbsPreferenceDataGenerator):
    """Base class incupsulating the required logic for generating random preferences"""

    def __init__(self, feedbackSource: AbsFeedbackSource):
        """
        Args:
            feedbackSource (AbsFeedbackSource): source to which to ask for preferences
                for action data
        """

        self.feedbackSource = feedbackSource

    def generate_preference_data(
        self, data: ActionData, limit: int
    ) -> tuple[ActionPairsData, PreferencePairsData]:
        """Creates all possible combinations out of provided actions and randomly shuffles them.
        Then asks for feebback for the first limit pairs.

        Args:
            data (ActionData): action list for which to generate preference data
            limit (int): number of action pairs to ask for feedback

        Returns:
            tuple[ActionPairsData, PreferencePairsData]: list of action pairs with corresponding preferences
        """

        actions_tensor = data.actions

        action_idx = list(range(len(actions_tensor)))
        pair_idx_tensor = torch.tensor(list(combinations(action_idx, 2)))
        shuffle_idx = torch.randperm(pair_idx_tensor.shape[0])
        pair_idx_tensor = pair_idx_tensor[shuffle_idx].view(pair_idx_tensor.size())
        pair_idx_tensor = pair_idx_tensor[:limit]

        action_pair_tensor = torch.stack(
            [
                actions_tensor[pair_idx_tensor[:, 0]],
                actions_tensor[pair_idx_tensor[:, 1]],
            ],
            dim=1,
        )

        action_pairs_data = ActionPairsData(action_pairs=action_pair_tensor)

        preference_data = self.feedbackSource.generate_feedback(action_pairs_data)

        return action_pairs_data, preference_data

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return "Random Feedback Manager"
