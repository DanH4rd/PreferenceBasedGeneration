import abc
from random import randint

import torch

from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)


class RandomFeedbackSource(object, metaclass=abc.ABCMeta):
    """
    Generates random feedback
    """

    possible_values = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]]

    def generate_feedback(
        self, action_pairs_data: ActionPairsData
    ) -> PreferencePairsData:
        """
        Returns feedback for provided data
        """

        action_pairs_num = action_pairs_data.action_pairs.shape[0]
        preference_list = []
        for i in range(action_pairs_num):
            pref_id = randint(0, 3)
            preference_list.append(self.possible_values[pref_id])

        preference_tensor = torch.tensor(preference_list)
        preference_data = PreferencePairsData(preference_pairs=preference_tensor)

        return preference_data

    def __str__(self) -> str:
        return "Random feedback"
