from random import randint
from dataclasses import dataclass
import torch

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.DataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.PreferencePairsData import PreferencePairsData


class RandomFeedbackSource(AbsFeedbackSource):
    """Generates random feedback for provided action pairs"""

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres
        """
        pass

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return RandomFeedbackSource()
    possible_values = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]]

    def generate_feedback(
        self, action_pairs_data: ActionPairsData
    ) -> PreferencePairsData:
        """

        Args:
            action_pairs_data (ActionPairsData): action pairs for which
                we want to generate feedback

        Returns:
            PreferencePairsData: preferences for each action pair
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
        """Returns string describing the object

        Returns:
            str:
        """
        return "Random feedback"
