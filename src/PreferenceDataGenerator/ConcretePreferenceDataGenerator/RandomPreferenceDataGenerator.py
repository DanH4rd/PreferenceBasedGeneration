import abc
from src.PreferenceDataGenerator.AbsPreferenceDataGenerator import AbsPreferenceDataGenerator
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
import math
import torch
from itertools import combinations
from src.DataStructures.ConcreteDataStructures.PairPreferenceData import PairPreferenceData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.FeedbackSource.AbsFeedbackSource import AbsFeedbackSource


class RandomPreferenceDataGenerator(AbsPreferenceDataGenerator):
    """
        Base class incupsulating the required logic for generating random pairs
    """

    def __init__(self, feedbackSource:AbsFeedbackSource):
        """
            Params:
                actions = ActionData object containing actions, out of which 
                pairs will be constructed
        """

        self.feedbackSource = feedbackSource

    def GeneratePreferenceData(self, data:ActionData, limit:int) -> tuple[ActionPairsData, PairPreferenceData]:
        """
            Creates all possible combinations out of provided actioms and randomly shuffles them.
            Then asks for feebback for the first limit pairs.
        """
        actions_tensor = data.actions

        action_idx = list(range(len(actions_tensor)))
        pair_idx_tensor = torch.tensor(list(combinations(action_idx, 2)))
        shuffle_idx = torch.randperm(pair_idx_tensor.shape[0])
        pair_idx_tensor = pair_idx_tensor[shuffle_idx].view(pair_idx_tensor.size())
        pair_idx_tensor = pair_idx_tensor[:limit]

        action_pair_tensor = torch.stack([actions_tensor[pair_idx_tensor[:, 0]],
                                            actions_tensor[pair_idx_tensor[:, 1]]],
                                            dim = 1)

        action_pairs_data = ActionPairsData(actions_pairs=action_pair_tensor)

        preference_data = self.feedbackSource.GenerateFeedback(action_pairs_data)
        
        return action_pairs_data, preference_data

    def __str__(self) -> str:
        return 'Random Feedback Manager'