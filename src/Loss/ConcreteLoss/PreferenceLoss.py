import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.Loss.AbsLoss import AbsLoss
from src.RewardModel.AbsRewardModel import AbsRewardModel


class PreferenceLoss(AbsLoss):
    """Class than calculates cross entrpy loss for preference probabilities
    for action pairs. Extimates preference probabilities based on action 
    rewards
    """

    def __init__(self, rewardModel: AbsRewardModel, decimals: None | int = None):
        """
        Args:
            rewardModel (AbsRewardModel): reward model to get action rewards from
            decimals (None | int, optional): if not None will round preference values to the given 
                decimals number. Defaults to None.

        Raises:
            Exception: if given decimals value is less than 1 or not an integer
        """        

        self.decimals = decimals
        self.rewardModel = rewardModel

        if self.decimals is not None:
            if self.decimals < 1 or not isinstance(self.decimals, int):
                raise Exception(f"Invalid decimals value: {self.decimals}")

    def ConvertRewards2Preferences(self, r1: torch.tensor, r2: torch.tensor) -> torch.tensor:
        """Function that converts rewards pairs to preferences using SoftMax

        Args:
            r1 (torch.tensor): list of first elements in reward pairs ([B,1] tensor), 
                B - number of pairs
            r2 (torch.tensor): list of second elements in reward pairs ([B,1] tensor) 
                B - number of pairs

        Returns:
            torch.tensor: preference probabilities for the first elements in pairs
        """        

        answer = torch.exp(r1) / (torch.exp(r1) + torch.exp(r2))
        return answer

    def calculate_loss(self, data: ActionPairsPrefPairsContainer) -> torch.tensor:
        """Calculates Cross Entropy loss on preference probabilities for the given 
        action pairs and real preferences

        Args:
            data (ActionPairsPrefPairsContainer): object containing list of action
                pairs and corresponding list of real preference values, serving
                as ground truth labels

        Returns:
            torch.tensor: mean of cross entropy loss with a grad

        TODO:
            Calculate preference probs only for the first elements, then
            round it to decimals and then calculate preference probs
            for 2nd pair elements
        """        

        action_pairs_tensor = data.action_pairs_data.action_pairs
        pref_pairs_tensor = data.pref_pairs_data.preference_pairs

        rewards_left_column = self.rewardModel.get_stable_rewards(
            ActionData(actions=action_pairs_tensor[:, 0, :])
        ).squeeze(1)
        rewards_right_column = self.rewardModel.get_stable_rewards(
            ActionData(actions=action_pairs_tensor[:, 1, :])
        ).squeeze(1)

        preferences_left_column = self.ConvertRewards2Preferences(
            rewards_left_column, rewards_right_column
        )
        preferences_right_column = self.ConvertRewards2Preferences(
            rewards_right_column, rewards_left_column
        )

        if self.decimals is not None:
            preferences_left_column = torch.round(
                preferences_left_column, decimals=self.decimals
            )
            preferences_right_column = torch.round(
                preferences_right_column, decimals=self.decimals
            )

        loss = pref_pairs_tensor[..., 0] * torch.log(
            preferences_left_column
        ) + pref_pairs_tensor[..., 1] * torch.log(preferences_right_column)

        loss = -loss.mean()

        return loss

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """        
        return f"Preference loss. Round to decimals: {self.decimals}"
