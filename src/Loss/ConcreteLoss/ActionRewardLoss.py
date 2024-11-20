import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.Loss.AbsLoss import AbsLoss

from src.RewardModel.AbsRewardModel import AbsRewardModel


class ActionRewardLoss(AbsLoss):
    """Negative reward values for given actions. Requires a reference reward model.
    """

    def __init__(self, rewardModel:AbsRewardModel):
        """
        Args:
            rewardModel (AbsRewardModel): reward model object to get rewards from
        """        

        self.rewardModel = rewardModel

    def calculate_loss(self, data: ActionData) -> torch.tensor:
        """Calculates negative action rewards for the given actions and sums it

        Args:
            data (ActionData): list of actions to calculate loss for

        Returns:
            torch.tensor: mean of negative action rewards values with grad attached
        """        

        loss = self.rewardModel.get_stable_rewards(data)

        loss = -loss.mean()

        return loss

    def __str__(self) -> str:
        return "Action Reward Loss"
