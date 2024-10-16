import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.Loss.AbsLoss import AbsLoss


class ActionRewardLoss(AbsLoss):
    """
    Negative reward values for given actions. Requires a reference reward model
    """

    def __init__(self, rewardModel):
        """
        Params:
            rewardModel - pointer to the reward model
            logger - if provided, will use it to log the loss
        """

        self.rewardModel = rewardModel

    def calculate_loss(self, data: ActionData) -> torch.tensor:
        """
        Calculates the action reward loss for the given actions.

        Parametres:
            X - [B, D] tensor, B - batch size, D - action dim


        Check the abstract base class for more info.
        """

        returnToTrainMode = False

        if self.rewardModel.model.is_train_mode():
            self.rewardModel.model.set_to_evaluaion_mode()
            returnToTrainMode = True

        loss = self.rewardModel(data)

        if returnToTrainMode:
            self.rewardModel.model.set_to_train_mode()

        loss = -loss.mean()

        return loss

    def __str__(self) -> str:
        return "Action Reward Loss"
