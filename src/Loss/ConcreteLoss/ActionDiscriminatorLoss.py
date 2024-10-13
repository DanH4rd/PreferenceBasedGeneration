import torch

from DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.Loss.AbsLoss import AbsLoss


class ActionDiscriminatorLoss(AbsLoss):
    """
    Negative discriminator scores for images generated from given actions.
    Requires a gen model object
    """

    def __init__(self, genModel, discModel):
        """
        Params:
            genModel - generator model object
            discModel - discriminator model object
            logger - if provided, will use it to log the loss
        """

        self.genModel = genModel
        self.discModel = discModel

    def CalculateLoss(self, data: ActionData) -> torch.tensor:
        """
        Calculates the action disriminator loss for the given actions.

        Parametres:
            X - [B, D] tensor, B - batch size, D - action dim


        Check the abstract base class for more info.
        """
        loss = self.discModel.Discriminate(self.genModel.Generate(data.actions))

        loss = -loss.mean()

        return loss

    def __str__(self) -> str:
        return "Action Reward Loss"
