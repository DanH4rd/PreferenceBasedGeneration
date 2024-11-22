import torch

from DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DiscModel.AbsDiscModel import AbsDiscModel
from src.GenModel.AbsGenModel import AbsGenModel
from src.Loss.AbsLoss import AbsLoss


class ActionDiscriminatorLoss(AbsLoss):
    """Negative GAN discriminator scores for images generated from
    actions given.
    """

    def __init__(self, genModel:AbsGenModel, discModel:AbsDiscModel):
        """
        Args:
            genModel (AbsGenModel): generator model object
            discModel (AbsDiscModel): discriminator model object
        """        

        self.genModel = genModel
        self.discModel = discModel

    def calculate_loss(self, data: ActionData) -> torch.tensor:
        """First generates images from the provided action list,
        then calculates discriminator scores for the given images.

        Args:
            data (ActionData): action list to get disc score for

        Returns:
            torch.tensor: mean of discrimination score values with grad attached
        """        
        loss = self.discModel.Discriminate(self.genModel.Generate(data.actions))

        loss = -loss.mean()

        return loss

    def __str__(self) -> str:
        """Returns a string describing an object

        Returns:
            str:
        """        
        return "Action Discriminator Loss"
