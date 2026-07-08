import torch

from src.Abstract.AbsLoss import AbsLoss
from src.DataStructures import ActionData
from src.DiscModel import StackGanDiscModel
from src.GenModel import StackGanGenModel


class ActionDiscriminatorLoss(AbsLoss[ActionData]):
    """Negative GAN discriminator scores for images generated from
    actions given.
    """

    def __init__(self, genModel: StackGanGenModel, discModel: StackGanDiscModel):
        """
        Args:
            genModel (StackGanGenModel): generator model object
            discModel (StackGanDiscModel): discriminator model object
        """

        self.genModel = genModel
        self.discModel = discModel

    def calculate_loss(self, data: ActionData) -> torch.Tensor:
        """First generates images from the provided action list,
        then calculates discriminator scores for the given images.

        Args:
            data (ActionData): action list to get disc score for

        Returns:
            torch.Tensor: mean of discrimination score values with grad attached
        """
        loss = self.discModel.discriminate(self.genModel.generate(data))

        loss = -loss.mean()

        return loss

    def __str__(self) -> str:
        """Returns a string describing an object

        Returns:
            str:
        """
        return "Action Discriminator Loss"
