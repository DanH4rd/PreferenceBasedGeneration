from src.Loss.AbsLoss import AbsLoss
from src.DataStructures.AbsData import AbsData
import torch

class CompositeLoss(AbsLoss):
    """
        Loss that is a composition of several other losses
    """

    def __init__(self, losses:list[AbsLoss]):
        """
            Params:
                losses - list of losses out of which the composite consists of 
                logger - if provided, will use it to log the loss
        """

        self.losses = losses


    def AddLoss(self, loss:AbsLoss|list[AbsLoss]) -> None:
        """
            Adds a loss to the composite elements list. Can accept a list
            of losses as a parametre, in this case it will concat
            the registered losses list with the passed loss list

            Params:
                loss - AbsLoss object or a list of those
        """

        if isinstance(loss, list):
            self.losses += loss
        else:
            self.losses.append(loss)

    def CalculateLoss(self, data:AbsData) -> torch.tensor:
        """
            Calculates the total sum of all composite losses.


            Check the abstract base class for more info.
        """
        total_loss = torch.tensor(0)

        for loss in self.losses:
            total_loss += loss.CalculateLoss(data)
            
        return total_loss

    def __str__(self) -> str:
        return f"Composite loss. Number of members: {len(self.loggers)}"