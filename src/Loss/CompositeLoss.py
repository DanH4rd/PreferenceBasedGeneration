import torch

from src.Abstract.AbsData import AbsData
from src.Abstract.AbsLoss import AbsLoss


class CompositeLoss(AbsLoss):
    """Loss that is a composition of several other losses.
    Returns sum of losses
    """

    def __init__(self, losses: list[AbsLoss] = []):
        """
        Args:
            losses (list[AbsLoss]): loss objects of which the composite consists
        """

        self.losses = losses

    def add_loss(self, loss: AbsLoss | list[AbsLoss]) -> None:
        """Adds a loss to the composite elements list. Can accept a list
        of losses as a parametre, in this case it will concat
        the registered losses list with the passed loss list

        Args:
            loss (AbsLoss | list[AbsLoss]): loss object or a list of those to
                add to the composite loss elements
        """

        if isinstance(loss, list):
            self.losses += loss
        else:
            self.losses.append(loss)

    def calculate_loss(self, data: AbsData) -> torch.tensor:
        """Calculates the total sum of all composite losses.

        Args:
            data (AbsData): data to calculate loss for

        Returns:
            torch.tensor: sum of all calculated loss values for given data
        """
        if self.is_empty:
            raise Exception("No losses are present in composite loss")
        
        total_loss = torch.tensor(0)

        for loss in self.losses:
            total_loss += loss.CalculateLoss(data)

        return total_loss

    def is_empty(self):
        return len(self.losses) == 0
    
    def __str__(self) -> str:
        """Returns a string describing an onject

        Returns:
            str
        """
        return f"Composite loss. Number of members: {len(self.loggers)}"
