from typing import Any

import torch

from src.Abstract.AbsData import AbsData
from src.Abstract.AbsLoss import AbsLoss


class CompositeLoss(AbsLoss[AbsData]):
    """Loss that is a composition of several other losses.
    Returns sum of losses
    """

    def __init__(self, losses: list[AbsLoss[Any]] = []):
        """
        Args:
            losses (list[AbsLoss[Any]]): loss objects of which the composite consists.
                Each must accept the same data type passed to calculate_loss.
        """

        self.losses = losses

    def add_loss(self, loss: AbsLoss[Any] | list[AbsLoss[Any]]) -> None:
        """Adds a loss to the composite elements list. Can accept a list
        of losses as a parametre, in this case it will concat
        the registered losses list with the passed loss list

        Args:
            loss (AbsLoss[Any] | list[AbsLoss[Any]]): loss object or a list of those to
                add to the composite loss elements
        """

        if isinstance(loss, list):
            self.losses += loss
        else:
            self.losses.append(loss)

    def calculate_loss(self, data: AbsData) -> torch.Tensor:
        """Calculates the total sum of all composite losses.

        Args:
            data (AbsData): data to calculate loss for

        Returns:
            torch.Tensor: sum of all calculated loss values for given data
        """
        if self.is_empty():
            raise Exception("No losses are present in composite loss")

        total_loss = torch.tensor(0)

        for loss in self.losses:
            total_loss += loss.calculate_loss(data)

        return total_loss

    def is_empty(self):
        return len(self.losses) == 0

    def __str__(self) -> str:
        """Returns a string describing an onject

        Returns:
            str
        """
        return f"Composite loss. Number of members: {len(self.losses)}"
