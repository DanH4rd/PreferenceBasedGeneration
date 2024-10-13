import torch

from src.DataStructures.AbsData import AbsData
from src.Loss.AbsLoss import AbsLoss
from src.MetricsLogger.AbsMetricsLogger import AbsMetricsLogger


class LogLossDecorator(AbsLoss):
    """
    A decorator class, which applied, will additionally log the loss
    value using a provided logger
    """

    def __init__(self, lossObject: AbsLoss, logger: AbsMetricsLogger):
        """
        Params:
            genModel - generator model object
            logger - if provided, will use it to log the loss
        """

        self.lossObject = lossObject
        self.logger = logger

    def calculate_loss(self, data: AbsData) -> torch.tensor:
        """
        Calculates the loss and logs its value.

        Parametres:
            X - [B, D] tensor, B - batch size, D - action dim


        Check the abstract base class for more info.
        """

        loss = self.lossObject.calculate_loss(data=data)

        self.logger.log(loss)

        return loss

    def __str__(self) -> str:
        return f"Log Loss Decorator of {str(self.lossObject)}"
