import torch

from src.Abstract.AbsData import AbsData
from src.Abstract.AbsLoss import AbsLoss
from src.Abstract.AbsMetricsLogger import AbsMetricsLogger


class LogLossDecorator(AbsLoss):
    """A decorator class for loss objects to automatically
    log the calculated loss value
    """

    def __init__(self, lossObject: AbsLoss, logger: AbsMetricsLogger):
        """
        Args:
            lossObject (AbsLoss): loss object which calculated loss values to log
            logger (AbsMetricsLogger): metrics logger object that would perform the logging
                of the loss value
        """

        self.lossObject = lossObject
        self.logger = logger

    def calculate_loss(self, data: AbsData) -> torch.tensor:
        """Calculates loss using given loss object value and
        invoke given metrics logger to log the calculated loss

        Args:
            data (AbsData): data to calculate loss for

        Returns:
            torch.tensor: loss value with grad
        """

        loss = self.lossObject.calculate_loss(data=data)

        self.logger.log(loss)

        return loss

    def __str__(self) -> str:
        """Returns a string describing an onject

        Returns:
            str
        """
        return f"Log Loss Decorator of {str(self.lossObject)}"
